import { rmSync } from "node:fs";
import { join } from "node:path";
import { defaultConfig, ModelConfig } from "../config/newModelConfig";
import { LearningNetwork } from "../core/evaluation";
import { loadNetworkFromExport } from "../export/networkLoader";
import { createNetworkExport, NetworkExport, readNetworkExport, writeNetworkExport } from "../export/networkExport";
import {
  blankChallengeScenario,
  ChallengeExperimentResult,
  ChallengeLearningMode,
  conflictChallengeScenario,
  createChallengeConfig,
  DEFAULT_EVAL_SEEDS,
  DEFAULT_TRAIN_SEEDS,
  runChallengeExperiment,
  sameActionCompositeChallengeScenario
} from "./challenge2d";

export interface TransferAuditReport {
  version: string;
  generatedAt: string;
  requiredPassed: boolean;
  summary: string;
  suites: TransferAuditSuiteResult[];
  cell?: TransferCellSummary;
}

export interface TransferAuditSuiteResult {
  name: string;
  required: boolean;
  passed: boolean;
  metrics: Record<string, number | string | boolean>;
  conclusion: string;
  notes: string[];
}

export interface TransferCellConfig {
  pretrainSeed: number;
  evalSeeds: number[];
  auditSeed: number;
  tmpDir: string;
  label: string;
}

export interface TransferCellSummary {
  pretrainSeed: number;
  evalSeeds: number[];
  auditSeed: number;
  label: string;
}

export interface TransferAuditOptions {
  cell?: Partial<TransferCellConfig>;
}

const TRANSFER_VERSION = "dg-snn-transfer-audit-v0.1";
export const DEFAULT_TRANSFER_EVAL_SEEDS = [201, 202, 203, 204, 205];
export const DEFAULT_TRANSFER_AUDIT_SEED = 211;
export const DEFAULT_PRETRAIN_SEED = 101;
const PRETRAIN_EPOCHS = 40;
const REQUIRED_SUPERVISED_TRANSFER_SUCCESS = 0.5;
const REQUIRED_SUPERVISED_TRANSFER_SEPARATION = 0.3;
const REQUIRED_LOADER_FIDELITY_SUCCESS = 0.8;
const DROPOUT_EVAL_MAX_STEPS = 4;
const CONTINUED_LEARNING_EPOCHS = 1;
const CONTINUED_LEARNING_TRAIN_SEEDS = [1];
const WRONG_PRIOR_CONTINUED_EPOCHS = 1;

const DEFAULT_CELL: TransferCellConfig = {
  pretrainSeed: DEFAULT_PRETRAIN_SEED,
  evalSeeds: DEFAULT_TRANSFER_EVAL_SEEDS,
  auditSeed: DEFAULT_TRANSFER_AUDIT_SEED,
  tmpDir: join("exports", "pretrained", "transfer-audit-tmp"),
  label: "default"
};

interface PretrainArtifact {
  result: ChallengeExperimentResult;
  snapshot: NetworkExport;
}

export function runTransferAudit(
  config: ModelConfig = defaultConfig,
  options: TransferAuditOptions = {}
): TransferAuditReport {
  const challengeConfig = createChallengeConfig(config);
  const cell: TransferCellConfig = { ...DEFAULT_CELL, ...options.cell };

  const supervised = pretrain(challengeConfig, "supervised", cell.pretrainSeed);
  const rewardOnly = pretrain(challengeConfig, "rewardOnly", cell.pretrainSeed);

  const supervisedPath = join(cell.tmpDir, "supervised-roundtrip.json");
  const rewardOnlyPath = join(cell.tmpDir, "rewardOnly-roundtrip.json");

  writeNetworkExport(supervisedPath, supervised.snapshot);
  writeNetworkExport(rewardOnlyPath, rewardOnly.snapshot);
  const supervisedReloaded = readNetworkExport(supervisedPath);
  const rewardOnlyReloaded = readNetworkExport(rewardOnlyPath);

  try {
    const suites = [
      auditLoaderRoundTrip(challengeConfig, supervised, supervisedReloaded, rewardOnly, rewardOnlyReloaded, cell),
      auditFrozenSeparation(challengeConfig, supervisedReloaded, "supervised", cell),
      auditFrozenSeparation(challengeConfig, rewardOnlyReloaded, "rewardOnly", cell),
      auditSeedIsolation(cell),
      auditConflictBoundary(challengeConfig, supervisedReloaded, cell),
      auditBlankWorld(challengeConfig, supervisedReloaded, cell),
      auditDropoutDiagnostic(challengeConfig, supervisedReloaded, rewardOnlyReloaded, 0.2, cell),
      auditDropoutDiagnostic(challengeConfig, supervisedReloaded, rewardOnlyReloaded, 0.3, cell),
      auditMultiObjectDiagnostic(challengeConfig, supervisedReloaded, cell),
      auditContinuedLearningDiagnostic(challengeConfig, supervisedReloaded, cell),
      auditWrongPriorDiagnostic(challengeConfig, cell)
    ];

    const requiredPassed = suites.filter((suite) => suite.required).every((suite) => suite.passed);

    return {
      version: TRANSFER_VERSION,
      generatedAt: new Date().toISOString(),
      requiredPassed,
      summary: requiredPassed
        ? "Required transfer checks passed; pretrained snapshots load faithfully and provide a measurable frozen-eval advantage over fresh models on held-out seeds."
        : "At least one required transfer check failed; pretrained snapshots do not yet demonstrate a reproducible initial advantage.",
      suites,
      cell: {
        pretrainSeed: cell.pretrainSeed,
        evalSeeds: [...cell.evalSeeds],
        auditSeed: cell.auditSeed,
        label: cell.label
      }
    };
  } finally {
    rmSync(cell.tmpDir, { recursive: true, force: true });
  }
}

export function formatTransferAuditReport(report: TransferAuditReport): string {
  const lines = [
    `Audit ${report.version}`,
    `requiredPassed=${report.requiredPassed}`,
    report.summary
  ];

  for (const suite of report.suites) {
    lines.push("");
    lines.push(`${suite.passed ? "PASS" : "FAIL"} ${suite.required ? "REQUIRED" : "DIAGNOSTIC"} ${suite.name}`);
    lines.push(`  metrics: ${JSON.stringify(suite.metrics)}`);
    lines.push(`  conclusion: ${suite.conclusion}`);
    for (const note of suite.notes) {
      lines.push(`  note: ${note}`);
    }
  }

  return lines.join("\n");
}

function pretrain(
  config: ModelConfig,
  mode: Extract<ChallengeLearningMode, "supervised" | "rewardOnly">,
  pretrainSeed: number
): PretrainArtifact {
  const result = runChallengeExperiment(config, {
    seed: pretrainSeed,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: PRETRAIN_EPOCHS,
    learningMode: mode
  });

  const snapshot = createNetworkExport({
    seed: pretrainSeed,
    config,
    neurons: result.network.neurons,
    synapses: result.network.synapses,
    pairMemory: result.network.pairMemory,
    metrics: {
      pretrainKind: "2d-challenge-transfer-audit",
      learningMode: mode,
      successRate: result.successRate,
      meanReward: result.meanReward
    }
  });

  return { result, snapshot };
}

function runFrozenEval(config: ModelConfig, network: LearningNetwork, cell: TransferCellConfig): ChallengeExperimentResult {
  return runChallengeExperiment(config, {
    seed: cell.auditSeed,
    trainSeeds: [],
    evalSeeds: cell.evalSeeds,
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: network
  });
}

function runFreshFrozenEval(config: ModelConfig, cell: TransferCellConfig): ChallengeExperimentResult {
  return runChallengeExperiment(config, {
    seed: cell.auditSeed,
    trainSeeds: [],
    evalSeeds: cell.evalSeeds,
    epochs: 0,
    learningMode: "frozen"
  });
}

function auditLoaderRoundTrip(
  config: ModelConfig,
  supervisedPretrain: PretrainArtifact,
  supervisedReloaded: NetworkExport,
  rewardOnlyPretrain: PretrainArtifact,
  rewardOnlyReloaded: NetworkExport,
  cell: TransferCellConfig
): TransferAuditSuiteResult {
  const supervisedBefore = runFrozenEval(config, supervisedPretrain.result.network, cell);
  const supervisedAfter = loadNetworkFromExport(supervisedReloaded);
  const supervisedAfterResult = runFrozenEval(config, supervisedAfter.network, cell);

  const rewardOnlyBefore = runFrozenEval(config, rewardOnlyPretrain.result.network, cell);
  const rewardOnlyAfter = loadNetworkFromExport(rewardOnlyReloaded);
  const rewardOnlyAfterResult = runFrozenEval(config, rewardOnlyAfter.network, cell);

  const supervisedMatch =
    supervisedBefore.successRate === supervisedAfterResult.successRate &&
    supervisedBefore.meanReward === supervisedAfterResult.meanReward &&
    supervisedBefore.noopRate === supervisedAfterResult.noopRate &&
    supervisedBefore.conflictRate === supervisedAfterResult.conflictRate;

  const rewardOnlyMatch =
    rewardOnlyBefore.successRate === rewardOnlyAfterResult.successRate &&
    rewardOnlyBefore.meanReward === rewardOnlyAfterResult.meanReward &&
    rewardOnlyBefore.noopRate === rewardOnlyAfterResult.noopRate &&
    rewardOnlyBefore.conflictRate === rewardOnlyAfterResult.conflictRate;

  const passed = supervisedMatch && rewardOnlyMatch && supervisedAfterResult.successRate >= REQUIRED_LOADER_FIDELITY_SUCCESS;

  return {
    name: "transfer loader preserves frozen-eval behavior (disk round-trip)",
    required: true,
    passed,
    metrics: {
      supervisedMatch,
      rewardOnlyMatch,
      supervisedBeforeSuccessRate: supervisedBefore.successRate,
      supervisedAfterSuccessRate: supervisedAfterResult.successRate,
      supervisedBeforeMeanReward: supervisedBefore.meanReward,
      supervisedAfterMeanReward: supervisedAfterResult.meanReward,
      rewardOnlyBeforeSuccessRate: rewardOnlyBefore.successRate,
      rewardOnlyAfterSuccessRate: rewardOnlyAfterResult.successRate,
      rewardOnlyBeforeMeanReward: rewardOnlyBefore.meanReward,
      rewardOnlyAfterMeanReward: rewardOnlyAfterResult.meanReward
    },
    conclusion: passed
      ? "Pretrained snapshots survive write->read->load and reproduce identical frozen-eval metrics."
      : "Loader round-trip diverged; downstream separation suites are not interpretable.",
    notes: [
      `Comparison is load-before (in-memory trained network) vs load-after (disk round-trip) under identical frozen eval seeds [${cell.evalSeeds.join(",")}].`,
      "Does not compare against snapshot.metrics training-period cumulative counts (rewardUpdateCount etc.), which frozen eval does not reproduce."
    ]
  };
}

function auditFrozenSeparation(
  config: ModelConfig,
  reloadedSnapshot: NetworkExport,
  mode: Extract<ChallengeLearningMode, "supervised" | "rewardOnly">,
  cell: TransferCellConfig
): TransferAuditSuiteResult {
  const pretrainedNetwork = loadNetworkFromExport(reloadedSnapshot).network;
  const pretrained = runFrozenEval(config, pretrainedNetwork, cell);
  const fresh = runFreshFrozenEval(config, cell);
  const separation = pretrained.successRate - fresh.successRate;

  let passed: boolean;
  let conclusion: string;

  if (mode === "supervised") {
    passed =
      pretrained.successRate >= REQUIRED_SUPERVISED_TRANSFER_SUCCESS &&
      separation >= REQUIRED_SUPERVISED_TRANSFER_SEPARATION &&
      fresh.noopRate === 1;
    conclusion = passed
      ? "Frozen-pretrained supervised network separates from frozen-fresh on held-out eval seeds."
      : "Pretrained supervised network does not separate from fresh; transfer advantage not established.";
  } else {
    passed =
      pretrained.meanReward > fresh.meanReward &&
      fresh.noopRate === 1 &&
      pretrained.successRate > 0;
    conclusion = passed
      ? "Frozen-pretrained rewardOnly network shows nonzero success and beats frozen-fresh mean reward."
      : "Pretrained rewardOnly network does not demonstrate transfer; mean-reward advantage or nonzero success missing.";
  }

  return {
    name: `transfer frozen-pretrained vs frozen-fresh separation (${mode})`,
    required: true,
    passed,
    metrics: {
      pretrainedSuccessRate: pretrained.successRate,
      freshSuccessRate: fresh.successRate,
      separation,
      pretrainedMeanReward: pretrained.meanReward,
      freshMeanReward: fresh.meanReward,
      pretrainedNoopRate: pretrained.noopRate,
      freshNoopRate: fresh.noopRate
    },
    conclusion,
    notes: [
      "Frozen mode disables all learning; eval reflects pure network state.",
      "Fresh-frozen outputs noop because initial interneuron->motor weights (0.35) do not reach motor threshold (1.0).",
      mode === "rewardOnly"
        ? "rewardOnly threshold adds pretrained.successRate > 0 to avoid a zero-success mean-reward-only false positive."
        : "supervised threshold is a lenient 'learned anything' test, not a 'learned well' test."
    ]
  };
}

function auditSeedIsolation(cell: TransferCellConfig): TransferAuditSuiteResult {
  const transferSet = new Set(cell.evalSeeds);
  const trainOverlap = DEFAULT_TRAIN_SEEDS.filter((seed) => transferSet.has(seed));
  const evalOverlap = DEFAULT_EVAL_SEEDS.filter((seed) => transferSet.has(seed));
  const overlap = [...trainOverlap, ...evalOverlap];
  const passed = overlap.length === 0;

  return {
    name: "transfer eval seed isolation",
    required: true,
    passed,
    metrics: {
      transferEvalSeeds: cell.evalSeeds.join(","),
      pretrainTrainSeeds: DEFAULT_TRAIN_SEEDS.join(","),
      pretrainEvalSeeds: DEFAULT_EVAL_SEEDS.join(","),
      seedOverlap: overlap.join(",")
    },
    conclusion: passed
      ? "Transfer eval seeds are disjoint from pretrain train/eval seeds; this tests transfer, not memorization."
      : "Transfer eval seeds overlap pretrain seeds; the separation test would be memorization, not transfer.",
    notes: [
      "Pretrain uses train [1..5] and eval [101..105]; transfer eval seeds are cell-specific.",
      "Without isolation, pretrained-vs-fresh separation could be replay of seen scenarios."
    ]
  };
}

function auditConflictBoundary(config: ModelConfig, reloadedSnapshot: NetworkExport, cell: TransferCellConfig): TransferAuditSuiteResult {
  const pretrainedNetwork = loadNetworkFromExport(reloadedSnapshot).network;
  const result = runChallengeExperiment(config, {
    seed: cell.auditSeed,
    trainSeeds: [],
    evalSeeds: [cell.auditSeed],
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: pretrainedNetwork,
    evaluationScenarios: [conflictChallengeScenario(cell.auditSeed)]
  });
  const episode = result.trace.episodes.find((candidate) => candidate.phase === "eval");
  const step = episode?.steps[0];
  const passed =
    step?.executedAction === "conflict" &&
    step.terminalReason === "conflict" &&
    result.successRate === 0;

  return {
    name: "transfer conflict boundary preservation",
    required: true,
    passed,
    metrics: {
      successRate: result.successRate,
      conflictRate: result.conflictRate,
      firstExecutedAction: step?.executedAction ?? "",
      firstTerminalReason: step?.terminalReason ?? ""
    },
    conclusion: passed
      ? "Pretrained network still records contradictory input as conflict, not success."
      : "Pretrained network lost the conflict boundary under transfer.",
    notes: [
      "Conflict remains a recorded boundary, not a solved arbitration policy.",
      "Pretraining must not have learned to suppress one motor to dodge conflict."
    ]
  };
}

function auditBlankWorld(config: ModelConfig, reloadedSnapshot: NetworkExport, cell: TransferCellConfig): TransferAuditSuiteResult {
  const pretrainedNetwork = loadNetworkFromExport(reloadedSnapshot).network;
  const result = runChallengeExperiment(config, {
    seed: cell.auditSeed,
    trainSeeds: [],
    evalSeeds: [cell.auditSeed],
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: pretrainedNetwork,
    evaluationScenarios: [blankChallengeScenario(cell.auditSeed)]
  });
  const passed = result.noopRate === 1 && result.meanReward === 0;

  return {
    name: "transfer blank world preservation",
    required: true,
    passed,
    metrics: {
      noopRate: result.noopRate,
      meanReward: result.meanReward,
      successRate: result.successRate
    },
    conclusion: passed
      ? "Pretrained network stays silent in a blank world; no spurious motor pressure."
      : "Pretrained network moved or scored in a blank world; transfer introduced noise.",
    notes: [
      "Blank world must produce noop with zero reward.",
      "This guards against pretraining producing a constant-action bias."
    ]
  };
}

function auditDropoutDiagnostic(
  config: ModelConfig,
  supervisedReloaded: NetworkExport,
  rewardOnlyReloaded: NetworkExport,
  dropout: number,
  cell: TransferCellConfig
): TransferAuditSuiteResult {
  const pretrainedNetwork = loadNetworkFromExport(supervisedReloaded).network;
  const pretrained = runChallengeExperiment(config, {
    seed: cell.auditSeed,
    trainSeeds: [],
    evalSeeds: cell.evalSeeds,
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: pretrainedNetwork,
    observationDropout: dropout,
    maxSteps: DROPOUT_EVAL_MAX_STEPS
  });
  const fresh = runChallengeExperiment(config, {
    seed: cell.auditSeed,
    trainSeeds: [],
    evalSeeds: cell.evalSeeds,
    epochs: 0,
    learningMode: "frozen",
    observationDropout: dropout,
    maxSteps: DROPOUT_EVAL_MAX_STEPS
  });
  const separation = pretrained.successRate - fresh.successRate;

  const rewardOnlyPretrainedNetwork = loadNetworkFromExport(rewardOnlyReloaded).network;
  const rewardOnlyPretrained = runChallengeExperiment(config, {
    seed: cell.auditSeed,
    trainSeeds: [],
    evalSeeds: cell.evalSeeds,
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: rewardOnlyPretrainedNetwork,
    observationDropout: dropout,
    maxSteps: DROPOUT_EVAL_MAX_STEPS
  });
  const rewardOnlyDelta = rewardOnlyPretrained.meanReward - fresh.meanReward;

  return {
    name: `transfer observation dropout ${dropout} diagnostic`,
    required: false,
    passed: pretrained.successRate >= 0.2,
    metrics: {
      observationDropout: dropout,
      pretrainedSuccessRate: pretrained.successRate,
      freshSuccessRate: fresh.successRate,
      separation,
      pretrainedNoopRate: pretrained.noopRate,
      freshNoopRate: fresh.noopRate,
      rewardOnlyPretrainedSuccessRate: rewardOnlyPretrained.successRate,
      rewardOnlyPretrainedMeanReward: rewardOnlyPretrained.meanReward,
      rewardOnlyDelta,
      rewardOnlyPretrainedNoopRate: rewardOnlyPretrained.noopRate
    },
    conclusion: pretrained.successRate >= 0.2
      ? `Pretrained network retains partial behavior under dropout ${dropout}.`
      : `Dropout ${dropout} substantially degrades pretrained behavior.`,
    notes: [
      "Diagnostic only; does not gate requiredPassed.",
      "Higher dropout probes real-world perception noise tolerance.",
      `maxSteps=${DROPOUT_EVAL_MAX_STEPS} chosen because the default 12-step budget absorbed dropout noise (frozen network needs only 2-3 visible steps to succeed). At dropout 0.3, expected visible steps ≈ ${DROPOUT_EVAL_MAX_STEPS} * (1 - 0.3) ≈ 2.8, creating real perception pressure.`,
      "rewardOnlyDelta = rewardOnlyPretrained.meanReward - fresh.meanReward; matrix aggregates this to watch for sign flips."
    ]
  };
}

function auditMultiObjectDiagnostic(config: ModelConfig, reloadedSnapshot: NetworkExport, cell: TransferCellConfig): TransferAuditSuiteResult {
  const pretrainedNetwork = loadNetworkFromExport(reloadedSnapshot).network;
  const pretrained = runChallengeExperiment(config, {
    seed: cell.auditSeed,
    trainSeeds: [],
    evalSeeds: [cell.auditSeed],
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: pretrainedNetwork,
    evaluationScenarios: [sameActionCompositeChallengeScenario(cell.auditSeed)]
  });
  const fresh = runChallengeExperiment(config, {
    seed: cell.auditSeed,
    trainSeeds: [],
    evalSeeds: [cell.auditSeed],
    epochs: 0,
    learningMode: "frozen",
    evaluationScenarios: [sameActionCompositeChallengeScenario(cell.auditSeed)]
  });

  return {
    name: "transfer multi-object composite diagnostic",
    required: false,
    passed: pretrained.successRate > fresh.successRate,
    metrics: {
      pretrainedSuccessRate: pretrained.successRate,
      freshSuccessRate: fresh.successRate,
      pretrainedNoopRate: pretrained.noopRate,
      pretrainedConflictRate: pretrained.conflictRate,
      freshNoopRate: fresh.noopRate
    },
    conclusion: pretrained.successRate > fresh.successRate
      ? "Pretrained network handles same-action composite (food-left + toxin-right) better than fresh."
      : "Pretrained network does not separate from fresh on multi-object composite.",
    notes: [
      "Diagnostic only; does not gate requiredPassed.",
      "Same-action composite: both objects imply moving left (toward food, away from toxin).",
      "Probes whether pretrained behavior survives multi-object observation."
    ]
  };
}

function auditContinuedLearningDiagnostic(config: ModelConfig, reloadedSnapshot: NetworkExport, cell: TransferCellConfig): TransferAuditSuiteResult {
  const continuedEpochs = CONTINUED_LEARNING_EPOCHS;
  const pretrainedNetwork = loadNetworkFromExport(reloadedSnapshot).network;
  const pretrained = runChallengeExperiment(config, {
    seed: cell.auditSeed,
    trainSeeds: CONTINUED_LEARNING_TRAIN_SEEDS,
    evalSeeds: cell.evalSeeds,
    epochs: continuedEpochs,
    learningMode: "supervised",
    initialNetwork: pretrainedNetwork
  });
  const fresh = runChallengeExperiment(config, {
    seed: cell.auditSeed,
    trainSeeds: CONTINUED_LEARNING_TRAIN_SEEDS,
    evalSeeds: cell.evalSeeds,
    epochs: continuedEpochs,
    learningMode: "supervised"
  });
  const separation = pretrained.successRate - fresh.successRate;

  return {
    name: "transfer continued-learning head-start diagnostic",
    required: false,
    passed: pretrained.successRate >= fresh.successRate,
    metrics: {
      pretrainedSuccessRate: pretrained.successRate,
      freshSuccessRate: fresh.successRate,
      separation,
      epochs: continuedEpochs
    },
    conclusion: pretrained.successRate >= fresh.successRate
      ? "Pretrained network reaches at least the same success as fresh after equal continued training."
      : "Fresh network caught up or surpassed pretrained under continued training.",
    notes: [
      "Diagnostic only; does not gate requiredPassed.",
      "Tests head-start, not final ceiling: both networks get the same small continued-training budget.",
      `${CONTINUED_LEARNING_EPOCHS} epoch on trainSeeds [${CONTINUED_LEARNING_TRAIN_SEEDS.join(",")}] chosen because 5 epochs on [1..5] saturated fresh to ceiling (successRate=1.0), making separation=0.000 vacuously pass. 1 epoch on a single seed leaves fresh on the rising part of the learning curve (~4 episodes, ~2-4 updates per synapse vs ~9 needed to reach threshold), exposing real head-start.`
    ]
  };
}

function auditWrongPriorDiagnostic(
  config: ModelConfig,
  cell: TransferCellConfig
): TransferAuditSuiteResult {
  const wrongPriorResult = runChallengeExperiment(config, {
    seed: cell.pretrainSeed,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: PRETRAIN_EPOCHS,
    learningMode: "supervised",
    reverseMapping: true
  });
  const wrongPriorSnapshot = createNetworkExport({
    seed: cell.pretrainSeed,
    config,
    neurons: wrongPriorResult.network.neurons,
    synapses: wrongPriorResult.network.synapses,
    pairMemory: wrongPriorResult.network.pairMemory,
    metrics: {
      pretrainKind: "2d-challenge-transfer-audit-wrong-prior",
      learningMode: "supervised",
      reverseMapping: true,
      successRate: wrongPriorResult.successRate,
      meanReward: wrongPriorResult.meanReward
    }
  });

  const wrongPriorPath = join(cell.tmpDir, "wrong-prior-roundtrip.json");
  writeNetworkExport(wrongPriorPath, wrongPriorSnapshot);
  const wrongPriorReloaded = readNetworkExport(wrongPriorPath);
  const wrongPriorNetwork = loadNetworkFromExport(wrongPriorReloaded).network;

  const synapseDump = dumpWrongPriorSynapseState(wrongPriorNetwork, config);

  const pretrained = runChallengeExperiment(config, {
    seed: cell.auditSeed,
    trainSeeds: CONTINUED_LEARNING_TRAIN_SEEDS,
    evalSeeds: cell.evalSeeds,
    epochs: WRONG_PRIOR_CONTINUED_EPOCHS,
    learningMode: "supervised",
    initialNetwork: wrongPriorNetwork
  });
  const postCDSynapseDump = dumpWrongPriorSynapseState(pretrained.network, config);
  const fresh = runChallengeExperiment(config, {
    seed: cell.auditSeed,
    trainSeeds: CONTINUED_LEARNING_TRAIN_SEEDS,
    evalSeeds: cell.evalSeeds,
    epochs: WRONG_PRIOR_CONTINUED_EPOCHS,
    learningMode: "supervised"
  });
  const separation = pretrained.successRate - fresh.successRate;

  return {
    name: "transfer wrong-prior continued-learning diagnostic",
    required: false,
    passed: separation < 0,
    metrics: {
      pretrainedSuccessRate: pretrained.successRate,
      freshSuccessRate: fresh.successRate,
      separation,
      epochs: WRONG_PRIOR_CONTINUED_EPOCHS,
      reverseMappingPretrain: true,
      wrongDirectionStableCount: synapseDump.wrongDirectionStableCount,
      wrongDirectionMaxStableWeight: synapseDump.wrongDirectionMaxStableWeight,
      wrongDirectionMaxFastWeight: synapseDump.wrongDirectionMaxFastWeight,
      correctDirectionMaxFastWeight: synapseDump.correctDirectionMaxFastWeight,
      dualLockConfirmed: synapseDump.wrongDirectionStableCount > 0,
      postCLWrongDirectionStableCount: postCDSynapseDump.wrongDirectionStableCount,
      postCLWrongDirectionMaxStableWeight: postCDSynapseDump.wrongDirectionMaxStableWeight,
      postCLWrongDirectionMaxFastWeight: postCDSynapseDump.wrongDirectionMaxFastWeight,
      postCLDualLockConfirmed: postCDSynapseDump.wrongDirectionStableCount > 0
    },
    conclusion:
      separation < 0
        ? "Wrong-prior pretrained network underperforms fresh under continued learning — gate has real signal."
        : "Wrong-prior pretrained network matches or beats fresh; wrong prior was unlearned within 1 epoch (task too trivial for wrong-prior to bite).",
    notes: [
      "Diagnostic only; does not gate requiredPassed.",
      "Pretrain uses reverseMapping=true (food-left->right, toxin-left->right, etc.); continued-learning uses correct mapping.",
      "passed=separation<0 means the test demonstrates wrong-prior hurts — this is the non-vacuity signal.",
      "If separation>=0 across all matrix cells, the task is too trivially unlearnable; document as structural limitation, do not promote to gate."
    ]
  };
}

interface WrongPriorSynapseDump {
  wrongDirectionStableCount: number;
  wrongDirectionMaxStableWeight: number;
  wrongDirectionMaxFastWeight: number;
  correctDirectionMaxFastWeight: number;
}

const CORRECT_MOTOR_FOR_INTER: Record<string, string> = {
  iFoodLeft: "leftMotor",
  iFoodRight: "rightMotor",
  iToxinLeft: "rightMotor",
  iToxinRight: "leftMotor"
};

function dumpWrongPriorSynapseState(
  network: LearningNetwork,
  config: ModelConfig
): WrongPriorSynapseDump {
  const interToMotor = network.synapses.filter((synapse) => {
    const pre = network.neurons.find((neuron) => neuron.id === synapse.preNeuronId);
    const post = network.neurons.find((neuron) => neuron.id === synapse.postNeuronId);
    return pre?.role === "interneuron" && post?.role === "motor";
  });

  const lines: string[] = [];
  lines.push("");
  lines.push("=== wrong-prior synapse state dump (interneuron -> motor) ===");
  lines.push(`maxWeight=${config.maxWeight} stableThreshold=${config.stableThreshold} stableDecay=${config.stableDecay}`);
  lines.push(`supervisedLearningRate=${config.supervisedLearningRate} stableCaptureRate=${config.stableCaptureRate} fastDecay=${config.fastDecay}`);

  let wrongDirectionStableCount = 0;
  let wrongDirectionMaxStableWeight = 0;
  let wrongDirectionMaxFastWeight = 0;
  let correctDirectionMaxFastWeight = 0;

  for (const synapse of interToMotor) {
    const correctMotor = CORRECT_MOTOR_FOR_INTER[synapse.preNeuronId];
    const isWrongDirection = correctMotor !== undefined && synapse.postNeuronId !== correctMotor;
    const direction = isWrongDirection ? "WRONG" : "CORRECT";
    const stableCaptured = synapse.stableWeight >= config.stableThreshold;

    if (isWrongDirection) {
      if (stableCaptured) {
        wrongDirectionStableCount += 1;
      }
      wrongDirectionMaxStableWeight = Math.max(wrongDirectionMaxStableWeight, synapse.stableWeight);
      wrongDirectionMaxFastWeight = Math.max(wrongDirectionMaxFastWeight, synapse.fastWeight);
    } else {
      correctDirectionMaxFastWeight = Math.max(correctDirectionMaxFastWeight, synapse.fastWeight);
    }

    lines.push(
      `  ${synapse.preNeuronId}->${synapse.postNeuronId} [${direction}] ` +
        `fast=${synapse.fastWeight.toFixed(4)} stable=${synapse.stableWeight.toFixed(4)} ` +
        `eff=${synapse.effectiveWeight.toFixed(4)} state=${synapse.state} ` +
        `recentUse=${synapse.recentUse.toFixed(4)} recentContrib=${synapse.recentContribution.toFixed(4)} ` +
        `stabilityScore=${synapse.stabilityScore.toFixed(4)}${stableCaptured ? " STABLE-CAPTURED" : ""}`
    );
  }

  lines.push("--- summary ---");
  lines.push(`wrong-direction: stableCount=${wrongDirectionStableCount}/4 maxStable=${wrongDirectionMaxStableWeight.toFixed(4)} maxFast=${wrongDirectionMaxFastWeight.toFixed(4)}`);
  lines.push(`correct-direction: maxFast=${correctDirectionMaxFastWeight.toFixed(4)}`);
  lines.push(`dualLock=${wrongDirectionStableCount > 0} (if true, stableWeight drives wrong motor even after fastWeight unlearn)`);
  lines.push("=== end dump ===");

  if (process.env.DEBUG_WRONG_PRIOR_DUMP === "1") {
    process.stderr.write(lines.join("\n") + "\n");
  }

  return {
    wrongDirectionStableCount,
    wrongDirectionMaxStableWeight,
    wrongDirectionMaxFastWeight,
    correctDirectionMaxFastWeight
  };
}
