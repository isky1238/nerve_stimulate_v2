import { createHash } from "node:crypto";
import { defaultConfig, ModelConfig } from "../config/newModelConfig";
import {
  blankChallengeScenario,
  conflictChallengeScenario,
  createChallengeConfig,
  DEFAULT_EVAL_SEEDS,
  DEFAULT_TRAIN_SEEDS,
  runChallengeExperiment,
  ChallengeExperimentResult
} from "./challenge2d";

export interface World2DChallengeAuditReport {
  version: string;
  generatedAt: string;
  requiredPassed: boolean;
  summary: string;
  suites: World2DChallengeAuditSuiteResult[];
}

export interface World2DChallengeAuditSuiteResult {
  name: string;
  required: boolean;
  passed: boolean;
  metrics: Record<string, number | string | boolean>;
  conclusion: string;
  notes: string[];
}

const REQUIRED_SUPERVISED_SUCCESS_RATE = 0.8;
const REQUIRED_BASELINE_SEPARATION = 0.3;
const DEFAULT_EPOCHS = 40;

export function runWorld2DChallengeAudit(config: ModelConfig = defaultConfig): World2DChallengeAuditReport {
  const auditConfig = createChallengeConfig(config);
  const suites = [
    auditChallengeDeterminism(auditConfig),
    auditSupervisedBaseline(auditConfig),
    auditFrozenSeparation(auditConfig),
    auditRewardOnlyFeasibility(auditConfig),
    auditTrainEvalSeedIsolation(auditConfig),
    auditBlankSparseWorld(auditConfig),
    auditConflictBoundary(auditConfig),
    auditNoiseDiagnostic(auditConfig)
  ];
  const requiredPassed = suites.filter((suite) => suite.required).every((suite) => suite.passed);

  return {
    version: "dg-snn-2d-challenge-audit-v0.1",
    generatedAt: new Date().toISOString(),
    requiredPassed,
    summary: requiredPassed
      ? "Required 2D-challenge checks passed; reward-only results remain feasibility evidence, not autonomous learning proof."
      : "At least one required 2D-challenge check failed; keep conclusions at the 2D-lite level.",
    suites
  };
}

export function formatWorld2DChallengeAuditReport(report: World2DChallengeAuditReport): string {
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

function auditChallengeDeterminism(config: ModelConfig): World2DChallengeAuditSuiteResult {
  const first = runChallengeExperiment(config, {
    seed: 1,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "supervised"
  });
  const second = runChallengeExperiment(config, {
    seed: 1,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "supervised"
  });
  const firstStable = stableChallengeProjection(first);
  const secondStable = stableChallengeProjection(second);
  const sameTrace = firstStable === secondStable;
  const passed = sameTrace && first.successRate >= REQUIRED_SUPERVISED_SUCCESS_RATE;

  return {
    name: "2D-challenge multi-step deterministic replay",
    required: true,
    passed,
    metrics: {
      sameStableTrace: sameTrace,
      successRate: first.successRate,
      meanReward: first.meanReward,
      meanStepsToTerminal: first.meanStepsToTerminal,
      normalizedTraceDigest: digest(firstStable)
    },
    conclusion: passed
      ? "The multi-step supervised challenge is replayable under the same seed."
      : "The multi-step supervised challenge is not stable enough for further interpretation.",
    notes: [
      "The digest excludes generatedAt and includes environment steps, actions, rewards, and learning counts.",
      "This proves deterministic replay for this controlled challenge only."
    ]
  };
}

function auditSupervisedBaseline(config: ModelConfig): World2DChallengeAuditSuiteResult {
  const result = runChallengeExperiment(config, {
    seed: 11,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "supervised"
  });
  const passed = result.successRate >= REQUIRED_SUPERVISED_SUCCESS_RATE && result.supervisedUpdateCount > 0;

  return {
    name: "2D-challenge supervised multi-step baseline",
    required: true,
    passed,
    metrics: commonMetrics(result),
    conclusion: passed
      ? "The supervised readout can drive a multi-step left/right environment policy."
      : "The supervised upper-bound baseline is not strong enough; do not interpret reward-only results yet.",
    notes: [
      "This is still target-motor supervision, not autonomous reward learning.",
      "It is the required upper-bound check before reading reward-only feasibility."
    ]
  };
}

function auditFrozenSeparation(config: ModelConfig): World2DChallengeAuditSuiteResult {
  const supervised = runChallengeExperiment(config, {
    seed: 12,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "supervised"
  });
  const frozen = runChallengeExperiment(config, {
    seed: 12,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "frozen"
  });
  const separation = supervised.successRate - frozen.successRate;
  const passed = supervised.successRate >= REQUIRED_SUPERVISED_SUCCESS_RATE && separation >= REQUIRED_BASELINE_SEPARATION;

  return {
    name: "2D-challenge frozen baseline separation",
    required: true,
    passed,
    metrics: {
      supervisedSuccessRate: supervised.successRate,
      frozenSuccessRate: frozen.successRate,
      separation,
      supervisedMeanReward: supervised.meanReward,
      frozenMeanReward: frozen.meanReward
    },
    conclusion: passed
      ? "The supervised multi-step result separates from the no-learning baseline."
      : "The challenge is not separating learned behavior from fixed initial behavior.",
    notes: [
      "Frozen mode disables supervised learning, reward learning, exploration, capture, and decay.",
      "This prevents calling a static readout a learned policy."
    ]
  };
}

function auditRewardOnlyFeasibility(config: ModelConfig): World2DChallengeAuditSuiteResult {
  const first = runChallengeExperiment(config, {
    seed: 21,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "rewardOnly"
  });
  const second = runChallengeExperiment(config, {
    seed: 21,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "rewardOnly"
  });
  const sameTrace = stableChallengeProjection(first) === stableChallengeProjection(second);
  const passed = sameTrace && first.rewardUpdateCount > 0;

  return {
    name: "2D-challenge reward-only feasibility",
    required: true,
    passed,
    metrics: {
      sameStableTrace: sameTrace,
      successRate: first.successRate,
      meanReward: first.meanReward,
      rewardUpdateCount: first.rewardUpdateCount,
      supervisedUpdateCount: first.supervisedUpdateCount,
      conflictRate: first.conflictRate,
      noopRate: first.noopRate,
      normalizedTraceDigest: digest(stableChallengeProjection(first))
    },
    conclusion: passed
      ? "Reward-only mode produces deterministic reward-driven updates; success rate is reported, not promoted to proof."
      : "Reward-only mode did not produce stable reward-learning evidence.",
    notes: [
      "Reward-only training uses deterministic exploration when the network has no usable motor action.",
      "Passing this suite does not mean autonomous reward learning has solved the task."
    ]
  };
}

function auditTrainEvalSeedIsolation(config: ModelConfig): World2DChallengeAuditSuiteResult {
  const result = runChallengeExperiment(config, {
    seed: 31,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "supervised"
  });
  const overlap = DEFAULT_TRAIN_SEEDS.filter((seed) => DEFAULT_EVAL_SEEDS.includes(seed));
  const passed = overlap.length === 0 && result.successRate >= REQUIRED_SUPERVISED_SUCCESS_RATE;

  return {
    name: "2D-challenge train/eval seed isolation",
    required: true,
    passed,
    metrics: {
      trainSeeds: DEFAULT_TRAIN_SEEDS.join(","),
      evalSeeds: DEFAULT_EVAL_SEEDS.join(","),
      seedOverlap: overlap.join(","),
      successRate: result.successRate,
      meanReward: result.meanReward
    },
    conclusion: passed
      ? "Evaluation uses held-out seeds while preserving the controlled left/right task family."
      : "Train and eval seeds overlap or held-out evaluation did not pass.",
    notes: [
      "Seed isolation is a pressure test against merely replaying the training scenarios.",
      "It is still not a broad layout generalization proof because the sensory vocabulary is unchanged."
    ]
  };
}

function auditBlankSparseWorld(config: ModelConfig): World2DChallengeAuditSuiteResult {
  const result = runChallengeExperiment(config, {
    seed: 41,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "supervised",
    evaluationScenarios: [blankChallengeScenario(41)]
  });
  const passed = result.noopRate === 1 && result.conflictRate === 0 && result.meanReward === 0;

  return {
    name: "2D-challenge blank sparse world",
    required: true,
    passed,
    metrics: commonMetrics(result),
    conclusion: passed
      ? "A trained multi-step policy remains inactive in a blank challenge world."
      : "The challenge policy moved or conflicted without visible objects.",
    notes: [
      "Blank sparse worlds are not counted as task success.",
      "The required behavior is no movement pressure and no reward."
    ]
  };
}

function auditConflictBoundary(config: ModelConfig): World2DChallengeAuditSuiteResult {
  const result = runChallengeExperiment(config, {
    seed: 51,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "supervised",
    evaluationScenarios: [conflictChallengeScenario(51)]
  });
  const episode = result.trace.episodes.find((candidate) => candidate.phase === "eval");
  const step = episode?.steps[0];
  const passed =
    step?.executedAction === "conflict" &&
    step.terminalReason === "conflict" &&
    result.successRate === 0;

  return {
    name: "2D-challenge conflict boundary",
    required: true,
    passed,
    metrics: {
      successRate: result.successRate,
      conflictRate: result.conflictRate,
      firstExecutedAction: step?.executedAction ?? "",
      firstTerminalReason: step?.terminalReason ?? "",
      firstActiveMotors: step?.learning.activeMotors.join(",") ?? ""
    },
    conclusion: passed
      ? "Contradictory multi-object input terminates as a recorded conflict and is not counted as success."
      : "Conflict input is not being preserved as an explicit challenge boundary.",
    notes: [
      "This remains a boundary marker, not a solved arbitration policy.",
      "Future work can replace conflict with a learned priority rule once it is tested."
    ]
  };
}

function auditNoiseDiagnostic(config: ModelConfig): World2DChallengeAuditSuiteResult {
  const result = runChallengeExperiment(config, {
    seed: 61,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "supervised",
    observationDropout: 0.1
  });
  const passed = result.successRate >= 0.5;

  return {
    name: "2D-challenge observation dropout diagnostic",
    required: false,
    passed,
    metrics: {
      observationDropout: 0.1,
      ...commonMetrics(result)
    },
    conclusion: passed
      ? "The supervised challenge has some tolerance to low deterministic observation dropout."
      : "Low observation dropout substantially degrades this controlled challenge.",
    notes: [
      "This diagnostic does not gate requiredPassed.",
      "Noise robustness should not be claimed until this becomes a required suite with stronger thresholds."
    ]
  };
}

function commonMetrics(result: ChallengeExperimentResult): Record<string, number> {
  return {
    successRate: result.successRate,
    meanReward: result.meanReward,
    meanStepsToTerminal: result.meanStepsToTerminal,
    conflictRate: result.conflictRate,
    noopRate: result.noopRate,
    rewardUpdateCount: result.rewardUpdateCount,
    supervisedUpdateCount: result.supervisedUpdateCount
  };
}

function stableChallengeProjection(result: ChallengeExperimentResult): string {
  return JSON.stringify({
    metrics: {
      successRate: result.successRate,
      meanReward: result.meanReward,
      meanStepsToTerminal: result.meanStepsToTerminal,
      conflictRate: result.conflictRate,
      noopRate: result.noopRate,
      rewardUpdateCount: result.rewardUpdateCount,
      supervisedUpdateCount: result.supervisedUpdateCount,
      captureUpdateCount: result.captureUpdateCount,
      decayUpdateCount: result.decayUpdateCount
    },
    trace: result.trace
  });
}

function digest(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}
