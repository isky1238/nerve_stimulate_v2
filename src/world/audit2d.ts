import { createHash } from "node:crypto";
import { defaultConfig, ModelConfig, withConfig } from "../config/newModelConfig";
import {
  blankWorldScenario,
  canonicalWorldScenarios,
  createWorldAuditConfig,
  mirroredScenarios,
  oppositeConflictScenario,
  runWorldExperiment,
  sameActionCompositeScenario,
  seededWorldScenarios,
  WorldExperimentResult
} from "./world2d";

export interface World2DAuditReport {
  version: string;
  generatedAt: string;
  requiredPassed: boolean;
  summary: string;
  suites: World2DAuditSuiteResult[];
}

export interface World2DAuditSuiteResult {
  name: string;
  required: boolean;
  passed: boolean;
  metrics: Record<string, number | string | boolean>;
  conclusion: string;
  notes: string[];
}

const REQUIRED_FINAL_ACCURACY = 0.95;
const MULTI_SEED_SET = [1, 2, 3, 4, 5];

export function runWorld2DAudit(config: ModelConfig = defaultConfig): World2DAuditReport {
  const auditConfig = createWorldAuditConfig(config);
  const suites = [
    auditWorldDeterminism(auditConfig),
    auditWorldMultiSeed(auditConfig),
    auditWorldMirror(auditConfig),
    auditWorldBlank(auditConfig),
    auditWorldLearningAblation(auditConfig),
    auditWorldPlasticityAblation(auditConfig),
    auditWorldConflictHandling(auditConfig)
  ];
  const requiredPassed = suites.filter((suite) => suite.required).every((suite) => suite.passed);

  return {
    version: "dg-snn-2d-lite-audit-v0.1",
    generatedAt: new Date().toISOString(),
    requiredPassed,
    summary: requiredPassed
      ? "Required 2D-lite checks passed; conclusions remain limited to deterministic fixed-topology supervised world tasks."
      : "At least one required 2D-lite check failed; do not proceed to broader 2D environment tests.",
    suites
  };
}

export function formatWorld2DAuditReport(report: World2DAuditReport): string {
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

function auditWorldDeterminism(config: ModelConfig): World2DAuditSuiteResult {
  const first = runWorldExperiment(config, { seed: 1, epochs: 12, learningOn: true });
  const second = runWorldExperiment(config, { seed: 1, epochs: 12, learningOn: true });
  const firstStable = stableWorldTraceProjection(first);
  const secondStable = stableWorldTraceProjection(second);
  const sameTrace = firstStable === secondStable;
  const passed = sameTrace && first.finalAccuracy >= REQUIRED_FINAL_ACCURACY;

  return {
    name: "2D-lite deterministic replay",
    required: true,
    passed,
    metrics: {
      sameStableTrace: sameTrace,
      finalAccuracy: first.finalAccuracy,
      finalTaskSuccessRate: first.finalTaskSuccessRate,
      supervisedUpdates: first.supervisedUpdates,
      normalizedTraceDigest: digest(firstStable)
    },
    conclusion: passed
      ? "Same seed and world setup produce identical environment-level traces after excluding timestamps."
      : "The 2D-lite replay is not deterministic or does not learn the canonical world set.",
    notes: [
      "The trace covers world state, observation, sensory mapping, motor arbitration, reward, and terminal reason.",
      "This is deterministic replay for a fixed 2D-lite task, not a stochastic generalization proof."
    ]
  };
}

function auditWorldMultiSeed(config: ModelConfig): World2DAuditSuiteResult {
  const results = MULTI_SEED_SET.map((seed) =>
    runWorldExperiment(config, {
      seed,
      epochs: 40,
      learningOn: true,
      evaluationScenarios: seededWorldScenarios(seed)
    })
  );
  const accuracies = results.map((result) => result.finalAccuracy);
  const rewards = results.map((result) => result.meanReward);
  const passed = accuracies.every((accuracy) => accuracy >= REQUIRED_FINAL_ACCURACY);

  return {
    name: "2D-lite multi-seed object placement",
    required: true,
    passed,
    metrics: {
      seeds: MULTI_SEED_SET.join(","),
      minFinalAccuracy: Math.min(...accuracies),
      meanFinalAccuracy: mean(accuracies),
      minMeanReward: Math.min(...rewards),
      meanReward: mean(rewards)
    },
    conclusion: passed
      ? "The trained fixed topology handles seeded left/right food/toxin placements across the required seed set."
      : "At least one seeded placement set failed the final action accuracy threshold.",
    notes: [
      "This varies object coordinates while preserving the four known sensory channels.",
      "It does not test visual occlusion, distance gradients, or noisy perception."
    ]
  };
}

function auditWorldMirror(config: ModelConfig): World2DAuditSuiteResult {
  const baseScenarios = seededWorldScenarios(21);
  const result = runWorldExperiment(config, {
    seed: 21,
    epochs: 40,
    learningOn: true,
    evaluationScenarios: mirroredScenarios(baseScenarios)
  });
  const passed = result.finalAccuracy >= REQUIRED_FINAL_ACCURACY;

  return {
    name: "2D-lite mirrored world positions",
    required: true,
    passed,
    metrics: {
      finalAccuracy: result.finalAccuracy,
      finalTaskSuccessRate: result.finalTaskSuccessRate,
      meanReward: result.meanReward,
      evaluatedScenarios: mirroredScenarios(baseScenarios).length
    },
    conclusion: passed
      ? "Mirroring object positions preserves the expected food/toxin action policy."
      : "Mirrored world placements failed after canonical training.",
    notes: [
      "This checks the environment mapping layer and left/right action policy together.",
      "It remains a controlled mirror test, not broad spatial generalization."
    ]
  };
}

function auditWorldBlank(config: ModelConfig): World2DAuditSuiteResult {
  const result = runWorldExperiment(config, {
    seed: 11,
    epochs: 40,
    learningOn: true,
    evaluationScenarios: [blankWorldScenario(11)]
  });
  const episode = result.trace.episodes.find((candidate) => candidate.phase === "eval");
  const step = episode?.steps[0];
  const passed = step?.decision.action === "noop" && step.correct;

  return {
    name: "2D-lite blank world silence",
    required: true,
    passed,
    metrics: {
      blankAction: step?.decision.action ?? "",
      blankActiveMotors: step?.activeMotors.join(",") ?? "",
      finalAccuracy: result.finalAccuracy,
      meanReward: result.meanReward
    },
    conclusion: passed
      ? "A trained network produces noop when the 2D-lite world has no visible object."
      : "The trained network fired or moved in a blank 2D-lite world.",
    notes: [
      "Blank worlds are required because sparse 2D environments often provide no actionable sensory input.",
      "Noop on blank input is a safety gate, not evidence of exploration."
    ]
  };
}

function auditWorldLearningAblation(config: ModelConfig): World2DAuditSuiteResult {
  const evaluationScenarios = seededWorldScenarios(31);
  const learningOn = runWorldExperiment(config, {
    seed: 31,
    epochs: 40,
    learningOn: true,
    evaluationScenarios
  });
  const learningOff = runWorldExperiment(config, {
    seed: 31,
    epochs: 40,
    learningOn: false,
    evaluationScenarios
  });
  const passed =
    learningOn.finalAccuracy >= REQUIRED_FINAL_ACCURACY &&
    learningOn.finalAccuracy > learningOff.finalAccuracy + 0.5 &&
    learningOn.supervisedUpdates > 0;

  return {
    name: "2D-lite learning-on versus learning-off",
    required: true,
    passed,
    metrics: {
      learningOnFinalAccuracy: learningOn.finalAccuracy,
      learningOffFinalAccuracy: learningOff.finalAccuracy,
      learningOnMeanReward: learningOn.meanReward,
      learningOffMeanReward: learningOff.meanReward,
      supervisedUpdates: learningOn.supervisedUpdates
    },
    conclusion: passed
      ? "2D-lite action accuracy depends on supervised plasticity rather than fixed initial readout weights."
      : "The learning-on versus learning-off separation is not strong enough in the 2D-lite setup.",
    notes: [
      "The learning signal is still supervised by the known world policy.",
      "This does not prove autonomous reward discovery."
    ]
  };
}

function auditWorldPlasticityAblation(config: ModelConfig): World2DAuditSuiteResult {
  const noSupervisedConfig = withConfig({
    ...config,
    supervisedLearningRate: 0
  });
  const baseline = runWorldExperiment(config, {
    seed: 41,
    epochs: 40,
    learningOn: true,
    evaluationScenarios: seededWorldScenarios(41)
  });
  const ablated = runWorldExperiment(noSupervisedConfig, {
    seed: 41,
    epochs: 40,
    learningOn: true,
    evaluationScenarios: seededWorldScenarios(41)
  });
  const passed =
    baseline.finalAccuracy >= REQUIRED_FINAL_ACCURACY &&
    ablated.finalAccuracy < REQUIRED_FINAL_ACCURACY &&
    ablated.supervisedUpdates === 0;

  return {
    name: "2D-lite supervised-plasticity ablation",
    required: true,
    passed,
    metrics: {
      baselineFinalAccuracy: baseline.finalAccuracy,
      ablatedFinalAccuracy: ablated.finalAccuracy,
      baselineSupervisedUpdates: baseline.supervisedUpdates,
      ablatedSupervisedUpdates: ablated.supervisedUpdates
    },
    conclusion: passed
      ? "Disabling supervised plasticity blocks the 2D-lite readout from learning the world policy."
      : "The 2D-lite result does not depend clearly enough on supervised plasticity.",
    notes: [
      "This is an additional guard against mistaking the initial fixed topology for learned behavior.",
      "Reward-only learning is intentionally not claimed by this suite."
    ]
  };
}

function auditWorldConflictHandling(config: ModelConfig): World2DAuditSuiteResult {
  const result = runWorldExperiment(config, {
    seed: 51,
    epochs: 40,
    learningOn: true,
    evaluationScenarios: [sameActionCompositeScenario(51), oppositeConflictScenario(52)]
  });
  const evalEpisodes = result.trace.episodes.filter((episode) => episode.phase === "eval");
  const sameActionStep = evalEpisodes.find((episode) => episode.scenarioId === "same-action-composite")?.steps[0];
  const conflictStep = evalEpisodes.find((episode) => episode.scenarioId === "opposite-action-conflict")?.steps[0];
  const passed = sameActionStep?.decision.action === "left" && conflictStep?.decision.action === "conflict";

  return {
    name: "2D-lite composite and conflict arbitration",
    required: true,
    passed,
    metrics: {
      sameActionDecision: sameActionStep?.decision.action ?? "",
      sameActionActiveMotors: sameActionStep?.activeMotors.join(",") ?? "",
      conflictDecision: conflictStep?.decision.action ?? "",
      conflictActiveMotors: conflictStep?.activeMotors.join(",") ?? "",
      conflictReward: conflictStep?.reward ?? 0,
      conflictTaskSuccess: conflictStep?.taskSuccess ?? false
    },
    conclusion: passed
      ? "The world layer records contradictory motor outputs as conflict instead of counting them as a successful action."
      : "Composite or contradictory inputs are not being handled by the 2D-lite arbitration layer.",
    notes: [
      "A conflict decision is a recorded boundary, not a solved behavior policy.",
      "This tightens the previous pre-2D diagnostic where raw motors could fire left and right together."
    ]
  };
}

function stableWorldTraceProjection(result: WorldExperimentResult): string {
  return JSON.stringify({
    metrics: {
      finalAccuracy: result.finalAccuracy,
      finalTaskSuccessRate: result.finalTaskSuccessRate,
      meanReward: result.meanReward,
      supervisedUpdates: result.supervisedUpdates,
      captureUpdates: result.captureUpdates,
      decayUpdates: result.decayUpdates
    },
    trace: result.trace
  });
}

function digest(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}
