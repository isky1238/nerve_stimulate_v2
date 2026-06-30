import { createHash } from "node:crypto";
import { defaultConfig, ModelConfig } from "../config/newModelConfig";
import {
  ChallengeExperimentResult,
  DEFAULT_EVAL_SEEDS,
  DEFAULT_TRAIN_SEEDS
} from "./challenge2d";
import {
  blankComplexScenario,
  compositeSameDirectionScenarios,
  createComplexConfig,
  distractorScenarios,
  priorityScenarios,
  runComplexExperiment,
  trueConflictScenarios
} from "./complex2d";

export interface World2DComplexAuditReport {
  version: string;
  generatedAt: string;
  requiredPassed: boolean;
  summary: string;
  suites: World2DComplexAuditSuiteResult[];
}

export interface World2DComplexAuditSuiteResult {
  name: string;
  required: boolean;
  passed: boolean;
  metrics: Record<string, number | string | boolean>;
  conclusion: string;
  notes: string[];
}

const REQUIRED_SUPERVISED_SUCCESS_RATE = 0.8;
const REQUIRED_FAMILY_SUCCESS_RATE = 0.5;
const REQUIRED_BASELINE_SEPARATION = 0.3;
const DEFAULT_EPOCHS = 40;

export function runWorld2DComplexAudit(config: ModelConfig = defaultConfig): World2DComplexAuditReport {
  const auditConfig = createComplexConfig(config);
  const suites = [
    auditComplexDeterminism(auditConfig),
    auditSupervisedBaseline(auditConfig),
    auditFrozenSeparation(auditConfig),
    auditCompositeSameDirection(auditConfig),
    auditDistractorImmunity(auditConfig),
    auditPriorityResolution(auditConfig),
    auditConflictBoundary(auditConfig),
    auditDropout02Robustness(auditConfig),
    auditBlankPreservation(auditConfig),
    auditDropout03Diagnostic(auditConfig),
    auditRewardOnlyFeasibility(auditConfig),
    auditRewardOnlyMultiObjectDiagnostic(auditConfig),
    auditTighterMaxStepsDiagnostic(auditConfig)
  ];
  const requiredPassed = suites.filter((suite) => suite.required).every((suite) => suite.passed);

  return {
    version: "dg-snn-2d-complex-audit-v0.1",
    generatedAt: new Date().toISOString(),
    requiredPassed,
    summary: requiredPassed
      ? "Required 2D-complex checks passed; multi-object compositional generalization and distance-weighted priority hold at conservative thresholds."
      : "At least one required 2D-complex check failed; keep conclusions at the 2D-challenge level until resolved.",
    suites
  };
}

export function formatWorld2DComplexAuditReport(report: World2DComplexAuditReport): string {
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

function auditComplexDeterminism(config: ModelConfig): World2DComplexAuditSuiteResult {
  const first = runComplexExperiment(config, {
    seed: 1,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "supervised"
  });
  const second = runComplexExperiment(config, {
    seed: 1,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "supervised"
  });
  const firstStable = stableComplexProjection(first);
  const secondStable = stableComplexProjection(second);
  const sameTrace = firstStable === secondStable;
  const passed = sameTrace && first.successRate >= REQUIRED_SUPERVISED_SUCCESS_RATE;

  return {
    name: "2D-complex multi-step deterministic replay",
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
      ? "The multi-step complex challenge is replayable under the same seed with spike-count arbitration."
      : "The complex challenge is not stable enough for further interpretation.",
    notes: [
      "Spike-count arbitration uses 3 sub-ticks per step; near objects fire more ticks than far ones.",
      "Determinism here covers the Family-A baseline only; multi-object families are checked separately."
    ]
  };
}

function auditSupervisedBaseline(config: ModelConfig): World2DComplexAuditSuiteResult {
  const result = runComplexExperiment(config, {
    seed: 11,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "supervised"
  });
  const passed = result.successRate >= REQUIRED_SUPERVISED_SUCCESS_RATE && result.supervisedUpdateCount > 0;

  return {
    name: "2D-complex supervised multi-step baseline (Family A)",
    required: true,
    passed,
    metrics: commonMetrics(result),
    conclusion: passed
      ? "The supervised readout drives a multi-step left/right policy under spike-count arbitration."
      : "The supervised upper-bound baseline is not strong enough; do not interpret multi-object results yet.",
    notes: [
      "Family A is the canonical single-object 4-pattern set, reused from 2D-challenge.",
      "This is the required upper-bound check before reading multi-object generalization."
    ]
  };
}

function auditFrozenSeparation(config: ModelConfig): World2DComplexAuditSuiteResult {
  const supervised = runComplexExperiment(config, {
    seed: 12,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "supervised"
  });
  const frozen = runComplexExperiment(config, {
    seed: 12,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "frozen"
  });
  const separation = supervised.successRate - frozen.successRate;
  const passed = supervised.successRate >= REQUIRED_SUPERVISED_SUCCESS_RATE && separation >= REQUIRED_BASELINE_SEPARATION;

  return {
    name: "2D-complex frozen baseline separation",
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
      ? "The supervised complex result separates from the no-learning baseline."
      : "The complex challenge is not separating learned behavior from fixed initial behavior.",
    notes: [
      "Frozen mode disables supervised learning, reward learning, exploration, capture, and decay.",
      "This prevents calling a static readout a learned policy."
    ]
  };
}

function auditCompositeSameDirection(config: ModelConfig): World2DComplexAuditSuiteResult {
  const pretrained = runComplexExperiment(config, {
    seed: 31,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "supervised"
  });
  const result = runComplexExperiment(config, {
    seed: 31,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: pretrained.network,
    evaluationScenarios: compositeSameDirectionScenarios()
  });
  const passed = result.successRate >= REQUIRED_FAMILY_SUCCESS_RATE;

  return {
    name: "2D-complex multi-object same-direction (Family B)",
    required: true,
    passed,
    metrics: {
      ...commonMetrics(result),
      pretrainedBaselineSuccessRate: pretrained.successRate
    },
    conclusion: passed
      ? "Pretrained network composes two same-direction object votes into the correct motor action."
      : "Compositional generalization to same-direction multi-object input failed.",
    notes: [
      "Family B: food-left + toxin-right (both vote left), food-right + toxin-left (both vote right).",
      "Network must sum two learned pathways into one motor decision."
    ]
  };
}

function auditDistractorImmunity(config: ModelConfig): World2DComplexAuditSuiteResult {
  const pretrained = runComplexExperiment(config, {
    seed: 41,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "supervised"
  });
  const result = runComplexExperiment(config, {
    seed: 41,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: pretrained.network,
    evaluationScenarios: distractorScenarios()
  });
  const passed = result.successRate >= REQUIRED_FAMILY_SUCCESS_RATE;

  return {
    name: "2D-complex distractor immunity (Family C)",
    required: true,
    passed,
    metrics: {
      ...commonMetrics(result),
      pretrainedBaselineSuccessRate: pretrained.successRate
    },
    conclusion: passed
      ? "Pretrained network ignores the far object and acts on the near one via spike-count weighting."
      : "Distractor (far object) incorrectly drove the motor action.",
    notes: [
      "Family C: same-kind near/far pairs (e.g., food-left-near + food-right-far).",
      "Distance encoded as sensory fire duration (near=3 ticks, far=1 tick); motor spike-count arbitrates."
    ]
  };
}

function auditPriorityResolution(config: ModelConfig): World2DComplexAuditSuiteResult {
  const pretrained = runComplexExperiment(config, {
    seed: 51,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "supervised"
  });
  const result = runComplexExperiment(config, {
    seed: 51,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: pretrained.network,
    evaluationScenarios: priorityScenarios()
  });
  const passed = result.successRate >= REQUIRED_FAMILY_SUCCESS_RATE;

  return {
    name: "2D-complex priority resolution (Family D)",
    required: true,
    passed,
    metrics: {
      ...commonMetrics(result),
      pretrainedBaselineSuccessRate: pretrained.successRate
    },
    conclusion: passed
      ? "Pretrained network resolves food+toxin same-side conflicts by distance priority."
      : "Priority resolution failed; network did not avoid the closer toxin or approach the closer food.",
    notes: [
      "Family D: food+toxin on the same side at different distances.",
      "Correct policy: approach food if food is closer, avoid toxin if toxin is closer or equidistant."
    ]
  };
}

function auditConflictBoundary(config: ModelConfig): World2DComplexAuditSuiteResult {
  const pretrained = runComplexExperiment(config, {
    seed: 61,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "supervised"
  });
  const result = runComplexExperiment(config, {
    seed: 61,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: pretrained.network,
    evaluationScenarios: trueConflictScenarios()
  });
  const episode = result.trace.episodes.find((candidate) => candidate.phase === "eval");
  const step = episode?.steps[0];
  const passed =
    step?.executedAction === "conflict" &&
    step.terminalReason === "conflict" &&
    result.successRate === 0;

  return {
    name: "2D-complex conflict boundary (Family E)",
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
      ? "Equidistant same-kind opposite-side input terminates as a recorded conflict."
      : "True conflict input is not being preserved as an explicit boundary.",
    notes: [
      "Family E: food-left + food-right equidistant, toxin-left + toxin-right equidistant.",
      "Equal spike counts yield 'conflict' — the network does not fabricate priority where none exists."
    ]
  };
}

function auditDropout02Robustness(config: ModelConfig): World2DComplexAuditSuiteResult {
  const result = runComplexExperiment(config, {
    seed: 71,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "supervised",
    observationDropout: 0.2
  });
  const passed = result.successRate >= REQUIRED_FAMILY_SUCCESS_RATE;

  return {
    name: "2D-complex dropout 0.2 robustness",
    required: true,
    passed,
    metrics: {
      observationDropout: 0.2,
      ...commonMetrics(result)
    },
    conclusion: passed
      ? "The supervised complex challenge tolerates 20% observation dropout at the conservative threshold."
      : "20% observation dropout substantially degrades the complex challenge.",
    notes: [
      "Dropout 0.2 is required at successRate >= 0.5; dropout 0.3 remains diagnostic.",
      "Promoted from 2D-challenge diagnostic per Level 4 pre-validation scope."
    ]
  };
}

function auditBlankPreservation(config: ModelConfig): World2DComplexAuditSuiteResult {
  const pretrained = runComplexExperiment(config, {
    seed: 81,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "supervised"
  });
  const result = runComplexExperiment(config, {
    seed: 81,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: pretrained.network,
    evaluationScenarios: [blankComplexScenario(81)]
  });
  const passed = result.noopRate === 1 && result.conflictRate === 0 && result.meanReward === 0;

  return {
    name: "2D-complex blank sparse world preservation",
    required: true,
    passed,
    metrics: commonMetrics(result),
    conclusion: passed
      ? "A trained complex policy remains inactive in a blank challenge world."
      : "The complex policy moved or conflicted without visible objects.",
    notes: [
      "Blank sparse worlds are not counted as task success.",
      "The required behavior is no movement pressure and no reward."
    ]
  };
}

function auditDropout03Diagnostic(config: ModelConfig): World2DComplexAuditSuiteResult {
  const result = runComplexExperiment(config, {
    seed: 91,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "supervised",
    observationDropout: 0.3
  });
  const passed = result.successRate >= 0.3;

  return {
    name: "2D-complex dropout 0.3 diagnostic",
    required: false,
    passed,
    metrics: {
      observationDropout: 0.3,
      ...commonMetrics(result)
    },
    conclusion: passed
      ? "The complex challenge has some tolerance to 30% observation dropout."
      : "30% observation dropout substantially degrades the complex challenge; recorded for future gate promotion.",
    notes: [
      "This diagnostic does not gate requiredPassed.",
      "If multi-matrix runs stabilize successRate >= 0.5 at 0.3 dropout, promote to required."
    ]
  };
}

function auditRewardOnlyFeasibility(config: ModelConfig): World2DComplexAuditSuiteResult {
  const first = runComplexExperiment(config, {
    seed: 101,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "rewardOnly"
  });
  const second = runComplexExperiment(config, {
    seed: 101,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "rewardOnly"
  });
  const sameTrace = stableComplexProjection(first) === stableComplexProjection(second);
  const passed = sameTrace && first.rewardUpdateCount > 0;

  return {
    name: "2D-complex reward-only feasibility (Family A)",
    required: false,
    passed,
    metrics: {
      sameStableTrace: sameTrace,
      successRate: first.successRate,
      meanReward: first.meanReward,
      rewardUpdateCount: first.rewardUpdateCount,
      supervisedUpdateCount: first.supervisedUpdateCount,
      conflictRate: first.conflictRate,
      noopRate: first.noopRate
    },
    conclusion: passed
      ? "Reward-only mode produces deterministic reward-driven updates on Family A."
      : "Reward-only mode did not produce stable reward-learning evidence.",
    notes: [
      "Diagnostic: 'can learn' preview alongside supervised 'can teach' baseline.",
      "Passing does not mean autonomous reward learning solved the task."
    ]
  };
}

function auditRewardOnlyMultiObjectDiagnostic(config: ModelConfig): World2DComplexAuditSuiteResult {
  const pretrained = runComplexExperiment(config, {
    seed: 111,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "rewardOnly"
  });
  const composite = runComplexExperiment(config, {
    seed: 111,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: pretrained.network,
    evaluationScenarios: [
      ...compositeSameDirectionScenarios(),
      ...distractorScenarios(),
      ...priorityScenarios()
    ]
  });

  return {
    name: "2D-complex reward-only multi-object diagnostic (Families B/C/D)",
    required: false,
    passed: true,
    metrics: {
      rewardOnlyBaselineSuccessRate: pretrained.successRate,
      multiObjectSuccessRate: composite.successRate,
      multiObjectConflictRate: composite.conflictRate,
      multiObjectNoopRate: composite.noopRate,
      multiObjectMeanReward: composite.meanReward
    },
    conclusion:
      "Reward-only multi-object success rate is recorded as a 'can learn compositional policy' signal; not gated.",
    notes: [
      "Compares rewardOnly pretrained network on multi-object families vs supervised pretrained.",
      "If rewardOnly successRate is near 0 here while supervised passes, compositional generalization requires supervised bootstrapping — an expected Level 4 finding."
    ]
  };
}

function auditTighterMaxStepsDiagnostic(config: ModelConfig): World2DComplexAuditSuiteResult {
  const result = runComplexExperiment(config, {
    seed: 121,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "supervised",
    maxSteps: 4
  });

  return {
    name: "2D-complex tighter maxSteps stress (maxSteps=4)",
    required: false,
    passed: result.successRate >= 0.5,
    metrics: {
      maxSteps: 4,
      ...commonMetrics(result)
    },
    conclusion:
      result.successRate >= 0.5
        ? "Family A tolerates maxSteps=4 (distance-3 traversal with 1 step margin)."
        : "maxSteps=4 is too tight for distance-3 traversal; recorded for boundary reference.",
    notes: [
      "Diagnostic: complex default is maxSteps=6; this stress tests the lower bound.",
      "If maxSteps=4 stabilizes, consider promoting to required for tighter task complexity."
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

function stableComplexProjection(result: ChallengeExperimentResult): string {
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
