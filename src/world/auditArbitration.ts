import { createHash } from "node:crypto";
import { defaultConfig, ModelConfig } from "../config/newModelConfig";
import { WorldAction } from "../core/arbitration";
import {
  LinearArbitrator,
  createInitialArbitrator,
  firstEvalSteps,
  inferArbitrator,
  recordArbitrationEvidence,
  reversePriorityAction,
  runArbitratedExperiment,
  scoreArbitrator,
  trainArbitrator,
  trainRewardArbitrator
} from "./arbitration2d";
import {
  ChallengeExperimentResult,
  DEFAULT_EVAL_SEEDS,
  DEFAULT_TRAIN_SEEDS,
  createChallengeScenarios
} from "./challenge2d";
import {
  DEFAULT_COMPLEX_MAX_STEPS,
  blankComplexScenario,
  compositeSameDirectionScenarios,
  createComplexConfig,
  distractorScenarios,
  priorityScenarios,
  runComplexExperiment,
  semanticConflictScenarios,
  trueConflictScenarios
} from "./complex2d";

export interface ArbitrationAuditReport {
  version: string;
  generatedAt: string;
  requiredPassed: boolean;
  summary: string;
  suites: ArbitrationAuditSuiteResult[];
}

export interface ArbitrationAuditSuiteResult {
  name: string;
  required: boolean;
  passed: boolean;
  metrics: Record<string, number | string | boolean>;
  conclusion: string;
  notes: string[];
}

interface TrainedArbitrationBundle {
  pretrain: ChallengeExperimentResult;
  arbitrator: LinearArbitrator;
  semanticTrainRaw: ChallengeExperimentResult;
  calibrationRaw: ChallengeExperimentResult;
  semanticRecordCount: number;
  calibrationRecordCount: number;
}

const DEFAULT_EPOCHS = 40;
const REQUIRED_ARBITRATION_SUCCESS_RATE = 0.9;
const REQUIRED_HELD_OUT_SUCCESS_RATE = 0.8;
const REQUIRED_FRESH_MAX_SUCCESS_RATE = 0.1;
const REQUIRED_SEPARATION = 0.8;
const SEMANTIC_RECORD_REPEAT = 12;

export function runArbitrationAudit(config: ModelConfig = defaultConfig): ArbitrationAuditReport {
  const auditConfig = createComplexConfig(config);
  const bundle = trainSupervisedArbitration(auditConfig, 701);
  const suites = [
    auditDeterminism(auditConfig),
    auditSemanticRawGate(auditConfig, bundle),
    auditSupervisedArbitration(auditConfig, bundle),
    auditHeldOutDistance(auditConfig, bundle),
    auditNonDegradation(auditConfig, bundle),
    auditTrueConflictPreservation(auditConfig, bundle),
    auditBlankPreservation(auditConfig, bundle),
    auditMotorOnlyAblation(auditConfig),
    auditReversePrior(auditConfig),
    auditWrongPriorNormalEval(auditConfig),
    auditRewardOnlyFeasibility(auditConfig),
    auditFrozenBaseline(auditConfig, bundle),
    auditMultiSeedMatrix(auditConfig)
  ];
  const requiredPassed = suites.filter((suite) => suite.required).every((suite) => suite.passed);

  return {
    version: "dg-snn-arbitration-audit-v0.1",
    generatedAt: new Date().toISOString(),
    requiredPassed,
    summary: requiredPassed
      ? "Required arbitration checks passed; a supervised linear resolver converts semantic raw conflicts into priority-correct actions while preserving true conflicts."
      : "At least one required arbitration check failed; keep conflict handling at the 2D-complex boundary level.",
    suites
  };
}

export function formatArbitrationAuditReport(report: ArbitrationAuditReport): string {
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

function auditDeterminism(config: ModelConfig): ArbitrationAuditSuiteResult {
  const first = trainSupervisedArbitration(config, 711);
  const second = trainSupervisedArbitration(config, 711);
  const firstEval = runSemanticHeldOut(config, first.pretrain, first.arbitrator, 711);
  const secondEval = runSemanticHeldOut(config, second.pretrain, second.arbitrator, 711);
  const firstStable = stableProjection(first.arbitrator, firstEval);
  const secondStable = stableProjection(second.arbitrator, secondEval);
  const sameStableTrace = firstStable === secondStable;
  const passed = sameStableTrace && firstEval.successRate >= REQUIRED_HELD_OUT_SUCCESS_RATE;

  return {
    name: "arbitration deterministic supervised replay",
    required: true,
    passed,
    metrics: {
      sameStableTrace,
      heldOutSuccessRate: firstEval.successRate,
      normalizedTraceDigest: digest(firstStable)
    },
    conclusion: passed
      ? "Linear arbitration training is deterministic under a fixed seed."
      : "Arbitration training is not stable enough for interpretation.",
    notes: [
      "Digest includes arbitrator weights and held-out arbitration trace.",
      "This covers the supervised resolver only; reward-only remains diagnostic."
    ]
  };
}

function auditSemanticRawGate(config: ModelConfig, bundle: TrainedArbitrationBundle): ArbitrationAuditSuiteResult {
  const rawSteps = firstEvalSteps(bundle.semanticTrainRaw);
  const rawConflictCount = rawSteps.filter((step) => step.networkDecision.action === "conflict").length;
  const expectedActionCount = rawSteps.filter((step) => step.expectedAction === "left" || step.expectedAction === "right").length;
  const passed = rawSteps.length > 0 && rawConflictCount === rawSteps.length && expectedActionCount === rawSteps.length;

  return {
    name: "arbitration semantic conflict raw gate (Family F)",
    required: true,
    passed,
    metrics: {
      scenarios: rawSteps.length,
      rawConflictCount,
      expectedActionCount,
      rawSuccessRate: bundle.semanticTrainRaw.successRate,
      firstActions: rawSteps.map((step) => step.networkDecision.action).join(",")
    },
    conclusion: passed
      ? "Family F is a real raw-conflict problem before arbitration."
      : "Family F did not produce the expected raw conflict gate.",
    notes: [
      "Semantic conflict is food+toxin on the same side at equal distance.",
      "Expected action is priority-correct toxin avoidance, but raw spike-count motor arbitration ties."
    ]
  };
}

function auditSupervisedArbitration(config: ModelConfig, bundle: TrainedArbitrationBundle): ArbitrationAuditSuiteResult {
  const trained = runSemanticTrain(config, bundle.pretrain, bundle.arbitrator, 721);
  const fresh = runSemanticTrain(config, bundle.pretrain, createInitialArbitrator(bundle.arbitrator.threshold), 721);
  const separation = trained.successRate - fresh.successRate;
  const trainedAccuracy = firstActionAccuracy(trained);
  const freshAccuracy = firstActionAccuracy(fresh);
  const passed =
    trained.successRate >= REQUIRED_ARBITRATION_SUCCESS_RATE &&
    fresh.successRate <= REQUIRED_FRESH_MAX_SUCCESS_RATE &&
    separation >= REQUIRED_SEPARATION;

  return {
    name: "arbitration supervised semantic conflict resolution (Family F train)",
    required: true,
    passed,
    metrics: {
      trainedSuccessRate: trained.successRate,
      freshSuccessRate: fresh.successRate,
      separation,
      trainedFirstActionAccuracy: trainedAccuracy,
      freshFirstActionAccuracy: freshAccuracy,
      semanticRecordCount: bundle.semanticRecordCount,
      calibrationRecordCount: bundle.calibrationRecordCount
    },
    conclusion: passed
      ? "The supervised linear resolver converts semantic raw conflicts into priority-correct actions."
      : "The supervised resolver failed to separate from an untrained resolver.",
    notes: [
      "Training uses Family F semantic conflict records plus Family A calibration evidence to disambiguate food/toxin semantics.",
      "The resolver still only intervenes when raw spike-count arbitration returns conflict."
    ]
  };
}

function auditHeldOutDistance(config: ModelConfig, bundle: TrainedArbitrationBundle): ArbitrationAuditSuiteResult {
  const result = runSemanticHeldOut(config, bundle.pretrain, bundle.arbitrator, 731);
  const accuracy = firstActionAccuracy(result);
  const passed = result.successRate >= REQUIRED_HELD_OUT_SUCCESS_RATE;

  return {
    name: "arbitration held-out distance and seed generalization",
    required: true,
    passed,
    metrics: {
      successRate: result.successRate,
      firstActionAccuracy: accuracy,
      meanReward: result.meanReward,
      meanStepsToTerminal: result.meanStepsToTerminal,
      conflictRate: result.conflictRate
    },
    conclusion: passed
      ? "The resolver generalizes from dist-2 semantic conflicts to held-out dist-1/dist-3 conflicts."
      : "The resolver did not generalize beyond the trained semantic distance.",
    notes: [
      "Train split uses F1/F2 at distance 2.",
      "Eval split uses F3-F6 at distances 1 and 3 with different scenario seeds."
    ]
  };
}

function auditNonDegradation(config: ModelConfig, bundle: TrainedArbitrationBundle): ArbitrationAuditSuiteResult {
  const familyA = runArbitratedExperiment(config, {
    seed: 741,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: bundle.pretrain.network,
    arbitrator: bundle.arbitrator,
    evaluationScenarios: createChallengeScenarios(DEFAULT_EVAL_SEEDS, DEFAULT_COMPLEX_MAX_STEPS)
  });
  const familyB = runArbitratedExperiment(config, {
    seed: 742,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: bundle.pretrain.network,
    arbitrator: bundle.arbitrator,
    evaluationScenarios: compositeSameDirectionScenarios()
  });
  const familyC = runArbitratedExperiment(config, {
    seed: 743,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: bundle.pretrain.network,
    arbitrator: bundle.arbitrator,
    evaluationScenarios: distractorScenarios()
  });
  const familyD = runArbitratedExperiment(config, {
    seed: 744,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: bundle.pretrain.network,
    arbitrator: bundle.arbitrator,
    evaluationScenarios: priorityScenarios()
  });
  const passed =
    familyA.successRate >= 0.8 &&
    familyB.successRate >= 0.5 &&
    familyC.successRate >= 0.5 &&
    familyD.successRate >= 0.5;

  return {
    name: "arbitration non-degradation on 2D-complex families",
    required: true,
    passed,
    metrics: {
      familyASuccessRate: familyA.successRate,
      familyBSuccessRate: familyB.successRate,
      familyCSuccessRate: familyC.successRate,
      familyDSuccessRate: familyD.successRate,
      familyCConflictRate: familyC.conflictRate
    },
    conclusion: passed
      ? "The resolver does not degrade existing 2D-complex required families at conservative thresholds."
      : "The resolver degraded an existing 2D-complex family.",
    notes: [
      "Raw non-conflict decisions pass through unchanged.",
      "Family C remains the conservative bottleneck and is held at the existing >=0.5 threshold."
    ]
  };
}

function auditTrueConflictPreservation(config: ModelConfig, bundle: TrainedArbitrationBundle): ArbitrationAuditSuiteResult {
  const result = runArbitratedExperiment(config, {
    seed: 751,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: bundle.pretrain.network,
    arbitrator: bundle.arbitrator,
    evaluationScenarios: trueConflictScenarios()
  });
  const steps = firstEvalSteps(result);
  const actions = steps.map((step) => step.executedAction);
  const deltas = steps.map((step) => step.complexEvidence ? Math.abs(scoreArbitrator(bundle.arbitrator, step.complexEvidence).delta) : 0);
  const maxAbsDelta = Math.max(0, ...deltas);
  const passed = actions.every((action) => action === "conflict") && result.successRate === 0;

  return {
    name: "arbitration true conflict preservation (Family E)",
    required: true,
    passed,
    metrics: {
      successRate: result.successRate,
      conflictRate: result.conflictRate,
      firstExecutedActions: actions.join(","),
      maxAbsDelta,
      threshold: bundle.arbitrator.threshold
    },
    conclusion: passed
      ? "Symmetric same-kind conflicts still fall back to conflict under the confidence threshold."
      : "The resolver forced a decision on a true symmetric conflict.",
    notes: [
      "Family E remains a boundary marker, not a learned decision target.",
      "No raw world object fields are passed to the resolver; fallback comes from balanced spike evidence and threshold tau."
    ]
  };
}

function auditBlankPreservation(config: ModelConfig, bundle: TrainedArbitrationBundle): ArbitrationAuditSuiteResult {
  const result = runArbitratedExperiment(config, {
    seed: 761,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: bundle.pretrain.network,
    arbitrator: bundle.arbitrator,
    evaluationScenarios: [blankComplexScenario(761)]
  });
  const passed = result.noopRate === 1 && result.meanReward === 0 && result.conflictRate === 0;

  return {
    name: "arbitration blank world preservation",
    required: true,
    passed,
    metrics: commonMetrics(result),
    conclusion: passed
      ? "The resolver leaves blank-world no-op behavior untouched."
      : "The resolver introduced movement or conflict in a blank world.",
    notes: [
      "The resolver only receives a call when raw decision is conflict.",
      "Blank worlds should remain noop with zero reward."
    ]
  };
}

function auditMotorOnlyAblation(config: ModelConfig): ArbitrationAuditSuiteResult {
  const bundle = trainSupervisedArbitration(config, 771, {
    featureMask: [false, false, false, false, true, true],
    includeCalibration: false
  });
  const result = runSemanticHeldOut(config, bundle.pretrain, bundle.arbitrator, 771);
  const passed = result.successRate <= 0.1;

  return {
    name: "arbitration motor-only ablation diagnostic",
    required: false,
    passed,
    metrics: {
      successRate: result.successRate,
      firstActionAccuracy: firstActionAccuracy(result),
      conflictRate: result.conflictRate
    },
    conclusion: passed
      ? "Motor counts alone cannot resolve semantic ties, so inter evidence is necessary."
      : "Motor-only features unexpectedly resolved semantic conflicts.",
    notes: [
      "Feature mask keeps only leftMotor/rightMotor counts.",
      "On Family F both motor counts tie, so a real semantic resolver needs interneuron evidence."
    ]
  };
}

function auditReversePrior(config: ModelConfig): ArbitrationAuditSuiteResult {
  const bundle = trainSupervisedArbitration(config, 781, {
    reversePrior: true,
    includeCalibration: false
  });
  const result = runSemanticHeldOut(config, bundle.pretrain, bundle.arbitrator, 781);
  const reverseAccuracy = firstActionAccuracy(result, (step) => reversePriorityAction(step.expectedAction));
  const passed = reverseAccuracy >= 0.8;

  return {
    name: "arbitration reverse-prior control diagnostic",
    required: false,
    passed,
    metrics: {
      reverseLabelAccuracy: reverseAccuracy,
      normalSuccessRate: result.successRate,
      conflictRate: result.conflictRate
    },
    conclusion: passed
      ? "The resolver follows reversed semantic labels, which argues against a fixed avoid-toxin hardcode."
      : "The resolver did not follow reversed semantic labels.",
    notes: [
      "Reverse-prior flips the semantic conflict priority label only for this diagnostic.",
      "Normal environment reward is expected to drop under reverse-prior behavior."
    ]
  };
}

function auditWrongPriorNormalEval(config: ModelConfig): ArbitrationAuditSuiteResult {
  const bundle = trainSupervisedArbitration(config, 791, {
    reversePrior: true,
    includeCalibration: false
  });
  const result = runSemanticHeldOut(config, bundle.pretrain, bundle.arbitrator, 791);
  const normalAccuracy = firstActionAccuracy(result);
  const passed = normalAccuracy < 0.5;

  return {
    name: "arbitration wrong-prior normal-eval diagnostic",
    required: false,
    passed,
    metrics: {
      normalFirstActionAccuracy: normalAccuracy,
      normalSuccessRate: result.successRate,
      conflictRate: result.conflictRate
    },
    conclusion: passed
      ? "A reversed-prior resolver fails under normal labels, so the diagnostic has non-vacuous signal."
      : "A reversed-prior resolver still passed normal eval; the task may be too weak.",
    notes: [
      "This is the counterpart to reverse-prior control.",
      "It should get worse under normal priority labels after reversed training."
    ]
  };
}

function auditRewardOnlyFeasibility(config: ModelConfig): ArbitrationAuditSuiteResult {
  const pretrain = pretrainComplex(config, 801);
  const raw = runSemanticTrain(config, pretrain, null, 801);
  const records = recordArbitrationEvidence(raw.trace.episodes, { onlyRawConflict: true });
  const arbitrator = trainRewardArbitrator(records, { seed: 801, threshold: 0.1, steps: 300 });
  const result = runSemanticHeldOut(config, pretrain, arbitrator, 801);

  return {
    name: "arbitration reward-only feasibility diagnostic",
    required: false,
    passed: true,
    metrics: {
      successRate: result.successRate,
      firstActionAccuracy: firstActionAccuracy(result),
      meanReward: result.meanReward,
      conflictRate: result.conflictRate,
      recordCount: records.length
    },
    conclusion: "Reward-shaped arbitration learning is recorded as a diagnostic and does not gate requiredPassed.",
    notes: [
      "This uses policy-gradient-style reward updates on the post-hoc resolver, not SNN synapse learning.",
      "Weak or unstable reward-only results are expected at this stage."
    ]
  };
}

function auditFrozenBaseline(config: ModelConfig, bundle: TrainedArbitrationBundle): ArbitrationAuditSuiteResult {
  const result = runSemanticHeldOut(config, bundle.pretrain, createInitialArbitrator(bundle.arbitrator.threshold), 811);
  const passed = result.successRate <= REQUIRED_FRESH_MAX_SUCCESS_RATE && result.conflictRate === 1;

  return {
    name: "arbitration frozen untrained resolver diagnostic",
    required: false,
    passed,
    metrics: {
      successRate: result.successRate,
      conflictRate: result.conflictRate,
      firstActionAccuracy: firstActionAccuracy(result)
    },
    conclusion: passed
      ? "An untrained resolver falls back to conflict on Family F."
      : "The untrained resolver produced nontrivial semantic behavior.",
    notes: [
      "Fresh linear weights and bias are zero.",
      "This guards against a vacuous pass from built-in left/right bias."
    ]
  };
}

function auditMultiSeedMatrix(config: ModelConfig): ArbitrationAuditSuiteResult {
  const trainSeeds = [821, 822, 823];
  const evalSeeds = [831, 832, 833];
  const successRates: number[] = [];

  for (const trainSeed of trainSeeds) {
    const bundle = trainSupervisedArbitration(config, trainSeed);
    for (const evalSeed of evalSeeds) {
      const result = runSemanticHeldOut(config, bundle.pretrain, bundle.arbitrator, evalSeed);
      successRates.push(result.successRate);
    }
  }

  const min = Math.min(...successRates);
  const max = Math.max(...successRates);
  const mean = successRates.reduce((sum, value) => sum + value, 0) / successRates.length;

  return {
    name: "arbitration multi-seed matrix diagnostic",
    required: false,
    passed: min >= 0.8,
    metrics: {
      cellsRun: successRates.length,
      successMin: min,
      successMean: mean,
      successMax: max
    },
    conclusion: "Multi-seed arbitration stability is recorded for future gate promotion.",
    notes: [
      "This is diagnostic because Family F scenario geometry is still controlled.",
      "Promote to required only after adding broader semantic-conflict scenario generation."
    ]
  };
}

function trainSupervisedArbitration(
  config: ModelConfig,
  seed: number,
  options: { reversePrior?: boolean; featureMask?: boolean[]; includeCalibration?: boolean } = {}
): TrainedArbitrationBundle {
  const pretrain = pretrainComplex(config, seed);
  const semanticTrainRaw = runSemanticTrain(config, pretrain, null, seed);
  const semanticRecords = recordArbitrationEvidence(semanticTrainRaw.trace.episodes, { onlyRawConflict: true });
  const calibrationRaw = runComplexExperiment(config, {
    seed,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: pretrain.network,
    evaluationScenarios: createChallengeScenarios(DEFAULT_TRAIN_SEEDS, DEFAULT_COMPLEX_MAX_STEPS)
  });
  const calibrationRecords = options.includeCalibration === false
    ? []
    : recordArbitrationEvidence(calibrationRaw.trace.episodes, { onlyRawConflict: false });
  const trainingRecords = [
    ...calibrationRecords,
    ...repeatRecords(semanticRecords, SEMANTIC_RECORD_REPEAT)
  ];
  const arbitrator = trainArbitrator(trainingRecords, {
    threshold: 0.1,
    learningRate: 0.08,
    steps: 400,
    reversePrior: options.reversePrior,
    featureMask: options.featureMask
  });

  return {
    pretrain,
    arbitrator,
    semanticTrainRaw,
    calibrationRaw,
    semanticRecordCount: semanticRecords.length,
    calibrationRecordCount: calibrationRecords.length
  };
}

function pretrainComplex(config: ModelConfig, seed: number): ChallengeExperimentResult {
  return runComplexExperiment(config, {
    seed,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "supervised"
  });
}

function runSemanticTrain(
  config: ModelConfig,
  pretrain: ChallengeExperimentResult,
  arbitrator: LinearArbitrator | null,
  seed: number
): ChallengeExperimentResult {
  return runArbitratedExperiment(config, {
    seed,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: pretrain.network,
    arbitrator,
    evaluationScenarios: semanticConflictScenarios().slice(0, 2)
  });
}

function runSemanticHeldOut(
  config: ModelConfig,
  pretrain: ChallengeExperimentResult,
  arbitrator: LinearArbitrator | null,
  seed: number
): ChallengeExperimentResult {
  return runArbitratedExperiment(config, {
    seed,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: pretrain.network,
    arbitrator,
    evaluationScenarios: semanticConflictScenarios().slice(2)
  });
}

function firstActionAccuracy(
  result: ChallengeExperimentResult,
  labeler: (step: ReturnType<typeof firstEvalSteps>[number]) => WorldAction = (step) => step.expectedAction
): number {
  const steps = firstEvalSteps(result);
  const labeled = steps.filter((step) => {
    const label = labeler(step);
    return label === "left" || label === "right";
  });
  const correct = labeled.filter((step) => step.executedAction === labeler(step)).length;
  return correct / Math.max(1, labeled.length);
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

function repeatRecords<T>(records: T[], count: number): T[] {
  const repeated: T[] = [];
  for (let index = 0; index < count; index += 1) {
    repeated.push(...records);
  }
  return repeated;
}

function stableProjection(arbitrator: LinearArbitrator, result: ChallengeExperimentResult): string {
  return JSON.stringify({
    arbitrator,
    metrics: commonMetrics(result),
    firstSteps: firstEvalSteps(result).map((step) => ({
      scenarioId: result.trace.episodes.find((episode) => episode.steps[0] === step)?.scenarioId ?? "",
      expectedAction: step.expectedAction,
      rawAction: step.networkDecision.action,
      executedAction: step.executedAction,
      evidence: step.complexEvidence,
      reward: step.reward,
      terminalReason: step.terminalReason
    }))
  });
}

function digest(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}
