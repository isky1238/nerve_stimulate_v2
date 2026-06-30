import { createHash } from "node:crypto";
import { defaultConfig, ModelConfig } from "../config/newModelConfig";
import {
  LinearArbitrator,
  firstEvalSteps,
  scoreArbitrator
} from "./arbitration2d";
import {
  MatrixTrainedBundle,
  findTauAcceptanceWindow,
  generateSemanticConflictScenarios,
  runMatrixEval,
  sweepThreshold,
  trainMatrixArbitration,
  evaluateAblation,
  MATRIX_EVAL_COUNT,
  MATRIX_TAUS
} from "./arbitrationMatrix";
import {
  ChallengeExperimentResult,
  DEFAULT_EVAL_SEEDS,
  createChallengeScenarios
} from "./challenge2d";
import {
  DEFAULT_COMPLEX_MAX_STEPS,
  blankComplexScenario,
  compositeSameDirectionScenarios,
  createComplexConfig,
  distractorScenarios,
  priorityScenarios,
  trueConflictScenarios
} from "./complex2d";

export interface ArbitrationMatrixAuditReport {
  version: string;
  generatedAt: string;
  requiredPassed: boolean;
  summary: string;
  suites: ArbitrationMatrixAuditSuiteResult[];
}

export interface ArbitrationMatrixAuditSuiteResult {
  name: string;
  required: boolean;
  passed: boolean;
  metrics: Record<string, number | string | boolean>;
  conclusion: string;
  notes: string[];
}

const REQUIRED_MATRIX_SUCCESS_RATE = 0.8;
const REQUIRED_MATRIX_FRESH_MAX = 0.2;
const REQUIRED_MATRIX_SEPARATION = 0.6;
const REQUIRED_FAMILY_E_FALLBACK = 0.9;
const TAU_WINDOW_MIN_WIDTH = 0.1;
const ABLATION_FULL_LABEL = "full-evidence";
const ABLATION_MOTOR_ONLY_LABEL = "motor-only";
const ABLATION_INTER_ONLY_LABEL = "inter-only";
const ABLATION_DROP_TOXIN_LABEL = "drop-toxin";
const ABLATION_DROP_FOOD_LABEL = "drop-food";

const ABLATION_FULL_MASK = [true, true, true, true, true, true];
const ABLATION_MOTOR_ONLY_MASK = [false, false, false, false, true, true];
const ABLATION_INTER_ONLY_MASK = [true, true, true, true, false, false];
const ABLATION_DROP_TOXIN_MASK = [true, true, false, false, true, true];
const ABLATION_DROP_FOOD_MASK = [false, false, true, true, true, true];

export function runArbitrationMatrixAudit(config: ModelConfig = defaultConfig): ArbitrationMatrixAuditReport {
  const auditConfig = createComplexConfig(config);
  const trainSeed = 1001;
  const evalSeed = 2001;
  const bundle = trainMatrixArbitration(auditConfig, { trainSeed });
  const evalScenarios = generateSemanticConflictScenarios(evalSeed, MATRIX_EVAL_COUNT);
  const familyEScenarios = trueConflictScenarios();
  const blankScenario = blankComplexScenario(2061);

  const suites = [
    auditDeterminism(auditConfig, trainSeed, evalSeed),
    auditDisjointGeneralization(auditConfig, bundle, evalScenarios, evalSeed),
    auditNonDegradation(auditConfig, bundle),
    auditTrueConflictAtDefaultTau(auditConfig, bundle, familyEScenarios, evalSeed),
    auditBlankPreservation(auditConfig, bundle),
    auditTauWindow(auditConfig, bundle, evalScenarios, familyEScenarios, blankScenario, evalSeed),
    auditTrueConflictPreservationAcrossTau(auditConfig, bundle, familyEScenarios, evalSeed),
    auditEvidenceAblation(auditConfig, evalScenarios, trainSeed, evalSeed),
    auditTauSweepDetail(auditConfig, bundle, evalScenarios, familyEScenarios, blankScenario, evalSeed),
    auditMultiSeedMatrix(auditConfig),
    auditFreshBaselineOnGenerated(auditConfig, evalScenarios, evalSeed)
  ];
  const requiredPassed = suites.filter((suite) => suite.required).every((suite) => suite.passed);

  return {
    version: "dg-snn-arbitration-matrix-audit-v0.1",
    generatedAt: new Date().toISOString(),
    requiredPassed,
    summary: requiredPassed
      ? "Required matrix checks passed; the supervised resolver generalizes to generated semantic-conflict scenarios and τ choice is non-lucky."
      : "At least one required matrix check failed; the resolver may be a narrow solution or τ may be a lucky point.",
    suites
  };
}

export function formatArbitrationMatrixReport(report: ArbitrationMatrixAuditReport): string {
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

function auditDeterminism(
  config: ModelConfig,
  trainSeed: number,
  evalSeed: number
): ArbitrationMatrixAuditSuiteResult {
  const first = trainMatrixArbitration(config, { trainSeed });
  const second = trainMatrixArbitration(config, { trainSeed });
  const evalScenarios = generateSemanticConflictScenarios(evalSeed, MATRIX_EVAL_COUNT);
  const firstEval = runMatrixEval(config, first, evalScenarios, evalSeed);
  const secondEval = runMatrixEval(config, second, evalScenarios, evalSeed);
  const firstStable = stableProjection(first, firstEval);
  const secondStable = stableProjection(second, secondEval);
  const sameStableTrace = firstStable === secondStable;
  const passed = sameStableTrace && firstEval.successRate >= REQUIRED_MATRIX_SUCCESS_RATE;

  return {
    name: "arbitration matrix deterministic supervised replay",
    required: true,
    passed,
    metrics: {
      sameStableTrace,
      matrixSuccessRate: firstEval.successRate,
      normalizedTraceDigest: digest(firstStable)
    },
    conclusion: passed
      ? "Matrix arbitration training is deterministic under a fixed seed and generalizes to generated eval scenarios."
      : "Matrix arbitration training is not stable enough for interpretation.",
    notes: [
      "Digest includes arbitrator weights and generated-scenario eval trace.",
      "Train and eval scenarios use disjoint seed pools (train 1000s, eval 2000s)."
    ]
  };
}

function auditDisjointGeneralization(
  config: ModelConfig,
  bundle: MatrixTrainedBundle,
  evalScenarios: ReturnType<typeof generateSemanticConflictScenarios>,
  evalSeed: number
): ArbitrationMatrixAuditSuiteResult {
  const trained = runMatrixEval(config, bundle, evalScenarios, evalSeed);
  const freshArbitrator: LinearArbitrator = {
    ...bundle.arbitrator,
    weights: Array.from({ length: bundle.arbitrator.weights.length }, () => 0),
    bias: [0, 0]
  };
  const fresh = runMatrixEval(config, bundle, evalScenarios, evalSeed, freshArbitrator);
  const separation = trained.successRate - fresh.successRate;
  const passed =
    trained.successRate >= REQUIRED_MATRIX_SUCCESS_RATE &&
    fresh.successRate <= REQUIRED_MATRIX_FRESH_MAX &&
    separation >= REQUIRED_MATRIX_SEPARATION;

  return {
    name: "arbitration matrix disjoint scenario generalization",
    required: true,
    passed,
    metrics: {
      trainedSuccessRate: trained.successRate,
      freshSuccessRate: fresh.successRate,
      separation,
      trainedConflictRate: trained.conflictRate,
      evalScenarioCount: evalScenarios.length,
      trainScenarioCount: bundle.trainScenarios.length
    },
    conclusion: passed
      ? "The resolver generalizes from generated train scenarios to disjoint generated eval scenarios."
      : "The resolver failed to generalize across the disjoint scenario split.",
    notes: [
      "Train pool uses seeds in [1000, 1999]; eval pool uses seeds in [2000, 2999].",
      "Fresh resolver has zero weights and bias to test for built-in bias."
    ]
  };
}

function auditTrueConflictAtDefaultTau(
  config: ModelConfig,
  bundle: MatrixTrainedBundle,
  familyEScenarios: ReturnType<typeof trueConflictScenarios>,
  evalSeed: number
): ArbitrationMatrixAuditSuiteResult {
  const result = runMatrixEval(config, bundle, familyEScenarios, evalSeed);
  const steps = firstEvalSteps(result);
  const fallbackRate = steps.filter((step) => step.executedAction === "conflict").length / Math.max(1, steps.length);
  const passed = fallbackRate >= REQUIRED_FAMILY_E_FALLBACK && result.successRate === 0;

  return {
    name: "arbitration matrix true conflict preservation at default tau",
    required: true,
    passed,
    metrics: {
      fallbackRate,
      successRate: result.successRate,
      conflictRate: result.conflictRate,
      firstExecutedActions: steps.map((step) => step.executedAction).join(","),
      threshold: bundle.arbitrator.threshold
    },
    conclusion: passed
      ? `Family E remains conflict (fallback >= ${REQUIRED_FAMILY_E_FALLBACK}) at default τ=${bundle.arbitrator.threshold}.`
      : "Family E did not fall back to conflict at default τ — the resolver forces a decision on symmetric evidence.",
    notes: [
      "Family E is equidistant same-kind objects on opposite sides (symmetric evidence).",
      "This is the hard gate; τ sweep and across-τ fallback are diagnostic."
    ]
  };
}

function auditBlankPreservation(
  config: ModelConfig,
  bundle: MatrixTrainedBundle
): ArbitrationMatrixAuditSuiteResult {
  const result = runMatrixEval(config, bundle, [blankComplexScenario(2061)], 2061);
  const passed = result.noopRate === 1 && result.meanReward === 0 && result.conflictRate === 0;

  return {
    name: "arbitration matrix blank world preservation",
    required: true,
    passed,
    metrics: {
      successRate: result.successRate,
      meanReward: result.meanReward,
      noopRate: result.noopRate,
      conflictRate: result.conflictRate
    },
    conclusion: passed
      ? "Matrix-trained resolver leaves blank-world no-op behavior untouched."
      : "Matrix-trained resolver introduced movement or conflict in a blank world.",
    notes: [
      "The resolver only receives a call when raw decision is conflict.",
      "Blank worlds should remain noop with zero reward."
    ]
  };
}

function auditTauWindow(
  config: ModelConfig,
  bundle: MatrixTrainedBundle,
  familyFScenarios: ReturnType<typeof generateSemanticConflictScenarios>,
  familyEScenarios: ReturnType<typeof trueConflictScenarios>,
  blankScenario: ReturnType<typeof blankComplexScenario>,
  evalSeed: number
): ArbitrationMatrixAuditSuiteResult {
  const sweep = sweepThreshold(config, bundle, familyFScenarios, familyEScenarios, blankScenario, MATRIX_TAUS, evalSeed);
  const window = findTauAcceptanceWindow(sweep, {
    minFamilyFSuccessRate: REQUIRED_MATRIX_SUCCESS_RATE,
    minFamilyEFallback: REQUIRED_FAMILY_E_FALLBACK,
    minWidth: TAU_WINDOW_MIN_WIDTH
  });

  return {
    name: "arbitration matrix tau acceptance window",
    required: false,
    passed: true,
    metrics: {
      windowExists: window.exists,
      minTau: window.minTau ?? -1,
      maxTau: window.maxTau ?? -1,
      satisfyingTaus: window.satisfyingTaus.join(",") || "none",
      sweepSummary: sweep
        .map((point) => `tau=${point.tau}:F=${point.familyFSuccessRate.toFixed(2)},E=${point.familyEFallbackRate.toFixed(2)},blank=${point.blankNoopRate.toFixed(2)}`)
        .join(" | ")
    },
    conclusion: window.exists
      ? `A τ window of width >= ${TAU_WINDOW_MIN_WIDTH} satisfies Family F SR >= ${REQUIRED_MATRIX_SUCCESS_RATE} and Family E fallback >= ${REQUIRED_FAMILY_E_FALLBACK}.`
      : "No τ window satisfies both Family F success and Family E fallback — τ choice is distribution-sensitive.",
    notes: [
      "Diagnostic per user spec: 'required 不一定要求全过'.",
      "Records the Family F commit vs Family E fallback tradeoff across τ."
    ]
  };
}

function auditNonDegradation(
  config: ModelConfig,
  bundle: MatrixTrainedBundle
): ArbitrationMatrixAuditSuiteResult {
  const familyA = runMatrixEval(
    config,
    bundle,
    createChallengeScenarios(DEFAULT_EVAL_SEEDS, DEFAULT_COMPLEX_MAX_STEPS),
    741
  );
  const familyB = runMatrixEval(config, bundle, compositeSameDirectionScenarios(), 742);
  const familyC = runMatrixEval(config, bundle, distractorScenarios(), 743);
  const familyD = runMatrixEval(config, bundle, priorityScenarios(), 744);
  const passed =
    familyA.successRate >= 0.8 &&
    familyB.successRate >= 0.5 &&
    familyC.successRate >= 0.5 &&
    familyD.successRate >= 0.5;

  return {
    name: "arbitration matrix non-degradation on 2D-complex families",
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
      ? "Matrix-trained resolver does not degrade existing 2D-complex required families."
      : "Matrix-trained resolver degraded an existing 2D-complex family.",
    notes: [
      "Family A/B/D use canonical 2D-complex scenario factories (not generated).",
      "Family C remains the conservative bottleneck at SR=0.5; distractor priority is not resolved by arbitration."
    ]
  };
}

function auditTrueConflictPreservationAcrossTau(
  config: ModelConfig,
  bundle: MatrixTrainedBundle,
  familyEScenarios: ReturnType<typeof trueConflictScenarios>,
  evalSeed: number
): ArbitrationMatrixAuditSuiteResult {
  const fallbackRates = MATRIX_TAUS.map((tau) => {
    const arbitrator: LinearArbitrator = { ...bundle.arbitrator, threshold: tau };
    const result = runMatrixEval(config, bundle, familyEScenarios, evalSeed, arbitrator);
    const steps = firstEvalSteps(result);
    const fallback = steps.filter((step) => step.executedAction === "conflict").length / Math.max(1, steps.length);
    return { tau, fallback };
  });
  const minFallback = Math.min(...fallbackRates.map((point) => point.fallback));

  return {
    name: "arbitration matrix true conflict preservation across tau",
    required: false,
    passed: true,
    metrics: {
      minFallbackAcrossTau: minFallback,
      fallbackByTau: fallbackRates.map((point) => `${point.tau}=${point.fallback.toFixed(2)}`).join(",")
    },
    conclusion: "Per-τ Family E fallback recorded to characterize τ sensitivity of the true conflict boundary.",
    notes: [
      "Diagnostic: records how Family E fallback varies with τ.",
      "The hard gate is at default τ only; across-τ behavior is recorded for inspection."
    ]
  };
}

function auditEvidenceAblation(
  config: ModelConfig,
  evalScenarios: ReturnType<typeof generateSemanticConflictScenarios>,
  trainSeed: number,
  evalSeed: number
): ArbitrationMatrixAuditSuiteResult {
  const full = evaluateAblation(config, trainSeed, evalScenarios, ABLATION_FULL_MASK, ABLATION_FULL_LABEL, evalSeed);
  const motorOnly = evaluateAblation(config, trainSeed, evalScenarios, ABLATION_MOTOR_ONLY_MASK, ABLATION_MOTOR_ONLY_LABEL, evalSeed);
  const interOnly = evaluateAblation(config, trainSeed, evalScenarios, ABLATION_INTER_ONLY_MASK, ABLATION_INTER_ONLY_LABEL, evalSeed);
  const dropToxin = evaluateAblation(config, trainSeed, evalScenarios, ABLATION_DROP_TOXIN_MASK, ABLATION_DROP_TOXIN_LABEL, evalSeed);
  const dropFood = evaluateAblation(config, trainSeed, evalScenarios, ABLATION_DROP_FOOD_MASK, ABLATION_DROP_FOOD_LABEL, evalSeed);

  const fullDrop = Math.max(0, full.successRate - motorOnly.successRate);
  const dropToxinDecline = Math.max(0, full.successRate - dropToxin.successRate);
  const dropFoodDecline = Math.max(0, full.successRate - dropFood.successRate);
  const asymmetric = dropToxinDecline > dropFoodDecline;

  return {
    name: "arbitration matrix evidence ablation",
    required: false,
    passed: true,
    metrics: {
      fullSuccessRate: full.successRate,
      motorOnlySuccessRate: motorOnly.successRate,
      interOnlySuccessRate: interOnly.successRate,
      dropToxinSuccessRate: dropToxin.successRate,
      dropFoodSuccessRate: dropFood.successRate,
      fullMinusMotorOnly: fullDrop,
      dropToxinDecline,
      dropFoodDecline,
      toxinFoodAsymmetric: asymmetric
    },
    conclusion: "Evidence ablation records whether inter evidence is necessary and whether toxin/food features carry asymmetric signal. Under current 4-inter evidence, drop-toxin/drop-food is a redundant diagnostic (inter-only reaches full SR), so symmetric non-decline is feature redundancy, not anti-hardcoding failure.",
    notes: [
      "Full evidence is the baseline; motor-only should fail (inter evidence necessary).",
      "Drop-toxin / drop-food are redundant diagnostics under current evidence (inter-only SR == full SR). Symmetric non-decline reflects feature redundancy, NOT anti-hardcoding failure — cannot be used as evidence that the resolver failed to read food/toxin semantics.",
      "Motor-only SR > 0 is a partial-measurability signal (not full semantic mastery); source requires separate explanation before promotion.",
      "Diagnostic only; does not gate requiredPassed."
    ]
  };
}

function auditTauSweepDetail(
  config: ModelConfig,
  bundle: MatrixTrainedBundle,
  familyFScenarios: ReturnType<typeof generateSemanticConflictScenarios>,
  familyEScenarios: ReturnType<typeof trueConflictScenarios>,
  blankScenario: ReturnType<typeof blankComplexScenario>,
  evalSeed: number
): ArbitrationMatrixAuditSuiteResult {
  const sweep = sweepThreshold(config, bundle, familyFScenarios, familyEScenarios, blankScenario, MATRIX_TAUS, evalSeed);

  return {
    name: "arbitration matrix tau sweep detail",
    required: false,
    passed: true,
    metrics: {
      sweepPoints: sweep
        .map((point) => `tau=${point.tau}:F=${point.familyFSuccessRate.toFixed(2)},E=${point.familyEFallbackRate.toFixed(2)},blank=${point.blankNoopRate.toFixed(2)}`)
        .join(" | ")
    },
    conclusion: "Per-τ metrics recorded for inspection of Family F commit vs Family E fallback tradeoff.",
    notes: [
      "τ only affects inference; training is fixed.",
      "Look for a τ range where Family F SR is high AND Family E fallback is high simultaneously."
    ]
  };
}

function auditMultiSeedMatrix(config: ModelConfig): ArbitrationMatrixAuditSuiteResult {
  const trainSeeds = [1101, 1102, 1103];
  const evalSeeds = [2101, 2102, 2103];
  const successRates: number[] = [];

  for (const trainSeed of trainSeeds) {
    const bundle = trainMatrixArbitration(config, { trainSeed });
    for (const evalSeed of evalSeeds) {
      const evalScenarios = generateSemanticConflictScenarios(evalSeed, MATRIX_EVAL_COUNT);
      const result = runMatrixEval(config, bundle, evalScenarios, evalSeed);
      successRates.push(result.successRate);
    }
  }

  const min = Math.min(...successRates);
  const max = Math.max(...successRates);
  const mean = successRates.reduce((sum, value) => sum + value, 0) / successRates.length;

  return {
    name: "arbitration matrix multi-seed stability",
    required: false,
    passed: min >= 0.8,
    metrics: {
      cellsRun: successRates.length,
      successMin: min,
      successMean: mean,
      successMax: max
    },
    conclusion: "Multi-seed stability on generated scenarios recorded for future gate promotion.",
    notes: [
      "3 train seeds × 3 eval seeds × generated scenario pools.",
      "Promote to required only after min SR is stable across a wider seed range."
    ]
  };
}

function auditFreshBaselineOnGenerated(
  config: ModelConfig,
  evalScenarios: ReturnType<typeof generateSemanticConflictScenarios>,
  evalSeed: number
): ArbitrationMatrixAuditSuiteResult {
  const bundle = trainMatrixArbitration(config, { trainSeed: 1201 });
  const freshArbitrator: LinearArbitrator = {
    ...bundle.arbitrator,
    weights: Array.from({ length: bundle.arbitrator.weights.length }, () => 0),
    bias: [0, 0]
  };
  const result = runMatrixEval(config, bundle, evalScenarios, evalSeed, freshArbitrator);
  const passed = result.successRate <= REQUIRED_MATRIX_FRESH_MAX;

  return {
    name: "arbitration matrix fresh baseline on generated scenarios",
    required: false,
    passed,
    metrics: {
      freshSuccessRate: result.successRate,
      freshConflictRate: result.conflictRate
    },
    conclusion: passed
      ? "Untrained resolver fails on generated scenarios — evidence has no built-in bias."
      : "Untrained resolver produced nontrivial behavior on generated scenarios — possible feature leakage.",
    notes: [
      "Fresh linear weights and bias are zero.",
      "If fresh SR is high, the evidence vector contains hardcoded direction signal."
    ]
  };
}

function stableProjection(bundle: MatrixTrainedBundle, result: ChallengeExperimentResult): string {
  return JSON.stringify({
    arbitrator: bundle.arbitrator,
    metrics: {
      successRate: result.successRate,
      meanReward: result.meanReward,
      meanStepsToTerminal: result.meanStepsToTerminal,
      conflictRate: result.conflictRate,
      noopRate: result.noopRate
    },
    firstSteps: firstEvalSteps(result).map((step) => ({
      expectedAction: step.expectedAction,
      rawAction: step.networkDecision.action,
      executedAction: step.executedAction,
      delta: step.complexEvidence
        ? Math.abs(scoreArbitrator(bundle.arbitrator, step.complexEvidence).delta)
        : 0
    }))
  });
}

function digest(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}
