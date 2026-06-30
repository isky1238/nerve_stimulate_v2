import { ModelConfig } from "../config/newModelConfig";
import { SeededRandom } from "../core/random";
import {
  ChallengeExperimentResult,
  ChallengeScenario,
  DEFAULT_EVAL_SEEDS,
  DEFAULT_TRAIN_SEEDS,
  createChallengeScenarios
} from "./challenge2d";
import {
  DEFAULT_COMPLEX_MAX_STEPS,
  createComplexConfig,
  runComplexExperiment,
  trueConflictScenarios
} from "./complex2d";
import {
  ArbitrationTrainingRecord,
  LinearArbitrator,
  TrainArbitratorOptions,
  firstEvalSteps,
  recordArbitrationEvidence,
  runArbitratedExperiment,
  trainArbitrator
} from "./arbitration2d";

export interface MatrixScenarioGeneratorOptions {
  maxSteps?: number;
  distances?: number[];
  sides?: Array<"left" | "right">;
}

export interface MatrixTrainingOptions {
  trainSeed: number;
  trainScenarioCount?: number;
  calibrationScenarios?: ChallengeScenario[];
  includeCalibration?: boolean;
  arbitratorOptions?: TrainArbitratorOptions;
}

export interface MatrixTrainedBundle {
  pretrain: ChallengeExperimentResult;
  arbitrator: LinearArbitrator;
  trainScenarios: ChallengeScenario[];
  semanticRecordCount: number;
  calibrationRecordCount: number;
}

export interface ThresholdSweepPoint {
  tau: number;
  familyFSuccessRate: number;
  familyEFallbackRate: number;
  blankNoopRate: number;
}

export interface AblationResult {
  label: string;
  featureMask: boolean[];
  successRate: number;
  firstActionAccuracy: number;
  conflictRate: number;
  recordCount: number;
}

const DEFAULT_MATRIX_TRAIN_COUNT = 24;
const DEFAULT_MATRIX_EVAL_COUNT = 48;
const DEFAULT_MATRIX_TAUS = [0.05, 0.1, 0.2, 0.3];
const SEMANTIC_RECORD_REPEAT = 1;
const FAMILY_E_RECORD_REPEAT = 24;
const CALIBRATION_RECORD_REPEAT = 6;
const DEFAULT_EPOCHS = 40;

export function generateSemanticConflictScenarios(
  seed: number,
  count: number,
  options: MatrixScenarioGeneratorOptions = {}
): ChallengeScenario[] {
  const maxSteps = options.maxSteps ?? DEFAULT_COMPLEX_MAX_STEPS;
  const distances = options.distances ?? [1, 2, 3];
  const sides = options.sides ?? ["left", "right"];
  const rng = new SeededRandom(seed);
  const width = 7;
  const height = 7;
  const center = { x: Math.floor(width / 2), y: Math.floor(height / 2) };
  const scenarios: ChallengeScenario[] = [];

  for (let index = 0; index < count; index += 1) {
    const distance = distances[Math.floor(rng.next() * distances.length)];
    const side = sides[Math.floor(rng.next() * sides.length)];
    const offset = side === "left" ? -distance : distance;
    const scenarioSeed = seed * 1000 + index;
    scenarios.push({
      id: `matrix-semantic-${seed}-${index}-d${distance}-${side}`,
      seed: scenarioSeed,
      width,
      height,
      maxSteps,
      agentStart: { ...center },
      objects: [
        {
          id: `toxin-${side}-d${distance}`,
          kind: "toxin",
          position: { x: center.x + offset, y: center.y }
        },
        {
          id: `food-${side}-d${distance}`,
          kind: "food",
          position: { x: center.x + offset, y: center.y }
        }
      ]
    });
  }

  return scenarios;
}

export function trainMatrixArbitration(
  config: ModelConfig,
  options: MatrixTrainingOptions
): MatrixTrainedBundle {
  const auditConfig = createComplexConfig(config);
  const trainScenarios = generateSemanticConflictScenarios(
    options.trainSeed,
    options.trainScenarioCount ?? DEFAULT_MATRIX_TRAIN_COUNT
  );
  const pretrain = runComplexExperiment(auditConfig, {
    seed: options.trainSeed,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: DEFAULT_EPOCHS,
    learningMode: "supervised"
  });

  const semanticTrainRaw = runArbitratedExperiment(auditConfig, {
    seed: options.trainSeed,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: pretrain.network,
    arbitrator: null,
    evaluationScenarios: trainScenarios
  });
  const semanticRecords = recordArbitrationEvidence(semanticTrainRaw.trace.episodes, {
    onlyRawConflict: true
  });

  const calibrationScenarios =
    options.calibrationScenarios ??
    createChallengeScenarios(DEFAULT_TRAIN_SEEDS, DEFAULT_COMPLEX_MAX_STEPS);
  const includeCalibration = options.includeCalibration ?? true;
  let calibrationRecords: ArbitrationTrainingRecord[] = [];
  if (includeCalibration) {
    const calibrationRaw = runComplexExperiment(auditConfig, {
      seed: options.trainSeed,
      trainSeeds: DEFAULT_TRAIN_SEEDS,
      evalSeeds: DEFAULT_EVAL_SEEDS,
      epochs: 0,
      learningMode: "frozen",
      initialNetwork: pretrain.network,
      evaluationScenarios: calibrationScenarios
    });
    calibrationRecords = recordArbitrationEvidence(calibrationRaw.trace.episodes, {
      onlyRawConflict: false
    });
  }

  const familyERaw = runComplexExperiment(auditConfig, {
    seed: options.trainSeed,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: pretrain.network,
    evaluationScenarios: trueConflictScenarios()
  });
  const familyERecords = recordArbitrationEvidence(familyERaw.trace.episodes, {
    onlyRawConflict: true,
    includeExpectedConflict: true
  });

  const repeatedSemantic: ArbitrationTrainingRecord[] = [];
  for (let repeat = 0; repeat < SEMANTIC_RECORD_REPEAT; repeat += 1) {
    repeatedSemantic.push(...semanticRecords);
  }
  const repeatedCalibration: ArbitrationTrainingRecord[] = [];
  for (let repeat = 0; repeat < CALIBRATION_RECORD_REPEAT; repeat += 1) {
    repeatedCalibration.push(...calibrationRecords);
  }
  const repeatedFamilyE: ArbitrationTrainingRecord[] = [];
  for (let repeat = 0; repeat < FAMILY_E_RECORD_REPEAT; repeat += 1) {
    repeatedFamilyE.push(...familyERecords);
  }
  const trainingRecords = [...repeatedCalibration, ...repeatedSemantic, ...repeatedFamilyE];
  const arbitrator = trainArbitrator(trainingRecords, {
    threshold: 0.1,
    learningRate: 0.08,
    steps: 400,
    ...options.arbitratorOptions
  });

  return {
    pretrain,
    arbitrator,
    trainScenarios,
    semanticRecordCount: semanticRecords.length,
    calibrationRecordCount: calibrationRecords.length
  };
}

export function runMatrixEval(
  config: ModelConfig,
  bundle: MatrixTrainedBundle,
  evalScenarios: ChallengeScenario[],
  seed: number,
  arbitratorOverride?: LinearArbitrator | null
): ChallengeExperimentResult {
  const auditConfig = createComplexConfig(config);
  return runArbitratedExperiment(auditConfig, {
    seed,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: bundle.pretrain.network,
    arbitrator: arbitratorOverride === undefined ? bundle.arbitrator : arbitratorOverride,
    evaluationScenarios: evalScenarios
  });
}

export function sweepThreshold(
  config: ModelConfig,
  bundle: MatrixTrainedBundle,
  familyFScenarios: ChallengeScenario[],
  familyEScenarios: ChallengeScenario[],
  blankScenario: ChallengeScenario,
  taus: number[] = DEFAULT_MATRIX_TAUS,
  seed: number
): ThresholdSweepPoint[] {
  return taus.map((tau) => {
    const arbitrator: LinearArbitrator = { ...bundle.arbitrator, threshold: tau };
    const familyF = runMatrixEval(config, bundle, familyFScenarios, seed, arbitrator);
    const familyE = runMatrixEval(config, bundle, familyEScenarios, seed, arbitrator);
    const blank = runMatrixEval(config, bundle, [blankScenario], seed, arbitrator);
    const familyEFirstSteps = firstEvalSteps(familyE);
    const familyEFallbackRate =
      familyEFirstSteps.filter((step) => step.executedAction === "conflict").length /
      Math.max(1, familyEFirstSteps.length);

    return {
      tau,
      familyFSuccessRate: familyF.successRate,
      familyEFallbackRate,
      blankNoopRate: blank.noopRate
    };
  });
}

export function evaluateAblation(
  config: ModelConfig,
  trainSeed: number,
  evalScenarios: ChallengeScenario[],
  featureMask: boolean[],
  label: string,
  evalSeed: number
): AblationResult {
  const bundle = trainMatrixArbitration(config, {
    trainSeed,
    arbitratorOptions: { featureMask }
  });
  const result = runMatrixEval(config, bundle, evalScenarios, evalSeed);
  const steps = firstEvalSteps(result);
  const labeled = steps.filter(
    (step) => step.expectedAction === "left" || step.expectedAction === "right"
  );
  const correct = labeled.filter((step) => step.executedAction === step.expectedAction).length;

  return {
    label,
    featureMask,
    successRate: result.successRate,
    firstActionAccuracy: correct / Math.max(1, labeled.length),
    conflictRate: result.conflictRate,
    recordCount: bundle.semanticRecordCount + bundle.calibrationRecordCount
  };
}

export function findTauAcceptanceWindow(
  sweep: ThresholdSweepPoint[],
  options: { minFamilyFSuccessRate?: number; minFamilyEFallback?: number; minWidth?: number } = {}
): { exists: boolean; minTau: number | null; maxTau: number | null; satisfyingTaus: number[] } {
  const minSR = options.minFamilyFSuccessRate ?? 0.8;
  const minFallback = options.minFamilyEFallback ?? 0.9;
  const minWidth = options.minWidth ?? 0.1;
  const satisfyingTaus = sweep
    .filter((point) => point.familyFSuccessRate >= minSR && point.familyEFallbackRate >= minFallback)
    .map((point) => point.tau)
    .sort((a, b) => a - b);

  if (satisfyingTaus.length === 0) {
    return { exists: false, minTau: null, maxTau: null, satisfyingTaus };
  }

  const minTau = satisfyingTaus[0];
  const maxTau = satisfyingTaus[satisfyingTaus.length - 1];
  const exists = maxTau - minTau >= minWidth || satisfyingTaus.length >= 2;

  return { exists, minTau, maxTau, satisfyingTaus };
}

export const MATRIX_TAUS = DEFAULT_MATRIX_TAUS;
export const MATRIX_TRAIN_COUNT = DEFAULT_MATRIX_TRAIN_COUNT;
export const MATRIX_EVAL_COUNT = DEFAULT_MATRIX_EVAL_COUNT;
