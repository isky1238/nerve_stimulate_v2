import { ModelConfig } from "../config/newModelConfig";
import { WorldAction } from "../core/arbitration";
import { createOfflineLearningNetwork, LearningNetwork } from "../core/evaluation";
import { SeededRandom } from "../core/random";
import {
  ChallengeComplexEvidence,
  ChallengeEpisodeTrace,
  ChallengeExperimentOptions,
  ChallengeExperimentResult,
  ChallengeScenario,
  ChallengeTraceStep,
  RewardAdvantageState,
  countLearningEvents,
  createChallengeScenarios,
  scoreChallengeStep,
  shuffleScenarios,
  stepChallengeWorld
} from "./challenge2d";
import { ComplexActionResolver, runComplexEpisode } from "./complex2d";
import { WorldState } from "./world2d";

export const ARBITRATION_FEATURE_IDS = [
  "iFoodLeft",
  "iFoodRight",
  "iToxinLeft",
  "iToxinRight",
  "leftMotor",
  "rightMotor"
] as const;

export type ArbitrationFeatureId = (typeof ARBITRATION_FEATURE_IDS)[number];

export interface LinearArbitrator {
  weights: number[];
  bias: number[];
  threshold: number;
}

export interface ArbitrationTrainingRecord {
  evidence: ChallengeComplexEvidence;
  expectedAction: WorldAction;
  rawAction: WorldAction;
  scenarioId: string;
  phase: string;
  before: WorldState;
}

export interface TrainArbitratorOptions {
  learningRate?: number;
  steps?: number;
  threshold?: number;
  reversePrior?: boolean;
  featureMask?: boolean[];
}

export interface ArbitratorScore {
  logitLeft: number;
  logitRight: number;
  delta: number;
  action: WorldAction;
}

export interface ArbitratedExperimentOptions extends ChallengeExperimentOptions {
  arbitrator?: LinearArbitrator | null;
}

const DEFAULT_THRESHOLD = 0.1;
const DEFAULT_LEARNING_RATE = 0.1;
const DEFAULT_TRAINING_STEPS = 200;

export function createInitialArbitrator(threshold = DEFAULT_THRESHOLD): LinearArbitrator {
  return {
    weights: Array.from({ length: ARBITRATION_FEATURE_IDS.length * 2 }, () => 0),
    bias: [0, 0],
    threshold
  };
}

export function trainArbitrator(
  records: ArbitrationTrainingRecord[],
  options: TrainArbitratorOptions = {}
): LinearArbitrator {
  const arbitrator = createInitialArbitrator(options.threshold ?? DEFAULT_THRESHOLD);
  const learningRate = options.learningRate ?? DEFAULT_LEARNING_RATE;
  const steps = options.steps ?? DEFAULT_TRAINING_STEPS;
  const samples = records
    .map((record) => ({
      features: maskedFeatures(arbitrationEvidenceVector(record.evidence), options.featureMask),
      label: options.reversePrior ? reversePriorityAction(record.expectedAction) : record.expectedAction
    }))
    .filter((sample) => sample.label === "left" || sample.label === "right" || sample.label === "conflict");

  if (samples.length === 0) {
    return arbitrator;
  }

  for (let step = 0; step < steps; step += 1) {
    const weightGradients = Array.from({ length: arbitrator.weights.length }, () => 0);
    const biasGradients = [0, 0];

    for (const sample of samples) {
      const logits = computeLogits(arbitrator, sample.features);
      const maxLogit = Math.max(logits[0], logits[1]);
      const expLeft = Math.exp(logits[0] - maxLogit);
      const expRight = Math.exp(logits[1] - maxLogit);
      const denominator = expLeft + expRight;
      const pLeft = expLeft / denominator;
      const pRight = expRight / denominator;
      const targetLeft = sample.label === "left" ? 1 : sample.label === "conflict" ? 0.5 : 0;
      const targetRight = sample.label === "right" ? 1 : sample.label === "conflict" ? 0.5 : 0;
      const gradLeft = pLeft - targetLeft;
      const gradRight = pRight - targetRight;

      for (let index = 0; index < sample.features.length; index += 1) {
        weightGradients[index] += gradLeft * sample.features[index];
        weightGradients[index + sample.features.length] += gradRight * sample.features[index];
      }
      biasGradients[0] += gradLeft;
      biasGradients[1] += gradRight;
    }

    for (let index = 0; index < arbitrator.weights.length; index += 1) {
      arbitrator.weights[index] -= learningRate * (weightGradients[index] / samples.length);
    }
    arbitrator.bias[0] -= learningRate * (biasGradients[0] / samples.length);
    arbitrator.bias[1] -= learningRate * (biasGradients[1] / samples.length);
  }

  return arbitrator;
}

export function trainRewardArbitrator(
  records: ArbitrationTrainingRecord[],
  options: TrainArbitratorOptions & { seed?: number } = {}
): LinearArbitrator {
  const arbitrator = createInitialArbitrator(options.threshold ?? DEFAULT_THRESHOLD);
  const learningRate = options.learningRate ?? 0.03;
  const steps = options.steps ?? 300;
  const rng = new SeededRandom(options.seed ?? 1);
  const samples = records.filter((record) => record.expectedAction === "left" || record.expectedAction === "right");

  if (samples.length === 0) {
    return arbitrator;
  }

  for (let step = 0; step < steps; step += 1) {
    const sample = samples[step % samples.length];
    const features = maskedFeatures(arbitrationEvidenceVector(sample.evidence), options.featureMask);
    const logits = computeLogits(arbitrator, features);
    const maxLogit = Math.max(logits[0], logits[1]);
    const expLeft = Math.exp(logits[0] - maxLogit);
    const expRight = Math.exp(logits[1] - maxLogit);
    const pLeft = expLeft / (expLeft + expRight);
    const action: Exclude<WorldAction, "noop" | "conflict"> = rng.next() < pLeft ? "left" : "right";
    const reward = scoreChallengeStep(sample.before, stepChallengeWorld(sample.before, action), action).reward;
    const selectedLeft = action === "left" ? 1 : 0;
    const selectedRight = action === "right" ? 1 : 0;

    for (let index = 0; index < features.length; index += 1) {
      arbitrator.weights[index] += learningRate * reward * (selectedLeft - pLeft) * features[index];
      arbitrator.weights[index + features.length] += learningRate * reward * (selectedRight - (1 - pLeft)) * features[index];
    }
    arbitrator.bias[0] += learningRate * reward * (selectedLeft - pLeft);
    arbitrator.bias[1] += learningRate * reward * (selectedRight - (1 - pLeft));
  }

  return arbitrator;
}

export function inferArbitrator(arbitrator: LinearArbitrator, evidence: ChallengeComplexEvidence): WorldAction {
  return scoreArbitrator(arbitrator, evidence).action;
}

export function scoreArbitrator(arbitrator: LinearArbitrator, evidence: ChallengeComplexEvidence): ArbitratorScore {
  const [logitLeft, logitRight] = computeLogits(arbitrator, arbitrationEvidenceVector(evidence));
  const delta = logitLeft - logitRight;
  const action =
    Math.abs(delta) < arbitrator.threshold
      ? "conflict"
      : delta > 0
        ? "left"
        : "right";

  return {
    logitLeft,
    logitRight,
    delta,
    action
  };
}

export function runArbitratedExperiment(
  config: ModelConfig,
  options: ArbitratedExperimentOptions
): ChallengeExperimentResult {
  const maxSteps = options.maxSteps ?? 6;
  const observationDropout = options.observationDropout ?? 0;
  const trainingScenarios =
    options.trainingScenarios ?? createChallengeScenarios(options.trainSeeds, maxSteps);
  const evaluationScenarios =
    options.evaluationScenarios ?? createChallengeScenarios(options.evalSeeds, maxSteps);
  const network = options.initialNetwork ?? createOfflineLearningNetwork(config);
  const episodes: ChallengeEpisodeTrace[] = [];
  const actionResolver = createActionResolver(options.arbitrator ?? null);
  let rewardUpdateCount = 0;
  let supervisedUpdateCount = 0;
  let captureUpdateCount = 0;
  let decayUpdateCount = 0;
  const rewardAdvantageState: RewardAdvantageState = { baseline: 0 };

  for (let epoch = 0; epoch < options.epochs; epoch += 1) {
    const epochScenarios = shuffleScenarios(trainingScenarios, options.seed + epoch);

    for (const scenario of epochScenarios) {
      const episode = runComplexEpisode(network, scenario, config, {
        phase: "train",
        learningMode: options.learningMode,
        learningEnabled: options.learningMode !== "frozen",
        seed: options.seed + epoch * 1000 + scenario.seed,
        observationDropout,
        reverseMapping: false,
        rewardAdvantageState,
        actionResolver
      });
      const counts = countLearningEvents(episode);
      rewardUpdateCount += counts.rewardUpdateCount;
      supervisedUpdateCount += counts.supervisedUpdateCount;
      captureUpdateCount += counts.captureUpdateCount;
      decayUpdateCount += counts.decayUpdateCount;
      episodes.push(episode);
    }
  }

  const evaluationEpisodes = evaluationScenarios.map((scenario, index) =>
    runComplexEpisode(network, scenario, config, {
      phase: "eval",
      learningMode: options.learningMode,
      learningEnabled: false,
      seed: options.seed * 100000 + index,
      observationDropout,
      reverseMapping: false,
      actionResolver
    })
  );
  episodes.push(...evaluationEpisodes);

  const evalSteps = evaluationEpisodes.flatMap((episode) => episode.steps);
  const successRate = evaluationEpisodes.filter((episode) => episode.success).length / Math.max(1, evaluationEpisodes.length);
  const meanReward =
    evaluationEpisodes.reduce((sum, episode) => sum + episode.totalReward, 0) / Math.max(1, evaluationEpisodes.length);
  const meanStepsToTerminal =
    evaluationEpisodes.reduce((sum, episode) => sum + episode.steps.length, 0) / Math.max(1, evaluationEpisodes.length);
  const conflictRate =
    evalSteps.filter((step) => step.executedAction === "conflict").length / Math.max(1, evalSteps.length);
  const noopRate = evalSteps.filter((step) => step.executedAction === "noop").length / Math.max(1, evalSteps.length);

  return {
    trace: {
      version: "dg-snn-arbitrated-2d-trace-v0.1",
      seed: options.seed,
      trainSeeds: [...options.trainSeeds],
      evalSeeds: [...options.evalSeeds],
      config: {
        width: 7,
        height: 7,
        maxSteps,
        epochs: options.epochs,
        learningMode: options.learningMode,
        observationDropout,
        reverseMapping: false,
        rewardAdvantageBaselineAlpha: config.rewardAdvantageBaselineAlpha,
        explorationStrategy: config.explorationStrategy,
        explorationEpsilon: config.explorationEpsilon
      },
      episodes
    },
    network,
    successRate,
    meanReward,
    meanStepsToTerminal,
    conflictRate,
    noopRate,
    rewardUpdateCount,
    supervisedUpdateCount,
    captureUpdateCount,
    decayUpdateCount
  };
}

export function recordArbitrationEvidence(
  episodes: ChallengeEpisodeTrace[],
  options: { onlyRawConflict?: boolean; includeExpectedConflict?: boolean } = {}
): ArbitrationTrainingRecord[] {
  const onlyRawConflict = options.onlyRawConflict ?? true;
  const includeExpectedConflict = options.includeExpectedConflict ?? false;
  const records: ArbitrationTrainingRecord[] = [];

  for (const episode of episodes) {
    for (const step of episode.steps) {
      if (!step.complexEvidence) {
        continue;
      }
      if (onlyRawConflict && step.networkDecision.action !== "conflict") {
        continue;
      }
      if (!includeExpectedConflict && step.expectedAction !== "left" && step.expectedAction !== "right") {
        continue;
      }
      records.push({
        evidence: step.complexEvidence,
        expectedAction: step.expectedAction,
        rawAction: step.networkDecision.action,
        scenarioId: episode.scenarioId,
        phase: episode.phase,
        before: step.before
      });
    }
  }

  return records;
}

export function arbitrationEvidenceVector(evidence: ChallengeComplexEvidence): number[] {
  return [
    evidence.interSpikeCounts.iFoodLeft ?? 0,
    evidence.interSpikeCounts.iFoodRight ?? 0,
    evidence.interSpikeCounts.iToxinLeft ?? 0,
    evidence.interSpikeCounts.iToxinRight ?? 0,
    evidence.motorSpikeCounts.leftMotor ?? 0,
    evidence.motorSpikeCounts.rightMotor ?? 0
  ];
}

export function reversePriorityAction(action: WorldAction): WorldAction {
  if (action === "left") {
    return "right";
  }
  if (action === "right") {
    return "left";
  }
  return action;
}

export function firstEvalSteps(result: ChallengeExperimentResult): ChallengeTraceStep[] {
  return result.trace.episodes
    .filter((episode) => episode.phase === "eval")
    .map((episode) => episode.steps[0])
    .filter((step): step is ChallengeTraceStep => step !== undefined);
}

function createActionResolver(arbitrator: LinearArbitrator | null): ComplexActionResolver | undefined {
  if (!arbitrator) {
    return undefined;
  }

  return ({ evidence }) => inferArbitrator(arbitrator, evidence);
}

function computeLogits(arbitrator: LinearArbitrator, features: number[]): [number, number] {
  let left = arbitrator.bias[0] ?? 0;
  let right = arbitrator.bias[1] ?? 0;
  for (let index = 0; index < features.length; index += 1) {
    left += (arbitrator.weights[index] ?? 0) * features[index];
    right += (arbitrator.weights[index + features.length] ?? 0) * features[index];
  }
  return [left, right];
}

function maskedFeatures(features: number[], featureMask?: boolean[]): number[] {
  if (!featureMask) {
    return features;
  }

  return features.map((value, index) => (featureMask[index] ? value : 0));
}
