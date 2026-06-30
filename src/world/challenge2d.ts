import { ModelConfig, withConfig } from "../config/newModelConfig";
import { ActionDecision, WorldAction, arbitrateMotorAction } from "../core/arbitration";
import type { LearningNetwork } from "../core/evaluation";
import {
  activeMotorIds,
  applyMaintenanceDecayAndCapture,
  applyRewardOutcomeLearning,
  applySupervisedMotorOutcomeLearning,
  computeRewardOutcomeSignal,
  clearSensoryOutputs,
  forceExplorationMotor,
  propagateAndIntegrateRole,
  resetNetworkRuntime,
  selectExplorationAction,
  setSensoryOutputs,
  updateNetworkEligibility
} from "../core/mechanism";
import { SeededRandom } from "../core/random";
import {
  blankChallengeScenario,
  CHALLENGE_HEIGHT,
  CHALLENGE_WIDTH,
  ChallengeAversiveTag,
  ChallengeRawObservation,
  ChallengeScenario,
  ChallengeTerminalReason,
  createChallengeScenarios,
  createChallengeWorldState,
  deriveChallengeAversiveTag,
  DEFAULT_CHALLENGE_MAX_STEPS,
  DEFAULT_EVAL_SEEDS,
  DEFAULT_TRAIN_SEEDS,
  expectedActionForChallengeState,
  conflictChallengeScenario,
  mapChallengeObservationToSensors,
  observeChallengeWorld,
  reverseAction,
  sameActionCompositeChallengeScenario,
  scoreChallengeStep,
  shuffleScenarios,
  stepChallengeWorld
} from "./challengeTask";
import { runExperimentWithRunner } from "./experimentRunner";
import { SensoryMapping, WorldState } from "./world2d";

export { activeMotorIds, forceExplorationMotor, selectExplorationAction } from "../core/mechanism";
export { countLearningEvents } from "./experimentRunner";
export {
  blankChallengeScenario,
  CHALLENGE_HEIGHT,
  CHALLENGE_WIDTH,
  conflictChallengeScenario,
  createChallengeScenarios,
  createChallengeWorldState,
  deriveChallengeAversiveTag,
  DEFAULT_CHALLENGE_MAX_STEPS,
  DEFAULT_EVAL_SEEDS,
  DEFAULT_TRAIN_SEEDS,
  expectedActionForChallengeState,
  mapChallengeObservationToSensors,
  observeChallengeWorld,
  reverseAction,
  sameActionCompositeChallengeScenario,
  scoreChallengeStep,
  shuffleScenarios,
  stepChallengeWorld
} from "./challengeTask";
export type {
  ChallengeObservedObject,
  ChallengeAversiveTag,
  ChallengeRawObservation,
  ChallengeScenario,
  ChallengeTerminalReason,
  ChallengeRewardResult
} from "./challengeTask";

export type ChallengeLearningMode = "supervised" | "rewardOnly" | "frozen";
export type ChallengeEpisodePhase = "train" | "eval";
export interface ChallengeLearningStep {
  rawActiveMotors: string[];
  activeMotors: string[];
  supervisedUpdates: number;
  rewardUpdates: number;
  captureUpdates: number;
  decayUpdates: number;
}

export interface ChallengeComplexEvidence {
  interSpikeCounts: Record<string, number>;
  motorSpikeCounts: Record<string, number>;
}

export interface ChallengeTraceStep {
  index: number;
  before: WorldState;
  rawObservation: ChallengeRawObservation;
  sensoryMapping: SensoryMapping;
  expectedAction: WorldAction;
  networkDecision: ActionDecision;
  explorationAction: WorldAction | null;
  executedAction: WorldAction;
  reward: number;
  rewardBaseline: number;
  rewardAdvantage: number;
  rewardSignal: number;
  aversiveTag?: ChallengeAversiveTag;
  distanceDelta: number;
  after: WorldState;
  terminalReason: ChallengeTerminalReason;
  terminal: boolean;
  success: boolean;
  learning: ChallengeLearningStep;
  complexEvidence?: ChallengeComplexEvidence;
}

export interface ChallengeEpisodeTrace {
  phase: ChallengeEpisodePhase;
  scenarioId: string;
  seed: number;
  episodeSeed: number;
  learningMode: ChallengeLearningMode;
  steps: ChallengeTraceStep[];
  totalReward: number;
  success: boolean;
  terminalReason: ChallengeTerminalReason;
}

export interface ChallengeExperimentTrace {
  version: string;
  seed: number;
  trainSeeds: number[];
  evalSeeds: number[];
  config: {
    width: number;
    height: number;
    maxSteps: number;
    epochs: number;
    learningMode: ChallengeLearningMode;
    observationDropout: number;
    reverseMapping: boolean;
    rewardAdvantageBaselineAlpha: number;
    aversiveTagStrategy: string;
    aversiveTagGain: number;
    aversiveAvoidanceBonus: number;
    aversiveDepotentiationRate: number;
    aversiveBadOutcomeThreshold: number;
    explorationStrategy: "conflictGated" | "epsilonGreedy";
    explorationEpsilon: number;
  };
  episodes: ChallengeEpisodeTrace[];
}

export interface ChallengeExperimentResult {
  trace: ChallengeExperimentTrace;
  network: LearningNetwork;
  successRate: number;
  meanReward: number;
  meanStepsToTerminal: number;
  conflictRate: number;
  noopRate: number;
  rewardUpdateCount: number;
  supervisedUpdateCount: number;
  captureUpdateCount: number;
  decayUpdateCount: number;
}

export interface ChallengeExperimentOptions {
  seed: number;
  trainSeeds: number[];
  evalSeeds: number[];
  epochs: number;
  learningMode: ChallengeLearningMode;
  observationDropout?: number;
  maxSteps?: number;
  trainingScenarios?: ChallengeScenario[];
  evaluationScenarios?: ChallengeScenario[];
  initialNetwork?: LearningNetwork;
  reverseMapping?: boolean;
  /**
   * Per-epoch diagnostic hook. Called at the end of each training epoch with the
   * current network and the training episodes run during that epoch. Used by
   * collapse diagnostics to sample bilateral fastWeight symmetry and per-epoch
   * conflict/noop rates. Undefined by default — does not affect existing runs.
   */
  epochProbe?: (
    epoch: number,
    network: LearningNetwork,
    epochEpisodes: ChallengeEpisodeTrace[]
  ) => void;
}

interface NetworkStepResult {
  rawActiveMotors: string[];
  activeMotors: string[];
  networkDecision: ActionDecision;
  explorationAction: WorldAction | null;
  executedAction: WorldAction;
  supervisedUpdates: number;
}

export interface RewardAdvantageState {
  baseline: number;
}

export function createChallengeConfig(config: ModelConfig): ModelConfig {
  return withConfig({
    ...config,
    leak: 1,
    branchLocalThreshold: 0.1,
    dendriteGateThreshold: 0.1,
    axonThreshold: 1,
    thresholdAdaptRate: 0,
    refractorySteps: 0,
    fastDecay: 0.9995,
    stableThreshold: 0.12,
    useThreshold: 0.08,
    depotentiationRate: 0.64
  });
}

export function runChallengeExperiment(
  config: ModelConfig,
  options: ChallengeExperimentOptions
): ChallengeExperimentResult {
  return runExperimentWithRunner(config, options, {
    traceVersion: "dg-snn-2d-challenge-trace-v0.1",
    width: CHALLENGE_WIDTH,
    height: CHALLENGE_HEIGHT,
    defaultMaxSteps: DEFAULT_CHALLENGE_MAX_STEPS,
    useReverseMapping: true,
    createScenarios: createChallengeScenarios,
    runEpisode: runChallengeEpisode
  });
}

export function runChallengeEpisode(
  network: LearningNetwork,
  scenario: ChallengeScenario,
  config: ModelConfig,
  options: {
    phase: ChallengeEpisodePhase;
    learningMode: ChallengeLearningMode;
    learningEnabled: boolean;
    seed: number;
    observationDropout: number;
    reverseMapping: boolean;
    rewardAdvantageState?: RewardAdvantageState;
  }
): ChallengeEpisodeTrace {
  const rng = new SeededRandom(options.seed);
  let state = createChallengeWorldState(scenario);
  const steps: ChallengeTraceStep[] = [];
  let totalReward = 0;
  let terminalReason: ChallengeTerminalReason = "step-limit";
  let success = false;

  for (let stepIndex = 0; stepIndex < scenario.maxSteps; stepIndex += 1) {
    const rawObservation = observeChallengeWorld(state, options.observationDropout, rng);
    const sensoryMapping = mapChallengeObservationToSensors(rawObservation);
    const rawExpectedAction = expectedActionForChallengeState(state);
    const expectedAction =
      options.reverseMapping && options.phase === "train"
        ? reverseAction(rawExpectedAction)
        : rawExpectedAction;
    const networkStep = runChallengeNetworkStep(network, sensoryMapping.activeSensorIds, expectedAction, config, {
      learningMode: options.learningMode,
      learningEnabled: options.learningEnabled,
      phase: options.phase,
      rng
    });
    const after = stepChallengeWorld(state, networkStep.executedAction);
    const reward = scoreChallengeStep(state, after, networkStep.executedAction);
    const rewardBaseline = options.rewardAdvantageState?.baseline ?? 0;
    const rewardAdvantage =
      options.learningEnabled && options.learningMode === "rewardOnly"
        ? reward.reward - rewardBaseline
        : reward.reward;
    const aversiveTag = deriveChallengeAversiveTag(
      rawObservation,
      reward,
      networkStep.executedAction,
      config.aversiveBadOutcomeThreshold
    );
    const rewardSignal =
      options.learningEnabled && options.learningMode === "rewardOnly"
        ? computeRewardOutcomeSignal(rewardAdvantage, config, aversiveTag)
        : rewardAdvantage;
    let rewardUpdates = 0;
    let captureUpdates = 0;
    let decayUpdates = 0;

    if (options.learningEnabled && options.learningMode === "rewardOnly") {
      rewardUpdates = applyRewardOutcomeLearning(network, rewardAdvantage, config, aversiveTag);
      updateRewardAdvantageBaseline(options.rewardAdvantageState, reward.reward, config);
      const maintenance = applyMaintenanceDecayAndCapture(network, config);
      captureUpdates = maintenance.captureUpdates;
      decayUpdates = maintenance.decayUpdates;
    } else if (options.learningEnabled && options.learningMode === "supervised") {
      const maintenance = applyMaintenanceDecayAndCapture(network, config);
      captureUpdates = maintenance.captureUpdates;
      decayUpdates = maintenance.decayUpdates;
    }

    const traceStep: ChallengeTraceStep = {
      index: stepIndex,
      before: state,
      rawObservation,
      sensoryMapping,
      expectedAction,
      networkDecision: networkStep.networkDecision,
      explorationAction: networkStep.explorationAction,
      executedAction: networkStep.executedAction,
      reward: reward.reward,
      rewardBaseline,
      rewardAdvantage,
      rewardSignal,
      aversiveTag,
      distanceDelta: reward.distanceDelta,
      after,
      terminalReason: reward.terminalReason,
      terminal: reward.terminal,
      success: reward.success,
      learning: {
        rawActiveMotors: networkStep.rawActiveMotors,
        activeMotors: networkStep.activeMotors,
        supervisedUpdates: networkStep.supervisedUpdates,
        rewardUpdates,
        captureUpdates,
        decayUpdates
      }
    };

    steps.push(traceStep);
    totalReward += reward.reward;
    state = after;

    if (reward.terminal) {
      terminalReason = reward.terminalReason;
      success = reward.success;
      break;
    }
  }

  return {
    phase: options.phase,
    scenarioId: scenario.id,
    seed: scenario.seed,
    episodeSeed: options.seed,
    learningMode: options.learningMode,
    steps,
    totalReward,
    success,
    terminalReason
  };
}

function runChallengeNetworkStep(
  network: LearningNetwork,
  activeSensorIds: string[],
  expectedAction: WorldAction,
  config: ModelConfig,
  options: {
    learningMode: ChallengeLearningMode;
    learningEnabled: boolean;
    phase: ChallengeEpisodePhase;
    rng: SeededRandom;
  }
): NetworkStepResult {
  resetNetworkRuntime(network);
  const activeSensors = new Set(activeSensorIds);
  setSensoryOutputs(network, activeSensors);
  propagateAndIntegrateRole(network, "interneuron", config);
  clearSensoryOutputs(network);
  propagateAndIntegrateRole(network, "motor", config);

  const rawActiveMotors = activeMotorIds(network);
  const networkDecision = arbitrateMotorAction(rawActiveMotors);
  const explorationAction = selectExplorationAction(networkDecision.action, options, config);
  const activeMotors = explorationAction
    ? forceExplorationMotor(network, explorationAction)
    : rawActiveMotors;
  const executedAction = explorationAction ?? networkDecision.action;
  let supervisedUpdates = 0;

  updateNetworkEligibility(network, config);

  if (options.learningEnabled && options.learningMode === "supervised") {
    supervisedUpdates = applySupervisedMotorOutcomeLearning(network, expectedAction, new Set(activeMotors), config);
  }

  return {
    rawActiveMotors,
    activeMotors,
    networkDecision,
    explorationAction,
    executedAction,
    supervisedUpdates
  };
}

export function updateRewardAdvantageBaseline(
  state: RewardAdvantageState | undefined,
  reward: number,
  config: ModelConfig
): void {
  if (!state) {
    return;
  }

  state.baseline =
    state.baseline * (1 - config.rewardAdvantageBaselineAlpha) + reward * config.rewardAdvantageBaselineAlpha;
}
