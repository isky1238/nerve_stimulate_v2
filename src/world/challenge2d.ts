import { ModelConfig, withConfig } from "../config/newModelConfig";
import { ActionDecision, WorldAction, arbitrateMotorAction, targetMotorForAction } from "../core/arbitration";
import { createOfflineLearningNetwork, LearningNetwork } from "../core/evaluation";
import { indexNeurons, integrateNeuron, resetBranchInputs, resetNeuronRuntime, setSensoryOutput } from "../core/neuron";
import {
  applyRewardLearning,
  applySupervisedMotorLearning,
  captureStableWeights,
  decayWeights,
  updateEligibility
} from "../core/plasticity";
import { SeededRandom } from "../core/random";
import { propagateSynapses } from "../core/synapse";
import { GridPosition, SensoryMapping, WorldObject, WorldObjectKind, WorldState } from "./world2d";

export type ChallengeLearningMode = "supervised" | "rewardOnly" | "frozen";
export type ChallengeEpisodePhase = "train" | "eval";
export type ChallengeTerminalReason =
  | "continue"
  | "food-contact"
  | "toxin-contact"
  | "toxin-avoided"
  | "conflict"
  | "step-limit";

export interface ChallengeObservedObject {
  id: string;
  kind: WorldObjectKind;
  dx: number;
  dy: number;
  distance: number;
  dropped: boolean;
}

export interface ChallengeRawObservation {
  visibleObjects: ChallengeObservedObject[];
  droppedObjects: ChallengeObservedObject[];
}

export interface ChallengeScenario {
  id: string;
  seed: number;
  width: number;
  height: number;
  maxSteps: number;
  agentStart: GridPosition;
  objects: WorldObject[];
}

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

interface RewardResult {
  reward: number;
  terminalReason: ChallengeTerminalReason;
  terminal: boolean;
  success: boolean;
  distanceDelta: number;
}

export interface RewardAdvantageState {
  baseline: number;
}

export const CHALLENGE_WIDTH = 7;
export const CHALLENGE_HEIGHT = 7;
export const DEFAULT_CHALLENGE_MAX_STEPS = 12;
export const DEFAULT_TRAIN_SEEDS = [1, 2, 3, 4, 5];
export const DEFAULT_EVAL_SEEDS = [101, 102, 103, 104, 105];

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
  const maxSteps = options.maxSteps ?? DEFAULT_CHALLENGE_MAX_STEPS;
  const observationDropout = options.observationDropout ?? 0;
  const reverseMapping = options.reverseMapping ?? false;
  const trainingScenarios =
    options.trainingScenarios ?? createChallengeScenarios(options.trainSeeds, maxSteps);
  const evaluationScenarios =
    options.evaluationScenarios ?? createChallengeScenarios(options.evalSeeds, maxSteps);
  const network = options.initialNetwork ?? createOfflineLearningNetwork(config);
  const episodes: ChallengeEpisodeTrace[] = [];
  let rewardUpdateCount = 0;
  let supervisedUpdateCount = 0;
  let captureUpdateCount = 0;
  let decayUpdateCount = 0;
  const rewardAdvantageState: RewardAdvantageState = { baseline: 0 };

  for (let epoch = 0; epoch < options.epochs; epoch += 1) {
    const epochScenarios = shuffleScenarios(trainingScenarios, options.seed + epoch);
    const epochEpisodes: ChallengeEpisodeTrace[] = [];

    for (const scenario of epochScenarios) {
      const episode = runChallengeEpisode(network, scenario, config, {
        phase: "train",
        learningMode: options.learningMode,
        learningEnabled: options.learningMode !== "frozen",
        seed: options.seed + epoch * 1000 + scenario.seed,
        observationDropout,
        reverseMapping,
        rewardAdvantageState
      });
      const counts = countLearningEvents(episode);
      rewardUpdateCount += counts.rewardUpdateCount;
      supervisedUpdateCount += counts.supervisedUpdateCount;
      captureUpdateCount += counts.captureUpdateCount;
      decayUpdateCount += counts.decayUpdateCount;
      episodes.push(episode);
      epochEpisodes.push(episode);
    }

    if (options.epochProbe) {
      options.epochProbe(epoch, network, epochEpisodes);
    }
  }

  const evaluationEpisodes = evaluationScenarios.map((scenario, index) =>
    runChallengeEpisode(network, scenario, config, {
      phase: "eval",
      learningMode: options.learningMode,
      learningEnabled: false,
      seed: options.seed * 100000 + index,
      observationDropout,
      reverseMapping: false
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
      version: "dg-snn-2d-challenge-trace-v0.1",
      seed: options.seed,
      trainSeeds: [...options.trainSeeds],
      evalSeeds: [...options.evalSeeds],
      config: {
        width: CHALLENGE_WIDTH,
        height: CHALLENGE_HEIGHT,
        maxSteps,
        epochs: options.epochs,
        learningMode: options.learningMode,
        observationDropout,
        reverseMapping,
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
    let rewardUpdates = 0;
    let captureUpdates = 0;
    let decayUpdates = 0;

    if (options.learningEnabled && options.learningMode === "rewardOnly") {
      const neuronsById = indexNeurons(network.neurons);
      rewardUpdates = applyRewardLearning(network.synapses, neuronsById, rewardAdvantage, config).length;
      updateRewardAdvantageBaseline(options.rewardAdvantageState, reward.reward, config);
      captureUpdates = captureStableWeights(network.synapses, config).length;
      decayUpdates = decayWeights(network.synapses, config).length;
    } else if (options.learningEnabled && options.learningMode === "supervised") {
      captureUpdates = captureStableWeights(network.synapses, config).length;
      decayUpdates = decayWeights(network.synapses, config).length;
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

export function createChallengeScenarios(seeds: number[], maxSteps = DEFAULT_CHALLENGE_MAX_STEPS): ChallengeScenario[] {
  return seeds.flatMap((seed) => {
    const rng = new SeededRandom(seed);
    const distance = 2 + rng.nextInt(2);

    return [
      createChallengeScenario(`seed-${seed}-food-left`, seed * 10 + 1, "food", "left", distance, maxSteps),
      createChallengeScenario(`seed-${seed}-food-right`, seed * 10 + 2, "food", "right", distance, maxSteps),
      createChallengeScenario(`seed-${seed}-toxin-left`, seed * 10 + 3, "toxin", "left", 2, maxSteps),
      createChallengeScenario(`seed-${seed}-toxin-right`, seed * 10 + 4, "toxin", "right", 2, maxSteps)
    ];
  });
}

export function blankChallengeScenario(seed = 0, maxSteps = DEFAULT_CHALLENGE_MAX_STEPS): ChallengeScenario {
  return {
    id: "challenge-blank",
    seed,
    width: CHALLENGE_WIDTH,
    height: CHALLENGE_HEIGHT,
    maxSteps,
    agentStart: centerPosition(),
    objects: []
  };
}

export function conflictChallengeScenario(seed = 0, maxSteps = DEFAULT_CHALLENGE_MAX_STEPS): ChallengeScenario {
  const center = centerPosition();

  return {
    id: "challenge-conflict",
    seed,
    width: CHALLENGE_WIDTH,
    height: CHALLENGE_HEIGHT,
    maxSteps,
    agentStart: center,
    objects: [
      { id: "food-left", kind: "food", position: { x: center.x - 1, y: center.y } },
      { id: "food-right", kind: "food", position: { x: center.x + 1, y: center.y } }
    ]
  };
}

export function sameActionCompositeChallengeScenario(
  seed = 0,
  maxSteps = DEFAULT_CHALLENGE_MAX_STEPS
): ChallengeScenario {
  const center = centerPosition();

  return {
    id: "challenge-composite-same-action",
    seed,
    width: CHALLENGE_WIDTH,
    height: CHALLENGE_HEIGHT,
    maxSteps,
    agentStart: center,
    objects: [
      { id: "food-left", kind: "food", position: { x: center.x - 2, y: center.y } },
      { id: "toxin-right", kind: "toxin", position: { x: center.x + 2, y: center.y } }
    ]
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
  const neuronsById = indexNeurons(network.neurons);

  for (const neuron of network.neurons) {
    resetNeuronRuntime(neuron);
  }

  const activeSensors = new Set(activeSensorIds);
  for (const neuron of network.neurons) {
    if (neuron.role === "sensory") {
      setSensoryOutput(neuron, activeSensors.has(neuron.id) ? 1 : 0);
    }
  }

  network.tick += 1;
  resetBranchInputs(network.neurons);
  propagateSynapses(neuronsById, network.synapses, network.tick, config);
  for (const neuron of network.neurons) {
    if (neuron.role === "interneuron") {
      integrateNeuron(neuron, config);
    }
  }

  for (const neuron of network.neurons) {
    if (neuron.role === "sensory") {
      setSensoryOutput(neuron, 0);
    }
  }

  network.tick += 1;
  resetBranchInputs(network.neurons);
  propagateSynapses(neuronsById, network.synapses, network.tick, config);
  for (const neuron of network.neurons) {
    if (neuron.role === "motor") {
      integrateNeuron(neuron, config);
    }
  }

  const rawActiveMotors = activeMotorIds(network);
  const networkDecision = arbitrateMotorAction(rawActiveMotors);
  const explorationAction = selectExplorationAction(networkDecision.action, options, config);
  const activeMotors = explorationAction
    ? forceExplorationMotor(network, explorationAction)
    : rawActiveMotors;
  const executedAction = explorationAction ?? networkDecision.action;
  let supervisedUpdates = 0;

  updateEligibility(network.synapses, neuronsById, config);

  if (options.learningEnabled && options.learningMode === "supervised") {
    const targetMotorId = targetMotorForAction(expectedAction);

    if (targetMotorId !== null) {
      supervisedUpdates = applySupervisedMotorLearning(
        network.synapses,
        neuronsById,
        targetMotorId,
        new Set(activeMotors),
        config
      ).length;
    }
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

export function selectExplorationAction(
  action: WorldAction,
  options: { learningMode: ChallengeLearningMode; learningEnabled: boolean; phase: ChallengeEpisodePhase; rng: SeededRandom },
  config: ModelConfig
): WorldAction | null {
  if (!options.learningEnabled || options.learningMode !== "rewardOnly" || options.phase !== "train") {
    return null;
  }

  if (config.explorationStrategy === "epsilonGreedy") {
    // ε-greedy: with probability ε override with a random motor action; otherwise
    // follow the network's own decision verbatim. Letting noop propagate (instead
    // of forcing a motor on every noop/conflict) makes the learner's inaction
    // visible and stops conflict-gated forcing from masking noop during training.
    if (options.rng.next() < config.explorationEpsilon) {
      return options.rng.nextInt(2) === 0 ? "left" : "right";
    }
    return null;
  }

  // Legacy "conflictGated": explore only when the network failed to commit.
  if (action === "left" || action === "right") {
    return null;
  }

  return options.rng.nextInt(2) === 0 ? "left" : "right";
}

export function forceExplorationMotor(network: LearningNetwork, action: WorldAction): string[] {
  const targetMotorId = targetMotorForAction(action);

  if (targetMotorId === null) {
    return activeMotorIds(network);
  }

  for (const neuron of network.neurons) {
    if (neuron.role === "motor") {
      neuron.outputSignal = neuron.id === targetMotorId ? 1 : 0;
      neuron.spike = neuron.id === targetMotorId;
    }
  }

  return activeMotorIds(network);
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

export function scoreChallengeStep(before: WorldState, after: WorldState, action: WorldAction): RewardResult {
  const beforeTarget = nearestObject(before);
  const afterTarget = nearestObject(after);

  if (action === "conflict") {
    return {
      reward: -0.2,
      terminalReason: "conflict",
      terminal: true,
      success: false,
      distanceDelta: 0
    };
  }

  if (!beforeTarget || !afterTarget) {
    return {
      reward: 0,
      terminalReason: "continue",
      terminal: false,
      success: false,
      distanceDelta: 0
    };
  }

  const beforeDistance = objectDistance(before, beforeTarget);
  const afterDistance = objectDistance(after, afterTarget);
  const distanceDelta = beforeDistance - afterDistance;

  if (afterTarget.kind === "food" && afterDistance === 0) {
    return {
      reward: 1,
      terminalReason: "food-contact",
      terminal: true,
      success: true,
      distanceDelta
    };
  }

  if (afterTarget.kind === "toxin" && afterDistance === 0) {
    return {
      reward: -1,
      terminalReason: "toxin-contact",
      terminal: true,
      success: false,
      distanceDelta
    };
  }

  if (afterTarget.kind === "toxin" && afterDistance >= 3) {
    return {
      reward: 1,
      terminalReason: "toxin-avoided",
      terminal: true,
      success: true,
      distanceDelta
    };
  }

  return {
    reward: shapedReward(afterTarget.kind, distanceDelta, action),
    terminalReason: "continue",
    terminal: false,
    success: false,
    distanceDelta
  };
}

function shapedReward(kind: WorldObjectKind, distanceDelta: number, action: WorldAction): number {
  if (action === "noop") {
    return 0;
  }

  if (kind === "food") {
    return distanceDelta > 0 ? 0.1 : -0.1;
  }

  return distanceDelta < 0 ? 0.1 : -0.1;
}

function expectedActionForChallengeState(state: WorldState): WorldAction {
  const votes = state.objects
    .map((object): WorldAction | null => {
      const dx = object.position.x - state.agent.position.x;

      if (dx === 0) {
        return "noop";
      }

      if (object.kind === "food") {
        return dx < 0 ? "left" : "right";
      }

      return dx < 0 ? "right" : "left";
    })
    .filter((action): action is WorldAction => action !== null && action !== "noop");
  const uniqueVotes = Array.from(new Set(votes));

  if (uniqueVotes.length === 0) {
    return "noop";
  }

  return uniqueVotes.length === 1 ? uniqueVotes[0] : "conflict";
}

function reverseAction(action: WorldAction): WorldAction {
  if (action === "left") {
    return "right";
  }
  if (action === "right") {
    return "left";
  }
  return action;
}

function createChallengeScenario(
  id: string,
  seed: number,
  kind: WorldObjectKind,
  side: "left" | "right",
  distance: number,
  maxSteps: number
): ChallengeScenario {
  const center = centerPosition();

  return {
    id,
    seed,
    width: CHALLENGE_WIDTH,
    height: CHALLENGE_HEIGHT,
    maxSteps,
    agentStart: center,
    objects: [
      {
        id: `${kind}-${side}-${distance}`,
        kind,
        position: {
          x: side === "left" ? center.x - distance : center.x + distance,
          y: center.y
        }
      }
    ]
  };
}

export function createChallengeWorldState(scenario: ChallengeScenario): WorldState {
  return {
    width: scenario.width,
    height: scenario.height,
    step: 0,
    agent: {
      position: { ...scenario.agentStart }
    },
    objects: scenario.objects.map((object) => ({
      ...object,
      position: { ...object.position }
    }))
  };
}

export function observeChallengeWorld(
  state: WorldState,
  observationDropout: number,
  rng: SeededRandom
): ChallengeRawObservation {
  const visibleObjects: ChallengeObservedObject[] = [];
  const droppedObjects: ChallengeObservedObject[] = [];

  for (const object of state.objects) {
    const dx = object.position.x - state.agent.position.x;
    const dy = object.position.y - state.agent.position.y;
    const observed = {
      id: object.id,
      kind: object.kind,
      dx,
      dy,
      distance: Math.abs(dx) + Math.abs(dy),
      dropped: false
    };

    if (dx === 0) {
      continue;
    }

    if (observationDropout > 0 && rng.next() < observationDropout) {
      droppedObjects.push({ ...observed, dropped: true });
    } else {
      visibleObjects.push(observed);
    }
  }

  return {
    visibleObjects,
    droppedObjects
  };
}

function mapChallengeObservationToSensors(observation: ChallengeRawObservation): SensoryMapping {
  const activeSensorIds: string[] = [];
  const sensorReasons: Record<string, string> = {};

  for (const object of observation.visibleObjects) {
    const side = object.dx < 0 ? "Left" : "Right";
    const sensorId = `${object.kind}${side}`;
    activeSensorIds.push(sensorId);
    sensorReasons[sensorId] = `${object.kind}:${side.toLowerCase()}:dx=${object.dx}:dy=${object.dy}:distance=${object.distance}`;
  }

  activeSensorIds.sort();
  return {
    activeSensorIds,
    sensorReasons
  };
}

export function stepChallengeWorld(state: WorldState, action: WorldAction): WorldState {
  const next: WorldState = {
    width: state.width,
    height: state.height,
    step: state.step + 1,
    agent: {
      position: { ...state.agent.position }
    },
    objects: state.objects.map((object) => ({
      ...object,
      position: { ...object.position }
    }))
  };

  if (action === "left") {
    next.agent.position.x = Math.max(0, next.agent.position.x - 1);
  }

  if (action === "right") {
    next.agent.position.x = Math.min(next.width - 1, next.agent.position.x + 1);
  }

  return next;
}

function nearestObject(state: WorldState): WorldObject | null {
  let nearest: WorldObject | null = null;
  let nearestDistance = Number.POSITIVE_INFINITY;

  for (const object of state.objects) {
    const distance = objectDistance(state, object);

    if (distance < nearestDistance) {
      nearest = object;
      nearestDistance = distance;
    }
  }

  return nearest;
}

function objectDistance(state: WorldState, object: WorldObject): number {
  return Math.abs(object.position.x - state.agent.position.x) + Math.abs(object.position.y - state.agent.position.y);
}

export function activeMotorIds(network: LearningNetwork): string[] {
  return network.neurons
    .filter((neuron) => neuron.role === "motor" && neuron.outputSignal !== 0)
    .map((neuron) => neuron.id)
    .sort();
}

export function countLearningEvents(episode: ChallengeEpisodeTrace): {
  rewardUpdateCount: number;
  supervisedUpdateCount: number;
  captureUpdateCount: number;
  decayUpdateCount: number;
} {
  return episode.steps.reduce(
    (counts, step) => ({
      rewardUpdateCount: counts.rewardUpdateCount + step.learning.rewardUpdates,
      supervisedUpdateCount: counts.supervisedUpdateCount + step.learning.supervisedUpdates,
      captureUpdateCount: counts.captureUpdateCount + step.learning.captureUpdates,
      decayUpdateCount: counts.decayUpdateCount + step.learning.decayUpdates
    }),
    {
      rewardUpdateCount: 0,
      supervisedUpdateCount: 0,
      captureUpdateCount: 0,
      decayUpdateCount: 0
    }
  );
}

function centerPosition(): GridPosition {
  return {
    x: Math.floor(CHALLENGE_WIDTH / 2),
    y: Math.floor(CHALLENGE_HEIGHT / 2)
  };
}

export function shuffleScenarios(scenarios: ChallengeScenario[], seed: number): ChallengeScenario[] {
  const rng = new SeededRandom(seed);
  const shuffled = [...scenarios];

  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = rng.nextInt(index + 1);
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }

  return shuffled;
}
