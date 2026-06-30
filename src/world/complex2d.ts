import { ModelConfig } from "../config/newModelConfig";
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
import { SensoryMapping, WorldObject, WorldState } from "./world2d";
import {
  CHALLENGE_HEIGHT,
  CHALLENGE_WIDTH,
  ChallengeEpisodePhase,
  ChallengeEpisodeTrace,
  ChallengeExperimentOptions,
  ChallengeExperimentResult,
  ChallengeLearningMode,
  ChallengeComplexEvidence,
  ChallengeRawObservation,
  ChallengeScenario,
  ChallengeTerminalReason,
  ChallengeTraceStep,
  activeMotorIds,
  countLearningEvents,
  createChallengeConfig,
  createChallengeScenarios,
  createChallengeWorldState,
  forceExplorationMotor,
  observeChallengeWorld,
  scoreChallengeStep,
  selectExplorationAction,
  shuffleScenarios,
  stepChallengeWorld
} from "./challenge2d";

export const DEFAULT_COMPLEX_MAX_STEPS = 6;
const COMPLEX_SPIKE_TICKS = 3;

export function createComplexConfig(config: ModelConfig): ModelConfig {
  return createChallengeConfig(config);
}

interface ComplexSensoryInput {
  mapping: SensoryMapping;
  fireDurations: Record<string, number>;
}

interface NetworkStepResult {
  rawActiveMotors: string[];
  activeMotors: string[];
  networkDecision: ActionDecision;
  explorationAction: WorldAction | null;
  executedAction: WorldAction;
  supervisedUpdates: number;
  complexEvidence: ChallengeComplexEvidence;
}

export interface ComplexActionResolverContext {
  rawDecision: ActionDecision;
  evidence: ChallengeComplexEvidence;
  expectedAction: WorldAction;
  phase: ChallengeEpisodePhase;
}

export type ComplexActionResolver = (context: ComplexActionResolverContext) => WorldAction | null;

export function runComplexExperiment(
  config: ModelConfig,
  options: ChallengeExperimentOptions
): ChallengeExperimentResult {
  const maxSteps = options.maxSteps ?? DEFAULT_COMPLEX_MAX_STEPS;
  const observationDropout = options.observationDropout ?? 0;
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

  for (let epoch = 0; epoch < options.epochs; epoch += 1) {
    const epochScenarios = shuffleScenarios(trainingScenarios, options.seed + epoch);

    for (const scenario of epochScenarios) {
      const episode = runComplexEpisode(network, scenario, config, {
        phase: "train",
        learningMode: options.learningMode,
        learningEnabled: options.learningMode !== "frozen",
        seed: options.seed + epoch * 1000 + scenario.seed,
        observationDropout,
        reverseMapping: false
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
      version: "dg-snn-2d-complex-trace-v0.1",
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
        reverseMapping: false
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

export function runComplexEpisode(
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
    actionResolver?: ComplexActionResolver;
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
    const sensoryInput = mapComplexObservationToSensors(rawObservation);
    const expectedAction = expectedActionForComplexState(state);
    const networkStep = runComplexNetworkStep(network, sensoryInput, expectedAction, config, {
      learningMode: options.learningMode,
      learningEnabled: options.learningEnabled,
      phase: options.phase,
      rng,
      actionResolver: options.actionResolver
    });
    const after = stepChallengeWorld(state, networkStep.executedAction);
    const reward = scoreChallengeStep(state, after, networkStep.executedAction);
    let rewardUpdates = 0;
    let captureUpdates = 0;
    let decayUpdates = 0;

    if (options.learningEnabled && options.learningMode === "rewardOnly") {
      const neuronsById = indexNeurons(network.neurons);
      rewardUpdates = applyRewardLearning(network.synapses, neuronsById, reward.reward, config).length;
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
      sensoryMapping: sensoryInput.mapping,
      expectedAction,
      networkDecision: networkStep.networkDecision,
      explorationAction: networkStep.explorationAction,
      executedAction: networkStep.executedAction,
      reward: reward.reward,
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
      },
      complexEvidence: networkStep.complexEvidence
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

function mapComplexObservationToSensors(observation: ChallengeRawObservation): ComplexSensoryInput {
  const activeSensorIds: string[] = [];
  const fireDurations: Record<string, number> = {};
  const sensorReasons: Record<string, string> = {};

  for (const object of observation.visibleObjects) {
    const side = object.dx < 0 ? "Left" : "Right";
    const sensorId = `${object.kind}${side}`;
    const duration = Math.max(1, COMPLEX_SPIKE_TICKS + 1 - object.distance);
    activeSensorIds.push(sensorId);
    fireDurations[sensorId] = duration;
    sensorReasons[sensorId] =
      `${object.kind}:${side.toLowerCase()}:dx=${object.dx}:dy=${object.dy}:distance=${object.distance}:fireTicks=${duration}`;
  }

  activeSensorIds.sort();
  return {
    mapping: { activeSensorIds, sensorReasons },
    fireDurations
  };
}

export function expectedActionForComplexState(state: WorldState): WorldAction {
  const objectsWithDx = state.objects.filter((object) => object.position.x !== state.agent.position.x);
  if (objectsWithDx.length === 0) {
    return "noop";
  }

  const foods = objectsWithDx
    .filter((object) => object.kind === "food")
    .map((object) => ({ object, distance: manhattan(state, object) }))
    .sort((a, b) => a.distance - b.distance);
  const toxins = objectsWithDx
    .filter((object) => object.kind === "toxin")
    .map((object) => ({ object, distance: manhattan(state, object) }))
    .sort((a, b) => a.distance - b.distance);

  const nearestFood = foods[0];
  const nearestToxin = toxins[0];

  const foodVote: WorldAction | null = nearestFood
    ? nearestFood.object.position.x < state.agent.position.x
      ? "left"
      : "right"
    : null;
  const toxinVote: WorldAction | null = nearestToxin
    ? nearestToxin.object.position.x > state.agent.position.x
      ? "left"
      : "right"
    : null;

  if (foodVote && toxinVote) {
    if (foodVote === toxinVote) {
      return foodVote;
    }
    if (nearestToxin.distance <= nearestFood.distance) {
      return toxinVote;
    }
    return foodVote;
  }

  if (foodVote) {
    if (foods.length >= 2) {
      const second = foods[1];
      const nearestSide = sideOf(nearestFood.object, state);
      const secondSide = sideOf(second.object, state);
      if (nearestSide !== secondSide && second.distance === nearestFood.distance) {
        return "conflict";
      }
    }
    return foodVote;
  }

  if (toxinVote) {
    if (toxins.length >= 2) {
      const second = toxins[1];
      const nearestSide = sideOf(nearestToxin.object, state);
      const secondSide = sideOf(second.object, state);
      if (nearestSide !== secondSide && second.distance === nearestToxin.distance) {
        return "conflict";
      }
    }
    return toxinVote;
  }

  return "noop";
}

function sideOf(object: WorldObject, state: WorldState): "left" | "right" {
  return object.position.x < state.agent.position.x ? "left" : "right";
}

function manhattan(state: WorldState, object: WorldObject): number {
  return Math.abs(object.position.x - state.agent.position.x) + Math.abs(object.position.y - state.agent.position.y);
}

function runComplexNetworkStep(
  network: LearningNetwork,
  sensoryInput: ComplexSensoryInput,
  expectedAction: WorldAction,
  config: ModelConfig,
  options: {
    learningMode: ChallengeLearningMode;
    learningEnabled: boolean;
    phase: ChallengeEpisodePhase;
    rng: SeededRandom;
    actionResolver?: ComplexActionResolver;
  }
): NetworkStepResult {
  const neuronsById = indexNeurons(network.neurons);

  for (const neuron of network.neurons) {
    resetNeuronRuntime(neuron);
  }

  const motorSpikeCounts: Record<string, number> = {};
  const interSpikeCounts: Record<string, number> = {};
  for (const neuron of network.neurons) {
    if (neuron.role === "motor") {
      motorSpikeCounts[neuron.id] = 0;
    }
    if (neuron.role === "interneuron") {
      interSpikeCounts[neuron.id] = 0;
    }
  }

  for (let tick = 0; tick < COMPLEX_SPIKE_TICKS; tick += 1) {
    for (const neuron of network.neurons) {
      if (neuron.role === "sensory") {
        const duration = sensoryInput.fireDurations[neuron.id] ?? 0;
        setSensoryOutput(neuron, tick < duration ? 1 : 0);
      }
    }

    network.tick += 1;
    resetBranchInputs(network.neurons);
    propagateSynapses(neuronsById, network.synapses, network.tick, config);
    for (const neuron of network.neurons) {
      if (neuron.role === "interneuron") {
        integrateNeuron(neuron, config);
        if (neuron.spike) {
          interSpikeCounts[neuron.id] += 1;
        }
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
        if (neuron.spike) {
          motorSpikeCounts[neuron.id] += 1;
        }
      }
    }
  }

  for (const neuron of network.neurons) {
    if (neuron.role === "interneuron" && (interSpikeCounts[neuron.id] ?? 0) > 0) {
      neuron.outputSignal = 1;
      neuron.spike = true;
    } else if (neuron.role === "interneuron") {
      neuron.outputSignal = 0;
      neuron.spike = false;
    }
    if (neuron.role === "motor" && motorSpikeCounts[neuron.id] > 0) {
      neuron.outputSignal = 1;
      neuron.spike = true;
    } else if (neuron.role === "motor") {
      neuron.outputSignal = 0;
      neuron.spike = false;
    }
  }

  const rawActiveMotors = Object.keys(motorSpikeCounts)
    .filter((id) => motorSpikeCounts[id] > 0)
    .sort();
  const networkDecision = arbitrateComplexMotorAction(motorSpikeCounts);
  const complexEvidence = {
    interSpikeCounts: { ...interSpikeCounts },
    motorSpikeCounts: { ...motorSpikeCounts }
  };
  const resolverAction =
    networkDecision.action === "conflict" && options.actionResolver
      ? options.actionResolver({
          rawDecision: networkDecision,
          evidence: complexEvidence,
          expectedAction,
          phase: options.phase
        })
      : null;
  const explorationAction =
    resolverAction === null ? selectExplorationAction(networkDecision.action, options) : null;
  const activeMotors = explorationAction
    ? forceExplorationMotor(network, explorationAction)
    : rawActiveMotors;
  const executedAction = resolverAction ?? explorationAction ?? networkDecision.action;
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
    supervisedUpdates,
    complexEvidence
  };
}

export function arbitrateComplexMotorAction(spikeCounts: Record<string, number>): ActionDecision {
  const left = spikeCounts["leftMotor"] ?? 0;
  const right = spikeCounts["rightMotor"] ?? 0;
  const activeMotors: string[] = [];
  if (left > 0) {
    activeMotors.push("leftMotor");
  }
  if (right > 0) {
    activeMotors.push("rightMotor");
  }
  activeMotors.sort();

  if (left === 0 && right === 0) {
    return {
      action: "noop",
      activeMotors: [],
      mappedActions: [],
      reason: "no-active-motor"
    };
  }

  if (left > right) {
    return {
      action: "left",
      activeMotors,
      mappedActions: ["left"],
      reason: "spike-count-left"
    };
  }

  if (right > left) {
    return {
      action: "right",
      activeMotors,
      mappedActions: ["right"],
      reason: "spike-count-right"
    };
  }

  return {
    action: "conflict",
    activeMotors,
    mappedActions: ["left", "right"],
    reason: "equal-spike-count"
  };
}

function centerPosition() {
  return {
    x: Math.floor(CHALLENGE_WIDTH / 2),
    y: Math.floor(CHALLENGE_HEIGHT / 2)
  };
}

export function compositeSameDirectionScenarios(maxSteps = DEFAULT_COMPLEX_MAX_STEPS): ChallengeScenario[] {
  const center = centerPosition();
  return [
    {
      id: "complex-composite-food-left-toxin-right",
      seed: 201,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "food-left", kind: "food", position: { x: center.x - 2, y: center.y } },
        { id: "toxin-right", kind: "toxin", position: { x: center.x + 2, y: center.y } }
      ]
    },
    {
      id: "complex-composite-food-right-toxin-left",
      seed: 202,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "food-right", kind: "food", position: { x: center.x + 2, y: center.y } },
        { id: "toxin-left", kind: "toxin", position: { x: center.x - 2, y: center.y } }
      ]
    }
  ];
}

export function distractorScenarios(maxSteps = DEFAULT_COMPLEX_MAX_STEPS): ChallengeScenario[] {
  const center = centerPosition();
  return [
    {
      id: "complex-distractor-food-left-near-food-right-far",
      seed: 211,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "food-left-near", kind: "food", position: { x: center.x - 1, y: center.y } },
        { id: "food-right-far", kind: "food", position: { x: center.x + 3, y: center.y } }
      ]
    },
    {
      id: "complex-distractor-food-right-near-food-left-far",
      seed: 212,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "food-right-near", kind: "food", position: { x: center.x + 1, y: center.y } },
        { id: "food-left-far", kind: "food", position: { x: center.x - 3, y: center.y } }
      ]
    },
    {
      id: "complex-distractor-toxin-left-near-toxin-right-far",
      seed: 213,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "toxin-left-near", kind: "toxin", position: { x: center.x - 1, y: center.y } },
        { id: "toxin-right-far", kind: "toxin", position: { x: center.x + 3, y: center.y } }
      ]
    },
    {
      id: "complex-distractor-toxin-right-near-toxin-left-far",
      seed: 214,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "toxin-right-near", kind: "toxin", position: { x: center.x + 1, y: center.y } },
        { id: "toxin-left-far", kind: "toxin", position: { x: center.x - 3, y: center.y } }
      ]
    }
  ];
}

export function priorityScenarios(maxSteps = DEFAULT_COMPLEX_MAX_STEPS): ChallengeScenario[] {
  const center = centerPosition();
  return [
    {
      id: "complex-priority-food-left-near-toxin-left-far",
      seed: 221,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "food-left-near", kind: "food", position: { x: center.x - 2, y: center.y } },
        { id: "toxin-left-far", kind: "toxin", position: { x: center.x - 3, y: center.y } }
      ]
    },
    {
      id: "complex-priority-food-left-far-toxin-left-near",
      seed: 222,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "food-left-far", kind: "food", position: { x: center.x - 3, y: center.y } },
        { id: "toxin-left-near", kind: "toxin", position: { x: center.x - 2, y: center.y } }
      ]
    },
    {
      id: "complex-priority-food-right-near-toxin-right-far",
      seed: 223,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "food-right-near", kind: "food", position: { x: center.x + 2, y: center.y } },
        { id: "toxin-right-far", kind: "toxin", position: { x: center.x + 3, y: center.y } }
      ]
    },
    {
      id: "complex-priority-food-right-far-toxin-right-near",
      seed: 224,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "food-right-far", kind: "food", position: { x: center.x + 3, y: center.y } },
        { id: "toxin-right-near", kind: "toxin", position: { x: center.x + 2, y: center.y } }
      ]
    }
  ];
}

export function trueConflictScenarios(maxSteps = DEFAULT_COMPLEX_MAX_STEPS): ChallengeScenario[] {
  const center = centerPosition();
  return [
    {
      id: "complex-conflict-food-left-food-right-equidistant",
      seed: 231,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "food-left", kind: "food", position: { x: center.x - 2, y: center.y } },
        { id: "food-right", kind: "food", position: { x: center.x + 2, y: center.y } }
      ]
    },
    {
      id: "complex-conflict-toxin-left-toxin-right-equidistant",
      seed: 232,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "toxin-left", kind: "toxin", position: { x: center.x - 2, y: center.y } },
        { id: "toxin-right", kind: "toxin", position: { x: center.x + 2, y: center.y } }
      ]
    }
  ];
}

export function semanticConflictScenarios(maxSteps = DEFAULT_COMPLEX_MAX_STEPS): ChallengeScenario[] {
  const center = centerPosition();
  const semanticScenario = (
    id: string,
    seed: number,
    side: "left" | "right",
    distance: number
  ): ChallengeScenario => {
    const offset = side === "left" ? -distance : distance;
    return {
      id,
      seed,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: `toxin-${side}-d${distance}`, kind: "toxin", position: { x: center.x + offset, y: center.y } },
        { id: `food-${side}-d${distance}`, kind: "food", position: { x: center.x + offset, y: center.y } }
      ]
    };
  };

  return [
    semanticScenario("complex-semantic-conflict-food-left-toxin-left-d2", 241, "left", 2),
    semanticScenario("complex-semantic-conflict-food-right-toxin-right-d2", 242, "right", 2),
    semanticScenario("complex-semantic-conflict-food-left-toxin-left-d1", 243, "left", 1),
    semanticScenario("complex-semantic-conflict-food-right-toxin-right-d1", 244, "right", 1),
    semanticScenario("complex-semantic-conflict-food-left-toxin-left-d3", 245, "left", 3),
    semanticScenario("complex-semantic-conflict-food-right-toxin-right-d3", 246, "right", 3)
  ];
}

export function blankComplexScenario(seed = 0, maxSteps = DEFAULT_COMPLEX_MAX_STEPS): ChallengeScenario {
  return {
    id: "complex-blank",
    seed,
    width: CHALLENGE_WIDTH,
    height: CHALLENGE_HEIGHT,
    maxSteps,
    agentStart: centerPosition(),
    objects: []
  };
}
