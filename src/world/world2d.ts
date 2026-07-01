import { ModelConfig, withConfig } from "../config/newModelConfig";
import { ActionDecision, WorldAction, arbitrateMotorAction, targetMotorForAction } from "../core/arbitration";
import { createOfflineLearningNetwork, LearningNetwork } from "../core/evaluation";
import { indexNeurons, integrateNeuron, resetBranchInputs, resetNeuronRuntime, setSensoryOutput } from "../core/neuron";
import { captureStableWeights, decayWeights, applySupervisedMotorLearning, updateEligibility } from "../core/plasticity";
import { SeededRandom } from "../core/random";
import { propagateSynapses } from "../core/synapse";

export type WorldObjectKind = "food" | "toxin";
export type HorizontalSide = "left" | "right";
export type WorldEpisodePhase = "train" | "eval";
export type TerminalReason = "one-step-evaluation";

export interface GridPosition {
  x: number;
  y: number;
}

export interface WorldObject {
  id: string;
  kind: WorldObjectKind;
  position: GridPosition;
}

export interface AgentState {
  position: GridPosition;
}

export interface WorldState {
  width: number;
  height: number;
  step: number;
  agent: AgentState;
  objects: WorldObject[];
}

export interface ObservedObject {
  id: string;
  kind: WorldObjectKind;
  side: HorizontalSide;
  dx: number;
  dy: number;
}

export interface RawObservation {
  visibleObjects: ObservedObject[];
}

export interface SensoryMapping {
  activeSensorIds: string[];
  sensorReasons: Record<string, string>;
}

export interface WorldScenario {
  id: string;
  seed: number;
  width: number;
  height: number;
  agentStart: GridPosition;
  objects: WorldObject[];
}

export interface NetworkWorldStep {
  activeMotors: string[];
  supervisedUpdates: number;
  captureUpdates: number;
  decayUpdates: number;
}

export interface WorldTraceStep {
  index: number;
  before: WorldState;
  rawObservation: RawObservation;
  sensoryMapping: SensoryMapping;
  expectedAction: WorldAction;
  activeMotors: string[];
  decision: ActionDecision;
  reward: number;
  after: WorldState;
  terminalReason: TerminalReason;
  correct: boolean;
  taskSuccess: boolean;
  learning: NetworkWorldStep;
}

export interface WorldEpisodeTrace {
  phase: WorldEpisodePhase;
  scenarioId: string;
  seed: number;
  episodeSeed: number;
  steps: WorldTraceStep[];
  totalReward: number;
  success: boolean;
}

export interface WorldExperimentTrace {
  version: string;
  seed: number;
  worldSeed: number;
  episodeSeed: number;
  config: {
    width: number;
    height: number;
    epochs: number;
    learningOn: boolean;
  };
  episodes: WorldEpisodeTrace[];
}

export interface WorldExperimentResult {
  trace: WorldExperimentTrace;
  network: LearningNetwork;
  finalAccuracy: number;
  finalTaskSuccessRate: number;
  meanReward: number;
  supervisedUpdates: number;
  captureUpdates: number;
  decayUpdates: number;
}

export interface WorldExperimentOptions {
  seed: number;
  epochs: number;
  learningOn: boolean;
  trainingScenarios?: WorldScenario[];
  evaluationScenarios?: WorldScenario[];
}

export const DEFAULT_WORLD_WIDTH = 7;
export const DEFAULT_WORLD_HEIGHT = 7;

export function createWorldAuditConfig(config: ModelConfig): ModelConfig {
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
    useThreshold: 0.08
  });
}

export function runWorldExperiment(
  config: ModelConfig,
  options: WorldExperimentOptions
): WorldExperimentResult {
  const trainingScenarios = options.trainingScenarios ?? canonicalWorldScenarios();
  const evaluationScenarios = options.evaluationScenarios ?? trainingScenarios;
  const network = createOfflineLearningNetwork(config);
  const episodes: WorldEpisodeTrace[] = [];
  let supervisedUpdates = 0;
  let captureUpdates = 0;
  let decayUpdates = 0;

  for (let epoch = 0; epoch < options.epochs; epoch += 1) {
    const epochScenarios = shuffleScenarios(trainingScenarios, options.seed + epoch);

    for (const scenario of epochScenarios) {
      const episode = runWorldEpisode(network, scenario, config, {
        phase: "train",
        learningOn: options.learningOn,
        episodeSeed: options.seed * 1000 + epoch
      });
      const step = episode.steps[0];
      supervisedUpdates += step.learning.supervisedUpdates;
      captureUpdates += step.learning.captureUpdates;
      decayUpdates += step.learning.decayUpdates;
      episodes.push(episode);
    }
  }

  const evaluationEpisodes = evaluationScenarios.map((scenario, index) =>
    runWorldEpisode(network, scenario, config, {
      phase: "eval",
      learningOn: false,
      episodeSeed: options.seed * 100000 + index
    })
  );
  episodes.push(...evaluationEpisodes);

  const finalCorrect = evaluationEpisodes.filter((episode) => episode.steps[0]?.correct).length;
  const finalTaskSuccesses = evaluationEpisodes.filter((episode) => episode.success).length;
  const rewardTotal = evaluationEpisodes.reduce((sum, episode) => sum + episode.totalReward, 0);

  return {
    trace: {
      version: "dg-snn-2d-lite-trace-v0.1",
      seed: options.seed,
      worldSeed: options.seed,
      episodeSeed: options.seed,
      config: {
        width: DEFAULT_WORLD_WIDTH,
        height: DEFAULT_WORLD_HEIGHT,
        epochs: options.epochs,
        learningOn: options.learningOn
      },
      episodes
    },
    network,
    finalAccuracy: finalCorrect / Math.max(1, evaluationEpisodes.length),
    finalTaskSuccessRate: finalTaskSuccesses / Math.max(1, evaluationEpisodes.length),
    meanReward: rewardTotal / Math.max(1, evaluationEpisodes.length),
    supervisedUpdates,
    captureUpdates,
    decayUpdates
  };
}

export function runWorldEpisode(
  network: LearningNetwork,
  scenario: WorldScenario,
  config: ModelConfig,
  options: { phase: WorldEpisodePhase; learningOn: boolean; episodeSeed: number }
): WorldEpisodeTrace {
  const before = createWorldState(scenario);
  const rawObservation = observeWorld(before);
  const sensoryMapping = mapObservationToSensors(rawObservation);
  const expectedAction = expectedActionForSensors(sensoryMapping.activeSensorIds);
  const networkStep = runNetworkForSensors(
    network,
    sensoryMapping.activeSensorIds,
    targetMotorForAction(expectedAction),
    config,
    options.learningOn
  );
  const decision = arbitrateMotorAction(networkStep.activeMotors);
  const reward = rewardForDecision(decision.action, expectedAction);
  const after = stepWorld(before, decision.action);
  const correct = decision.action === expectedAction;
  const taskSuccess = correct && expectedAction !== "conflict";
  const step: WorldTraceStep = {
    index: 0,
    before,
    rawObservation,
    sensoryMapping,
    expectedAction,
    activeMotors: networkStep.activeMotors,
    decision,
    reward,
    after,
    terminalReason: "one-step-evaluation",
    correct,
    taskSuccess,
    learning: networkStep
  };

  return {
    phase: options.phase,
    scenarioId: scenario.id,
    seed: scenario.seed,
    episodeSeed: options.episodeSeed,
    steps: [step],
    totalReward: reward,
    success: taskSuccess
  };
}

export function canonicalWorldScenarios(): WorldScenario[] {
  const center = centerPosition(DEFAULT_WORLD_WIDTH, DEFAULT_WORLD_HEIGHT);

  return [
    createSingleObjectScenario("canonical-food-left", 1, "food", { x: center.x - 1, y: center.y }),
    createSingleObjectScenario("canonical-food-right", 2, "food", { x: center.x + 1, y: center.y }),
    createSingleObjectScenario("canonical-toxin-left", 3, "toxin", { x: center.x - 1, y: center.y - 1 }),
    createSingleObjectScenario("canonical-toxin-right", 4, "toxin", { x: center.x + 1, y: center.y - 1 })
  ];
}

export function seededWorldScenarios(seed: number): WorldScenario[] {
  const rng = new SeededRandom(seed);
  const scenarios: WorldScenario[] = [];
  let index = 0;

  for (const kind of ["food", "toxin"] as const) {
    for (const side of ["left", "right"] as const) {
      index += 1;
      const position = positionForSide(side, rng);
      scenarios.push(createSingleObjectScenario(`seed-${seed}-${kind}-${side}`, seed + index, kind, position));
    }
  }

  return scenarios;
}

export function mirroredScenarios(scenarios: WorldScenario[]): WorldScenario[] {
  return scenarios.map((scenario) => ({
    ...scenario,
    id: `${scenario.id}-mirror`,
    objects: scenario.objects.map((object) => ({
      ...object,
      position: {
        x: scenario.width - 1 - object.position.x,
        y: object.position.y
      }
    }))
  }));
}

export function blankWorldScenario(seed = 0): WorldScenario {
  return {
    id: "blank-world",
    seed,
    width: DEFAULT_WORLD_WIDTH,
    height: DEFAULT_WORLD_HEIGHT,
    agentStart: centerPosition(DEFAULT_WORLD_WIDTH, DEFAULT_WORLD_HEIGHT),
    objects: []
  };
}

export function sameActionCompositeScenario(seed = 0): WorldScenario {
  const center = centerPosition(DEFAULT_WORLD_WIDTH, DEFAULT_WORLD_HEIGHT);

  return {
    id: "same-action-composite",
    seed,
    width: DEFAULT_WORLD_WIDTH,
    height: DEFAULT_WORLD_HEIGHT,
    agentStart: center,
    objects: [
      { id: "food-left", kind: "food", position: { x: center.x - 1, y: center.y } },
      { id: "toxin-right", kind: "toxin", position: { x: center.x + 1, y: center.y } }
    ]
  };
}

export function oppositeConflictScenario(seed = 0): WorldScenario {
  const center = centerPosition(DEFAULT_WORLD_WIDTH, DEFAULT_WORLD_HEIGHT);

  return {
    id: "opposite-action-conflict",
    seed,
    width: DEFAULT_WORLD_WIDTH,
    height: DEFAULT_WORLD_HEIGHT,
    agentStart: center,
    objects: [
      { id: "food-left", kind: "food", position: { x: center.x - 1, y: center.y } },
      { id: "food-right", kind: "food", position: { x: center.x + 1, y: center.y } }
    ]
  };
}

export function expectedActionForSensors(sensorIds: string[]): WorldAction {
  const votes = sensorIds
    .map((sensorId): WorldAction | null => {
      if (sensorId === "foodLeft" || sensorId === "toxinRight") {
        return "left";
      }

      if (sensorId === "foodRight" || sensorId === "toxinLeft") {
        return "right";
      }

      return null;
    })
    .filter((action): action is Exclude<WorldAction, "noop" | "conflict"> => action !== null);

  if (votes.length === 0) {
    return "noop";
  }

  const uniqueVotes = Array.from(new Set(votes));
  return uniqueVotes.length === 1 ? uniqueVotes[0] : "conflict";
}

function runNetworkForSensors(
  network: LearningNetwork,
  activeSensorIds: string[],
  targetMotorId: string | null,
  config: ModelConfig,
  learningOn: boolean
): NetworkWorldStep {
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

  updateEligibility(network.synapses, neuronsById, config);
  const activeMotors = network.neurons
    .filter((neuron) => neuron.role === "motor" && neuron.outputSignal !== 0)
    .map((neuron) => neuron.id)
    .sort();
  const activeMotorSet = new Set(activeMotors);
  let supervisedUpdates = 0;
  let captureUpdates = 0;
  let decayUpdates = 0;

  if (learningOn && targetMotorId !== null) {
    supervisedUpdates = applySupervisedMotorLearning(
      network.synapses,
      neuronsById,
      targetMotorId,
      activeMotorSet,
      1,
      config
    ).length;
    captureUpdates = captureStableWeights(network.synapses, neuronsById, network.globalAversiveLoad, config).length;
    decayUpdates = decayWeights(network.synapses, config).length;
  }

  return {
    activeMotors,
    supervisedUpdates,
    captureUpdates,
    decayUpdates
  };
}

function createWorldState(scenario: WorldScenario): WorldState {
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

function observeWorld(state: WorldState): RawObservation {
  return {
    visibleObjects: state.objects
      .map((object): ObservedObject | null => {
        const dx = object.position.x - state.agent.position.x;
        const dy = object.position.y - state.agent.position.y;

        if (dx === 0) {
          return null;
        }

        return {
          id: object.id,
          kind: object.kind,
          side: dx < 0 ? "left" : "right",
          dx,
          dy
        };
      })
      .filter((object): object is ObservedObject => object !== null)
  };
}

function mapObservationToSensors(observation: RawObservation): SensoryMapping {
  const activeSensorIds: string[] = [];
  const sensorReasons: Record<string, string> = {};

  for (const object of observation.visibleObjects) {
    const sensorId = `${object.kind}${capitalize(object.side)}`;
    activeSensorIds.push(sensorId);
    sensorReasons[sensorId] = `${object.kind}:${object.side}:dx=${object.dx}:dy=${object.dy}`;
  }

  activeSensorIds.sort();
  return {
    activeSensorIds,
    sensorReasons
  };
}

function stepWorld(state: WorldState, action: WorldAction): WorldState {
  const next = createWorldState({
    id: "next",
    seed: 0,
    width: state.width,
    height: state.height,
    agentStart: state.agent.position,
    objects: state.objects
  });
  next.step = state.step + 1;

  if (action === "left") {
    next.agent.position.x = Math.max(0, next.agent.position.x - 1);
  }

  if (action === "right") {
    next.agent.position.x = Math.min(next.width - 1, next.agent.position.x + 1);
  }

  return next;
}

function rewardForDecision(action: WorldAction, expectedAction: WorldAction): number {
  if (expectedAction === "conflict") {
    return action === "conflict" ? -0.1 : -1;
  }

  if (action === expectedAction) {
    return expectedAction === "noop" ? 0 : 1;
  }

  return -1;
}

function createSingleObjectScenario(
  id: string,
  seed: number,
  kind: WorldObjectKind,
  objectPosition: GridPosition
): WorldScenario {
  return {
    id,
    seed,
    width: DEFAULT_WORLD_WIDTH,
    height: DEFAULT_WORLD_HEIGHT,
    agentStart: centerPosition(DEFAULT_WORLD_WIDTH, DEFAULT_WORLD_HEIGHT),
    objects: [
      {
        id: `${kind}-${objectPosition.x}-${objectPosition.y}`,
        kind,
        position: objectPosition
      }
    ]
  };
}

function positionForSide(side: HorizontalSide, rng: SeededRandom): GridPosition {
  const center = centerPosition(DEFAULT_WORLD_WIDTH, DEFAULT_WORLD_HEIGHT);
  const distance = 1 + rng.nextInt(3);
  const y = rng.nextInt(DEFAULT_WORLD_HEIGHT);

  return {
    x: side === "left" ? center.x - distance : center.x + distance,
    y
  };
}

function centerPosition(width: number, height: number): GridPosition {
  return {
    x: Math.floor(width / 2),
    y: Math.floor(height / 2)
  };
}

function shuffleScenarios(scenarios: WorldScenario[], seed: number): WorldScenario[] {
  const rng = new SeededRandom(seed);
  const shuffled = [...scenarios];

  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = rng.nextInt(index + 1);
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }

  return shuffled;
}

function capitalize(value: string): string {
  return `${value.charAt(0).toUpperCase()}${value.slice(1)}`;
}
