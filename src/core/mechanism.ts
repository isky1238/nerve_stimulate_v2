import { ModelConfig } from "../config/newModelConfig";
import { ActionDecision, WorldAction, targetMotorForAction } from "./arbitration";
import type { LearningNetwork } from "./evaluation";
import { indexNeurons, integrateNeuron, NeuronRole, resetBranchInputs, resetNeuronRuntime, setSensoryOutput } from "./neuron";
import {
  applyRewardLearning,
  applySupervisedMotorLearning,
  captureStableWeights,
  decayWeights,
  updateEligibility
} from "./plasticity";
import { SeededRandom } from "./random";
import { propagateSynapses } from "./synapse";

export interface ExplorationSelectionOptions {
  learningMode: "supervised" | "rewardOnly" | "frozen";
  learningEnabled: boolean;
  phase: "train" | "eval";
  rng: SeededRandom;
}

export interface MaintenanceUpdateCounts {
  captureUpdates: number;
  decayUpdates: number;
}

export function resetNetworkRuntime(network: LearningNetwork): void {
  for (const neuron of network.neurons) {
    resetNeuronRuntime(neuron);
  }
}

export function setSensoryOutputs(network: LearningNetwork, activeSensorIds: Set<string>): void {
  for (const neuron of network.neurons) {
    if (neuron.role === "sensory") {
      setSensoryOutput(neuron, activeSensorIds.has(neuron.id) ? 1 : 0);
    }
  }
}

export function clearSensoryOutputs(network: LearningNetwork): void {
  for (const neuron of network.neurons) {
    if (neuron.role === "sensory") {
      setSensoryOutput(neuron, 0);
    }
  }
}

export function propagateAndIntegrateRole(
  network: LearningNetwork,
  role: Exclude<NeuronRole, "sensory">,
  config: ModelConfig
): void {
  const neuronsById = indexNeurons(network.neurons);
  network.tick += 1;
  resetBranchInputs(network.neurons);
  propagateSynapses(neuronsById, network.synapses, network.tick, config);

  for (const neuron of network.neurons) {
    if (neuron.role === role) {
      integrateNeuron(neuron, config);
    }
  }
}

export function updateNetworkEligibility(network: LearningNetwork, config: ModelConfig): void {
  updateEligibility(network.synapses, indexNeurons(network.neurons), config);
}

export function applyRewardOutcomeLearning(
  network: LearningNetwork,
  rewardAdvantage: number,
  config: ModelConfig
): number {
  return applyRewardLearning(network.synapses, indexNeurons(network.neurons), rewardAdvantage, config).length;
}

export function applySupervisedMotorOutcomeLearning(
  network: LearningNetwork,
  expectedAction: WorldAction,
  activeMotors: Set<string>,
  config: ModelConfig
): number {
  const targetMotorId = targetMotorForAction(expectedAction);

  if (targetMotorId === null) {
    return 0;
  }

  return applySupervisedMotorLearning(
    network.synapses,
    indexNeurons(network.neurons),
    targetMotorId,
    activeMotors,
    config
  ).length;
}

export function applyMaintenanceDecayAndCapture(
  network: LearningNetwork,
  config: ModelConfig
): MaintenanceUpdateCounts {
  return {
    captureUpdates: captureStableWeights(network.synapses, config).length,
    decayUpdates: decayWeights(network.synapses, config).length
  };
}

export function activeMotorIds(network: LearningNetwork): string[] {
  return network.neurons
    .filter((neuron) => neuron.role === "motor" && neuron.outputSignal !== 0)
    .map((neuron) => neuron.id)
    .sort();
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

export function selectExplorationAction(
  action: ActionDecision["action"],
  options: ExplorationSelectionOptions,
  config: ModelConfig
): WorldAction | null {
  if (!options.learningEnabled || options.learningMode !== "rewardOnly" || options.phase !== "train") {
    return null;
  }

  if (config.explorationStrategy === "epsilonGreedy") {
    // With epsilon-greedy, noop remains visible unless epsilon triggers a motor override.
    if (options.rng.next() < config.explorationEpsilon) {
      return options.rng.nextInt(2) === 0 ? "left" : "right";
    }
    return null;
  }

  if (action === "left" || action === "right") {
    return null;
  }

  return options.rng.nextInt(2) === 0 ? "left" : "right";
}
