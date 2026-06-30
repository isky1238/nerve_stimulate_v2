import { defaultConfig, ModelConfig, withConfig } from "../config/newModelConfig";
import { PairMemory, tryFormConnections, updateConnectionStates } from "./development";
import { createNeuron, indexNeurons, integrateNeuron, Neuron, resetBranchInputs, resetNeuronRuntime, setSensoryOutput } from "./neuron";
import { applySupervisedMotorLearning, captureStableWeights, decayWeights, updateEligibility } from "./plasticity";
import { SeededRandom } from "./random";
import { Signal } from "./signal";
import { createSynapse, propagateSynapses, Synapse } from "./synapse";
import { createLearningNetworkFromBlueprint, offlineLearningTopologyBlueprint } from "./topologyBlueprint";

export interface EvaluationResult {
  name: string;
  passed: boolean;
  metrics: Record<string, number | string | boolean>;
  notes: string[];
}

export interface LearningNetwork {
  neurons: Neuron[];
  synapses: Synapse[];
  pairMemory: PairMemory[];
  tick: number;
}

export interface Pattern {
  sensorId: string;
  targetMotorId: string;
}

export function runAllEvaluations(config: ModelConfig = defaultConfig): EvaluationResult[] {
  return [
    testSingleNeuronGate(config),
    testSingleSynapsePropagation(config),
    testSmallCircuitGate(config),
    testConnectionFormation(config),
    testOfflineLearning(config)
  ];
}

export function runLearningDemo(config: ModelConfig = defaultConfig): {
  network: LearningNetwork;
  config: ModelConfig;
  metrics: Record<string, number | string | boolean>;
  events: unknown[];
} {
  const evalConfig = withConfig({
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
  const network = createOfflineLearningNetwork(evalConfig);
  const events: unknown[] = [];
  const patterns = offlinePatterns();

  for (let epoch = 0; epoch < 60; epoch += 1) {
    for (const pattern of patterns) {
      const result = runMappingTrial(network, pattern, evalConfig, true);
      events.push({ epoch, ...result });
    }
  }

  const accuracy = evaluateMappingAccuracy(network, patterns, evalConfig);

  return {
    network,
    config: evalConfig,
    metrics: {
      accuracy,
      activeSynapses: network.synapses.filter((synapse) => synapse.state === "active").length,
      stableSynapses: network.synapses.filter((synapse) => synapse.state === "stable").length
    },
    events
  };
}

function testSingleNeuronGate(config: ModelConfig): EvaluationResult {
  const evalConfig = withConfig({
    ...config,
    leak: 1,
    branchLocalThreshold: 0.3,
    dendriteGateThreshold: 0.3,
    axonThreshold: 1,
    thresholdAdaptRate: 0,
    refractorySteps: 2
  });
  const neuron = createNeuron(
    {
      id: "n1",
      role: "interneuron",
      branchCount: 1
    },
    evalConfig
  );

  const stimulate = (value: number): Signal => {
    resetBranchInputs([neuron]);
    neuron.branches[0].inputSum = value;
    integrateNeuron(neuron, evalConfig);
    return neuron.outputSignal;
  };

  const weakA = stimulate(0.1);
  const weakB = stimulate(0.2);
  const strong = stimulate(1.1);
  const refractory = stimulate(1.1);
  stimulate(0);
  const recovered = stimulate(1.1);
  const passed = weakA === 0 && weakB === 0 && strong === 1 && refractory === 0 && recovered === 1;

  return {
    name: "Test A - single neuron gates",
    passed,
    metrics: {
      weakA,
      weakB,
      strong,
      refractory,
      recovered
    },
    notes: ["Validates dendrite gate, axon gate, and refractory behavior."]
  };
}

function testSingleSynapsePropagation(config: ModelConfig): EvaluationResult {
  const evalConfig = withConfig({
    ...config,
    leak: 1,
    branchLocalThreshold: 0.1,
    dendriteGateThreshold: 0.1,
    axonThreshold: 0.5,
    thresholdAdaptRate: 0,
    refractorySteps: 0
  });
  const source = createNeuron({ id: "sensory", role: "sensory" }, evalConfig);
  const excited = createNeuron({ id: "excited", role: "interneuron", branchCount: 1 }, evalConfig);
  const inhibited = createNeuron({ id: "inhibited", role: "interneuron", branchCount: 1 }, evalConfig);
  const neurons = [source, excited, inhibited];
  const neuronsById = indexNeurons(neurons);

  const excitatory = createSynapse(
    {
      id: "s-excited",
      preNeuronId: source.id,
      postNeuronId: excited.id,
      postBranchId: excited.branches[0].id,
      effectSign: 1,
      fastWeight: 0.8
    },
    evalConfig
  );
  const inhibitory = createSynapse(
    {
      id: "s-inhibited",
      preNeuronId: source.id,
      postNeuronId: inhibited.id,
      postBranchId: inhibited.branches[0].id,
      effectSign: -1,
      fastWeight: 0.8
    },
    evalConfig
  );

  setSensoryOutput(source, 1);
  resetBranchInputs(neurons);
  propagateSynapses(neuronsById, [excitatory], 1, evalConfig);
  integrateNeuron(excited, evalConfig);
  const excitedOutput = excited.outputSignal;

  resetNeuronRuntime(inhibited);
  resetBranchInputs(neurons);
  inhibited.branches[0].inputSum = 0.8;
  propagateSynapses(neuronsById, [inhibitory], 2, evalConfig);
  integrateNeuron(inhibited, evalConfig);
  const inhibitedOutput = inhibited.outputSignal;
  const passed = excitedOutput === 1 && inhibitedOutput === 0 && inhibited.branches[0].inhibitionLoad > 0;

  return {
    name: "Test B - single synapse propagation",
    passed,
    metrics: {
      excitedOutput,
      inhibitedOutput,
      inhibitionLoad: inhibited.branches[0].inhibitionLoad
    },
    notes: ["Validates excitatory propagation and local inhibitory suppression."]
  };
}

function testSmallCircuitGate(config: ModelConfig): EvaluationResult {
  const evalConfig = withConfig({
    ...config,
    leak: 1,
    branchLocalThreshold: 0.1,
    dendriteGateThreshold: 0.1,
    axonThreshold: 0.9,
    thresholdAdaptRate: 0,
    refractorySteps: 0
  });
  const sensory = createNeuron({ id: "S", role: "sensory" }, evalConfig);
  const inhibitor = createNeuron({ id: "I", role: "sensory" }, evalConfig);
  const inter = createNeuron({ id: "E", role: "interneuron", branchCount: 1 }, evalConfig);
  const motor = createNeuron({ id: "M", role: "motor", branchCount: 1 }, evalConfig);
  const neurons = [sensory, inhibitor, inter, motor];
  const neuronsById = indexNeurons(neurons);
  const sToE = createSynapse(
    {
      id: "S-E",
      preNeuronId: sensory.id,
      postNeuronId: inter.id,
      postBranchId: inter.branches[0].id,
      fastWeight: 1.2
    },
    evalConfig
  );
  const eToM = createSynapse(
    {
      id: "E-M",
      preNeuronId: inter.id,
      postNeuronId: motor.id,
      postBranchId: motor.branches[0].id,
      fastWeight: 1.2
    },
    evalConfig
  );
  const iToE = createSynapse(
    {
      id: "I-E",
      preNeuronId: inhibitor.id,
      postNeuronId: inter.id,
      postBranchId: inter.branches[0].id,
      effectSign: -1,
      fastWeight: 1.2
    },
    evalConfig
  );

  const withoutInhibition = runTwoStepCircuit(neurons, neuronsById, [sToE, eToM], sensory, null, inter, motor, evalConfig);
  const withInhibition = runTwoStepCircuit(
    neurons,
    neuronsById,
    [sToE, eToM, iToE],
    sensory,
    inhibitor,
    inter,
    motor,
    evalConfig
  );
  const recovered = runTwoStepCircuit(neurons, neuronsById, [sToE, eToM], sensory, null, inter, motor, evalConfig);
  const passed = withoutInhibition === 1 && withInhibition === 0 && recovered === 1;

  return {
    name: "Test C - small circuit gates",
    passed,
    metrics: {
      withoutInhibition,
      withInhibition,
      recovered
    },
    notes: ["Validates S -> E -> M propagation, local inhibition, and recovery."]
  };
}

function testConnectionFormation(config: ModelConfig): EvaluationResult {
  const evalConfig = withConfig({
    ...config,
    connectionDistanceLambda: 10,
    connectionThreshold: 0.2,
    candidateMaxAge: 1,
    dormantLimit: 1,
    baseCooldown: 10,
    fastWeightInit: 0.04
  });
  const neurons = [
    createNeuron({ id: "S1", role: "sensory", position: { x: 0, y: 0 } }, evalConfig),
    createNeuron({ id: "E1", role: "interneuron", position: { x: 1, y: 0 }, branchCount: 1 }, evalConfig),
    createNeuron({ id: "M1", role: "motor", position: { x: 2, y: 0 }, branchCount: 1 }, evalConfig),
    createNeuron({ id: "E_far", role: "interneuron", position: { x: 100, y: 100 }, branchCount: 1 }, evalConfig)
  ];
  const synapses: Synapse[] = [];
  const pairMemory: PairMemory[] = [];
  const rng = new SeededRandom(1);
  const formed = tryFormConnections(neurons, synapses, pairMemory, 1, evalConfig, rng, 4);

  const victim = synapses[0];
  if (victim) {
    victim.age = 2;
    victim.recentUse = 0;
  }

  const pruned = updateConnectionStates(neurons, synapses, pairMemory, 2, evalConfig);
  const blocked = tryFormConnections(neurons, synapses, pairMemory, 3, evalConfig, rng, 4);
  const passed = formed.formed > 0 && pruned.pruned > 0 && blocked.tombstoneHit > 0;

  return {
    name: "Test D - connection formation",
    passed,
    metrics: {
      formed: formed.formed,
      pruned: pruned.pruned,
      tombstoneHit: blocked.tombstoneHit,
      pairMemory: pairMemory.length
    },
    notes: ["Validates near-neighbor growth, candidate pruning, and reconnect cooldown."]
  };
}

function testOfflineLearning(config: ModelConfig): EvaluationResult {
  const evalConfig = withConfig({
    ...config,
    leak: 1,
    branchLocalThreshold: 0.1,
    dendriteGateThreshold: 0.1,
    axonThreshold: 1,
    thresholdAdaptRate: 0,
    refractorySteps: 0,
    fastDecay: 0.9995,
    supervisedLearningRate: 0.08,
    useThreshold: 0.08
  });
  const learningOnNetwork = createOfflineLearningNetwork(evalConfig);
  const learningOffNetwork = createOfflineLearningNetwork(evalConfig);
  const patterns = offlinePatterns();

  const learningOnAccuracy = trainAndScore(learningOnNetwork, patterns, evalConfig, true);
  const learningOffAccuracy = trainAndScore(learningOffNetwork, patterns, evalConfig, false);
  const stableWeightRatio =
    learningOnNetwork.synapses.reduce((sum, synapse) => sum + synapse.stableWeight, 0) /
    Math.max(1e-9, learningOnNetwork.synapses.reduce((sum, synapse) => sum + synapse.fastWeight + synapse.stableWeight, 0));
  const passed = learningOnAccuracy >= 0.95 && learningOnAccuracy > learningOffAccuracy;

  return {
    name: "Test E - offline input-output learning",
    passed,
    metrics: {
      learningOnAccuracy,
      learningOffAccuracy,
      stableWeightRatio
    },
    notes: ["Validates food/toxin channel-separated supervised mapping before 2D world integration."]
  };
}

function runTwoStepCircuit(
  neurons: Neuron[],
  neuronsById: Map<string, Neuron>,
  synapses: Synapse[],
  sensory: Neuron,
  inhibitor: Neuron | null,
  inter: Neuron,
  motor: Neuron,
  config: ModelConfig
): Signal {
  for (const neuron of neurons) {
    resetNeuronRuntime(neuron);
  }

  setSensoryOutput(sensory, 1);
  if (inhibitor) {
    setSensoryOutput(inhibitor, 1);
  }

  resetBranchInputs(neurons);
  propagateSynapses(neuronsById, synapses, 1, config);
  integrateNeuron(inter, config);

  resetBranchInputs(neurons);
  setSensoryOutput(sensory, 0);
  if (inhibitor) {
    setSensoryOutput(inhibitor, 0);
  }
  propagateSynapses(neuronsById, synapses, 2, config);
  integrateNeuron(motor, config);
  return motor.outputSignal;
}

export function createOfflineLearningNetwork(config: ModelConfig): LearningNetwork {
  return createLearningNetworkFromBlueprint(offlineLearningTopologyBlueprint, config);
}

export function offlinePatterns(): Pattern[] {
  return [
    { sensorId: "foodLeft", targetMotorId: "leftMotor" },
    { sensorId: "foodRight", targetMotorId: "rightMotor" },
    { sensorId: "toxinLeft", targetMotorId: "rightMotor" },
    { sensorId: "toxinRight", targetMotorId: "leftMotor" }
  ];
}

function trainAndScore(
  network: LearningNetwork,
  patterns: Pattern[],
  config: ModelConfig,
  learningOn: boolean
): number {
  const scoreWindowStart = 30;
  let correct = 0;
  let total = 0;

  for (let epoch = 0; epoch < 40; epoch += 1) {
    for (const pattern of patterns) {
      const result = runMappingTrial(network, pattern, config, learningOn);

      if (epoch >= scoreWindowStart) {
        total += 1;
        correct += result.correct ? 1 : 0;
      }
    }
  }

  return correct / total;
}

function evaluateMappingAccuracy(network: LearningNetwork, patterns: Pattern[], config: ModelConfig): number {
  let correct = 0;

  for (const pattern of patterns) {
    const result = runMappingTrial(network, pattern, config, false);
    correct += result.correct ? 1 : 0;
  }

  return correct / patterns.length;
}

function runMappingTrial(
  network: LearningNetwork,
  pattern: Pattern,
  config: ModelConfig,
  learningOn: boolean
): { correct: boolean; activeMotors: string[]; targetMotorId: string } {
  const neuronsById = indexNeurons(network.neurons);

  for (const neuron of network.neurons) {
    resetNeuronRuntime(neuron);
  }

  for (const neuron of network.neurons) {
    if (neuron.role === "sensory") {
      setSensoryOutput(neuron, neuron.id === pattern.sensorId ? 1 : 0);
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
    .map((neuron) => neuron.id);
  const activeMotorSet = new Set(activeMotors);
  const correct = activeMotorSet.has(pattern.targetMotorId) && activeMotorSet.size === 1;

  if (learningOn) {
    applySupervisedMotorLearning(network.synapses, neuronsById, pattern.targetMotorId, activeMotorSet, config);
    captureStableWeights(network.synapses, config);
  }

  decayWeights(network.synapses, config);
  return {
    correct,
    activeMotors,
    targetMotorId: pattern.targetMotorId
  };
}
