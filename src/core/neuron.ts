import { ModelConfig } from "../config/newModelConfig";
import { Signal, ema } from "./signal";

export type NeuronRole = "sensory" | "interneuron" | "motor";
export type NeuronSubtype = "excitatory" | "inhibitory" | "modulatory";

export interface Position {
  x: number;
  y: number;
}

export interface Branch {
  id: string;
  neuronId: string;
  inputSum: number;
  inhibitionLoad: number;
  gain: number;
  localThreshold: number;
  active: boolean;
  plasticityGate: number;
  output: number;
  recentActiveRate: number;
  dormantTime: number;
}

export interface Gate {
  threshold: number;
  openRatio: number;
}

export interface Neuron {
  id: string;
  role: NeuronRole;
  subtype: NeuronSubtype;
  position: Position;
  branches: Branch[];

  somaPotential: number;
  axonDrive: number;
  dendriteToAxonGate: Gate;
  axonOutputGate: Gate;
  baseThreshold: number;
  dynamicThreshold: number;
  refractory: number;
  refractorySteps: number;

  inputSlots: Array<string | null>;
  outputSlots: Array<string | null>;
  maxInputSlots: number;
  maxOutputSlots: number;

  outputSignal: Signal;
  spike: boolean;
  recentSpikeRate: number;
  targetSpikeRate: number;
  dormantTime: number;
  overactiveTime: number;

  /**
   * Carried tag load from a tagged impulse (toxin sensory origin). Rides the
   * active conduction path: set on toxin sensory firing, propagated forward
   * to a postsynaptic neuron when that neuron fires after receiving tagged
   * input. Decays per tick when not re-driven. Read at capture time to decide
   * whether to flip accumulation direction on this neuron's readout synapses.
   */
  tagLoad: number;
  /**
   * Per-tick accumulator of incoming tag from active presynaptic partners.
   * Reset at the start of each propagation tick (resetBranchInputs); folded
   * into tagLoad during integrateNeuron if this neuron fires.
   */
  tagInputAccum: number;
}

export interface CreateNeuronParams {
  id: string;
  role: NeuronRole;
  subtype?: NeuronSubtype;
  position?: Position;
  branchCount?: number;
  maxInputSlots?: number;
  maxOutputSlots?: number;
  axonThreshold?: number;
  dendriteGateThreshold?: number;
  branchLocalThreshold?: number;
}

export function createBranch(
  neuronId: string,
  index: number,
  config: ModelConfig,
  localThreshold = config.branchLocalThreshold
): Branch {
  return {
    id: `${neuronId}:b${index}`,
    neuronId,
    inputSum: 0,
    inhibitionLoad: 0,
    gain: 1,
    localThreshold,
    active: false,
    plasticityGate: 1,
    output: 0,
    recentActiveRate: 0,
    dormantTime: 0
  };
}

export function createNeuron(params: CreateNeuronParams, config: ModelConfig): Neuron {
  const roleSlots = defaultSlotsForRole(params.role, config);
  const branchCount = params.branchCount ?? (params.role === "sensory" ? 0 : 1);
  const maxInputSlots = params.maxInputSlots ?? roleSlots.maxInputs;
  const maxOutputSlots = params.maxOutputSlots ?? roleSlots.maxOutputs;
  const axonThreshold = params.axonThreshold ?? config.axonThreshold;

  return {
    id: params.id,
    role: params.role,
    subtype: params.subtype ?? "excitatory",
    position: params.position ?? { x: 0, y: 0 },
    branches: Array.from({ length: branchCount }, (_, index) =>
      createBranch(params.id, index, config, params.branchLocalThreshold)
    ),

    somaPotential: 0,
    axonDrive: 0,
    dendriteToAxonGate: {
      threshold: params.dendriteGateThreshold ?? config.dendriteGateThreshold,
      openRatio: 0
    },
    axonOutputGate: {
      threshold: axonThreshold,
      openRatio: 0
    },
    baseThreshold: axonThreshold,
    dynamicThreshold: axonThreshold,
    refractory: 0,
    refractorySteps: config.refractorySteps,

    inputSlots: Array.from({ length: maxInputSlots }, () => null),
    outputSlots: Array.from({ length: maxOutputSlots }, () => null),
    maxInputSlots,
    maxOutputSlots,

    outputSignal: 0,
    spike: false,
    recentSpikeRate: 0,
    targetSpikeRate: config.targetSpikeRate,
    dormantTime: 0,
    overactiveTime: 0,
    tagLoad: 0,
    tagInputAccum: 0
  };
}

export function indexNeurons(neurons: Neuron[]): Map<string, Neuron> {
  return new Map(neurons.map((neuron) => [neuron.id, neuron]));
}

export function resetBranchInputs(neurons: Neuron[]): void {
  for (const neuron of neurons) {
    for (const branch of neuron.branches) {
      branch.inputSum = 0;
      branch.inhibitionLoad = 0;
      branch.gain = 1;
      branch.output = 0;
      branch.active = false;
      branch.plasticityGate = 1;
    }
    // Per-tick incoming tag accumulator is cleared before each propagation
    // pass; tagLoad (the carried-forward tag) is NOT cleared here — it decays
    // in integrateNeuron.
    neuron.tagInputAccum = 0;
  }
}

export function setSensoryOutput(neuron: Neuron, signal: Signal): void {
  if (neuron.role !== "sensory") {
    throw new Error(`Neuron ${neuron.id} is not sensory.`);
  }

  neuron.outputSignal = signal;
  neuron.spike = signal !== 0;
}

export function resetNeuronRuntime(neuron: Neuron, clearActivity = false): void {
  neuron.somaPotential = 0;
  neuron.axonDrive = 0;
  neuron.dendriteToAxonGate.openRatio = 0;
  neuron.axonOutputGate.openRatio = 0;
  neuron.refractory = 0;
  neuron.outputSignal = 0;
  neuron.spike = false;
  // Tag is per-trial transient runtime state; cleared on network reset so a
  // prior trial's tag does not leak into the next. The toxin sensory origin
  // re-sets tagLoad fresh each trial after reset.
  neuron.tagLoad = 0;
  neuron.tagInputAccum = 0;

  for (const branch of neuron.branches) {
    branch.inputSum = 0;
    branch.inhibitionLoad = 0;
    branch.gain = 1;
    branch.active = false;
    branch.output = 0;
    branch.plasticityGate = 1;

    if (clearActivity) {
      branch.recentActiveRate = 0;
      branch.dormantTime = 0;
    }
  }

  if (clearActivity) {
    neuron.recentSpikeRate = 0;
    neuron.dormantTime = 0;
    neuron.overactiveTime = 0;
  }
}

export function integrateNeuron(neuron: Neuron, config: ModelConfig): void {
  if (neuron.role === "sensory") {
    return;
  }

  let somaInput = 0;

  for (const branch of neuron.branches) {
    branch.gain = 1 / (1 + Math.max(0, branch.inhibitionLoad) * config.inhibitionShuntScale);
    const effectiveInput = branch.inputSum * branch.gain;
    branch.active = Math.abs(effectiveInput) >= branch.localThreshold;
    branch.output = branch.active ? effectiveInput : 0;
    branch.plasticityGate = branch.inhibitionLoad > config.inhibitionFreezeThreshold ? 0 : 1;
    branch.recentActiveRate = ema(branch.recentActiveRate, branch.active ? 1 : 0, config.emaAlpha);
    branch.dormantTime = branch.active ? 0 : branch.dormantTime + 1;
    somaInput += branch.output;
  }

  neuron.dendriteToAxonGate.openRatio =
    Math.abs(somaInput) >= neuron.dendriteToAxonGate.threshold ? 1 : 0;

  if (neuron.dendriteToAxonGate.openRatio > 0) {
    neuron.somaPotential = neuron.somaPotential * config.leak + somaInput;
  } else {
    neuron.somaPotential *= config.leak;
  }

  neuron.axonDrive = neuron.somaPotential;
  neuron.spike = false;
  neuron.outputSignal = 0;

  if (neuron.refractory > 0) {
    neuron.refractory -= 1;
  } else if (neuron.axonDrive >= neuron.dynamicThreshold) {
    neuron.spike = true;
    neuron.outputSignal = outputSignalForSubtype(neuron.subtype);
    neuron.axonOutputGate.openRatio = 1;
    neuron.somaPotential = 0;
    neuron.refractory = neuron.refractorySteps;
  } else {
    neuron.axonOutputGate.openRatio = 0;
  }

  neuron.recentSpikeRate = ema(neuron.recentSpikeRate, neuron.spike ? 1 : 0, config.emaAlpha);
  neuron.dormantTime = neuron.spike ? 0 : neuron.dormantTime + 1;
  neuron.overactiveTime = neuron.recentSpikeRate > neuron.targetSpikeRate * 2 ? neuron.overactiveTime + 1 : 0;

  // Tag propagation: a tagged impulse rides the active path. If this neuron
  // fired AND received tagged input this tick, it carries the tag forward
  // (so the next layer's synapses can be marked). If it did not fire, its
  // carried tag decays. A firing neuron with no tagged input also decays its
  // carried tag (tag does not spontaneously arise — only toxin sensory
  // origin sets it, and only active propagation carries it forward).
  if (neuron.spike && neuron.tagInputAccum > 0) {
    neuron.tagLoad = Math.max(neuron.tagLoad * config.tagDecay, neuron.tagInputAccum);
  } else {
    neuron.tagLoad *= config.tagDecay;
  }

  const thresholdDelta = (neuron.recentSpikeRate - neuron.targetSpikeRate) * config.thresholdAdaptRate;
  neuron.dynamicThreshold = Math.min(
    config.thresholdMax,
    Math.max(config.thresholdMin, neuron.dynamicThreshold + thresholdDelta)
  );
}

function defaultSlotsForRole(
  role: NeuronRole,
  config: ModelConfig
): { maxInputs: number; maxOutputs: number } {
  if (role === "sensory") {
    return { maxInputs: config.sensoryMaxInputs, maxOutputs: config.sensoryMaxOutputs };
  }

  if (role === "motor") {
    return { maxInputs: config.motorMaxInputs, maxOutputs: config.motorMaxOutputs };
  }

  return { maxInputs: config.interneuronMaxInputs, maxOutputs: config.interneuronMaxOutputs };
}

function outputSignalForSubtype(subtype: NeuronSubtype): Signal {
  if (subtype === "inhibitory") {
    return -1;
  }

  return 1;
}
