import type { OfflineLearningTopologyBlueprint, TopologyNeuronBlueprint, TopologySynapseBlueprint } from "./topologyBlueprint";

export interface LayerCounts {
  inputCount: number;
  mediumCount: number;
  outputCount: number;
}

export interface ReducedLayerRatio extends LayerCounts {
  commonScale: number;
}

export interface NearestLayeredTopologyOptions extends LayerCounts {
  synapsesPerInput: number;
  synapsesPerMedium: number;
  structuralStableWeight?: number;
  readoutFastWeight?: number;
  /**
   * Readout wiring mode.
   * - "prewired" (default): stem (sensory->inter, decayProtected) AND readout
   *   (inter->motor, plastic) edges both pre-built. Existing behavior.
   * - "stem": only stem edges pre-built; inter->motor readout left empty for
   *   the developmental loop (tryFormConnections) to grow spontaneously.
   * - "empty": no edges at all; only neurons placed. Both stem and readout
   *   must grow spontaneously (may bootstrap-deadlock without a seed).
   *
   * In stem/empty modes, motorGrowthSlots reserves input slots on motor
   * (and output slots on sensory / both on interneuron for empty) so the
   * developmental loop has free slots to attach into.
   */
  readoutMode?: "prewired" | "stem" | "empty";
  motorGrowthSlots?: number;
}

export interface UniformNaturalLayeredTopologyOptions {
  layerSize?: number;
  inputCount?: number;
  mediumCount?: number;
  outputCount?: number;
  slotsPerNeuron: number;
}

export function reduceLayerCounts(counts: LayerCounts): ReducedLayerRatio {
  const inputCount = positiveInteger(counts.inputCount, "inputCount");
  const mediumCount = positiveInteger(counts.mediumCount, "mediumCount");
  const outputCount = positiveInteger(counts.outputCount, "outputCount");
  const commonScale = gcd(gcd(inputCount, mediumCount), outputCount);

  return {
    inputCount: inputCount / commonScale,
    mediumCount: mediumCount / commonScale,
    outputCount: outputCount / commonScale,
    commonScale
  };
}

export function sameLayerRatio(a: LayerCounts, b: LayerCounts): boolean {
  const left = reduceLayerCounts(a);
  const right = reduceLayerCounts(b);
  return (
    left.inputCount === right.inputCount &&
    left.mediumCount === right.mediumCount &&
    left.outputCount === right.outputCount
  );
}

export function createNearestLayeredTopologyBlueprint(
  options: NearestLayeredTopologyOptions
): OfflineLearningTopologyBlueprint {
  const inputCount = positiveInteger(options.inputCount, "inputCount");
  const mediumCount = positiveInteger(options.mediumCount, "mediumCount");
  const outputCount = positiveInteger(options.outputCount, "outputCount");
  const synapsesPerInput = boundedFanout(options.synapsesPerInput, "synapsesPerInput");
  const synapsesPerMedium = boundedFanout(options.synapsesPerMedium, "synapsesPerMedium");
  const structuralStableWeight = options.structuralStableWeight ?? 1.1;
  const readoutFastWeight = options.readoutFastWeight ?? 0.35;
  const readoutMode = options.readoutMode ?? "prewired";
  // Growth slot budget for stem/empty modes: room for the developmental loop
  // to attach readout synapses. Default = mediumCount (every medium could in
  // principle connect to a given motor).
  const growthSlots = options.motorGrowthSlots ?? mediumCount;

  const sensoryNodes = createLayerNodes("input", "sensory", inputCount, 0);
  const mediumNodes = createLayerNodes("medium", "interneuron", mediumCount, 1, 1);
  const motorNodes = createLayerNodes("output", "motor", outputCount, 2, 1);
  const synapses: TopologySynapseBlueprint[] = [];
  const buildStem = readoutMode === "prewired" || readoutMode === "stem";
  const buildReadout = readoutMode === "prewired";

  if (buildStem) {
    for (const input of sensoryNodes) {
      for (const medium of nearestNodes(input, mediumNodes, Math.min(synapsesPerInput, mediumNodes.length))) {
        synapses.push({
          kind: "structuralStem",
          preNeuronId: input.id,
          postNeuronId: medium.id,
          postBranchIndex: 0,
          fastWeight: 0,
          stableWeight: structuralStableWeight,
          decayProtected: true
        });
      }
    }
  }

  if (buildReadout) {
    for (const medium of mediumNodes) {
      for (const output of nearestNodes(medium, motorNodes, Math.min(synapsesPerMedium, motorNodes.length))) {
        synapses.push({
          kind: "plasticReadout",
          preNeuronId: medium.id,
          postNeuronId: output.id,
          postBranchIndex: 0,
          fastWeight: readoutFastWeight,
          stableWeight: 0,
          decayProtected: false
        });
      }
    }
  }

  // Slot accounting. In prewired mode the declared edges fully determine slots
  // (existing behavior). In stem/empty modes, motor (and in empty mode, all
  // nodes) have fewer/no declared edges, so applySlotCounts would zero their
  // slots and the developmental loop could never attach (hasFreeSlot always
  // false). Reserve growth slots explicitly in those modes.
  applySlotCounts([...sensoryNodes, ...mediumNodes, ...motorNodes], synapses);
  if (readoutMode === "stem") {
    // Readout (inter->motor) is not pre-built, so interneurons need free
    // OUTPUT slots and motors need free INPUT slots for the developmental
    // loop to attach readout synapses.
    for (const medium of mediumNodes) {
      medium.maxOutputSlots = Math.max(medium.maxOutputSlots ?? 0, growthSlots);
    }
    for (const motor of motorNodes) {
      motor.maxInputSlots = Math.max(motor.maxInputSlots ?? 0, growthSlots);
    }
  } else if (readoutMode === "empty") {
    for (const sensory of sensoryNodes) {
      sensory.maxOutputSlots = Math.max(sensory.maxOutputSlots ?? 0, growthSlots);
    }
    for (const medium of mediumNodes) {
      medium.maxInputSlots = Math.max(medium.maxInputSlots ?? 0, inputCount);
      medium.maxOutputSlots = Math.max(medium.maxOutputSlots ?? 0, growthSlots);
    }
    for (const motor of motorNodes) {
      motor.maxInputSlots = Math.max(motor.maxInputSlots ?? 0, growthSlots);
    }
  }

  return Object.freeze({
    sensoryNodes: Object.freeze(sensoryNodes),
    interneuronNodes: Object.freeze(mediumNodes),
    motorNodes: Object.freeze(motorNodes),
    synapses: Object.freeze(synapses)
  });
}

export function createUniformNaturalLayeredTopologyBlueprint(
  options: UniformNaturalLayeredTopologyOptions
): OfflineLearningTopologyBlueprint {
  const inputCount = naturalLayerCount(options, "inputCount");
  const mediumCount = naturalLayerCount(options, "mediumCount");
  const outputCount = naturalLayerCount(options, "outputCount");
  const slotsPerNeuron = boundedFanout(options.slotsPerNeuron, "slotsPerNeuron");

  return Object.freeze({
    sensoryNodes: Object.freeze(createUniformLayerNodes("input", "sensory", inputCount, 0, slotsPerNeuron)),
    interneuronNodes: Object.freeze(createUniformLayerNodes("medium", "interneuron", mediumCount, 1, slotsPerNeuron, 1)),
    motorNodes: Object.freeze(createUniformLayerNodes("output", "motor", outputCount, 2, slotsPerNeuron, 1)),
    synapses: Object.freeze([])
  });
}

function naturalLayerCount(options: UniformNaturalLayeredTopologyOptions, key: "inputCount" | "mediumCount" | "outputCount"): number {
  const value = options[key] ?? options.layerSize;
  if (value === undefined) {
    throw new Error(`${key} is required when layerSize is omitted.`);
  }
  return positiveInteger(value, key);
}

function createLayerNodes(
  prefix: string,
  role: TopologyNeuronBlueprint["role"],
  count: number,
  x: number,
  branchCount?: number
): TopologyNeuronBlueprint[] {
  return Array.from({ length: count }, (_, index): TopologyNeuronBlueprint => ({
    id: `${prefix}${index}`,
    role,
    position: { x, y: normalizedLayerY(index, count) },
    branchCount
  }));
}

function createUniformLayerNodes(
  prefix: string,
  role: TopologyNeuronBlueprint["role"],
  count: number,
  x: number,
  slotsPerNeuron: number,
  branchCount?: number
): TopologyNeuronBlueprint[] {
  return Array.from({ length: count }, (_, index): TopologyNeuronBlueprint => ({
    id: `${prefix}${index}`,
    role,
    position: { x, y: normalizedLayerY(index, count) },
    branchCount,
    maxInputSlots: slotsPerNeuron,
    maxOutputSlots: slotsPerNeuron
  }));
}

function normalizedLayerY(index: number, count: number): number {
  return count === 1 ? 0.5 : index / (count - 1);
}

function nearestNodes<T extends TopologyNeuronBlueprint>(
  source: TopologyNeuronBlueprint,
  targets: readonly T[],
  limit: number
): T[] {
  return [...targets]
    .sort((a, b) => {
      const distance = Math.abs(source.position.y - a.position.y) - Math.abs(source.position.y - b.position.y);
      return distance === 0 ? a.id.localeCompare(b.id) : distance;
    })
    .slice(0, limit);
}

function applySlotCounts(
  nodes: TopologyNeuronBlueprint[],
  synapses: readonly TopologySynapseBlueprint[]
): void {
  const inputs = new Map<string, number>();
  const outputs = new Map<string, number>();

  for (const synapse of synapses) {
    outputs.set(synapse.preNeuronId, (outputs.get(synapse.preNeuronId) ?? 0) + 1);
    inputs.set(synapse.postNeuronId, (inputs.get(synapse.postNeuronId) ?? 0) + 1);
  }

  for (const node of nodes) {
    node.maxInputSlots = inputs.get(node.id) ?? 0;
    node.maxOutputSlots = outputs.get(node.id) ?? 0;
  }
}

function boundedFanout(value: number, name: string): number {
  const intValue = positiveInteger(value, name);
  if (intValue > 5) {
    throw new Error(`${name} must be between 1 and 5.`);
  }
  return intValue;
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
  return value;
}

function gcd(a: number, b: number): number {
  let left = Math.abs(a);
  let right = Math.abs(b);

  while (right !== 0) {
    const next = left % right;
    left = right;
    right = next;
  }

  return left;
}
