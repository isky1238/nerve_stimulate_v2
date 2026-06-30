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

  const sensoryNodes = createLayerNodes("input", "sensory", inputCount, 0);
  const mediumNodes = createLayerNodes("medium", "interneuron", mediumCount, 1, 1);
  const motorNodes = createLayerNodes("output", "motor", outputCount, 2, 1);
  const synapses: TopologySynapseBlueprint[] = [];

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

  applySlotCounts([...sensoryNodes, ...mediumNodes, ...motorNodes], synapses);

  return Object.freeze({
    sensoryNodes: Object.freeze(sensoryNodes),
    interneuronNodes: Object.freeze(mediumNodes),
    motorNodes: Object.freeze(motorNodes),
    synapses: Object.freeze(synapses)
  });
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
