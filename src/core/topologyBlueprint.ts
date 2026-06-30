import { ModelConfig } from "../config/newModelConfig";
import { attachSynapseToSlots } from "./development";
import type { LearningNetwork } from "./evaluation";
import { createNeuron, indexNeurons, NeuronRole, Position } from "./neuron";
import { createSynapse, Synapse } from "./synapse";

export type TopologySynapseKind = "structuralStem" | "plasticReadout";

export interface TopologyNeuronBlueprint {
  id: string;
  role: NeuronRole;
  position: Position;
  branchCount?: number;
  maxInputSlots?: number;
  maxOutputSlots?: number;
}

export interface TopologySynapseBlueprint {
  kind: TopologySynapseKind;
  preNeuronId: string;
  postNeuronId: string;
  postBranchIndex: number;
  fastWeight: number;
  stableWeight: number;
  decayProtected: boolean;
}

export interface OfflineLearningTopologyBlueprint {
  sensoryNodes: readonly TopologyNeuronBlueprint[];
  interneuronNodes: readonly TopologyNeuronBlueprint[];
  motorNodes: readonly TopologyNeuronBlueprint[];
  synapses: readonly TopologySynapseBlueprint[];
}

export interface ScaledTopologyOptions {
  interneuronCopiesPerSensor: number;
  normalizeReadoutByCopies?: boolean;
}

const CHANNELS = [
  { sensoryId: "foodLeft", interId: "iFoodLeft", y: 0 },
  { sensoryId: "foodRight", interId: "iFoodRight", y: 1 },
  { sensoryId: "toxinLeft", interId: "iToxinLeft", y: 2 },
  { sensoryId: "toxinRight", interId: "iToxinRight", y: 3 }
] as const;

const SENSORY_NODES: TopologyNeuronBlueprint[] = CHANNELS.map((channel) => ({
  id: channel.sensoryId,
  role: "sensory",
  position: { x: 0, y: channel.y }
}));

const INTERNEURON_NODES: TopologyNeuronBlueprint[] = CHANNELS.map((channel) => ({
  id: channel.interId,
  role: "interneuron",
  position: { x: 1, y: channel.y },
  branchCount: 1
}));

const MOTOR_NODES: TopologyNeuronBlueprint[] = [
  { id: "leftMotor", role: "motor", position: { x: 2, y: 0 }, branchCount: 1 },
  { id: "rightMotor", role: "motor", position: { x: 2, y: 1 }, branchCount: 1 }
];

export const offlineLearningTopologyBlueprint: OfflineLearningTopologyBlueprint = Object.freeze({
  sensoryNodes: Object.freeze([...SENSORY_NODES]),
  interneuronNodes: Object.freeze([...INTERNEURON_NODES]),
  motorNodes: Object.freeze([...MOTOR_NODES]),
  synapses: Object.freeze([
    ...SENSORY_NODES.map((sensory, index): TopologySynapseBlueprint => ({
      kind: "structuralStem",
      preNeuronId: sensory.id,
      postNeuronId: INTERNEURON_NODES[index].id,
      postBranchIndex: 0,
      fastWeight: 0,
      stableWeight: 1.1,
      decayProtected: true
    })),
    ...INTERNEURON_NODES.flatMap((inter): TopologySynapseBlueprint[] =>
      MOTOR_NODES.map((motor) => ({
        kind: "plasticReadout",
        preNeuronId: inter.id,
        postNeuronId: motor.id,
        postBranchIndex: 0,
        fastWeight: 0.35,
        stableWeight: 0,
        decayProtected: false
      }))
    )
  ])
});

export function createScaledOfflineLearningTopologyBlueprint(
  options: ScaledTopologyOptions
): OfflineLearningTopologyBlueprint {
  const copies = Math.max(1, Math.floor(options.interneuronCopiesPerSensor));

  if (copies === 1) {
    return offlineLearningTopologyBlueprint;
  }

  const readoutFastWeight = options.normalizeReadoutByCopies === false ? 0.35 : 0.35 / copies;
  const sensoryNodes: TopologyNeuronBlueprint[] = CHANNELS.map((channel) => ({
    id: channel.sensoryId,
    role: "sensory",
    position: { x: 0, y: channel.y },
    maxOutputSlots: copies
  }));
  const interneuronNodes: TopologyNeuronBlueprint[] = CHANNELS.flatMap((channel) =>
    Array.from({ length: copies }, (_, index): TopologyNeuronBlueprint => ({
      id: scaledInterneuronId(channel.interId, index),
      role: "interneuron",
      position: { x: 1, y: channel.y + index / Math.max(10, copies * 2) },
      branchCount: 1,
      maxInputSlots: 1,
      maxOutputSlots: MOTOR_NODES.length
    }))
  );
  const motorNodes: TopologyNeuronBlueprint[] = MOTOR_NODES.map((motor) => ({
    ...motor,
    maxInputSlots: CHANNELS.length * copies
  }));
  const synapses: TopologySynapseBlueprint[] = [
    ...CHANNELS.flatMap((channel) =>
      Array.from({ length: copies }, (_, index): TopologySynapseBlueprint => ({
        kind: "structuralStem",
        preNeuronId: channel.sensoryId,
        postNeuronId: scaledInterneuronId(channel.interId, index),
        postBranchIndex: 0,
        fastWeight: 0,
        stableWeight: 1.1,
        decayProtected: true
      }))
    ),
    ...interneuronNodes.flatMap((inter): TopologySynapseBlueprint[] =>
      MOTOR_NODES.map((motor) => ({
        kind: "plasticReadout",
        preNeuronId: inter.id,
        postNeuronId: motor.id,
        postBranchIndex: 0,
        fastWeight: readoutFastWeight,
        stableWeight: 0,
        decayProtected: false
      }))
    )
  ];

  return Object.freeze({
    sensoryNodes: Object.freeze(sensoryNodes),
    interneuronNodes: Object.freeze(interneuronNodes),
    motorNodes: Object.freeze(motorNodes),
    synapses: Object.freeze(synapses)
  });
}

function scaledInterneuronId(baseId: string, copyIndex: number): string {
  return copyIndex === 0 ? baseId : `${baseId}_copy${copyIndex + 1}`;
}

export function createLearningNetworkFromBlueprint(
  blueprint: OfflineLearningTopologyBlueprint,
  config: ModelConfig
): LearningNetwork {
  const neurons = [
    ...blueprint.sensoryNodes,
    ...blueprint.interneuronNodes,
    ...blueprint.motorNodes
  ].map((node) =>
    createNeuron(
      {
        id: node.id,
        role: node.role,
        position: node.position,
        branchCount: node.branchCount,
        maxInputSlots: node.maxInputSlots,
        maxOutputSlots: node.maxOutputSlots
      },
      config
    )
  );
  const neuronsById = indexNeurons(neurons);
  const synapses: Synapse[] = [];

  for (const edge of blueprint.synapses) {
    addBlueprintSynapse(synapses, neuronsById, edge, config);
  }

  return {
    neurons,
    synapses,
    pairMemory: [],
    tick: 0
  };
}

function addBlueprintSynapse(
  synapses: Synapse[],
  neuronsById: ReturnType<typeof indexNeurons>,
  edge: TopologySynapseBlueprint,
  config: ModelConfig
): void {
  const post = neuronsById.get(edge.postNeuronId);
  const postBranch = post?.branches[edge.postBranchIndex];

  if (!post || !postBranch) {
    throw new Error(`Unable to resolve blueprint post branch ${edge.postNeuronId}:${edge.postBranchIndex}.`);
  }

  const synapse = createSynapse(
    {
      id: `fixed-${edge.preNeuronId}-${edge.postNeuronId}-${synapses.length}`,
      preNeuronId: edge.preNeuronId,
      postNeuronId: edge.postNeuronId,
      postBranchId: postBranch.id,
      effectSign: 1,
      state: "active",
      fastWeight: edge.fastWeight,
      stableWeight: edge.stableWeight,
      decayProtected: edge.decayProtected
    },
    config
  );
  const pre = neuronsById.get(edge.preNeuronId);

  if (!pre || !attachSynapseToSlots(pre, post, synapse)) {
    throw new Error(`Unable to attach synapse ${synapse.id}.`);
  }

  synapses.push(synapse);
}
