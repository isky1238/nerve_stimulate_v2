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

const SENSORY_NODES: TopologyNeuronBlueprint[] = [
  { id: "foodLeft", role: "sensory", position: { x: 0, y: 0 } },
  { id: "foodRight", role: "sensory", position: { x: 0, y: 1 } },
  { id: "toxinLeft", role: "sensory", position: { x: 0, y: 2 } },
  { id: "toxinRight", role: "sensory", position: { x: 0, y: 3 } }
];

const INTERNEURON_NODES: TopologyNeuronBlueprint[] = [
  { id: "iFoodLeft", role: "interneuron", position: { x: 1, y: 0 }, branchCount: 1 },
  { id: "iFoodRight", role: "interneuron", position: { x: 1, y: 1 }, branchCount: 1 },
  { id: "iToxinLeft", role: "interneuron", position: { x: 1, y: 2 }, branchCount: 1 },
  { id: "iToxinRight", role: "interneuron", position: { x: 1, y: 3 }, branchCount: 1 }
];

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
        branchCount: node.branchCount
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
