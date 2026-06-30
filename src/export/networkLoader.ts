import { ModelConfig, withConfig } from "../config/newModelConfig";
import { LearningNetwork, createOfflineLearningNetwork } from "../core/evaluation";
import { Neuron, resetNeuronRuntime } from "../core/neuron";
import { refreshSynapseWeights, Synapse } from "../core/synapse";
import { NetworkExport } from "./networkExport";

interface SnapshotNeuron {
  id: string;
  role: string;
  subtype: string;
  threshold: number;
  dynamicThreshold: number;
  recentSpikeRate: number;
  dormantTime: number;
  position: { x: number; y: number };
  inputSlots: Array<string | null>;
  outputSlots: Array<string | null>;
}

interface SnapshotBranch {
  neuronId: string;
  branchId: string;
  inputSum: number;
  inhibitionLoad: number;
  gain: number;
  plasticityGate: number;
  activeRate: number;
}

export function loadNetworkFromExport(snapshot: NetworkExport): {
  network: LearningNetwork;
  config: ModelConfig;
} {
  if (snapshot.version !== "dg-snn-v0.1") {
    throw new Error(`Unsupported export version: ${snapshot.version}`);
  }

  const config = withConfig(snapshot.config);
  const network = createOfflineLearningNetwork(config);

  validateStructure(snapshot, network);

  applySynapseState(network.synapses, snapshot.synapses);
  applyNeuronState(network.neurons, snapshot.neurons as SnapshotNeuron[]);
  applyBranchState(network.neurons, snapshot.branches as SnapshotBranch[]);

  refreshSynapseWeights(network.synapses, config);

  for (const neuron of network.neurons) {
    resetNeuronRuntime(neuron);
  }

  network.tick = 0;
  network.pairMemory = snapshot.pairMemory.map((entry) => ({ ...entry }));

  return { network, config };
}

function validateStructure(snapshot: NetworkExport, network: LearningNetwork): void {
  if (snapshot.synapses.length !== network.synapses.length) {
    throw new Error(
      `Synapse count mismatch: skeleton=${network.synapses.length} snapshot=${snapshot.synapses.length}`
    );
  }

  const snapshotSynapsesById = new Map(snapshot.synapses.map((synapse) => [synapse.id, synapse]));
  for (const synapse of network.synapses) {
    if (!snapshotSynapsesById.has(synapse.id)) {
      throw new Error(`Skeleton synapse ${synapse.id} missing from snapshot`);
    }
  }

  const snapshotNeurons = snapshot.neurons as SnapshotNeuron[];
  if (snapshotNeurons.length !== network.neurons.length) {
    throw new Error(
      `Neuron count mismatch: skeleton=${network.neurons.length} snapshot=${snapshotNeurons.length}`
    );
  }
  const snapshotNeuronsById = new Map(snapshotNeurons.map((neuron) => [neuron.id, neuron]));
  for (const neuron of network.neurons) {
    if (!snapshotNeuronsById.has(neuron.id)) {
      throw new Error(`Skeleton neuron ${neuron.id} missing from snapshot`);
    }
  }
  const skeletonNeuronIds = new Set(network.neurons.map((neuron) => neuron.id));
  for (const snapshotNeuron of snapshotNeurons) {
    if (!skeletonNeuronIds.has(snapshotNeuron.id)) {
      throw new Error(`Snapshot neuron ${snapshotNeuron.id} not in skeleton (extra)`);
    }
  }

  const snapshotBranches = snapshot.branches as SnapshotBranch[];
  const skeletonBranchKeys: string[] = [];
  for (const neuron of network.neurons) {
    for (const branch of neuron.branches) {
      skeletonBranchKeys.push(`${neuron.id}:${branch.id}`);
    }
  }
  if (snapshotBranches.length !== skeletonBranchKeys.length) {
    throw new Error(
      `Branch count mismatch: skeleton=${skeletonBranchKeys.length} snapshot=${snapshotBranches.length}`
    );
  }
  const snapshotBranchesByKey = new Map(
    snapshotBranches.map((branch) => [`${branch.neuronId}:${branch.branchId}`, branch])
  );
  for (const key of skeletonBranchKeys) {
    if (!snapshotBranchesByKey.has(key)) {
      throw new Error(`Skeleton branch ${key} missing from snapshot`);
    }
  }
  const skeletonBranchKeySet = new Set(skeletonBranchKeys);
  for (const snapshotBranch of snapshotBranches) {
    const key = `${snapshotBranch.neuronId}:${snapshotBranch.branchId}`;
    if (!skeletonBranchKeySet.has(key)) {
      throw new Error(`Snapshot branch ${key} not in skeleton (extra)`);
    }
  }
}

function applySynapseState(target: Synapse[], source: Synapse[]): void {
  const sourceById = new Map(source.map((synapse) => [synapse.id, synapse]));

  for (const synapse of target) {
    const snap = sourceById.get(synapse.id);
    if (!snap) {
      continue;
    }

    synapse.effectSign = snap.effectSign;
    synapse.connected = snap.connected;
    synapse.state = snap.state;
    synapse.fastWeight = snap.fastWeight;
    synapse.stableWeight = snap.stableWeight;
    synapse.age = snap.age;
    synapse.dormantTicks = snap.dormantTicks;
    synapse.lastUsedTime = snap.lastUsedTime;
    synapse.recentUse = snap.recentUse;
    synapse.recentContribution = snap.recentContribution;
    synapse.preTrace = snap.preTrace;
    synapse.postTrace = snap.postTrace;
    synapse.eligibilityTrace = snap.eligibilityTrace;
    synapse.reconnectCooldown = snap.reconnectCooldown;
    synapse.pruneMark = snap.pruneMark;
    synapse.stabilityScore = snap.stabilityScore;
    // Preserve the structural-stem flag (older snapshots written before this
    // field default to false; the skeleton rebuilt by createOfflineLearningNetwork
    // already carries the correct value, so only overwrite when the snapshot
    // actually records it).
    if (typeof snap.decayProtected === "boolean") {
      synapse.decayProtected = snap.decayProtected;
    }
  }
}

function applyNeuronState(target: Neuron[], source: SnapshotNeuron[]): void {
  const sourceById = new Map(source.map((neuron) => [neuron.id, neuron]));

  for (const neuron of target) {
    const snap = sourceById.get(neuron.id);
    if (!snap) {
      continue;
    }

    neuron.dynamicThreshold = snap.dynamicThreshold;
    neuron.recentSpikeRate = snap.recentSpikeRate;
    neuron.dormantTime = snap.dormantTime;
  }
}

function applyBranchState(target: Neuron[], source: SnapshotBranch[]): void {
  const sourceByKey = new Map(
    source.map((branch) => [`${branch.neuronId}:${branch.branchId}`, branch])
  );

  for (const neuron of target) {
    for (const branch of neuron.branches) {
      const snap = sourceByKey.get(`${neuron.id}:${branch.id}`);
      if (!snap) {
        continue;
      }

      branch.recentActiveRate = snap.activeRate;
    }
  }
}
