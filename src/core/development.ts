import { ModelConfig } from "../config/newModelConfig";
import { Neuron, indexNeurons } from "./neuron";
import { SeededRandom } from "./random";
import { createSynapse, Synapse } from "./synapse";

export interface PairMemory {
  preNeuronId: string;
  postNeuronId: string;
  postBranchId: string;
  lastPrunedTime: number;
  failureCount: number;
  cooldownUntil: number;
}

export interface DevelopmentMetrics {
  formed: number;
  activated: number;
  dormant: number;
  pruned: number;
  tombstoneHit: number;
}

export function createDevelopmentMetrics(): DevelopmentMetrics {
  return {
    formed: 0,
    activated: 0,
    dormant: 0,
    pruned: 0,
    tombstoneHit: 0
  };
}

export function roleCompatible(pre: Neuron, post: Neuron): boolean {
  if (pre.id === post.id) {
    return false;
  }

  if (pre.role === "sensory") {
    return post.role === "interneuron";
  }

  if (pre.role === "interneuron") {
    return post.role === "interneuron" || post.role === "motor";
  }

  return false;
}

export function attachSynapseToSlots(pre: Neuron, post: Neuron, synapse: Synapse): boolean {
  const outputSlot = pre.outputSlots.findIndex((slot) => slot === null);
  const inputSlot = post.inputSlots.findIndex((slot) => slot === null);

  if (outputSlot < 0 || inputSlot < 0) {
    return false;
  }

  pre.outputSlots[outputSlot] = synapse.id;
  post.inputSlots[inputSlot] = synapse.id;
  return true;
}

export function releaseSynapseFromSlots(pre: Neuron, post: Neuron, synapseId: string): void {
  replaceSlotValue(pre.outputSlots, synapseId, null);
  replaceSlotValue(post.inputSlots, synapseId, null);
}

export function tryFormConnections(
  neurons: Neuron[],
  synapses: Synapse[],
  pairMemory: PairMemory[],
  tick: number,
  config: ModelConfig,
  rng: SeededRandom,
  maxNewConnections = 8
): DevelopmentMetrics {
  const metrics = createDevelopmentMetrics();
  let remaining = maxNewConnections;

  for (const pre of neurons) {
    // Skip pres with no free output slot (e.g. a sensory whose stem edges
    // already used its full slot budget), but keep scanning other pres — a
    // later interneuron may still form readout connections. Only the global
    // remaining-budget exhaustion should stop the whole sweep.
    if (remaining <= 0) {
      break;
    }
    if (!hasFreeSlot(pre.outputSlots)) {
      continue;
    }

    for (const post of neurons) {
      if (remaining <= 0 || !hasFreeSlot(pre.outputSlots)) {
        break;
      }

      if (!roleCompatible(pre, post) || !hasFreeSlot(post.inputSlots)) {
        continue;
      }

      for (const branch of post.branches) {
        if (remaining <= 0) {
          break;
        }

        if (hasLiveConnection(synapses, pre.id, post.id, branch.id)) {
          continue;
        }

        const memory = findPairMemory(pairMemory, pre.id, post.id, branch.id);
        if (memory && memory.cooldownUntil > tick) {
          metrics.tombstoneHit += 1;
          continue;
        }

        const score = connectionScore(pre, post, config);
        if (score < config.connectionThreshold || rng.next() > score) {
          continue;
        }

        const synapse = createSynapse(
          {
            id: `syn-${synapses.length + 1}`,
            preNeuronId: pre.id,
            postNeuronId: post.id,
            postBranchId: branch.id,
            effectSign: 1,
            state: "candidate",
            fastWeight: config.fastWeightInit,
            stableWeight: 0
          },
          config
        );

        if (!attachSynapseToSlots(pre, post, synapse)) {
          continue;
        }

        synapses.push(synapse);
        metrics.formed += 1;
        remaining -= 1;
      }
    }
  }

  return metrics;
}

export function updateConnectionStates(
  neurons: Neuron[],
  synapses: Synapse[],
  pairMemory: PairMemory[],
  tick: number,
  config: ModelConfig
): DevelopmentMetrics {
  const metrics = createDevelopmentMetrics();
  const neuronsById = indexNeurons(neurons);

  for (const synapse of synapses) {
    if (synapse.state === "pruned") {
      continue;
    }

    if (
      synapse.state === "candidate" &&
      synapse.age > config.candidateMaxAge &&
      synapse.recentUse >= config.useThreshold &&
      synapse.recentContribution > 0
    ) {
      synapse.state = "active";
      metrics.activated += 1;
      continue;
    }

    if (
      synapse.state === "candidate" &&
      synapse.age > config.candidateMaxAge &&
      synapse.recentUse < config.useThreshold
    ) {
      pruneSynapse(neuronsById, synapse, pairMemory, tick, config);
      metrics.pruned += 1;
      continue;
    }

    if (
      synapse.state !== "stable" &&
      synapse.state !== "dormant" &&
      synapse.age > config.minConnectionAge &&
      synapse.recentUse < config.useThreshold &&
      synapse.fastWeight + synapse.stableWeight < config.weakWeightThreshold &&
      synapse.recentContribution <= 0
    ) {
      synapse.state = "dormant";
      synapse.dormantTicks = 0;
      metrics.dormant += 1;
      continue;
    }

    if (synapse.state === "dormant") {
      synapse.dormantTicks += 1;

      if (synapse.dormantTicks > config.dormantLimit) {
        pruneSynapse(neuronsById, synapse, pairMemory, tick, config);
        metrics.pruned += 1;
      }
    }
  }

  return metrics;
}

function pruneSynapse(
  neuronsById: Map<string, Neuron>,
  synapse: Synapse,
  pairMemory: PairMemory[],
  tick: number,
  config: ModelConfig
): void {
  const pre = neuronsById.get(synapse.preNeuronId);
  const post = neuronsById.get(synapse.postNeuronId);

  if (pre && post) {
    releaseSynapseFromSlots(pre, post, synapse.id);
  }

  synapse.connected = false;
  synapse.state = "pruned";
  synapse.reconnectCooldown = config.baseCooldown;
  synapse.pruneMark = tick;
  recordPairMemory(pairMemory, synapse, tick, config);
}

function recordPairMemory(pairMemory: PairMemory[], synapse: Synapse, tick: number, config: ModelConfig): void {
  const existing = findPairMemory(
    pairMemory,
    synapse.preNeuronId,
    synapse.postNeuronId,
    synapse.postBranchId
  );

  if (existing) {
    existing.failureCount += 1;
    existing.lastPrunedTime = tick;
    existing.cooldownUntil = tick + config.baseCooldown * (1 + existing.failureCount) ** 2;
    return;
  }

  pairMemory.push({
    preNeuronId: synapse.preNeuronId,
    postNeuronId: synapse.postNeuronId,
    postBranchId: synapse.postBranchId,
    lastPrunedTime: tick,
    failureCount: 1,
    cooldownUntil: tick + config.baseCooldown * 4
  });
}

function hasFreeSlot(slots: Array<string | null>): boolean {
  return slots.some((slot) => slot === null);
}

function replaceSlotValue(slots: Array<string | null>, from: string, to: string | null): void {
  const index = slots.findIndex((slot) => slot === from);

  if (index >= 0) {
    slots[index] = to;
  }
}

function findPairMemory(
  pairMemory: PairMemory[],
  preNeuronId: string,
  postNeuronId: string,
  postBranchId: string
): PairMemory | undefined {
  return pairMemory.find(
    (memory) =>
      memory.preNeuronId === preNeuronId &&
      memory.postNeuronId === postNeuronId &&
      memory.postBranchId === postBranchId
  );
}

function hasLiveConnection(
  synapses: Synapse[],
  preNeuronId: string,
  postNeuronId: string,
  postBranchId: string
): boolean {
  return synapses.some(
    (synapse) =>
      synapse.preNeuronId === preNeuronId &&
      synapse.postNeuronId === postNeuronId &&
      synapse.postBranchId === postBranchId &&
      synapse.state !== "pruned"
  );
}

function connectionScore(pre: Neuron, post: Neuron, config: ModelConfig): number {
  const dx = pre.position.x - post.position.x;
  const dy = pre.position.y - post.position.y;
  const distance = Math.sqrt(dx * dx + dy * dy);
  return Math.exp(-distance / config.connectionDistanceLambda);
}
