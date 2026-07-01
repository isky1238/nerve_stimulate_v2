import { ModelConfig } from "../config/newModelConfig";
import { Branch, Neuron } from "./neuron";
import { Signal, clampMagnitude, ema, isActiveSignal } from "./signal";

export type SynapseState = "candidate" | "active" | "dormant" | "pruned" | "stable";
export type EffectSign = 1 | -1;

export interface Synapse {
  id: string;
  preNeuronId: string;
  postNeuronId: string;
  postBranchId: string;

  effectSign: EffectSign;
  connected: boolean;
  state: SynapseState;

  fastWeight: number;
  stableWeight: number;
  effectiveWeight: number;

  age: number;
  dormantTicks: number;
  lastUsedTime: number;
  recentUse: number;
  recentContribution: number;

  preTrace: number;
  postTrace: number;
  eligibilityTrace: number;

  reconnectCooldown: number;
  pruneMark: number;
  stabilityScore: number;
  /**
   * Structural hardware stem flag. True for hardwired sensory-input干线
   * (e.g. sensory→interneuron fixed synapses) that carry the network's
   * only afferent signal and must NOT be eroded by passive stableDecay —
   * eroding them past the post-synaptic axon threshold silently severs the
   * whole downstream motor chain (observed as the long-range rewardOnly
   * noop cliff around epoch ~200-250). fastWeight decay, learning updates,
   * and effectiveWeight computation are unchanged; only stableWeight passive
   * decay is skipped. Learned/plastic synapses stay false.
   */
  decayProtected: boolean;
  /**
   * Tag load reaching this synapse from a tagged impulse along the active
   * path. Set during propagateSynapses when the presynaptic partner carries
   * tagLoad and fires. Read at capture time to decide whether to flip the
   * accumulation direction (de-consolidation). Cleared on network reset.
   */
  tagLoad: number;
}

export interface CreateSynapseParams {
  id: string;
  preNeuronId: string;
  postNeuronId: string;
  postBranchId: string;
  effectSign?: EffectSign;
  state?: SynapseState;
  fastWeight?: number;
  stableWeight?: number;
  decayProtected?: boolean;
}

export interface PropagationEvent {
  synapseId: string;
  preSignal: Signal;
  effect: number;
}

export function createSynapse(params: CreateSynapseParams, config: ModelConfig): Synapse {
  const synapse: Synapse = {
    id: params.id,
    preNeuronId: params.preNeuronId,
    postNeuronId: params.postNeuronId,
    postBranchId: params.postBranchId,
    effectSign: params.effectSign ?? 1,
    connected: true,
    state: params.state ?? "active",
    fastWeight: params.fastWeight ?? config.fastWeightInit,
    stableWeight: params.stableWeight ?? config.stableWeightInit,
    effectiveWeight: 0,
    age: 0,
    dormantTicks: 0,
    lastUsedTime: -1,
    recentUse: 0,
    recentContribution: 0,
    preTrace: 0,
    postTrace: 0,
    eligibilityTrace: 0,
    reconnectCooldown: 0,
    pruneMark: 0,
    stabilityScore: 0,
    decayProtected: params.decayProtected ?? false,
    tagLoad: 0
  };

  refreshSynapseWeight(synapse, config);
  return synapse;
}

export function refreshSynapseWeight(synapse: Synapse, config: ModelConfig): void {
  const magnitude = clampMagnitude(synapse.fastWeight + synapse.stableWeight, 0, config.maxWeight);
  synapse.effectiveWeight = synapse.effectSign * magnitude;
}

export function refreshSynapseWeights(synapses: Synapse[], config: ModelConfig): void {
  for (const synapse of synapses) {
    refreshSynapseWeight(synapse, config);
  }
}

export function isConductingSynapse(synapse: Synapse): boolean {
  return synapse.connected && synapse.state !== "pruned" && synapse.state !== "dormant";
}

export function propagateSynapses(
  neuronsById: Map<string, Neuron>,
  synapses: Synapse[],
  tick: number,
  config: ModelConfig
): PropagationEvent[] {
  const events: PropagationEvent[] = [];

  for (const synapse of synapses) {
    synapse.age += 1;
    refreshSynapseWeight(synapse, config);

    if (!isConductingSynapse(synapse)) {
      continue;
    }

    const pre = neuronsById.get(synapse.preNeuronId);
    const post = neuronsById.get(synapse.postNeuronId);
    const postBranch = post?.branches.find((branch) => branch.id === synapse.postBranchId);

    if (!pre || !post || !postBranch) {
      continue;
    }

    if (!isActiveSignal(pre.outputSignal)) {
      synapse.recentUse = ema(synapse.recentUse, 0, config.emaAlpha);
      // No active impulse traverses this synapse this tick, so it carries no
      // tag forward. Decay any residual tag (e.g. from a prior tick).
      synapse.tagLoad = ema(synapse.tagLoad, 0, config.emaAlpha);
      continue;
    }

    const effect = pre.outputSignal * synapse.effectiveWeight;
    applyEffectToBranch(postBranch, effect);
    synapse.lastUsedTime = tick;
    synapse.recentUse = ema(synapse.recentUse, 1, config.emaAlpha);

    // Tagged-impulse hitchhiking: if the presynaptic partner carries tagLoad,
    // the active impulse marks this synapse and deposits tag into the post
    // neuron's per-tick accumulator (folded into post.tagLoad at integrate
    // time iff the post fires). This does NOT alter the forward effect — tag
    // only travels along the active path, it does not drive activity.
    if (pre.tagLoad > 0) {
      const transferred = pre.tagLoad * config.tagTransferRate;
      synapse.tagLoad = Math.max(synapse.tagLoad, transferred);
      post.tagInputAccum += transferred;
    } else {
      synapse.tagLoad = ema(synapse.tagLoad, 0, config.emaAlpha);
    }

    events.push({
      synapseId: synapse.id,
      preSignal: pre.outputSignal,
      effect
    });
  }

  return events;
}

function applyEffectToBranch(branch: Branch, effect: number): void {
  branch.inputSum += effect;

  if (effect < 0) {
    branch.inhibitionLoad += Math.abs(effect);
  }
}
