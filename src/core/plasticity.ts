import { ModelConfig } from "../config/newModelConfig";
import { Neuron } from "./neuron";
import { clampMagnitude, ema, isActiveSignal } from "./signal";
import { Synapse, isConductingSynapse, refreshSynapseWeight } from "./synapse";
import {
  AversiveLearningTag,
  computeAversiveStableDepotentiationDelta,
  computeRewardFastDelta,
  computeStableCaptureAmount,
  computeStableDepotentiationDelta,
  computeStdpEligibilityDelta,
  computeSupervisedFastDelta,
  nextActivityTrace,
  nextEligibilityTrace,
  positiveEligibilityScale,
  shouldApplyAversiveStableDepotentiation
} from "./plasticityMechanisms";

export interface LearningEvent {
  synapseId: string;
  kind: "reward" | "supervised" | "capture" | "decay";
  deltaFast: number;
  deltaStable: number;
}

export function updateEligibility(
  synapses: Synapse[],
  neuronsById: Map<string, Neuron>,
  config: ModelConfig
): void {
  for (const synapse of synapses) {
    if (!isConductingSynapse(synapse)) {
      continue;
    }

    const pre = neuronsById.get(synapse.preNeuronId);
    const post = neuronsById.get(synapse.postNeuronId);

    if (!pre || !post) {
      continue;
    }

    const preActive = isActiveSignal(pre.outputSignal) ? 1 : 0;
    const postActive = post.spike ? 1 : 0;
    synapse.preTrace = nextActivityTrace(synapse.preTrace, preActive, config.traceDecay);
    synapse.postTrace = nextActivityTrace(synapse.postTrace, postActive, config.traceDecay);

    const stdp = computeStdpEligibilityDelta(
      {
        preTrace: synapse.preTrace,
        postTrace: synapse.postTrace,
        preActive,
        postActive,
        effectSign: synapse.effectSign,
        effectiveWeight: synapse.effectiveWeight
      },
      config
    );
    synapse.eligibilityTrace = nextEligibilityTrace(
      synapse.eligibilityTrace,
      stdp.eligibilityDelta,
      config
    );

    // recentContribution still tracks absolute contribution magnitude (used by
    // captureStableWeights gates). BAP weighting changes its meaning slightly
    // (driven by post firing rather than binary coactivity) but the magnitude
    // semantics for capture are preserved.
    synapse.recentContribution = ema(
      synapse.recentContribution,
      postActive ? Math.abs(synapse.effectiveWeight) : 0,
      config.emaAlpha
    );
  }
}

/**
 * Phase 2: per-post-spike divisive normalization of positive (LTP) eligibility.
 * For each post that just spiked, scale its incoming synapses' positive
 * eligibility by 1/Σ(positive eligibility) so total LTP credit per post-spike is
 * bounded at 1.0 — counters the "normal-summation" risk where many weak synapses
 * each claim full credit. LTD (negative eligibility) is left untouched.
 */
export function normalizeEligibility(
  synapses: Synapse[],
  neuronsById: Map<string, Neuron>
): void {
  const byPost = new Map<string, Synapse[]>();
  for (const synapse of synapses) {
    if (!isConductingSynapse(synapse)) {
      continue;
    }
    const list = byPost.get(synapse.postNeuronId);
    if (list) {
      list.push(synapse);
    } else {
      byPost.set(synapse.postNeuronId, [synapse]);
    }
  }

  for (const [postId, incoming] of byPost) {
    const post = neuronsById.get(postId);
    if (!post || !post.spike) {
      continue;
    }
    let ltpSum = 0;
    for (const synapse of incoming) {
      if (synapse.eligibilityTrace > 0) {
        ltpSum += synapse.eligibilityTrace;
      }
    }
    if (ltpSum <= 1e-12) {
      continue;
    }
    const scale = positiveEligibilityScale(ltpSum);
    for (const synapse of incoming) {
      if (synapse.eligibilityTrace > 0) {
        synapse.eligibilityTrace *= scale;
      }
    }
  }
}

export function applyRewardLearning(
  synapses: Synapse[],
  neuronsById: Map<string, Neuron>,
  rewardSignal: number,
  modulator: number,
  config: ModelConfig
): LearningEvent[] {
  const events: LearningEvent[] = [];

  for (const synapse of synapses) {
    if (!isConductingSynapse(synapse) || synapse.eligibilityTrace === 0) {
      continue;
    }

    const post = neuronsById.get(synapse.postNeuronId);
    const branch = post?.branches.find((candidate) => candidate.id === synapse.postBranchId);
    const plasticityGate = branch?.plasticityGate ?? 1;
    const before = synapse.fastWeight;
    const delta = computeRewardFastDelta(
      rewardSignal,
      synapse.eligibilityTrace,
      plasticityGate,
      modulator,
      config
    );
    synapse.fastWeight = clampMagnitude(synapse.fastWeight + delta, 0, config.maxWeight);
    refreshSynapseWeight(synapse, config);

    if (synapse.fastWeight !== before) {
      events.push({
        synapseId: synapse.id,
        kind: "reward",
        deltaFast: synapse.fastWeight - before,
        deltaStable: 0
      });
    }
  }

  return events;
}

export function applyAversiveStableDepotentiation(
  synapses: Synapse[],
  neuronsById: Map<string, Neuron>,
  aversiveTag: AversiveLearningTag | undefined,
  config: ModelConfig
): LearningEvent[] {
  if (!shouldApplyAversiveStableDepotentiation(aversiveTag, config)) {
    return [];
  }

  const events: LearningEvent[] = [];
  const intensity = aversiveTag?.intensity ?? 0;

  for (const synapse of synapses) {
    if (!isConductingSynapse(synapse) || synapse.stableWeight <= 0 || synapse.eligibilityTrace === 0) {
      continue;
    }

    const pre = neuronsById.get(synapse.preNeuronId);
    const post = neuronsById.get(synapse.postNeuronId);

    if (!pre || !post || post.role !== "motor" || !isActiveSignal(pre.outputSignal) || !isActiveSignal(post.outputSignal)) {
      continue;
    }

    const branch = post.branches.find((candidate) => candidate.id === synapse.postBranchId);
    const plasticityGate = branch?.plasticityGate ?? 1;
    const stableBefore = synapse.stableWeight;
    const stableDelta = computeAversiveStableDepotentiationDelta(
      synapse.eligibilityTrace,
      plasticityGate,
      intensity,
      config
    );

    synapse.stableWeight = clampMagnitude(synapse.stableWeight + stableDelta, 0, config.maxWeight);
    if (synapse.stableWeight < config.stableThreshold) {
      synapse.state = "active";
    }
    refreshSynapseWeight(synapse, config);

    if (synapse.stableWeight !== stableBefore) {
      events.push({
        synapseId: synapse.id,
        kind: "reward",
        deltaFast: 0,
        deltaStable: synapse.stableWeight - stableBefore
      });
    }
  }

  return events;
}

export function applySupervisedMotorLearning(
  synapses: Synapse[],
  neuronsById: Map<string, Neuron>,
  targetMotorId: string,
  activeMotorIds: Set<string>,
  modulator: number,
  config: ModelConfig
): LearningEvent[] {
  const events: LearningEvent[] = [];

  for (const synapse of synapses) {
    if (!isConductingSynapse(synapse)) {
      continue;
    }

    const pre = neuronsById.get(synapse.preNeuronId);
    const post = neuronsById.get(synapse.postNeuronId);

    if (!pre || !post || post.role !== "motor" || !isActiveSignal(pre.outputSignal)) {
      continue;
    }

    const branch = post.branches.find((candidate) => candidate.id === synapse.postBranchId);
    const plasticityGate = branch?.plasticityGate ?? 1;
    const isTarget = post.id === targetMotorId;
    const wasWronglyActive = !isTarget && activeMotorIds.has(post.id);
    const before = synapse.fastWeight;
    const delta = computeSupervisedFastDelta(isTarget, wasWronglyActive, plasticityGate, modulator, config);

    synapse.fastWeight = clampMagnitude(synapse.fastWeight + delta, 0, config.maxWeight);

    if (wasWronglyActive && synapse.state === "stable") {
      const stableBefore = synapse.stableWeight;
      const stableDelta = computeStableDepotentiationDelta(
        synapse.eligibilityTrace,
        plasticityGate,
        modulator,
        config
      );
      synapse.stableWeight = clampMagnitude(synapse.stableWeight + stableDelta, 0, config.maxWeight);
      if (synapse.stableWeight < config.stableThreshold) {
        synapse.state = "active";
      }
      if (synapse.stableWeight !== stableBefore) {
        events.push({
          synapseId: synapse.id,
          kind: "supervised",
          deltaFast: 0,
          deltaStable: synapse.stableWeight - stableBefore
        });
      }
    }

    synapse.recentContribution = ema(
      synapse.recentContribution,
      isTarget ? Math.abs(synapse.effectiveWeight) : -Math.abs(synapse.effectiveWeight),
      config.emaAlpha
    );
    refreshSynapseWeight(synapse, config);

    if (synapse.state === "candidate" && isTarget) {
      synapse.state = "active";
    }

    if (synapse.fastWeight !== before) {
      events.push({
        synapseId: synapse.id,
        kind: "supervised",
        deltaFast: synapse.fastWeight - before,
        deltaStable: 0
      });
    }
  }

  return events;
}

export function decayWeights(synapses: Synapse[], config: ModelConfig): LearningEvent[] {
  const events: LearningEvent[] = [];

  for (const synapse of synapses) {
    if (synapse.state === "pruned") {
      continue;
    }

    const beforeFast = synapse.fastWeight;
    const beforeStable = synapse.stableWeight;
    synapse.fastWeight *= config.fastDecay;
    // Structural sensory stems (decayProtected) carry the network's only
    // afferent signal and must not be passively eroded past the post-synaptic
    // threshold — doing so silently severs the downstream chain (the long-range
    // rewardOnly noop cliff). Learned stable weights still decay normally.
    if (!synapse.decayProtected) {
      synapse.stableWeight *= config.stableDecay;
    }
    refreshSynapseWeight(synapse, config);

    if (beforeFast !== synapse.fastWeight || beforeStable !== synapse.stableWeight) {
      events.push({
        synapseId: synapse.id,
        kind: "decay",
        deltaFast: synapse.fastWeight - beforeFast,
        deltaStable: synapse.stableWeight - beforeStable
      });
    }
  }

  return events;
}

export function captureStableWeights(synapses: Synapse[], config: ModelConfig): LearningEvent[] {
  const events: LearningEvent[] = [];

  for (const synapse of synapses) {
    if (!isConductingSynapse(synapse)) {
      continue;
    }

    if (
      synapse.recentUse < config.useThreshold ||
      synapse.recentContribution < config.contributionThreshold ||
      synapse.fastWeight <= 0
    ) {
      continue;
    }

    const beforeFast = synapse.fastWeight;
    const beforeStable = synapse.stableWeight;
    const captured = computeStableCaptureAmount(synapse.fastWeight, config);
    synapse.stableWeight = clampMagnitude(synapse.stableWeight + captured, 0, config.maxWeight);
    synapse.fastWeight = clampMagnitude(synapse.fastWeight - captured, 0, config.maxWeight);
    synapse.stabilityScore = ema(synapse.stabilityScore, 1, config.emaAlpha);

    if (synapse.stableWeight >= config.stableThreshold) {
      synapse.state = "stable";
    }

    refreshSynapseWeight(synapse, config);
    events.push({
      synapseId: synapse.id,
      kind: "capture",
      deltaFast: synapse.fastWeight - beforeFast,
      deltaStable: synapse.stableWeight - beforeStable
    });
  }

  return events;
}
