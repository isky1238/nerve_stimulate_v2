import { ModelConfig } from "../config/newModelConfig";
import { Neuron } from "./neuron";
import { clampMagnitude, ema, isActiveSignal } from "./signal";
import { Synapse, isConductingSynapse, refreshSynapseWeight } from "./synapse";

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
    // preTrace / postTrace are decaying accumulators of recent pre / post activity.
    // They were previously computed but unused (dead code). They now carry the STDP
    // time window: at the moment of a post spike, preTrace holds the memory of
    // recent pre activity (pre-before-post → LTP); at a pre spike, postTrace holds
    // the memory of recent post activity (post-before-pre → LTD).
    // Steady-state normalized: a constantly-firing neuron's trace saturates at
    // 1/(1-traceDecay); we divide by that so the trace expresses "recent activity
    // rate" in [0,1] regardless of traceDecay. Without this, a motor that fires
    // every step saturates postTrace to ~6.7 (traceDecay=0.85) and the LTD term
    // dominates LTP — collapsing all learning.
    const traceSteadyState = 1 / (1 - config.traceDecay);
    synapse.preTrace = (synapse.preTrace * config.traceDecay + preActive) / traceSteadyState;
    synapse.postTrace = (synapse.postTrace * config.traceDecay + postActive) / traceSteadyState;

    // BAP-weighted STDP eligibility. bapWeight keeps the synapse's effectSign
    // (so inhibitory synapses keep negative credit) and scales by |effectiveWeight|
    // — the "how much did this synapse drive the post" contribution that pure binary
    // coactivity discarded. eligibilityTrace is now a SIGNED scalar
    // (positive = LTP bias, negative = LTD bias), peaking at ~|effectiveWeight|
    // (same order as the old coactivity*sign(eff)=±1 baseline) so downstream
    // fastLearningRate scaling is unchanged. stdpLtpRate/stdpLtdRate act as
    // LTP/LTD relative-asymmetry factors (default 1.0 each), NOT tiny rate constants.
    const bapWeight = synapse.effectSign * Math.abs(synapse.effectiveWeight);
    const ltpElig = config.stdpLtpRate * synapse.preTrace * postActive * bapWeight;
    const ltdElig = config.stdpLtdRate * synapse.postTrace * preActive * bapWeight;
    synapse.eligibilityTrace =
      synapse.eligibilityTrace * config.eligibilityDecay + ltpElig - ltdElig;

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
    const scale = 1 / ltpSum;
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
    // Phase 2: modulator (reward-derived intensity) composes with the binary
    // inhibition-freeze plasticityGate. rewardSignal (advantage) carries the sign;
    // signed eligibilityTrace carries the LTP/LTD credit. Their product gives the
    // four-quadrant update; modulator scales how strongly this step learns at all.
    const delta = config.fastLearningRate * rewardSignal * synapse.eligibilityTrace * plasticityGate * modulator;
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
    // modulator is 1 for supervised (explicit target), kept in the formula for
    // symmetry with rewardOnly and future reward-driven supervised variants.
    const delta = isTarget
      ? config.supervisedLearningRate * plasticityGate * modulator
      : -config.supervisedLearningRate * (wasWronglyActive ? 1 : 0.7) * plasticityGate * modulator;

    synapse.fastWeight = clampMagnitude(synapse.fastWeight + delta, 0, config.maxWeight);

    if (wasWronglyActive && synapse.state === "stable") {
      // Stable depotentiation: trigger on wasWronglyActive (the motor fired when it
      // shouldn't have) — pre activity is already guaranteed by the isActiveSignal
      // gate above, so this synapse contributed to the wrong firing. Under STDP the
      // eligibilityTrace is signed (some wrong synapses go negative via post-before-pre
      // LTD); the depotentiation magnitude uses |eligibilityTrace| so the sign cannot
      // shield a wrong-direction stable synapse from depotentiation.
      const stableBefore = synapse.stableWeight;
      const stableDelta = -config.depotentiationRate * Math.abs(synapse.eligibilityTrace) * plasticityGate * modulator;
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
    const captured = synapse.fastWeight * config.stableCaptureRate;
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
