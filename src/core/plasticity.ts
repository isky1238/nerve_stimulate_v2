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
    synapse.preTrace = synapse.preTrace * config.traceDecay + preActive;
    synapse.postTrace = synapse.postTrace * config.traceDecay + postActive;

    const coactivity = preActive * postActive;
    synapse.eligibilityTrace =
      synapse.eligibilityTrace * config.eligibilityDecay + coactivity * Math.sign(synapse.effectiveWeight || 1);
    synapse.recentContribution = ema(
      synapse.recentContribution,
      coactivity > 0 ? Math.abs(synapse.effectiveWeight) : 0,
      config.emaAlpha
    );
  }
}

export function applyRewardLearning(
  synapses: Synapse[],
  neuronsById: Map<string, Neuron>,
  rewardSignal: number,
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
    const delta = config.fastLearningRate * rewardSignal * synapse.eligibilityTrace * plasticityGate;
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
    const delta = isTarget
      ? config.supervisedLearningRate * plasticityGate
      : -config.supervisedLearningRate * (wasWronglyActive ? 1 : 0.7) * plasticityGate;

    synapse.fastWeight = clampMagnitude(synapse.fastWeight + delta, 0, config.maxWeight);

    if (wasWronglyActive && synapse.state === "stable" && synapse.eligibilityTrace > 0) {
      const stableBefore = synapse.stableWeight;
      const stableDelta = -config.depotentiationRate * synapse.eligibilityTrace * plasticityGate;
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
