import { ModelConfig } from "../config/newModelConfig";
import { Neuron } from "./neuron";
import { Synapse } from "./synapse";

export interface AversiveLearningTag {
  present: boolean;
  badOutcome: boolean;
  goodAvoidance: boolean;
  intensity: number;
}

export interface StdpEligibilityInput {
  preTrace: number;
  postTrace: number;
  preActive: number;
  postActive: number;
  effectSign: number;
  effectiveWeight: number;
}

export interface StdpEligibilityDelta {
  ltpEligibility: number;
  ltdEligibility: number;
  eligibilityDelta: number;
}

export function nextActivityTrace(currentTrace: number, active: number, traceDecay: number): number {
  const traceAlpha = 1 - traceDecay;
  return currentTrace * traceDecay + active * traceAlpha;
}

export function bapContributionWeight(effectSign: number, effectiveWeight: number): number {
  return effectSign * Math.abs(effectiveWeight);
}

export function computeStdpEligibilityDelta(
  input: StdpEligibilityInput,
  config: ModelConfig
): StdpEligibilityDelta {
  const bapWeight = bapContributionWeight(input.effectSign, input.effectiveWeight);
  const ltpEligibility =
    config.stdpLtpRate * input.preTrace * input.postActive * input.preActive * bapWeight;
  const ltdEligibility = config.stdpLtdRate * input.postTrace * input.preActive * bapWeight;

  return {
    ltpEligibility,
    ltdEligibility,
    eligibilityDelta: ltpEligibility - ltdEligibility
  };
}

export function nextEligibilityTrace(
  currentEligibility: number,
  eligibilityDelta: number,
  config: ModelConfig
): number {
  return currentEligibility * config.eligibilityDecay + eligibilityDelta;
}

export function positiveEligibilityScale(positiveEligibilitySum: number): number {
  return positiveEligibilitySum > 1e-12 ? 1 / positiveEligibilitySum : 1;
}

export function computeRewardModulator(rewardAdvantage: number, config: ModelConfig): number {
  return Math.tanh(Math.abs(rewardAdvantage) * config.modulatorGain);
}

export function computeAversiveRewardSignal(
  rewardSignal: number,
  aversiveTag: AversiveLearningTag | undefined,
  config: ModelConfig
): number {
  if (
    !aversiveTag?.present ||
    !aversiveTag.goodAvoidance ||
    (config.aversiveTagStrategy !== "avoidanceMarker" && config.aversiveTagStrategy !== "combined")
  ) {
    return rewardSignal;
  }

  return rewardSignal + config.aversiveAvoidanceBonus * aversiveTag.intensity;
}

export function computeAversiveModulator(
  baseModulator: number,
  aversiveTag: AversiveLearningTag | undefined,
  config: ModelConfig
): number {
  if (
    !aversiveTag?.present ||
    config.aversiveTagStrategy !== "modulatorOnly" ||
    config.aversiveTagGain <= 0
  ) {
    return baseModulator;
  }

  return Math.min(1, baseModulator + Math.tanh(aversiveTag.intensity * config.aversiveTagGain));
}

/**
 * Whether a synapse should undergo tagged-impulse depotentiation (the
 * flip-accumulation-direction mechanism that replaces the old reverse-term
 * B channel). Conditions:
 *  - mode != "off"
 *  - synapse is on the tagged active path (synapse.tagLoad > 0)
 *  - post is a motor neuron (readout only — protects sensory->inter stems;
 *    decayProtected stems are also excluded as belt-and-suspenders)
 *  - variant 1 (specificFactor): AND the global aversive load exceeds the
 *    sensitization threshold (the specific-factor hormone gate)
 *
 * Pure compute; the actual stable erode + fast reverse-migration is committed
 * in captureStableWeights using computeTaggedCaptureAmount.
 */
export function isTaggedDepotentiationActive(
  synapse: Synapse,
  neuronsById: Map<string, Neuron>,
  globalAversiveLoad: number,
  config: ModelConfig
): boolean {
  if (config.taggedDepotentiationMode === "off") {
    return false;
  }
  if (synapse.tagLoad <= 0) {
    return false;
  }
  if (synapse.decayProtected) {
    return false;
  }
  const post = neuronsById.get(synapse.postNeuronId);
  if (!post || post.role !== "motor") {
    return false;
  }
  if (config.taggedDepotentiationMode === "specificFactor") {
    return globalAversiveLoad > config.globalSensitizationThreshold;
  }
  // taggedImpulse: tag reaching the readout alone flips capture.
  return true;
}

/**
 * Amount of stable weight to erode (and reverse-migrate to fast) when a
 * tagged impulse flips the capture direction. Based on stableWeight (the
 * consolidation being de-consolidated), unlike normal capture which is based
 * on fastWeight. Gain scales the rate relative to stableCaptureRate.
 */
export function computeTaggedCaptureAmount(stableWeight: number, config: ModelConfig): number {
  return stableWeight * config.stableCaptureRate * config.taggedCaptureGain;
}

export function computeRewardFastDelta(
  rewardSignal: number,
  eligibilityTrace: number,
  plasticityGate: number,
  modulator: number,
  config: ModelConfig
): number {
  return config.fastLearningRate * rewardSignal * eligibilityTrace * plasticityGate * modulator;
}

export function computeSupervisedFastDelta(
  isTarget: boolean,
  wasWronglyActive: boolean,
  plasticityGate: number,
  modulator: number,
  config: ModelConfig
): number {
  const direction = isTarget ? 1 : -(wasWronglyActive ? 1 : 0.7);
  return config.supervisedLearningRate * direction * plasticityGate * modulator;
}

export function computeStableDepotentiationDelta(
  eligibilityTrace: number,
  plasticityGate: number,
  modulator: number,
  config: ModelConfig
): number {
  return -config.depotentiationRate * Math.abs(eligibilityTrace) * plasticityGate * modulator;
}

export function computeStableCaptureAmount(fastWeight: number, config: ModelConfig): number {
  return fastWeight * config.stableCaptureRate;
}
