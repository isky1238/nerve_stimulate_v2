import { ModelConfig } from "../config/newModelConfig";

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

export function shouldApplyAversiveStableDepotentiation(
  aversiveTag: AversiveLearningTag | undefined,
  config: ModelConfig
): boolean {
  return Boolean(
    aversiveTag?.present &&
    aversiveTag.badOutcome &&
    config.aversiveDepotentiationRate > 0 &&
    (config.aversiveTagStrategy === "badOutcomeDepotentiation" ||
      config.aversiveTagStrategy === "combined")
  );
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

export function computeAversiveStableDepotentiationDelta(
  eligibilityTrace: number,
  plasticityGate: number,
  aversiveIntensity: number,
  config: ModelConfig
): number {
  return -config.aversiveDepotentiationRate * Math.abs(eligibilityTrace) * plasticityGate * aversiveIntensity;
}

export function computeStableCaptureAmount(fastWeight: number, config: ModelConfig): number {
  return fastWeight * config.stableCaptureRate;
}
