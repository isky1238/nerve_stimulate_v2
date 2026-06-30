export type AversiveTagStrategy =
  | "off"
  | "modulatorOnly"
  | "avoidanceMarker"
  | "badOutcomeDepotentiation"
  | "combined";

export interface ModelConfig {
  // Experiment clocking / smoothing defaults.
  dt: number;
  emaAlpha: number;

  // Thresholds and gate bounds.
  leak: number;
  branchLocalThreshold: number;
  dendriteGateThreshold: number;
  axonThreshold: number;
  refractorySteps: number;
  thresholdMin: number;
  thresholdMax: number;
  thresholdAdaptRate: number;
  targetSpikeRate: number;

  inhibitionShuntScale: number;
  inhibitionFreezeThreshold: number;

  // Weight bounds, traces, plasticity rates, and mechanism toggles.
  fastWeightInit: number;
  stableWeightInit: number;
  maxWeight: number;
  weakWeightThreshold: number;
  eligibilityDecay: number;
  traceDecay: number;
  fastLearningRate: number;
  rewardAdvantageBaselineAlpha: number;
  /**
   * Relative LTP/LTD factors for the BAP-weighted STDP eligibility rule.
   * The actual formula lives in core/plasticityMechanisms.ts.
   */
  stdpLtpRate: number;
  stdpLtdRate: number;
  /**
   * RewardOnly exploration policy selector. Historical experiment notes belong in
   * EVAL_TODO.md; this field only selects the mechanism.
   */
  explorationStrategy: "conflictGated" | "epsilonGreedy";
  explorationEpsilon: number;
  supervisedLearningRate: number;
  stableCaptureRate: number;
  stableThreshold: number;
  useThreshold: number;
  contributionThreshold: number;
  /**
   * Mechanism toggle for positive-eligibility normalization.
   */
  eligibilityNormalization: boolean;
  /**
   * Gain for the reward-derived plasticity modulator.
   */
  modulatorGain: number;
  aversiveTagStrategy: AversiveTagStrategy;
  aversiveTagGain: number;
  aversiveAvoidanceBonus: number;
  fastDecay: number;
  stableDecay: number;
  depotentiationRate: number;
  aversiveDepotentiationRate: number;
  negativeThreshold: number;
  aversiveBadOutcomeThreshold: number;

  // Structural growth / topology limits.
  // Synapse.decayProtected is also a structural property. It is intentionally
  // stored on fixed topology edges rather than in ModelConfig.
  connectionDistanceLambda: number;
  connectionThreshold: number;
  candidateMaxAge: number;
  minConnectionAge: number;
  dormantLimit: number;
  baseCooldown: number;

  sensoryMaxInputs: number;
  sensoryMaxOutputs: number;
  interneuronMaxInputs: number;
  interneuronMaxOutputs: number;
  motorMaxInputs: number;
  motorMaxOutputs: number;
}

export type ConfigCategory =
  | "structural"
  | "thresholds"
  | "neuralDynamics"
  | "weightBounds"
  | "plasticityTimescales"
  | "plasticityRates"
  | "plasticityMechanisms"
  | "signalModulation"
  | "exploration"
  | "experimentDefaults"
  | "auditThresholds";

export const configFieldGroups = Object.freeze({
  structural: [
    "connectionDistanceLambda",
    "connectionThreshold",
    "candidateMaxAge",
    "minConnectionAge",
    "dormantLimit",
    "baseCooldown",
    "sensoryMaxInputs",
    "sensoryMaxOutputs",
    "interneuronMaxInputs",
    "interneuronMaxOutputs",
    "motorMaxInputs",
    "motorMaxOutputs"
  ],
  thresholds: [
    "branchLocalThreshold",
    "dendriteGateThreshold",
    "axonThreshold",
    "thresholdMin",
    "thresholdMax",
    "inhibitionFreezeThreshold",
    "weakWeightThreshold",
    "stableThreshold",
    "useThreshold",
    "contributionThreshold",
    "negativeThreshold",
    "aversiveBadOutcomeThreshold"
  ],
  neuralDynamics: [
    "leak",
    "refractorySteps",
    "thresholdAdaptRate",
    "targetSpikeRate",
    "inhibitionShuntScale"
  ],
  weightBounds: [
    "fastWeightInit",
    "stableWeightInit",
    "maxWeight"
  ],
  plasticityTimescales: [
    "eligibilityDecay",
    "traceDecay",
    "fastDecay",
    "stableDecay"
  ],
  plasticityRates: [
    "fastLearningRate",
    "supervisedLearningRate",
    "stableCaptureRate",
    "depotentiationRate",
    "aversiveDepotentiationRate",
    "stdpLtpRate",
    "stdpLtdRate"
  ],
  plasticityMechanisms: [
    "eligibilityNormalization",
    "aversiveTagStrategy"
  ],
  signalModulation: [
    "rewardAdvantageBaselineAlpha",
    "modulatorGain",
    "aversiveTagGain",
    "aversiveAvoidanceBonus"
  ],
  exploration: ["explorationStrategy", "explorationEpsilon"],
  experimentDefaults: ["dt", "emaAlpha"],
  auditThresholds: []
}) satisfies Readonly<Record<ConfigCategory, readonly (keyof ModelConfig)[]>>;

export const defaultConfig: ModelConfig = Object.freeze({
  dt: 0.01,
  emaAlpha: 0.1,

  leak: 0.92,
  branchLocalThreshold: 0.4,
  dendriteGateThreshold: 0.4,
  axonThreshold: 1.0,
  refractorySteps: 2,
  thresholdMin: 0.7,
  thresholdMax: 1.5,
  thresholdAdaptRate: 0.0005,
  targetSpikeRate: 0.05,

  inhibitionShuntScale: 1.0,
  inhibitionFreezeThreshold: 0.8,

  fastWeightInit: 0.12,
  stableWeightInit: 0,
  maxWeight: 2.0,
  weakWeightThreshold: 0.05,
  eligibilityDecay: 0.9,
  traceDecay: 0.85,
  fastLearningRate: 0.01,
  rewardAdvantageBaselineAlpha: 0.1,
  stdpLtpRate: 1.0,
  stdpLtdRate: 1.0,
  eligibilityNormalization: true,
  modulatorGain: 1.0,
  aversiveTagStrategy: "off",
  aversiveTagGain: 0,
  aversiveAvoidanceBonus: 0,
  explorationStrategy: "conflictGated",
  explorationEpsilon: 0.2,
  supervisedLearningRate: 0.08,
  stableCaptureRate: 0.02,
  stableThreshold: 0.25,
  useThreshold: 0.2,
  contributionThreshold: 0.15,
  fastDecay: 0.999,
  stableDecay: 0.99999,
  depotentiationRate: 0.02,
  aversiveDepotentiationRate: 0,
  negativeThreshold: -0.15,
  aversiveBadOutcomeThreshold: 0,

  connectionDistanceLambda: 8,
  connectionThreshold: 0.25,
  candidateMaxAge: 1000,
  minConnectionAge: 20,
  dormantLimit: 3000,
  baseCooldown: 500,

  sensoryMaxInputs: 0,
  sensoryMaxOutputs: 3,
  interneuronMaxInputs: 4,
  interneuronMaxOutputs: 4,
  motorMaxInputs: 6,
  motorMaxOutputs: 0
});

export function withConfig(overrides: Partial<ModelConfig>): ModelConfig {
  return {
    ...defaultConfig,
    ...overrides
  };
}
