export interface ModelConfig {
  // Experiment clocking / smoothing defaults.
  dt: number;
  emaAlpha: number;

  // Threshold and gate dynamics.
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

  // Learning dynamics and weight timescales.
  fastWeightInit: number;
  stableWeightInit: number;
  maxWeight: number;
  weakWeightThreshold: number;
  eligibilityDecay: number;
  traceDecay: number;
  fastLearningRate: number;
  rewardAdvantageBaselineAlpha: number;
  /**
   * STDP LTP/LTD relative-asymmetry factors for the BAP-weighted eligibility rule
   * (Phase 1 of the STDP/BAP baseline refactor). Eligibility is a SIGNED scalar:
   *   ltp = stdpLtpRate * preTrace * postActive * bapWeight   (pre-before-post → strengthen)
   *   ltd = stdpLtdRate * postTrace * preActive * bapWeight   (post-before-pre → weaken)
   *   eligibilityTrace = eligibilityTrace * eligibilityDecay + ltp - ltd
   * bapWeight = effectSign * |effectiveWeight| (keeps inhibitory sign, scales by
   * synaptic contribution). preTrace/postTrace are steady-state-normalized to [0,1]
   * so eligibility peaks at ~|effectiveWeight| (same order as the old ±1 coactivity
   * baseline), keeping downstream fastLearningRate scaling unchanged. These factors
   * default to 1.0 (symmetric LTP/LTD); tune the ratio for LTD>LTP asymmetry.
   * Coarse-grained: the time window is traceDecay-relative, NOT physical ms (deferred
   * to the unify-units pass).
   */
  stdpLtpRate: number;
  stdpLtdRate: number;
  /**
   * Exploration strategy for rewardOnly training.
   * - "conflictGated" (default): force a random motor only when the network fails
   *   to commit (noop/conflict). Masks noop from the learner during training, but
   *   stops forcing once the network commits — which turned out to be essential
   *   for rewardOnly convergence.
   * - "epsilonGreedy": with probability explorationEpsilon force a random motor;
   *   otherwise follow the network's own decision verbatim (noop stays noop, so
   *   the learner sees its own inaction and committed directions get truthful
   *   credit). NOTE: constant (non-annealed) ε-greedy was tried as C-tier step 1
   *   and REGRESSED rewardOnly to 100% noop collapse on every axis (challenge
   *   noopRate 0.857 -> 1.0, transfer pretrained noopRate 1). Kept as a toggle for
   *   future annealed/variant experiments; not the default.
   */
  explorationStrategy: "conflictGated" | "epsilonGreedy";
  explorationEpsilon: number;
  supervisedLearningRate: number;
  stableCaptureRate: number;
  stableThreshold: number;
  useThreshold: number;
  contributionThreshold: number;
  /**
   * Phase 2: per-post-spike divisive normalization of positive (LTP) eligibility.
   * When true, for each post that just spiked, its incoming synapses' positive
   * eligibility is scaled by 1/Σ(positive eligibility) so total LTP credit per
   * post-spike is bounded at 1.0. Directly counters the "normal-summation" risk
   * where many weak synapses each claim full credit for a co-driven firing.
   * LTD (negative eligibility) is NOT normalized — it is a distinct credit signal.
   */
  eligibilityNormalization: boolean;
  /**
   * Phase 2: gain for the reward-derived modulator. The modulator is a global
   * scalar modulating plasticityGate: modulator = tanh(|rewardAdvantage| * gain).
   * rewardOnly: advantage≈0 → modulator≈0 (no learning); large |advantage| → 1.
   * supervised: modulator is forced to 1 (explicit target signal, no modulation).
   * Composes with (does not replace) the inhibition-freeze binary gate.
   */
  modulatorGain: number;
  fastDecay: number;
  stableDecay: number;
  depotentiationRate: number;
  negativeThreshold: number;

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
  | "learningDynamics"
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
    "negativeThreshold"
  ],
  learningDynamics: [
    "leak",
    "refractorySteps",
    "thresholdAdaptRate",
    "targetSpikeRate",
    "inhibitionShuntScale",
    "fastWeightInit",
    "stableWeightInit",
    "maxWeight",
    "eligibilityDecay",
    "traceDecay",
    "fastLearningRate",
    "rewardAdvantageBaselineAlpha",
    "stdpLtpRate",
    "stdpLtdRate",
    "eligibilityNormalization",
    "modulatorGain",
    "supervisedLearningRate",
    "stableCaptureRate",
    "fastDecay",
    "stableDecay",
    "depotentiationRate"
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
  negativeThreshold: -0.15,

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
