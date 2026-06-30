export interface ModelConfig {
  dt: number;
  emaAlpha: number;

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

  fastWeightInit: number;
  stableWeightInit: number;
  maxWeight: number;
  weakWeightThreshold: number;
  eligibilityDecay: number;
  traceDecay: number;
  fastLearningRate: number;
  rewardAdvantageBaselineAlpha: number;
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
  fastDecay: number;
  stableDecay: number;
  depotentiationRate: number;
  negativeThreshold: number;

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
