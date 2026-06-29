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
  supervisedLearningRate: number;
  stableCaptureRate: number;
  stableThreshold: number;
  useThreshold: number;
  contributionThreshold: number;
  fastDecay: number;
  stableDecay: number;

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
  supervisedLearningRate: 0.08,
  stableCaptureRate: 0.02,
  stableThreshold: 0.25,
  useThreshold: 0.2,
  contributionThreshold: 0.15,
  fastDecay: 0.999,
  stableDecay: 0.99999,

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
