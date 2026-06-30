import { defaultConfig, ModelConfig } from "../config/newModelConfig";
import {
  createChallengeConfig,
  DEFAULT_EVAL_SEEDS,
  DEFAULT_TRAIN_SEEDS,
  runChallengeExperiment
} from "./challenge2d";
import {
  compositeSameDirectionScenarios,
  createComplexConfig,
  distractorScenarios,
  priorityScenarios,
  runComplexExperiment
} from "./complex2d";
import { CollapseEpochSample, sampleCollapseEpoch } from "./diagnostics";
import {
  formatRewardOnlyChallengeCollapseReport,
  formatRewardOnlyComplexCollapseReport
} from "./rewardOnlyCollapseReport";

const COLLAPSE_EPOCHS = 40;
const COLLAPSE_SEED = 101;
const CHALLENGE_COLLAPSE_SEED = 21;

/**
 * RewardOnly collapse diagnostic. Trains a rewardOnly network on complex Family A
 * for COLLAPSE_EPOCHS epochs and samples, per epoch, the bilateral motor fastWeight
 * symmetry (leftFastSum / rightFastSum / asymmetry) and the training-step conflict /
 * noop rates. After training it evaluates the same network frozen on Family A
 * (expected ~1.0/0/0 — advantage broke the collapse) and on multi-object composite
 * scenarios (Families B/C/D, expected ~0.5/0.3125 — residual conflict).
 *
 * The contrast answers whether advantage's fastWeight depotentiation broke bilateral
 * co-enhancement for Family A, and whether the residual multi-object conflict is
 * still a bilateral-symmetry artifact.
 */
export function runRewardOnlyCollapseAudit(config: ModelConfig = defaultConfig): string {
  const complexConfig = createComplexConfig(config);
  const samples: CollapseEpochSample[] = [];

  const familyA = runComplexExperiment(complexConfig, {
    seed: COLLAPSE_SEED,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: COLLAPSE_EPOCHS,
    learningMode: "rewardOnly",
    epochProbe: (epoch, network, epochEpisodes) => {
      samples.push(sampleCollapseEpoch(epoch, network, epochEpisodes));
    }
  });

  const composite = runComplexExperiment(complexConfig, {
    seed: COLLAPSE_SEED,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: 0,
    learningMode: "frozen",
    initialNetwork: familyA.network,
    evaluationScenarios: [
      ...compositeSameDirectionScenarios(),
      ...distractorScenarios(),
      ...priorityScenarios()
    ]
  });

  return formatRewardOnlyComplexCollapseReport(familyA, composite, samples);
}

/**
 * RewardOnly challenge-collapse diagnostic. Same probe as the complex version but
 * on 2D-challenge (binary, single-tick, maxSteps=12) where rewardOnly is known to
 * fail by NOOP (noopRate~0.857) rather than by conflict. Samples per-epoch bilateral
 * motor fastWeight symmetry + training conflict/noop rates.
 *
 * Contrast with the complex collapse audit isolates the failure-mode split: challenge
 * = bilateral mutual-cancellation -> noop, complex = bilateral co-firing -> conflict.
 */
export function runRewardOnlyChallengeCollapseAudit(config: ModelConfig = defaultConfig): string {
  const challengeConfig = createChallengeConfig(config);
  const samples: CollapseEpochSample[] = [];

  const result = runChallengeExperiment(challengeConfig, {
    seed: CHALLENGE_COLLAPSE_SEED,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: COLLAPSE_EPOCHS,
    learningMode: "rewardOnly",
    epochProbe: (epoch, network, epochEpisodes) => {
      samples.push(sampleCollapseEpoch(epoch, network, epochEpisodes));
    }
  });

  return formatRewardOnlyChallengeCollapseReport(result, samples);
}
