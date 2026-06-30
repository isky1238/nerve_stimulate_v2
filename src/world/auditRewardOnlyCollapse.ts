import { defaultConfig, ModelConfig } from "../config/newModelConfig";
import { LearningNetwork } from "../core/evaluation";
import {
  ChallengeEpisodeTrace,
  ChallengeExperimentResult,
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

const COLLAPSE_EPOCHS = 40;
const COLLAPSE_SEED = 101;
const CHALLENGE_COLLAPSE_SEED = 21;

const INTERNEURON_IDS = ["iFoodLeft", "iFoodRight", "iToxinLeft", "iToxinRight"];
const MOTOR_IDS = ["leftMotor", "rightMotor"] as const;

interface EpochSample {
  epoch: number;
  leftFastSum: number;
  rightFastSum: number;
  asymmetry: number;
  trainConflictRate: number;
  trainNoopRate: number;
}

interface MotorFastSums {
  leftFastSum: number;
  rightFastSum: number;
}

function computeMotorFastSums(network: LearningNetwork): MotorFastSums {
  let leftFastSum = 0;
  let rightFastSum = 0;
  for (const synapse of network.synapses) {
    if (!INTERNEURON_IDS.includes(synapse.preNeuronId)) {
      continue;
    }
    if (synapse.postNeuronId === "leftMotor") {
      leftFastSum += synapse.fastWeight;
    } else if (synapse.postNeuronId === "rightMotor") {
      rightFastSum += synapse.fastWeight;
    }
  }
  return { leftFastSum, rightFastSum };
}

function trainRatesFromEpisodes(episodes: ChallengeEpisodeTrace[]): { conflictRate: number; noopRate: number } {
  let conflict = 0;
  let noop = 0;
  let total = 0;
  for (const episode of episodes) {
    for (const step of episode.steps) {
      total += 1;
      if (step.executedAction === "conflict") {
        conflict += 1;
      } else if (step.executedAction === "noop") {
        noop += 1;
      }
    }
  }
  const denom = Math.max(1, total);
  return { conflictRate: conflict / denom, noopRate: noop / denom };
}

function sampleEpoch(epoch: number, network: LearningNetwork, epochEpisodes: ChallengeEpisodeTrace[]): EpochSample {
  const { leftFastSum, rightFastSum } = computeMotorFastSums(network);
  const asymmetry = Math.abs(leftFastSum - rightFastSum) / (leftFastSum + rightFastSum + 1e-9);
  const { conflictRate, noopRate } = trainRatesFromEpisodes(epochEpisodes);
  return { epoch, leftFastSum, rightFastSum, asymmetry, trainConflictRate: conflictRate, trainNoopRate: noopRate };
}

function fmt(value: number, digits = 3): string {
  return value.toFixed(digits);
}

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
  const samples: EpochSample[] = [];

  const familyA = runComplexExperiment(complexConfig, {
    seed: COLLAPSE_SEED,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: COLLAPSE_EPOCHS,
    learningMode: "rewardOnly",
    epochProbe: (epoch, network, epochEpisodes) => {
      samples.push(sampleEpoch(epoch, network, epochEpisodes));
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

  return formatCollapseReport(familyA, composite, samples);
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
  const samples: EpochSample[] = [];

  const result = runChallengeExperiment(challengeConfig, {
    seed: CHALLENGE_COLLAPSE_SEED,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: COLLAPSE_EPOCHS,
    learningMode: "rewardOnly",
    epochProbe: (epoch, network, epochEpisodes) => {
      samples.push(sampleEpoch(epoch, network, epochEpisodes));
    }
  });

  return formatChallengeCollapseReport(result, samples);
}

function formatChallengeCollapseReport(
  result: ChallengeExperimentResult,
  samples: EpochSample[]
): string {
  const lines: string[] = [];
  lines.push("Audit dg-snn-rewardonly-challenge-collapse-v0.1");
  lines.push(`seed=${CHALLENGE_COLLAPSE_SEED} epochs=${COLLAPSE_EPOCHS} learningMode=rewardOnly (2D-challenge Family A)`);
  lines.push("");
  lines.push("Per-epoch bilateral fastWeight symmetry + training conflict/noop:");
  lines.push("  epoch  leftFast  rightFast  asymmetry  trainConflict  trainNoop");
  for (const s of samples) {
    lines.push(
      `  ${String(s.epoch).padStart(5)}  ${fmt(s.leftFastSum).padStart(8)}  ${fmt(s.rightFastSum).padStart(9)}  ` +
        `${fmt(s.asymmetry).padStart(9)}  ${fmt(s.trainConflictRate).padStart(13)}  ${fmt(s.trainNoopRate).padStart(9)}`
    );
  }
  lines.push("");
  lines.push("Final eval (frozen):");
  lines.push(
    `  2D-challenge Family A:   SR=${fmt(result.successRate)}  conflict=${fmt(result.conflictRate)}  noop=${fmt(result.noopRate)}  meanReward=${fmt(result.meanReward)}`
  );
  lines.push("");
  const first = samples[0];
  const last = samples[samples.length - 1];
  if (first && last) {
    lines.push("Reading guide:");
    lines.push(
      `  fastWeight drift: left ${fmt(first.leftFastSum)} -> ${fmt(last.leftFastSum)}, ` +
        `right ${fmt(first.rightFastSum)} -> ${fmt(last.rightFastSum)}, ` +
        `asymmetry ${fmt(first.asymmetry)} -> ${fmt(last.asymmetry)}`
    );
    lines.push("  - High trainNoopRate sustained (~0.85) with low asymmetry: bilateral pathways mutually cancel");
    lines.push("    below motor threshold -> noop. Advantage did NOT break this; commit-then-no-explore locks it in.");
    lines.push("  - If asymmetry grows and noop drops: advantage selected one side -> noop resolved.");
  }
  return lines.join("\n");
}

function formatCollapseReport(
  familyA: ChallengeExperimentResult,
  composite: ChallengeExperimentResult,
  samples: EpochSample[]
): string {
  const lines: string[] = [];
  lines.push("Audit dg-snn-rewardonly-collapse-v0.1");
  lines.push(`seed=${COLLAPSE_SEED} epochs=${COLLAPSE_EPOCHS} learningMode=rewardOnly (complex Family A pretrain)`);
  lines.push("");
  lines.push("Per-epoch bilateral fastWeight symmetry + training conflict/noop:");
  lines.push("  epoch  leftFast  rightFast  asymmetry  trainConflict  trainNoop");
  for (const s of samples) {
    lines.push(
      `  ${String(s.epoch).padStart(5)}  ${fmt(s.leftFastSum).padStart(8)}  ${fmt(s.rightFastSum).padStart(9)}  ` +
        `${fmt(s.asymmetry).padStart(9)}  ${fmt(s.trainConflictRate).padStart(13)}  ${fmt(s.trainNoopRate).padStart(9)}`
    );
  }
  lines.push("");
  lines.push("Final eval (frozen):");
  lines.push(
    `  Family A   (4-pattern single-object):    SR=${fmt(familyA.successRate)}  conflict=${fmt(familyA.conflictRate)}  noop=${fmt(familyA.noopRate)}  meanReward=${fmt(familyA.meanReward)}`
  );
  lines.push(
    `  Multi-obj  (Families B/C/D composite):   SR=${fmt(composite.successRate)}  conflict=${fmt(composite.conflictRate)}  noop=${fmt(composite.noopRate)}  meanReward=${fmt(composite.meanReward)}`
  );
  lines.push("");
  const first = samples[0];
  const last = samples[samples.length - 1];
  if (first && last) {
    lines.push("Reading guide:");
    lines.push(
      `  fastWeight drift: left ${fmt(first.leftFastSum)} -> ${fmt(last.leftFastSum)}, ` +
        `right ${fmt(first.rightFastSum)} -> ${fmt(last.rightFastSum)}, ` +
        `asymmetry ${fmt(first.asymmetry)} -> ${fmt(last.asymmetry)}`
    );
    lines.push("  - If leftFast/rightFast track together (low asymmetry) and Family A conflict=0: advantage broke");
    lines.push("    collapse via spike-count arbitration, NOT via asymmetric weights -> structural bilateral wiring persists.");
    lines.push("  - If asymmetry grows: advantage's fastWeight depotentiation selected one side -> bilateral co-enhancement broken.");
    lines.push("  - Multi-object conflict>0 with low asymmetry: residual conflict is compositional vote-tie, not bilateral co-enhancement.");
  }
  return lines.join("\n");
}
