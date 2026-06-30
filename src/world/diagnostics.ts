import { ModelConfig } from "../config/newModelConfig";
import type { LearningNetwork } from "../core/evaluation";
import type { ChallengeEpisodeTrace } from "./challenge2d";

const INTERNEURON_IDS = ["iFoodLeft", "iFoodRight", "iToxinLeft", "iToxinRight"];
const CORRECT_MOTOR_FOR_INTER: Record<string, string> = {
  iFoodLeft: "leftMotor",
  iFoodRight: "rightMotor",
  iToxinLeft: "rightMotor",
  iToxinRight: "leftMotor"
};

export interface ActionRates {
  conflictRate: number;
  noopRate: number;
  totalSteps: number;
}

export interface MotorWeightSums {
  leftFastSum: number;
  rightFastSum: number;
  leftStableSum: number;
  rightStableSum: number;
  fastAsymmetry: number;
}

export interface CollapseEpochSample extends MotorWeightSums {
  epoch: number;
  trainConflictRate: number;
  trainNoopRate: number;
}

export interface WrongPriorSynapseDump {
  wrongDirectionStableCount: number;
  wrongDirectionMaxStableWeight: number;
  wrongDirectionMaxFastWeight: number;
  correctDirectionMaxFastWeight: number;
}

export interface LongRangeCliffIndicators {
  fastSum: number;
  stableSum: number;
  effectiveSum: number;
  noopStuck: boolean;
  partialSuccess: boolean;
  solved: boolean;
}

export function actionRatesFromEpisodes(episodes: ChallengeEpisodeTrace[]): ActionRates {
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
  return { conflictRate: conflict / denom, noopRate: noop / denom, totalSteps: total };
}

export function motorWeightSums(network: LearningNetwork): MotorWeightSums {
  let leftFastSum = 0;
  let rightFastSum = 0;
  let leftStableSum = 0;
  let rightStableSum = 0;

  for (const synapse of network.synapses) {
    if (!INTERNEURON_IDS.includes(synapse.preNeuronId)) {
      continue;
    }
    if (synapse.postNeuronId === "leftMotor") {
      leftFastSum += synapse.fastWeight;
      leftStableSum += synapse.stableWeight;
    } else if (synapse.postNeuronId === "rightMotor") {
      rightFastSum += synapse.fastWeight;
      rightStableSum += synapse.stableWeight;
    }
  }

  return {
    leftFastSum,
    rightFastSum,
    leftStableSum,
    rightStableSum,
    fastAsymmetry: Math.abs(leftFastSum - rightFastSum) / (leftFastSum + rightFastSum + 1e-9)
  };
}

export function sampleCollapseEpoch(
  epoch: number,
  network: LearningNetwork,
  epochEpisodes: ChallengeEpisodeTrace[]
): CollapseEpochSample {
  const sums = motorWeightSums(network);
  const rates = actionRatesFromEpisodes(epochEpisodes);
  return {
    epoch,
    ...sums,
    trainConflictRate: rates.conflictRate,
    trainNoopRate: rates.noopRate
  };
}

export function dumpWrongPriorSynapseState(
  network: LearningNetwork,
  config: ModelConfig
): WrongPriorSynapseDump {
  const interToMotor = network.synapses.filter((synapse) => {
    const pre = network.neurons.find((neuron) => neuron.id === synapse.preNeuronId);
    const post = network.neurons.find((neuron) => neuron.id === synapse.postNeuronId);
    return pre?.role === "interneuron" && post?.role === "motor";
  });

  const lines: string[] = [];
  lines.push("");
  lines.push("=== wrong-prior synapse state dump (interneuron -> motor) ===");
  lines.push(`maxWeight=${config.maxWeight} stableThreshold=${config.stableThreshold} stableDecay=${config.stableDecay}`);
  lines.push(`supervisedLearningRate=${config.supervisedLearningRate} stableCaptureRate=${config.stableCaptureRate} fastDecay=${config.fastDecay}`);

  let wrongDirectionStableCount = 0;
  let wrongDirectionMaxStableWeight = 0;
  let wrongDirectionMaxFastWeight = 0;
  let correctDirectionMaxFastWeight = 0;

  for (const synapse of interToMotor) {
    const correctMotor = CORRECT_MOTOR_FOR_INTER[synapse.preNeuronId];
    const isWrongDirection = correctMotor !== undefined && synapse.postNeuronId !== correctMotor;
    const direction = isWrongDirection ? "WRONG" : "CORRECT";
    const stableCaptured = synapse.stableWeight >= config.stableThreshold;

    if (isWrongDirection) {
      if (stableCaptured) {
        wrongDirectionStableCount += 1;
      }
      wrongDirectionMaxStableWeight = Math.max(wrongDirectionMaxStableWeight, synapse.stableWeight);
      wrongDirectionMaxFastWeight = Math.max(wrongDirectionMaxFastWeight, synapse.fastWeight);
    } else {
      correctDirectionMaxFastWeight = Math.max(correctDirectionMaxFastWeight, synapse.fastWeight);
    }

    lines.push(
      `  ${synapse.preNeuronId}->${synapse.postNeuronId} [${direction}] ` +
        `fast=${synapse.fastWeight.toFixed(4)} stable=${synapse.stableWeight.toFixed(4)} ` +
        `eff=${synapse.effectiveWeight.toFixed(4)} state=${synapse.state} ` +
        `recentUse=${synapse.recentUse.toFixed(4)} recentContrib=${synapse.recentContribution.toFixed(4)} ` +
        `stabilityScore=${synapse.stabilityScore.toFixed(4)}${stableCaptured ? " STABLE-CAPTURED" : ""}`
    );
  }

  lines.push("--- summary ---");
  lines.push(`wrong-direction: stableCount=${wrongDirectionStableCount}/4 maxStable=${wrongDirectionMaxStableWeight.toFixed(4)} maxFast=${wrongDirectionMaxFastWeight.toFixed(4)}`);
  lines.push(`correct-direction: maxFast=${correctDirectionMaxFastWeight.toFixed(4)}`);
  lines.push(`dualLock=${wrongDirectionStableCount > 0} (if true, stableWeight drives wrong motor even after fastWeight unlearn)`);
  lines.push("=== end dump ===");

  if (process.env.DEBUG_WRONG_PRIOR_DUMP === "1") {
    process.stderr.write(lines.join("\n") + "\n");
  }

  return {
    wrongDirectionStableCount,
    wrongDirectionMaxStableWeight,
    wrongDirectionMaxFastWeight,
    correctDirectionMaxFastWeight
  };
}

export function longRangeCliffIndicators(params: {
  successRate: number;
  noopRate: number;
  motorWeights: MotorWeightSums;
}): LongRangeCliffIndicators {
  const fastSum = params.motorWeights.leftFastSum + params.motorWeights.rightFastSum;
  const stableSum = params.motorWeights.leftStableSum + params.motorWeights.rightStableSum;
  const effectiveSum = fastSum + stableSum;

  return {
    fastSum,
    stableSum,
    effectiveSum,
    noopStuck: params.noopRate >= 0.8,
    partialSuccess: params.successRate > 0 && params.successRate < 0.8,
    solved: params.successRate >= 0.8
  };
}
