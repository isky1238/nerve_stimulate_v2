import { join } from "node:path";
import { ModelConfig } from "../config/newModelConfig";
import {
  ChallengeExperimentResult,
  ChallengeLearningMode,
  DEFAULT_EVAL_SEEDS,
  DEFAULT_TRAIN_SEEDS,
  createChallengeConfig,
  runChallengeExperiment
} from "../world/challenge2d";
import { createNetworkExport, NetworkExport, writeNetworkExport } from "./networkExport";

export interface ChallengePretrainExport {
  mode: Extract<ChallengeLearningMode, "supervised" | "rewardOnly">;
  filePath: string;
  snapshot: NetworkExport;
}

export interface ChallengePretrainExportOptions {
  seed?: number;
  epochs?: number;
  outputDir?: string;
}

const DEFAULT_PRETRAIN_SEED = 101;
const DEFAULT_PRETRAIN_EPOCHS = 40;

export function createChallengePretrainExports(
  config: ModelConfig,
  options: ChallengePretrainExportOptions = {}
): ChallengePretrainExport[] {
  const seed = options.seed ?? DEFAULT_PRETRAIN_SEED;
  const epochs = options.epochs ?? DEFAULT_PRETRAIN_EPOCHS;
  const outputDir = options.outputDir ?? join("exports", "pretrained");
  const challengeConfig = createChallengeConfig(config);

  return (["supervised", "rewardOnly"] as const).map((mode) => {
    const result = runChallengeExperiment(challengeConfig, {
      seed,
      trainSeeds: DEFAULT_TRAIN_SEEDS,
      evalSeeds: DEFAULT_EVAL_SEEDS,
      epochs,
      learningMode: mode
    });
    const snapshot = createNetworkExport({
      seed,
      config: challengeConfig,
      neurons: result.network.neurons,
      synapses: result.network.synapses,
      pairMemory: result.network.pairMemory,
      metrics: challengeMetrics(mode, result),
      events: [challengePretrainMetadata(mode, result)]
    });

    return {
      mode,
      filePath: join(outputDir, `2d-challenge-${mode}-pretrained.json`),
      snapshot
    };
  });
}

export function writeChallengePretrainExports(
  config: ModelConfig,
  options: ChallengePretrainExportOptions = {}
): ChallengePretrainExport[] {
  const exports = createChallengePretrainExports(config, options);

  for (const item of exports) {
    writeNetworkExport(item.filePath, item.snapshot);
  }

  return exports;
}

function challengeMetrics(
  mode: Extract<ChallengeLearningMode, "supervised" | "rewardOnly">,
  result: ChallengeExperimentResult
): Record<string, number | string | boolean> {
  return {
    pretrainKind: "2d-challenge",
    learningMode: mode,
    successRate: result.successRate,
    meanReward: result.meanReward,
    meanStepsToTerminal: result.meanStepsToTerminal,
    conflictRate: result.conflictRate,
    noopRate: result.noopRate,
    rewardUpdateCount: result.rewardUpdateCount,
    supervisedUpdateCount: result.supervisedUpdateCount,
    captureUpdateCount: result.captureUpdateCount,
    decayUpdateCount: result.decayUpdateCount,
    trainSeeds: DEFAULT_TRAIN_SEEDS.join(","),
    evalSeeds: DEFAULT_EVAL_SEEDS.join(","),
    epochs: result.trace.config.epochs,
    maxSteps: result.trace.config.maxSteps,
    observationDropout: result.trace.config.observationDropout,
    claimBoundary: "pretrained candidate only; not proof of real-world autonomous learning"
  };
}

function challengePretrainMetadata(
  mode: Extract<ChallengeLearningMode, "supervised" | "rewardOnly">,
  result: ChallengeExperimentResult
): Record<string, unknown> {
  const evaluationEpisodes = result.trace.episodes.filter((episode) => episode.phase === "eval");

  return {
    kind: "2d-challenge-pretrain-metadata",
    mode,
    traceVersion: result.trace.version,
    trainSeeds: result.trace.trainSeeds,
    evalSeeds: result.trace.evalSeeds,
    config: result.trace.config,
    metrics: {
      successRate: result.successRate,
      meanReward: result.meanReward,
      meanStepsToTerminal: result.meanStepsToTerminal,
      conflictRate: result.conflictRate,
      noopRate: result.noopRate,
      rewardUpdateCount: result.rewardUpdateCount,
      supervisedUpdateCount: result.supervisedUpdateCount
    },
    evaluationSummary: evaluationEpisodes.map((episode) => ({
      scenarioId: episode.scenarioId,
      totalReward: episode.totalReward,
      success: episode.success,
      terminalReason: episode.terminalReason,
      steps: episode.steps.length
    })),
    intendedUse: [
      "future real-world transfer attempt",
      "pre-learning baseline comparison",
      "regression snapshot for trained synapse weights"
    ],
    boundary:
      mode === "rewardOnly"
        ? "Reward-only used deterministic exploration in the controlled challenge."
        : "Supervised pretrain is an upper-bound baseline with target-motor feedback."
  };
}
