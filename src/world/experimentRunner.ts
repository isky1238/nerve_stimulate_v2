import { ModelConfig } from "../config/newModelConfig";
import { createOfflineLearningNetwork, LearningNetwork } from "../core/evaluation";
import type {
  ChallengeEpisodePhase,
  ChallengeEpisodeTrace,
  ChallengeExperimentOptions,
  ChallengeExperimentResult,
  ChallengeLearningMode,
  RewardAdvantageState
} from "./challenge2d";
import { ChallengeScenario, shuffleScenarios } from "./challengeTask";

interface EpisodeRunnerOptions {
  phase: ChallengeEpisodePhase;
  learningMode: ChallengeLearningMode;
  learningEnabled: boolean;
  seed: number;
  observationDropout: number;
  reverseMapping: boolean;
  rewardAdvantageState?: RewardAdvantageState;
}

interface ExperimentRunnerSpec {
  traceVersion: string;
  width: number;
  height: number;
  defaultMaxSteps: number;
  useReverseMapping: boolean;
  createScenarios: (seeds: number[], maxSteps: number) => ChallengeScenario[];
  runEpisode: (
    network: LearningNetwork,
    scenario: ChallengeScenario,
    config: ModelConfig,
    options: EpisodeRunnerOptions
  ) => ChallengeEpisodeTrace;
}

export function runExperimentWithRunner(
  config: ModelConfig,
  options: ChallengeExperimentOptions,
  spec: ExperimentRunnerSpec
): ChallengeExperimentResult {
  const maxSteps = options.maxSteps ?? spec.defaultMaxSteps;
  const observationDropout = options.observationDropout ?? 0;
  const reverseMapping = spec.useReverseMapping ? options.reverseMapping ?? false : false;
  const trainingScenarios = options.trainingScenarios ?? spec.createScenarios(options.trainSeeds, maxSteps);
  const evaluationScenarios = options.evaluationScenarios ?? spec.createScenarios(options.evalSeeds, maxSteps);
  const network = options.initialNetwork ?? createOfflineLearningNetwork(config);
  const episodes: ChallengeEpisodeTrace[] = [];
  let rewardUpdateCount = 0;
  let supervisedUpdateCount = 0;
  let captureUpdateCount = 0;
  let decayUpdateCount = 0;
  const rewardAdvantageState: RewardAdvantageState = { baseline: 0 };

  for (let epoch = 0; epoch < options.epochs; epoch += 1) {
    const epochScenarios = shuffleScenarios(trainingScenarios, options.seed + epoch);
    const epochEpisodes: ChallengeEpisodeTrace[] = [];

    for (const scenario of epochScenarios) {
      const episode = spec.runEpisode(network, scenario, config, {
        phase: "train",
        learningMode: options.learningMode,
        learningEnabled: options.learningMode !== "frozen",
        seed: options.seed + epoch * 1000 + scenario.seed,
        observationDropout,
        reverseMapping,
        rewardAdvantageState
      });
      const counts = countLearningEvents(episode);
      rewardUpdateCount += counts.rewardUpdateCount;
      supervisedUpdateCount += counts.supervisedUpdateCount;
      captureUpdateCount += counts.captureUpdateCount;
      decayUpdateCount += counts.decayUpdateCount;
      episodes.push(episode);
      epochEpisodes.push(episode);
    }

    if (options.epochProbe) {
      options.epochProbe(epoch, network, epochEpisodes);
    }
  }

  const evaluationEpisodes = evaluationScenarios.map((scenario, index) =>
    spec.runEpisode(network, scenario, config, {
      phase: "eval",
      learningMode: options.learningMode,
      learningEnabled: false,
      seed: options.seed * 100000 + index,
      observationDropout,
      reverseMapping: false
    })
  );
  episodes.push(...evaluationEpisodes);

  const evalSteps = evaluationEpisodes.flatMap((episode) => episode.steps);
  const successRate = evaluationEpisodes.filter((episode) => episode.success).length / Math.max(1, evaluationEpisodes.length);
  const meanReward =
    evaluationEpisodes.reduce((sum, episode) => sum + episode.totalReward, 0) / Math.max(1, evaluationEpisodes.length);
  const meanStepsToTerminal =
    evaluationEpisodes.reduce((sum, episode) => sum + episode.steps.length, 0) / Math.max(1, evaluationEpisodes.length);
  const conflictRate =
    evalSteps.filter((step) => step.executedAction === "conflict").length / Math.max(1, evalSteps.length);
  const noopRate = evalSteps.filter((step) => step.executedAction === "noop").length / Math.max(1, evalSteps.length);

  return {
    trace: {
      version: spec.traceVersion,
      seed: options.seed,
      trainSeeds: [...options.trainSeeds],
      evalSeeds: [...options.evalSeeds],
      config: {
        width: spec.width,
        height: spec.height,
        maxSteps,
        epochs: options.epochs,
        learningMode: options.learningMode,
        observationDropout,
        reverseMapping,
        rewardAdvantageBaselineAlpha: config.rewardAdvantageBaselineAlpha,
        explorationStrategy: config.explorationStrategy,
        explorationEpsilon: config.explorationEpsilon
      },
      episodes
    },
    network,
    successRate,
    meanReward,
    meanStepsToTerminal,
    conflictRate,
    noopRate,
    rewardUpdateCount,
    supervisedUpdateCount,
    captureUpdateCount,
    decayUpdateCount
  };
}

export function countLearningEvents(episode: ChallengeEpisodeTrace): {
  rewardUpdateCount: number;
  supervisedUpdateCount: number;
  captureUpdateCount: number;
  decayUpdateCount: number;
} {
  return episode.steps.reduce(
    (counts, step) => ({
      rewardUpdateCount: counts.rewardUpdateCount + step.learning.rewardUpdates,
      supervisedUpdateCount: counts.supervisedUpdateCount + step.learning.supervisedUpdates,
      captureUpdateCount: counts.captureUpdateCount + step.learning.captureUpdates,
      decayUpdateCount: counts.decayUpdateCount + step.learning.decayUpdates
    }),
    {
      rewardUpdateCount: 0,
      supervisedUpdateCount: 0,
      captureUpdateCount: 0,
      decayUpdateCount: 0
    }
  );
}
