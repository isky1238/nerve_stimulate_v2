import { ModelConfig } from "../config/newModelConfig";
import { ActionDecision, WorldAction } from "../core/arbitration";
import type { LearningNetwork } from "../core/evaluation";
import {
  applyMaintenanceDecayAndCapture,
  applyRewardOutcomeLearning,
  applySupervisedMotorOutcomeLearning,
  forceExplorationMotor,
  selectExplorationAction,
  updateNetworkEligibility
} from "../core/mechanism";
import { indexNeurons, integrateNeuron, resetBranchInputs, resetNeuronRuntime, setSensoryOutput } from "../core/neuron";
import { SeededRandom } from "../core/random";
import { propagateSynapses } from "../core/synapse";
import { WorldState } from "./world2d";
import {
  CHALLENGE_HEIGHT,
  CHALLENGE_WIDTH,
  ChallengeEpisodePhase,
  ChallengeEpisodeTrace,
  ChallengeExperimentOptions,
  ChallengeExperimentResult,
  ChallengeLearningMode,
  ChallengeComplexEvidence,
  ChallengeRawObservation,
  RewardAdvantageState,
  ChallengeScenario,
  ChallengeTerminalReason,
  ChallengeTraceStep,
  createChallengeConfig,
  createChallengeScenarios,
  createChallengeWorldState,
  observeChallengeWorld,
  scoreChallengeStep,
  stepChallengeWorld,
  updateRewardAdvantageBaseline
} from "./challenge2d";
import { runExperimentWithRunner } from "./experimentRunner";
import {
  arbitrateComplexMotorAction,
  blankComplexScenario,
  COMPLEX_SPIKE_TICKS,
  ComplexSensoryInput,
  compositeSameDirectionScenarios,
  DEFAULT_COMPLEX_MAX_STEPS,
  distractorScenarios,
  expectedActionForComplexState,
  mapComplexObservationToSensors,
  priorityScenarios,
  semanticConflictScenarios,
  trueConflictScenarios
} from "./complexTask";

export {
  arbitrateComplexMotorAction,
  blankComplexScenario,
  COMPLEX_SPIKE_TICKS,
  compositeSameDirectionScenarios,
  DEFAULT_COMPLEX_MAX_STEPS,
  distractorScenarios,
  expectedActionForComplexState,
  mapComplexObservationToSensors,
  priorityScenarios,
  semanticConflictScenarios,
  trueConflictScenarios
} from "./complexTask";
export type { ComplexSensoryInput } from "./complexTask";

export function createComplexConfig(config: ModelConfig): ModelConfig {
  return createChallengeConfig(config);
}

interface NetworkStepResult {
  rawActiveMotors: string[];
  activeMotors: string[];
  networkDecision: ActionDecision;
  explorationAction: WorldAction | null;
  executedAction: WorldAction;
  supervisedUpdates: number;
  complexEvidence: ChallengeComplexEvidence;
}

export interface ComplexActionResolverContext {
  rawDecision: ActionDecision;
  evidence: ChallengeComplexEvidence;
  expectedAction: WorldAction;
  phase: ChallengeEpisodePhase;
}

export type ComplexActionResolver = (context: ComplexActionResolverContext) => WorldAction | null;

export function runComplexExperiment(
  config: ModelConfig,
  options: ChallengeExperimentOptions
): ChallengeExperimentResult {
  return runExperimentWithRunner(config, options, {
    traceVersion: "dg-snn-2d-complex-trace-v0.1",
    width: CHALLENGE_WIDTH,
    height: CHALLENGE_HEIGHT,
    defaultMaxSteps: DEFAULT_COMPLEX_MAX_STEPS,
    useReverseMapping: false,
    createScenarios: createChallengeScenarios,
    runEpisode: runComplexEpisode
  });
}

export function runComplexEpisode(
  network: LearningNetwork,
  scenario: ChallengeScenario,
  config: ModelConfig,
  options: {
    phase: ChallengeEpisodePhase;
    learningMode: ChallengeLearningMode;
    learningEnabled: boolean;
    seed: number;
    observationDropout: number;
    reverseMapping: boolean;
    rewardAdvantageState?: RewardAdvantageState;
    actionResolver?: ComplexActionResolver;
  }
): ChallengeEpisodeTrace {
  const rng = new SeededRandom(options.seed);
  let state = createChallengeWorldState(scenario);
  const steps: ChallengeTraceStep[] = [];
  let totalReward = 0;
  let terminalReason: ChallengeTerminalReason = "step-limit";
  let success = false;

  for (let stepIndex = 0; stepIndex < scenario.maxSteps; stepIndex += 1) {
    const rawObservation = observeChallengeWorld(state, options.observationDropout, rng);
    const sensoryInput = mapComplexObservationToSensors(rawObservation);
    const expectedAction = expectedActionForComplexState(state);
    const networkStep = runComplexNetworkStep(network, sensoryInput, expectedAction, config, {
      learningMode: options.learningMode,
      learningEnabled: options.learningEnabled,
      phase: options.phase,
      rng,
      actionResolver: options.actionResolver
    });
    const after = stepChallengeWorld(state, networkStep.executedAction);
    const reward = scoreChallengeStep(state, after, networkStep.executedAction);
    const rewardBaseline = options.rewardAdvantageState?.baseline ?? 0;
    const rewardAdvantage =
      options.learningEnabled && options.learningMode === "rewardOnly"
        ? reward.reward - rewardBaseline
        : reward.reward;
    let rewardUpdates = 0;
    let captureUpdates = 0;
    let decayUpdates = 0;

    if (options.learningEnabled && options.learningMode === "rewardOnly") {
      rewardUpdates = applyRewardOutcomeLearning(network, rewardAdvantage, config);
      updateRewardAdvantageBaseline(options.rewardAdvantageState, reward.reward, config);
      const maintenance = applyMaintenanceDecayAndCapture(network, config);
      captureUpdates = maintenance.captureUpdates;
      decayUpdates = maintenance.decayUpdates;
    } else if (options.learningEnabled && options.learningMode === "supervised") {
      const maintenance = applyMaintenanceDecayAndCapture(network, config);
      captureUpdates = maintenance.captureUpdates;
      decayUpdates = maintenance.decayUpdates;
    }

    const traceStep: ChallengeTraceStep = {
      index: stepIndex,
      before: state,
      rawObservation,
      sensoryMapping: sensoryInput.mapping,
      expectedAction,
      networkDecision: networkStep.networkDecision,
      explorationAction: networkStep.explorationAction,
      executedAction: networkStep.executedAction,
      reward: reward.reward,
      rewardBaseline,
      rewardAdvantage,
      rewardSignal: rewardAdvantage,
      distanceDelta: reward.distanceDelta,
      after,
      terminalReason: reward.terminalReason,
      terminal: reward.terminal,
      success: reward.success,
      learning: {
        rawActiveMotors: networkStep.rawActiveMotors,
        activeMotors: networkStep.activeMotors,
        supervisedUpdates: networkStep.supervisedUpdates,
        rewardUpdates,
        captureUpdates,
        decayUpdates
      },
      complexEvidence: networkStep.complexEvidence
    };

    steps.push(traceStep);
    totalReward += reward.reward;
    state = after;

    if (reward.terminal) {
      terminalReason = reward.terminalReason;
      success = reward.success;
      break;
    }
  }

  return {
    phase: options.phase,
    scenarioId: scenario.id,
    seed: scenario.seed,
    episodeSeed: options.seed,
    learningMode: options.learningMode,
    steps,
    totalReward,
    success,
    terminalReason
  };
}

function runComplexNetworkStep(
  network: LearningNetwork,
  sensoryInput: ComplexSensoryInput,
  expectedAction: WorldAction,
  config: ModelConfig,
  options: {
    learningMode: ChallengeLearningMode;
    learningEnabled: boolean;
    phase: ChallengeEpisodePhase;
    rng: SeededRandom;
    actionResolver?: ComplexActionResolver;
  }
): NetworkStepResult {
  const neuronsById = indexNeurons(network.neurons);

  for (const neuron of network.neurons) {
    resetNeuronRuntime(neuron);
  }

  const motorSpikeCounts: Record<string, number> = {};
  const interSpikeCounts: Record<string, number> = {};
  for (const neuron of network.neurons) {
    if (neuron.role === "motor") {
      motorSpikeCounts[neuron.id] = 0;
    }
    if (neuron.role === "interneuron") {
      interSpikeCounts[neuron.id] = 0;
    }
  }

  for (let tick = 0; tick < COMPLEX_SPIKE_TICKS; tick += 1) {
    for (const neuron of network.neurons) {
      if (neuron.role === "sensory") {
        const duration = sensoryInput.fireDurations[neuron.id] ?? 0;
        setSensoryOutput(neuron, tick < duration ? 1 : 0);
      }
    }

    network.tick += 1;
    resetBranchInputs(network.neurons);
    propagateSynapses(neuronsById, network.synapses, network.tick, config);
    for (const neuron of network.neurons) {
      if (neuron.role === "interneuron") {
        integrateNeuron(neuron, config);
        if (neuron.spike) {
          interSpikeCounts[neuron.id] += 1;
        }
      }
    }

    for (const neuron of network.neurons) {
      if (neuron.role === "sensory") {
        setSensoryOutput(neuron, 0);
      }
    }

    network.tick += 1;
    resetBranchInputs(network.neurons);
    propagateSynapses(neuronsById, network.synapses, network.tick, config);
    for (const neuron of network.neurons) {
      if (neuron.role === "motor") {
        integrateNeuron(neuron, config);
        if (neuron.spike) {
          motorSpikeCounts[neuron.id] += 1;
        }
      }
    }

    // Update eligibility per micro-tick so each tick's spike events feed the STDP
    // time window (preTrace / postTrace accumulate across micro-ticks via traceDecay).
    // Previously this ran once after the loop, collapsing all micro-tick timing.
    updateNetworkEligibility(network, config);
  }

  for (const neuron of network.neurons) {
    if (neuron.role === "interneuron" && (interSpikeCounts[neuron.id] ?? 0) > 0) {
      neuron.outputSignal = 1;
      neuron.spike = true;
    } else if (neuron.role === "interneuron") {
      neuron.outputSignal = 0;
      neuron.spike = false;
    }
    if (neuron.role === "motor" && motorSpikeCounts[neuron.id] > 0) {
      neuron.outputSignal = 1;
      neuron.spike = true;
    } else if (neuron.role === "motor") {
      neuron.outputSignal = 0;
      neuron.spike = false;
    }
  }

  const rawActiveMotors = Object.keys(motorSpikeCounts)
    .filter((id) => motorSpikeCounts[id] > 0)
    .sort();
  const networkDecision = arbitrateComplexMotorAction(motorSpikeCounts);
  const complexEvidence = {
    interSpikeCounts: { ...interSpikeCounts },
    motorSpikeCounts: { ...motorSpikeCounts }
  };
  const resolverAction =
    networkDecision.action === "conflict" && options.actionResolver
      ? options.actionResolver({
          rawDecision: networkDecision,
          evidence: complexEvidence,
          expectedAction,
          phase: options.phase
        })
      : null;
  const explorationAction =
    resolverAction === null ? selectExplorationAction(networkDecision.action, options, config) : null;
  const activeMotors = explorationAction
    ? forceExplorationMotor(network, explorationAction)
    : rawActiveMotors;
  const executedAction = resolverAction ?? explorationAction ?? networkDecision.action;
  let supervisedUpdates = 0;

  if (options.learningEnabled && options.learningMode === "supervised") {
    supervisedUpdates = applySupervisedMotorOutcomeLearning(network, expectedAction, new Set(activeMotors), config);
  }

  return {
    rawActiveMotors,
    activeMotors,
    networkDecision,
    explorationAction,
    executedAction,
    supervisedUpdates,
    complexEvidence
  };
}
