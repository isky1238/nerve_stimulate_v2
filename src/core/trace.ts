import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { ModelConfig, defaultConfig, withConfig } from "../config/newModelConfig";
import { createOfflineLearningNetwork, offlinePatterns, Pattern, LearningNetwork } from "./evaluation";
import {
  Branch,
  Neuron,
  indexNeurons,
  integrateNeuron,
  resetBranchInputs,
  resetNeuronRuntime,
  setSensoryOutput
} from "./neuron";
import {
  LearningEvent,
  applySupervisedMotorLearning,
  captureStableWeights,
  decayWeights,
  updateEligibility
} from "./plasticity";
import { Signal } from "./signal";
import { PropagationEvent, Synapse, propagateSynapses } from "./synapse";

export interface TraceOptions {
  seed?: number;
  epochs?: number;
  learningOn?: boolean;
}

export interface LearningTrace {
  version: string;
  generatedAt: string;
  seed: number;
  config: ModelConfig;
  metrics: TraceMetrics;
  episodes: TraceEpisode[];
}

export interface TraceMetrics {
  epochs: number;
  totalTrials: number;
  correctTrials: number;
  accuracy: number;
  finalEpochAccuracy: number;
  learningOn: boolean;
  supervisedUpdateCount: number;
  captureUpdateCount: number;
  decayUpdateCount: number;
}

export interface TraceEpisode {
  epoch: number;
  patternIndex: number;
  inputLabel: string;
  targetMotorId: string;
  activeMotors: string[];
  correct: boolean;
  phases: TracePhase[];
  weightEvents: TraceWeightEvent[];
}

export interface TracePhase {
  name: "sensory-to-interneuron" | "interneuron-to-motor";
  tick: number;
  propagationEvents: TracePropagationEvent[];
  neurons: TraceNeuronSnapshot[];
}

export interface TracePropagationEvent extends PropagationEvent {
  preNeuronId: string;
  postNeuronId: string;
  postBranchId: string;
  effectSign: number;
  effectiveWeight: number;
  synapseState: string;
}

export interface TraceNeuronSnapshot {
  id: string;
  role: string;
  outputSignal: Signal;
  spike: boolean;
  somaPotential: number;
  axonDrive: number;
  dendriteGateOpen: number;
  axonGateOpen: number;
  dynamicThreshold: number;
  branches: TraceBranchSnapshot[];
}

export interface TraceBranchSnapshot {
  id: string;
  inputSum: number;
  inhibitionLoad: number;
  gain: number;
  localThreshold: number;
  active: boolean;
  output: number;
  plasticityGate: number;
  recentActiveRate: number;
}

export interface TraceWeightEvent {
  synapseId: string;
  kind: LearningEvent["kind"];
  feedback: string;
  preNeuronId: string;
  postNeuronId: string;
  postBranchId: string;
  targetMotorId: string;
  activeMotorIds: string[];
  beforeFastWeight: number;
  afterFastWeight: number;
  beforeStableWeight: number;
  afterStableWeight: number;
  deltaFast: number;
  deltaStable: number;
  eligibilityTrace: number;
  preTrace: number;
  postTrace: number;
  plasticityGate: number;
}

interface SynapseWeightSnapshot {
  fastWeight: number;
  stableWeight: number;
}

const DEFAULT_TRACE_EPOCHS = 40;

export function runLearningTrace(
  config: ModelConfig = defaultConfig,
  options: TraceOptions = {}
): LearningTrace {
  const traceConfig = createTraceConfig(config);
  const network = createOfflineLearningNetwork(traceConfig);
  const patterns = offlinePatterns();
  const epochs = options.epochs ?? DEFAULT_TRACE_EPOCHS;
  const learningOn = options.learningOn ?? true;
  const episodes: TraceEpisode[] = [];

  for (let epoch = 0; epoch < epochs; epoch += 1) {
    for (let patternIndex = 0; patternIndex < patterns.length; patternIndex += 1) {
      episodes.push(
        runTraceTrial(network, patterns[patternIndex], patternIndex, epoch, traceConfig, learningOn)
      );
    }
  }

  const correctTrials = episodes.filter((episode) => episode.correct).length;
  const finalEpochEpisodes = episodes.filter((episode) => episode.epoch === epochs - 1);
  const finalEpochCorrect = finalEpochEpisodes.filter((episode) => episode.correct).length;

  return {
    version: "dg-snn-trace-v0.1",
    generatedAt: new Date().toISOString(),
    seed: options.seed ?? 1,
    config: traceConfig,
    metrics: {
      epochs,
      totalTrials: episodes.length,
      correctTrials,
      accuracy: correctTrials / Math.max(1, episodes.length),
      finalEpochAccuracy: finalEpochCorrect / Math.max(1, finalEpochEpisodes.length),
      learningOn,
      supervisedUpdateCount: countWeightEvents(episodes, "supervised"),
      captureUpdateCount: countWeightEvents(episodes, "capture"),
      decayUpdateCount: countWeightEvents(episodes, "decay")
    },
    episodes
  };
}

export function writeTraceExport(filePath: string, trace: LearningTrace): void {
  mkdirSync(dirname(filePath), { recursive: true });
  writeFileSync(filePath, `${JSON.stringify(trace, null, 2)}\n`, "utf8");
}

export function readTraceExport(filePath: string): LearningTrace {
  return JSON.parse(readFileSync(filePath, "utf8")) as LearningTrace;
}

export function explainTrace(trace: LearningTrace): string {
  const lines = [
    `Trace ${trace.version}`,
    `episodes=${trace.metrics.totalTrials}, allTrialAccuracy=${formatNumber(trace.metrics.accuracy)}, finalEpochAccuracy=${formatNumber(trace.metrics.finalEpochAccuracy)}, learningOn=${trace.metrics.learningOn}`,
    `updates: supervised=${trace.metrics.supervisedUpdateCount}, capture=${trace.metrics.captureUpdateCount}, decay=${trace.metrics.decayUpdateCount}`
  ];
  const episode =
    trace.episodes.find((candidate) => candidate.weightEvents.some((event) => event.kind !== "decay")) ??
    trace.episodes[0];

  if (!episode) {
    return `${lines.join("\n")}\nNo episodes recorded.`;
  }

  lines.push(
    `Episode epoch=${episode.epoch}, pattern=${episode.patternIndex}: input ${episode.inputLabel} -> target ${episode.targetMotorId}; active=${formatList(episode.activeMotors)}; correct=${episode.correct}`
  );

  const activePaths = inferActivePaths(episode);
  lines.push(`Active paths: ${activePaths.length > 0 ? activePaths.join("; ") : "none"}`);

  const informativeEvents = episode.weightEvents
    .filter((event) => event.kind !== "decay")
    .slice(0, 8);

  if (informativeEvents.length === 0) {
    lines.push("No supervised or stable-capture weight updates were recorded for the selected episode.");
    return lines.join("\n");
  }

  lines.push("Weight changes:");
  for (const event of informativeEvents) {
    lines.push(
      `${event.kind} ${event.feedback}: ${event.preNeuronId} -> ${event.postNeuronId} (${event.synapseId}) fast ${formatNumber(event.beforeFastWeight)} -> ${formatNumber(event.afterFastWeight)} (${formatSigned(event.deltaFast)}), stable ${formatNumber(event.beforeStableWeight)} -> ${formatNumber(event.afterStableWeight)} (${formatSigned(event.deltaStable)}), eligibility=${formatNumber(event.eligibilityTrace)}`
    );
  }

  return lines.join("\n");
}

function createTraceConfig(config: ModelConfig): ModelConfig {
  return withConfig({
    ...config,
    leak: 1,
    branchLocalThreshold: 0.1,
    dendriteGateThreshold: 0.1,
    axonThreshold: 1,
    thresholdAdaptRate: 0,
    refractorySteps: 0,
    fastDecay: 0.9995,
    stableThreshold: 0.12,
    useThreshold: 0.08
  });
}

function runTraceTrial(
  network: LearningNetwork,
  pattern: Pattern,
  patternIndex: number,
  epoch: number,
  config: ModelConfig,
  learningOn: boolean
): TraceEpisode {
  const neuronsById = indexNeurons(network.neurons);

  for (const neuron of network.neurons) {
    resetNeuronRuntime(neuron);
  }

  for (const neuron of network.neurons) {
    if (neuron.role === "sensory") {
      setSensoryOutput(neuron, neuron.id === pattern.sensorId ? 1 : 0);
    }
  }

  network.tick += 1;
  resetBranchInputs(network.neurons);
  const sensoryEvents = propagateSynapses(neuronsById, network.synapses, network.tick, config);
  for (const neuron of network.neurons) {
    if (neuron.role === "interneuron") {
      integrateNeuron(neuron, config);
    }
  }
  const sensoryPhase = createPhaseSnapshot(
    "sensory-to-interneuron",
    network.tick,
    network,
    sensoryEvents
  );

  for (const neuron of network.neurons) {
    if (neuron.role === "sensory") {
      setSensoryOutput(neuron, 0);
    }
  }

  network.tick += 1;
  resetBranchInputs(network.neurons);
  const motorEvents = propagateSynapses(neuronsById, network.synapses, network.tick, config);
  for (const neuron of network.neurons) {
    if (neuron.role === "motor") {
      integrateNeuron(neuron, config);
    }
  }
  const motorPhase = createPhaseSnapshot("interneuron-to-motor", network.tick, network, motorEvents);

  updateEligibility(network.synapses, neuronsById, config);
  const activeMotors = network.neurons
    .filter((neuron) => neuron.role === "motor" && neuron.outputSignal !== 0)
    .map((neuron) => neuron.id);
  const activeMotorSet = new Set(activeMotors);
  const correct = activeMotorSet.has(pattern.targetMotorId) && activeMotorSet.size === 1;
  const weightEvents: TraceWeightEvent[] = [];

  if (learningOn) {
    weightEvents.push(
      ...recordWeightEvents(network, neuronsById, pattern.targetMotorId, activeMotors, () =>
        applySupervisedMotorLearning(network.synapses, neuronsById, pattern.targetMotorId, activeMotorSet, 1, config)
      )
    );
    weightEvents.push(
      ...recordWeightEvents(network, neuronsById, pattern.targetMotorId, activeMotors, () =>
        captureStableWeights(network.synapses, config)
      )
    );
  }

  weightEvents.push(
    ...recordWeightEvents(network, neuronsById, pattern.targetMotorId, activeMotors, () =>
      decayWeights(network.synapses, config)
    )
  );

  return {
    epoch,
    patternIndex,
    inputLabel: pattern.sensorId,
    targetMotorId: pattern.targetMotorId,
    activeMotors,
    correct,
    phases: [sensoryPhase, motorPhase],
    weightEvents
  };
}

function createPhaseSnapshot(
  name: TracePhase["name"],
  tick: number,
  network: LearningNetwork,
  events: PropagationEvent[]
): TracePhase {
  const synapsesById = new Map(network.synapses.map((synapse) => [synapse.id, synapse]));

  return {
    name,
    tick,
    propagationEvents: events.map((event) => enrichPropagationEvent(event, synapsesById)),
    neurons: network.neurons.map(snapshotNeuron)
  };
}

function enrichPropagationEvent(
  event: PropagationEvent,
  synapsesById: Map<string, Synapse>
): TracePropagationEvent {
  const synapse = synapsesById.get(event.synapseId);

  if (!synapse) {
    throw new Error(`Missing synapse for propagation event ${event.synapseId}.`);
  }

  return {
    ...event,
    preNeuronId: synapse.preNeuronId,
    postNeuronId: synapse.postNeuronId,
    postBranchId: synapse.postBranchId,
    effectSign: synapse.effectSign,
    effectiveWeight: synapse.effectiveWeight,
    synapseState: synapse.state
  };
}

function snapshotNeuron(neuron: Neuron): TraceNeuronSnapshot {
  return {
    id: neuron.id,
    role: neuron.role,
    outputSignal: neuron.outputSignal,
    spike: neuron.spike,
    somaPotential: neuron.somaPotential,
    axonDrive: neuron.axonDrive,
    dendriteGateOpen: neuron.dendriteToAxonGate.openRatio,
    axonGateOpen: neuron.axonOutputGate.openRatio,
    dynamicThreshold: neuron.dynamicThreshold,
    branches: neuron.branches.map(snapshotBranch)
  };
}

function snapshotBranch(branch: Branch): TraceBranchSnapshot {
  return {
    id: branch.id,
    inputSum: branch.inputSum,
    inhibitionLoad: branch.inhibitionLoad,
    gain: branch.gain,
    localThreshold: branch.localThreshold,
    active: branch.active,
    output: branch.output,
    plasticityGate: branch.plasticityGate,
    recentActiveRate: branch.recentActiveRate
  };
}

function recordWeightEvents(
  network: LearningNetwork,
  neuronsById: Map<string, Neuron>,
  targetMotorId: string,
  activeMotorIds: string[],
  runUpdate: () => LearningEvent[]
): TraceWeightEvent[] {
  const before = new Map(
    network.synapses.map((synapse) => [
      synapse.id,
      {
        fastWeight: synapse.fastWeight,
        stableWeight: synapse.stableWeight
      }
    ])
  );
  const events = runUpdate();

  return events.map((event) =>
    createWeightEvent(event, network.synapses, neuronsById, before, targetMotorId, activeMotorIds)
  );
}

function createWeightEvent(
  event: LearningEvent,
  synapses: Synapse[],
  neuronsById: Map<string, Neuron>,
  before: Map<string, SynapseWeightSnapshot>,
  targetMotorId: string,
  activeMotorIds: string[]
): TraceWeightEvent {
  const synapse = synapses.find((candidate) => candidate.id === event.synapseId);
  const prior = before.get(event.synapseId);

  if (!synapse || !prior) {
    throw new Error(`Missing synapse state for weight event ${event.synapseId}.`);
  }

  const post = neuronsById.get(synapse.postNeuronId);
  const branch = post?.branches.find((candidate) => candidate.id === synapse.postBranchId);

  return {
    synapseId: synapse.id,
    kind: event.kind,
    feedback: feedbackForEvent(event.kind, synapse, targetMotorId, activeMotorIds),
    preNeuronId: synapse.preNeuronId,
    postNeuronId: synapse.postNeuronId,
    postBranchId: synapse.postBranchId,
    targetMotorId,
    activeMotorIds,
    beforeFastWeight: prior.fastWeight,
    afterFastWeight: synapse.fastWeight,
    beforeStableWeight: prior.stableWeight,
    afterStableWeight: synapse.stableWeight,
    deltaFast: event.deltaFast,
    deltaStable: event.deltaStable,
    eligibilityTrace: synapse.eligibilityTrace,
    preTrace: synapse.preTrace,
    postTrace: synapse.postTrace,
    plasticityGate: branch?.plasticityGate ?? 1
  };
}

function feedbackForEvent(
  kind: LearningEvent["kind"],
  synapse: Synapse,
  targetMotorId: string,
  activeMotorIds: string[]
): string {
  if (kind === "supervised") {
    if (synapse.postNeuronId === targetMotorId) {
      return "supervised-target-reinforce";
    }

    return activeMotorIds.includes(synapse.postNeuronId)
      ? "supervised-wrong-active-suppress"
      : "supervised-non-target-suppress";
  }

  if (kind === "capture") {
    return "stable-capture";
  }

  if (kind === "decay") {
    return "maintenance-decay";
  }

  return "reward-modulated";
}

function inferActivePaths(episode: TraceEpisode): string[] {
  const sensoryPhase = episode.phases.find((phase) => phase.name === "sensory-to-interneuron");
  const motorPhase = episode.phases.find((phase) => phase.name === "interneuron-to-motor");

  if (!sensoryPhase || !motorPhase) {
    return [];
  }

  const paths: string[] = [];
  for (const motorEvent of motorPhase.propagationEvents) {
    const upstream = sensoryPhase.propagationEvents.find(
      (event) => event.postNeuronId === motorEvent.preNeuronId
    );
    paths.push(
      upstream
        ? `${upstream.preNeuronId} -> ${motorEvent.preNeuronId} -> ${motorEvent.postNeuronId}`
        : `${motorEvent.preNeuronId} -> ${motorEvent.postNeuronId}`
    );
  }

  return Array.from(new Set(paths));
}

function countWeightEvents(episodes: TraceEpisode[], kind: LearningEvent["kind"]): number {
  return episodes.reduce(
    (sum, episode) => sum + episode.weightEvents.filter((event) => event.kind === kind).length,
    0
  );
}

function formatList(items: string[]): string {
  return items.length > 0 ? items.join(",") : "none";
}

function formatNumber(value: number): string {
  return Number.isInteger(value) ? `${value}` : value.toFixed(4);
}

function formatSigned(value: number): string {
  const formatted = formatNumber(value);
  return value >= 0 ? `+${formatted}` : formatted;
}
