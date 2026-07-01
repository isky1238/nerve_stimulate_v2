"use strict";

const fs = require("node:fs/promises");
const http = require("node:http");
const path = require("node:path");

const ROOT = path.join(__dirname, "..");
const WEB_ROOT = path.join(ROOT, "webui");
const DEFAULT_SEEDS = [21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171];

const { defaultConfig, withConfig } = require(path.join(ROOT, "dist/src/config/newModelConfig"));
const {
  createUniformNaturalLayeredTopologyBlueprint
} = require(path.join(ROOT, "dist/src/core/layeredTopologyBlueprint"));
const { createLearningNetworkFromBlueprint } = require(path.join(ROOT, "dist/src/core/topologyBlueprint"));
const {
  tryFormNearestLayeredConnections,
  updateConnectionStates
} = require(path.join(ROOT, "dist/src/core/development"));
const {
  activeMotorIds,
  applyMaintenanceDecayAndCapture,
  applyRewardOutcomeLearning,
  clearSensoryOutputs,
  propagateAndIntegrateRole,
  resetNetworkRuntime,
  setSensoryOutputs,
  updateNetworkEligibility
} = require(path.join(ROOT, "dist/src/core/mechanism"));
const { SeededRandom } = require(path.join(ROOT, "dist/src/core/random"));

const VALENCE_IDS = {
  nutrientGate: "nutrientGate",
  toxinGate: "toxinGate",
  objectStimA: "objectStimA",
  objectStimB: "objectStimB",
  toxinMotor: "toxinMotor",
  nutrientMotor: "nutrientMotor"
};

const GRID_IDS = {
  nutrientSense: "nutrientSense",
  toxinSense: "toxinSense",
  leftMotor: "leftMotor",
  rightMotor: "rightMotor"
};

function parsePort(argv) {
  const index = argv.indexOf("--port");
  if (index >= 0 && argv[index + 1]) {
    const value = Number(argv[index + 1]);
    if (Number.isFinite(value) && value > 0) return value;
  }
  const env = Number(process.env.PORT);
  return Number.isFinite(env) && env > 0 ? env : 4173;
}

function positiveInt(value, fallback, max = 1000) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.max(1, Math.min(max, Math.floor(number)));
}

function boundedNumber(value, fallback, min, max) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.max(min, Math.min(max, number));
}

function boolValue(value) {
  return value === true || value === "true" || value === "1" || value === 1;
}

function checkpointSet(epochs) {
  const base = [1, 5, 10, 20, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000];
  return new Set([...base.filter((epoch) => epoch <= epochs), epochs].sort((a, b) => a - b));
}

function normalizedY(index, count) {
  return count === 1 ? 0.5 : index / (count - 1);
}

function mean(items, selector) {
  return items.length === 0 ? 0 : items.reduce((sum, item) => sum + selector(item), 0) / items.length;
}

function shuffle(items, rng) {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = rng.nextInt(i + 1);
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function stateCounts(network) {
  const roles = new Map(network.neurons.map((neuron) => [neuron.id, neuron.role]));
  const stats = {
    stem: 0,
    readout: 0,
    candidate: 0,
    active: 0,
    stable: 0,
    dormant: 0,
    pruned: 0,
    live: 0
  };

  for (const synapse of network.synapses) {
    stats[synapse.state] = (stats[synapse.state] ?? 0) + 1;
    if (synapse.state !== "pruned" && synapse.state !== "dormant") {
      stats.live += 1;
    }
    if (synapse.state === "pruned") continue;
    const pre = roles.get(synapse.preNeuronId);
    const post = roles.get(synapse.postNeuronId);
    if (pre === "sensory" && post === "interneuron") stats.stem += 1;
    if (pre === "interneuron" && post === "motor") stats.readout += 1;
  }

  return stats;
}

function graphSnapshot(network) {
  const roles = new Map(network.neurons.map((neuron) => [neuron.id, neuron.role]));
  return {
    nodes: network.neurons.map((neuron) => ({
      id: neuron.id,
      role: neuron.role,
      x: neuron.position.x,
      y: neuron.position.y,
      spike: Boolean(neuron.spike),
      outputSignal: neuron.outputSignal,
      tagLoad: neuron.tagLoad,
      inputSlotsUsed: neuron.inputSlots.filter(Boolean).length,
      outputSlotsUsed: neuron.outputSlots.filter(Boolean).length,
      maxInputSlots: neuron.maxInputSlots,
      maxOutputSlots: neuron.maxOutputSlots
    })),
    edges: network.synapses.map((synapse) => ({
      id: synapse.id,
      pre: synapse.preNeuronId,
      post: synapse.postNeuronId,
      preRole: roles.get(synapse.preNeuronId),
      postRole: roles.get(synapse.postNeuronId),
      state: synapse.state,
      connected: synapse.connected,
      effectiveWeight: synapse.effectiveWeight,
      fastWeight: synapse.fastWeight,
      stableWeight: synapse.stableWeight,
      recentUse: synapse.recentUse,
      recentContribution: synapse.recentContribution,
      age: synapse.age,
      dormantTicks: synapse.dormantTicks,
      decayProtected: synapse.decayProtected,
      tagLoad: synapse.tagLoad
    }))
  };
}

function runNaturalStep(network, inputId, config) {
  resetNetworkRuntime(network);
  setSensoryOutputs(network, new Set([inputId]));
  propagateAndIntegrateRole(network, "interneuron", config);
  clearSensoryOutputs(network);
  propagateAndIntegrateRole(network, "motor", config);
  return activeMotorIds(network);
}

function classifyNatural(activeOutputs, targetOutputId) {
  const unique = Array.from(new Set(activeOutputs)).sort();
  if (unique.length === 0) return { action: "noop", activeOutputs: unique };
  if (unique.length > 1) return { action: "conflict", activeOutputs: unique };
  return { action: unique[0] === targetOutputId ? "correct" : "wrong", activeOutputs: unique };
}

function forceOutput(network, outputId) {
  for (const neuron of network.neurons) {
    if (neuron.role === "motor") {
      neuron.outputSignal = neuron.id === outputId ? 1 : 0;
      neuron.spike = neuron.id === outputId;
    }
  }
  return activeMotorIds(network);
}

function naturalReward(action) {
  if (action === "correct") return 1;
  if (action === "wrong" || action === "conflict") return -1;
  return -0.1;
}

function createNaturalConfig(params) {
  return withConfig({
    ...defaultConfig,
    fastWeightInit: boundedNumber(params.fastInit, defaultConfig.fastWeightInit, 0, 2),
    leak: 1,
    branchLocalThreshold: boundedNumber(params.branchThreshold, 0.1, 0, 5),
    dendriteGateThreshold: boundedNumber(params.branchThreshold, 0.1, 0, 5),
    axonThreshold: boundedNumber(params.axonThreshold, 1, 0, 5),
    thresholdAdaptRate: 0,
    refractorySteps: 0,
    fastDecay: 0.9995,
    stableThreshold: 0.12,
    useThreshold: 0.08,
    contributionThreshold: 0.05,
    candidateMaxAge: positiveInt(params.candidateMaxAge, 20, 1000),
    minConnectionAge: 10,
    dormantLimit: positiveInt(params.dormantLimit, 50, 5000),
    connectionDistanceLambda: 1.5,
    connectionThreshold: 0.1,
    rewardAdvantageBaselineAlpha: 0.1
  });
}

function naturalParams(input) {
  const scale = positiveInt(input.scale, 1, 20);
  const baseInputCount = positiveInt(input.inputCount, 2, 100);
  const baseMediumCount = positiveInt(input.mediumCount, 10, 500);
  const baseOutputCount = positiveInt(input.outputCount, 2, 100);
  const params = {
    inputCount: baseInputCount * scale,
    mediumCount: baseMediumCount * scale,
    outputCount: baseOutputCount * scale,
    baseInputCount,
    baseMediumCount,
    baseOutputCount,
    scale,
    slotsPerNeuron: positiveInt(input.slotsPerNeuron, 5, 50),
    epochs: positiveInt(input.epochs, 100, 1000),
    seedCount: positiveInt(input.seedCount, 1, DEFAULT_SEEDS.length),
    fastInit: boundedNumber(input.fastInit, defaultConfig.fastWeightInit, 0, 2),
    axonThreshold: boundedNumber(input.axonThreshold, 1, 0, 5),
    branchThreshold: boundedNumber(input.branchThreshold, 0.1, 0, 5),
    candidateMaxAge: positiveInt(input.candidateMaxAge, 20, 1000),
    dormantLimit: positiveInt(input.dormantLimit, 50, 5000)
  };
  params.maxNewConnections = positiveInt(
    input.maxNewConnections,
    (params.inputCount + params.mediumCount) * params.slotsPerNeuron,
    100000
  );
  return params;
}

function naturalPatterns(params) {
  return Array.from({ length: params.inputCount }, (_, index) => {
    const targetIndex = params.outputCount === 1
      ? 0
      : Math.max(0, Math.min(params.outputCount - 1, Math.round(normalizedY(index, params.inputCount) * (params.outputCount - 1))));
    return {
      inputId: `input${index}`,
      inputIndex: index,
      targetOutputId: `output${targetIndex}`,
      targetOutputIndex: targetIndex
    };
  });
}

function trainNaturalTrial(network, pattern, params, config, rng, baselineState) {
  let activeOutputs = runNaturalStep(network, pattern.inputId, config);
  let classification = classifyNatural(activeOutputs, pattern.targetOutputId);

  if (classification.action === "noop" || classification.action === "conflict") {
    activeOutputs = forceOutput(network, `output${rng.nextInt(params.outputCount)}`);
    classification = classifyNatural(activeOutputs, pattern.targetOutputId);
  }

  updateNetworkEligibility(network, config);
  const reward = naturalReward(classification.action);
  const rewardAdvantage = reward - baselineState.baseline;
  applyRewardOutcomeLearning(network, rewardAdvantage, config);
  baselineState.baseline =
    baselineState.baseline * (1 - config.rewardAdvantageBaselineAlpha) +
    reward * config.rewardAdvantageBaselineAlpha;
  applyMaintenanceDecayAndCapture(network, config);
}

function evaluateNatural(network, patterns, config) {
  const rows = patterns.map((pattern) => {
    const clone = structuredClone(network);
    const activeOutputs = runNaturalStep(clone, pattern.inputId, config);
    return { ...pattern, ...classifyNatural(activeOutputs, pattern.targetOutputId) };
  });
  const denom = Math.max(1, rows.length);
  return {
    sr: rows.filter((row) => row.action === "correct").length / denom,
    noop: rows.filter((row) => row.action === "noop").length / denom,
    conflict: rows.filter((row) => row.action === "conflict").length / denom,
    wrong: rows.filter((row) => row.action === "wrong").length / denom,
    rows
  };
}

function naturalPathSnapshot(network, pattern, params, config) {
  const clone = structuredClone(network);
  resetNetworkRuntime(clone);
  setSensoryOutputs(clone, new Set([pattern.inputId]));
  propagateAndIntegrateRole(clone, "interneuron", config);

  const roles = new Map(clone.neurons.map((neuron) => [neuron.id, neuron.role]));
  const firingMediumIds = new Set(
    clone.neurons
      .filter((neuron) => neuron.role === "interneuron" && neuron.spike)
      .map((neuron) => neuron.id)
  );
  const liveStemIds = new Set();
  const driveByOutput = new Map();
  let liveStemEff = 0;
  let activeStemEff = 0;

  for (const synapse of clone.synapses) {
    if (synapse.state === "pruned" || synapse.state === "dormant") continue;
    const preRole = roles.get(synapse.preNeuronId);
    const postRole = roles.get(synapse.postNeuronId);

    if (preRole === "sensory" && postRole === "interneuron" && synapse.preNeuronId === pattern.inputId) {
      liveStemIds.add(synapse.postNeuronId);
      liveStemEff += synapse.effectiveWeight;
      if (firingMediumIds.has(synapse.postNeuronId)) activeStemEff += synapse.effectiveWeight;
    }

    if (preRole === "interneuron" && postRole === "motor" && firingMediumIds.has(synapse.preNeuronId)) {
      driveByOutput.set(synapse.postNeuronId, (driveByOutput.get(synapse.postNeuronId) ?? 0) + synapse.effectiveWeight);
    }
  }

  clearSensoryOutputs(clone);
  propagateAndIntegrateRole(clone, "motor", config);
  const activeOutputs = activeMotorIds(clone);
  const classified = classifyNatural(activeOutputs, pattern.targetOutputId);
  const correctDrive = driveByOutput.get(pattern.targetOutputId) ?? 0;
  const wrongEntries = [...driveByOutput.entries()].filter(([outputId]) => outputId !== pattern.targetOutputId);
  wrongEntries.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]) || a[0].localeCompare(b[0]));
  const [wrongOutputId, wrongDrive] = wrongEntries[0] ?? [null, 0];

  return {
    ...pattern,
    action: classified.action,
    activeOutputs: classified.activeOutputs,
    firingMediumIds: [...firingMediumIds].sort(),
    liveStemCount: liveStemIds.size,
    firingMediumCount: firingMediumIds.size,
    liveStemEff,
    activeStemEff,
    correctDrive,
    wrongOutputId,
    wrongMaxDrive: Math.abs(wrongDrive),
    outputDrives: Array.from({ length: params.outputCount }, (_, index) => {
      const outputId = `output${index}`;
      return { outputId, drive: driveByOutput.get(outputId) ?? 0 };
    })
  };
}

function naturalWeightTotals(network, params) {
  const roles = new Map(network.neurons.map((neuron) => [neuron.id, neuron.role]));
  const totals = Array.from({ length: params.outputCount }, (_, index) => ({
    outputId: `output${index}`,
    live: 0,
    eff: 0,
    fast: 0,
    stable: 0
  }));
  const byOutput = new Map(totals.map((item) => [item.outputId, item]));

  for (const synapse of network.synapses) {
    if (synapse.state === "pruned" || synapse.state === "dormant") continue;
    if (roles.get(synapse.preNeuronId) !== "interneuron" || roles.get(synapse.postNeuronId) !== "motor") continue;
    const total = byOutput.get(synapse.postNeuronId);
    if (!total) continue;
    total.live += 1;
    total.eff += synapse.effectiveWeight;
    total.fast += synapse.fastWeight;
    total.stable += synapse.stableWeight;
  }

  return totals;
}

function naturalCheckpoint(network, patterns, params, config, cumulative, includeGraph) {
  return {
    metrics: evaluateNatural(network, patterns, config),
    stats: stateCounts(network),
    cumulative: { ...cumulative },
    graph: includeGraph ? graphSnapshot(network) : null,
    paths: includeGraph ? patterns.map((pattern) => naturalPathSnapshot(network, pattern, params, config)) : [],
    weights: includeGraph ? naturalWeightTotals(network, params) : []
  };
}

function runNaturalSeed(seed, params, includeGraph) {
  const config = createNaturalConfig(params);
  const topology = createUniformNaturalLayeredTopologyBlueprint({
    inputCount: params.inputCount,
    mediumCount: params.mediumCount,
    outputCount: params.outputCount,
    slotsPerNeuron: params.slotsPerNeuron
  });
  const network = createLearningNetworkFromBlueprint(topology, config);
  const patterns = naturalPatterns(params);
  const rng = new SeededRandom(seed);
  const checkpoints = checkpointSet(params.epochs);
  const baselineState = { baseline: 0 };
  const cumulative = { formed: 0, activated: 0, dormant: 0, pruned: 0, tombstoneHit: 0 };
  const rows = [];

  for (let epoch = 1; epoch <= params.epochs; epoch += 1) {
    const formed = tryFormNearestLayeredConnections(
      network.neurons,
      network.synapses,
      network.pairMemory,
      network.tick,
      config,
      params.maxNewConnections
    );
    cumulative.formed += formed.formed;
    cumulative.tombstoneHit += formed.tombstoneHit;

    for (const pattern of shuffle(patterns, rng)) {
      trainNaturalTrial(network, pattern, params, config, rng, baselineState);
    }

    const state = updateConnectionStates(network.neurons, network.synapses, network.pairMemory, network.tick, config);
    cumulative.activated += state.activated;
    cumulative.dormant += state.dormant;
    cumulative.pruned += state.pruned;
    network.tick += 1;

    if (checkpoints.has(epoch)) {
      rows.push({
        epoch,
        ...naturalCheckpoint(network, patterns, params, config, cumulative, includeGraph)
      });
    }
  }

  return {
    seed,
    checkpoints: rows,
    final: naturalCheckpoint(network, patterns, params, config, cumulative, includeGraph)
  };
}

function aggregateNatural(seedRuns, params) {
  const first = seedRuns[0];
  const checkpoints = first.checkpoints.map((base, index) => {
    const rows = seedRuns.map((run) => run.checkpoints[index]).filter(Boolean);
    return {
      epoch: base.epoch,
      metrics: {
        sr: mean(rows, (row) => row.metrics.sr),
        noop: mean(rows, (row) => row.metrics.noop),
        conflict: mean(rows, (row) => row.metrics.conflict),
        wrong: mean(rows, (row) => row.metrics.wrong)
      },
      stats: aggregateStats(rows),
      cumulative: {
        formed: mean(rows, (row) => row.cumulative.formed),
        activated: mean(rows, (row) => row.cumulative.activated),
        dormant: mean(rows, (row) => row.cumulative.dormant),
        pruned: mean(rows, (row) => row.cumulative.pruned),
        tombstoneHit: mean(rows, (row) => row.cumulative.tombstoneHit)
      },
      graph: base.graph,
      paths: base.paths,
      weights: base.weights
    };
  });
  const finalRows = seedRuns.map((run) => run.final);
  return {
    mode: "natural",
    params,
    seeds: seedRuns.map((run) => run.seed),
    checkpoints,
    final: {
      metrics: {
        sr: mean(finalRows, (row) => row.metrics.sr),
        noop: mean(finalRows, (row) => row.metrics.noop),
        conflict: mean(finalRows, (row) => row.metrics.conflict),
        wrong: mean(finalRows, (row) => row.metrics.wrong)
      },
      stats: aggregateStats(finalRows),
      solved: seedRuns.filter((run) => run.final.metrics.sr >= 0.99).length,
      stuck: seedRuns.filter((run) => run.final.metrics.noop >= 0.8).length
    }
  };
}

function aggregateStats(rows) {
  return {
    stem: mean(rows, (row) => row.stats.stem),
    readout: mean(rows, (row) => row.stats.readout),
    candidate: mean(rows, (row) => row.stats.candidate),
    active: mean(rows, (row) => row.stats.active),
    stable: mean(rows, (row) => row.stats.stable),
    dormant: mean(rows, (row) => row.stats.dormant),
    pruned: mean(rows, (row) => row.stats.pruned),
    live: mean(rows, (row) => row.stats.live)
  };
}

function createValenceConfig(params) {
  return withConfig({
    ...defaultConfig,
    leak: 1,
    branchLocalThreshold: boundedNumber(params.branchThreshold, 0.1, 0, 5),
    dendriteGateThreshold: boundedNumber(params.branchThreshold, 0.1, 0, 5),
    axonThreshold: boundedNumber(params.axonThreshold, 1, 0, 5),
    thresholdAdaptRate: 0,
    refractorySteps: 0,
    fastDecay: 0.9995,
    stableThreshold: 0.12,
    useThreshold: 0.08,
    depotentiationRate: 0.64,
    aversiveTagStrategy: "off",
    aversiveTagGain: 0,
    aversiveAvoidanceBonus: 0,
    aversiveDepotentiationRate: 0,
    aversiveBadOutcomeThreshold: 0,
    taggedDepotentiationMode: params.taggedMode,
    taggedCaptureGain: 1.0,
    globalAversiveLoadIncrement: params.globalIncrement,
    globalAversiveLoadDecay: params.globalDecay,
    globalSensitizationThreshold: params.globalThreshold,
    rewardAdvantageBaselineAlpha: 0
  });
}

function valenceParams(input) {
  const taggedMode = ["off", "taggedImpulse", "specificFactor"].includes(input.taggedMode)
    ? input.taggedMode
    : "specificFactor";
  return {
    epochs: positiveInt(input.epochs, 300, 1000),
    seedCount: positiveInt(input.seedCount, 1, DEFAULT_SEEDS.length),
    trialsPerEpoch: positiveInt(input.trialsPerEpoch, 10, 200),
    mediumCount: positiveInt(input.mediumCount, 12, 200),
    stemFanout: positiveInt(input.stemFanout, 5, 50),
    readoutFast: boundedNumber(input.readoutFast, 0.35, 0, 5),
    axonThreshold: boundedNumber(input.axonThreshold, 1, 0, 5),
    branchThreshold: boundedNumber(input.branchThreshold, 0.1, 0, 5),
    taggedMode,
    tagObjectStimWithToxin: boolValue(input.tagObjectStimWithToxin),
    globalIncrement: boundedNumber(input.globalIncrement, 1, -10, 10),
    globalDecay: boundedNumber(input.globalDecay, 0.9, 0, 1),
    globalThreshold: boundedNumber(input.globalThreshold, 0.5, -10, 10)
  };
}

function nearestNodes(source, targets, limit) {
  return [...targets]
    .sort((a, b) => {
      const distance = Math.abs(source.position.y - a.position.y) - Math.abs(source.position.y - b.position.y);
      return distance === 0 ? a.id.localeCompare(b.id) : distance;
    })
    .slice(0, limit);
}

function createValenceTopology(params) {
  const sensoryNodes = [
    { id: VALENCE_IDS.toxinGate, role: "sensory", position: { x: 0, y: 0.0 }, maxOutputSlots: params.stemFanout },
    { id: VALENCE_IDS.objectStimA, role: "sensory", position: { x: 0, y: 0.35 }, maxOutputSlots: params.stemFanout },
    { id: VALENCE_IDS.objectStimB, role: "sensory", position: { x: 0, y: 0.65 }, maxOutputSlots: params.stemFanout },
    { id: VALENCE_IDS.nutrientGate, role: "sensory", position: { x: 0, y: 1.0 }, maxOutputSlots: params.stemFanout }
  ];
  const interneuronNodes = Array.from({ length: params.mediumCount }, (_, index) => ({
    id: `medium${index}`,
    role: "interneuron",
    position: { x: 1, y: normalizedY(index, params.mediumCount) },
    branchCount: 1,
    maxInputSlots: 2,
    maxOutputSlots: 1
  }));
  const motorNodes = [
    { id: VALENCE_IDS.toxinMotor, role: "motor", position: { x: 2, y: 0 }, branchCount: 1, maxInputSlots: params.mediumCount },
    { id: VALENCE_IDS.nutrientMotor, role: "motor", position: { x: 2, y: 1 }, branchCount: 1, maxInputSlots: params.mediumCount }
  ];
  const synapses = [];

  for (const sensory of sensoryNodes) {
    for (const medium of nearestNodes(sensory, interneuronNodes, Math.min(params.stemFanout, interneuronNodes.length))) {
      synapses.push({
        kind: "structuralStem",
        preNeuronId: sensory.id,
        postNeuronId: medium.id,
        postBranchIndex: 0,
        fastWeight: 0,
        stableWeight: 1.1,
        decayProtected: true
      });
    }
  }

  for (const medium of interneuronNodes) {
    const motor = nearestNodes(medium, motorNodes, 1)[0];
    synapses.push({
      kind: "plasticReadout",
      preNeuronId: medium.id,
      postNeuronId: motor.id,
      postBranchIndex: 0,
      fastWeight: params.readoutFast,
      stableWeight: 0,
      decayProtected: false
    });
  }

  return Object.freeze({
    sensoryNodes: Object.freeze(sensoryNodes),
    interneuronNodes: Object.freeze(interneuronNodes),
    motorNodes: Object.freeze(motorNodes),
    synapses: Object.freeze(synapses)
  });
}

function midpointInputs() {
  return new Set([
    VALENCE_IDS.toxinGate,
    VALENCE_IDS.objectStimA,
    VALENCE_IDS.objectStimB,
    VALENCE_IDS.nutrientGate
  ]);
}

function valenceCases() {
  return [
    { label: "MIDPOINT", active: midpointInputs() },
    { label: "TOXIN_OBJECT", active: new Set([VALENCE_IDS.toxinGate, VALENCE_IDS.objectStimA, VALENCE_IDS.objectStimB]) },
    { label: "NUTRI_OBJECT", active: new Set([VALENCE_IDS.nutrientGate, VALENCE_IDS.objectStimA, VALENCE_IDS.objectStimB]) },
    { label: "PURE_OBJECT", active: new Set([VALENCE_IDS.objectStimA, VALENCE_IDS.objectStimB]) },
    { label: "TOXIN_GATE", active: new Set([VALENCE_IDS.toxinGate]) },
    { label: "NUTR_GATE", active: new Set([VALENCE_IDS.nutrientGate]) },
    { label: "OBJECT_A", active: new Set([VALENCE_IDS.objectStimA]) },
    { label: "OBJECT_B", active: new Set([VALENCE_IDS.objectStimB]) }
  ];
}

function markValenceToxinTag(network, activeIds, params, config) {
  if (!activeIds.has(VALENCE_IDS.toxinGate)) return;
  const objectStimuli = new Set([VALENCE_IDS.objectStimA, VALENCE_IDS.objectStimB]);

  for (const neuron of network.neurons) {
    if (neuron.role !== "sensory") continue;
    if (
      neuron.id === VALENCE_IDS.toxinGate ||
      (params.tagObjectStimWithToxin && objectStimuli.has(neuron.id) && activeIds.has(neuron.id))
    ) {
      neuron.tagLoad = 1;
    }
  }

  if (config.taggedDepotentiationMode === "specificFactor") {
    network.globalAversiveLoad += config.globalAversiveLoadIncrement;
  }
}

function runValenceStep(network, activeIds, params, config) {
  resetNetworkRuntime(network);
  setSensoryOutputs(network, activeIds);
  markValenceToxinTag(network, activeIds, params, config);
  propagateAndIntegrateRole(network, "interneuron", config);
  clearSensoryOutputs(network);
  propagateAndIntegrateRole(network, "motor", config);
  return activeMotorIds(network);
}

function classifyValence(activeOutputs) {
  const active = Array.from(new Set(activeOutputs)).sort();
  const toxin = active.includes(VALENCE_IDS.toxinMotor);
  const nutrient = active.includes(VALENCE_IDS.nutrientMotor);
  if (toxin && nutrient) return { action: "conflict", activeOutputs: active, contact: null };
  if (toxin) return { action: "toxin", activeOutputs: active, contact: "toxin" };
  if (nutrient) return { action: "nutrient", activeOutputs: active, contact: "nutrient" };
  return { action: "noop", activeOutputs: active, contact: null };
}

function trainValenceTrial(network, params, config, rng) {
  let activeOutputs = runValenceStep(network, midpointInputs(), params, config);
  let choice = classifyValence(activeOutputs);

  if (choice.action === "noop" || choice.action === "conflict") {
    const forced = rng.next() < 0.5 ? VALENCE_IDS.toxinMotor : VALENCE_IDS.nutrientMotor;
    activeOutputs = forceOutput(network, forced);
    choice = classifyValence(activeOutputs);
  }

  updateNetworkEligibility(network, config);
  applyRewardOutcomeLearning(network, 0, config);
  applyMaintenanceDecayAndCapture(network, config);
}

function valenceSnapshotCase(network, item, params, config) {
  const clone = structuredClone(network);
  const activeOutputs = runValenceStep(clone, item.active, params, config);
  const choice = classifyValence(activeOutputs);
  const roles = new Map(clone.neurons.map((neuron) => [neuron.id, neuron.role]));
  const firingMediumIds = new Set(
    clone.neurons
      .filter((neuron) => neuron.role === "interneuron" && neuron.spike)
      .map((neuron) => neuron.id)
  );
  const activeInputIds = new Set(item.active);
  let toxinDrive = 0;
  let nutrientDrive = 0;
  let liveStemEff = 0;
  let activeStemEff = 0;

  for (const synapse of clone.synapses) {
    if (synapse.state === "pruned" || synapse.state === "dormant") continue;
    const preRole = roles.get(synapse.preNeuronId);
    const postRole = roles.get(synapse.postNeuronId);

    if (preRole === "sensory" && postRole === "interneuron" && activeInputIds.has(synapse.preNeuronId)) {
      liveStemEff += synapse.effectiveWeight;
      if (firingMediumIds.has(synapse.postNeuronId)) activeStemEff += synapse.effectiveWeight;
    }

    if (preRole === "interneuron" && postRole === "motor" && firingMediumIds.has(synapse.preNeuronId)) {
      if (synapse.postNeuronId === VALENCE_IDS.toxinMotor) toxinDrive += synapse.effectiveWeight;
      if (synapse.postNeuronId === VALENCE_IDS.nutrientMotor) nutrientDrive += synapse.effectiveWeight;
    }
  }

  return {
    label: item.label,
    action: choice.action,
    activeInputs: [...item.active].sort(),
    activeOutputs: choice.activeOutputs,
    firingMediumIds: [...firingMediumIds].sort(),
    firingMediumCount: firingMediumIds.size,
    liveStemEff,
    activeStemEff,
    toxinDrive,
    nutrientDrive
  };
}

function valenceCheckpoint(network, params, config, includeGraph) {
  const cases = valenceCases().map((item) => valenceSnapshotCase(network, item, params, config));
  const midpoint = cases.find((item) => item.label === "MIDPOINT");
  return {
    metrics: {
      toxin: midpoint.action === "toxin" ? 1 : 0,
      nutrient: midpoint.action === "nutrient" ? 1 : 0,
      noop: midpoint.action === "noop" ? 1 : 0,
      conflict: midpoint.action === "conflict" ? 1 : 0,
      toxinDrive: midpoint.toxinDrive,
      nutrientDrive: midpoint.nutrientDrive
    },
    stats: stateCounts(network),
    graph: includeGraph ? graphSnapshot(network) : null,
    cases: includeGraph ? cases : []
  };
}

function runValenceSeed(seed, params, includeGraph) {
  const config = createValenceConfig(params);
  const network = createLearningNetworkFromBlueprint(createValenceTopology(params), config);
  const rng = new SeededRandom(seed);
  const checkpoints = checkpointSet(params.epochs);
  const rows = [];

  for (let epoch = 1; epoch <= params.epochs; epoch += 1) {
    for (let trial = 0; trial < params.trialsPerEpoch; trial += 1) {
      trainValenceTrial(network, params, config, rng);
    }
    if (checkpoints.has(epoch)) {
      rows.push({ epoch, ...valenceCheckpoint(network, params, config, includeGraph) });
    }
  }

  return {
    seed,
    checkpoints: rows,
    final: valenceCheckpoint(network, params, config, includeGraph)
  };
}

function aggregateValence(seedRuns, params) {
  const first = seedRuns[0];
  const checkpoints = first.checkpoints.map((base, index) => {
    const rows = seedRuns.map((run) => run.checkpoints[index]).filter(Boolean);
    return {
      epoch: base.epoch,
      metrics: {
        toxin: mean(rows, (row) => row.metrics.toxin),
        nutrient: mean(rows, (row) => row.metrics.nutrient),
        noop: mean(rows, (row) => row.metrics.noop),
        conflict: mean(rows, (row) => row.metrics.conflict),
        toxinDrive: mean(rows, (row) => row.metrics.toxinDrive),
        nutrientDrive: mean(rows, (row) => row.metrics.nutrientDrive)
      },
      stats: aggregateStats(rows),
      graph: base.graph,
      cases: base.cases
    };
  });
  const finalRows = seedRuns.map((run) => run.final);
  return {
    mode: "valence",
    params,
    seeds: seedRuns.map((run) => run.seed),
    checkpoints,
    final: {
      metrics: {
        toxin: mean(finalRows, (row) => row.metrics.toxin),
        nutrient: mean(finalRows, (row) => row.metrics.nutrient),
        noop: mean(finalRows, (row) => row.metrics.noop),
        conflict: mean(finalRows, (row) => row.metrics.conflict),
        toxinDrive: mean(finalRows, (row) => row.metrics.toxinDrive),
        nutrientDrive: mean(finalRows, (row) => row.metrics.nutrientDrive)
      },
      stats: aggregateStats(finalRows)
    }
  };
}

function gridWorldParams(input) {
  const taggedMode = ["off", "taggedImpulse", "specificFactor"].includes(input.taggedMode)
    ? input.taggedMode
    : "specificFactor";
  const params = {
    objectSensorCount: positiveInt(input.objectSensorCount, 5, 100),
    mediumRows: positiveInt(input.mediumRows, 3, 50),
    mediumCols: positiveInt(input.mediumCols, 5, 50),
    slotsPerNeuron: positiveInt(input.slotsPerNeuron, 5, 50),
    epochs: positiveInt(input.epochs, 300, 1000),
    seedCount: positiveInt(input.seedCount, 1, DEFAULT_SEEDS.length),
    fastInit: boundedNumber(input.fastInit, defaultConfig.fastWeightInit, 0, 2),
    axonThreshold: boundedNumber(input.axonThreshold, 1, 0, 5),
    branchThreshold: boundedNumber(input.branchThreshold, 0.1, 0, 5),
    candidateMaxAge: positiveInt(input.candidateMaxAge, 20, 1000),
    dormantLimit: positiveInt(input.dormantLimit, 50, 5000),
    taggedMode,
    tagObjectStimWithToxin: boolValue(input.tagObjectStimWithToxin),
    globalIncrement: boundedNumber(input.globalIncrement, 1, -10, 10),
    globalDecay: boundedNumber(input.globalDecay, 0.9, 0, 1),
    globalThreshold: boundedNumber(input.globalThreshold, 0.5, -10, 10)
  };
  params.mediumCount = params.mediumRows * params.mediumCols;
  params.maxNewConnections = positiveInt(
    input.maxNewConnections,
    (params.objectSensorCount + 2 + params.mediumCount) * params.slotsPerNeuron,
    100000
  );
  return params;
}

function createGridWorldConfig(params) {
  return withConfig({
    ...defaultConfig,
    fastWeightInit: params.fastInit,
    leak: 1,
    branchLocalThreshold: params.branchThreshold,
    dendriteGateThreshold: params.branchThreshold,
    axonThreshold: params.axonThreshold,
    thresholdAdaptRate: 0,
    refractorySteps: 0,
    fastDecay: 0.9995,
    stableThreshold: 0.12,
    useThreshold: 0.08,
    contributionThreshold: 0.05,
    candidateMaxAge: params.candidateMaxAge,
    minConnectionAge: 10,
    dormantLimit: params.dormantLimit,
    connectionDistanceLambda: 1.5,
    connectionThreshold: 0.1,
    rewardAdvantageBaselineAlpha: 0.1,
    depotentiationRate: 0.64,
    taggedDepotentiationMode: params.taggedMode,
    taggedCaptureGain: 1.0,
    globalAversiveLoadIncrement: params.globalIncrement,
    globalAversiveLoadDecay: params.globalDecay,
    globalSensitizationThreshold: params.globalThreshold
  });
}

function gridObjectId(index) {
  return `object${index}`;
}

function createGridWorldTopology(params) {
  const objectNodes = Array.from({ length: params.objectSensorCount }, (_, index) => ({
    id: gridObjectId(index),
    role: "sensory",
    position: { x: normalizedY(index, params.objectSensorCount), y: 0.08 },
    maxInputSlots: params.slotsPerNeuron,
    maxOutputSlots: params.slotsPerNeuron
  }));
  const sensoryNodes = [
    {
      id: GRID_IDS.nutrientSense,
      role: "sensory",
      position: { x: 0.12, y: 0 },
      maxInputSlots: params.slotsPerNeuron,
      maxOutputSlots: params.slotsPerNeuron
    },
    ...objectNodes,
    {
      id: GRID_IDS.toxinSense,
      role: "sensory",
      position: { x: 0.88, y: 0 },
      maxInputSlots: params.slotsPerNeuron,
      maxOutputSlots: params.slotsPerNeuron
    }
  ];
  const interneuronNodes = [];
  for (let row = 0; row < params.mediumRows; row += 1) {
    for (let col = 0; col < params.mediumCols; col += 1) {
      interneuronNodes.push({
        id: `medium${row}_${col}`,
        role: "interneuron",
        position: {
          x: normalizedY(col, params.mediumCols),
          y: 0.34 + normalizedY(row, params.mediumRows) * 0.34
        },
        branchCount: 1,
        maxInputSlots: params.slotsPerNeuron,
        maxOutputSlots: params.slotsPerNeuron
      });
    }
  }
  const motorNodes = [
    {
      id: GRID_IDS.leftMotor,
      role: "motor",
      position: { x: 0.28, y: 0.96 },
      branchCount: 1,
      maxInputSlots: params.slotsPerNeuron,
      maxOutputSlots: params.slotsPerNeuron
    },
    {
      id: GRID_IDS.rightMotor,
      role: "motor",
      position: { x: 0.72, y: 0.96 },
      branchCount: 1,
      maxInputSlots: params.slotsPerNeuron,
      maxOutputSlots: params.slotsPerNeuron
    }
  ];

  return Object.freeze({
    sensoryNodes: Object.freeze(sensoryNodes),
    interneuronNodes: Object.freeze(interneuronNodes),
    motorNodes: Object.freeze(motorNodes),
    synapses: Object.freeze([])
  });
}

function gridWorldCases(params) {
  const leftObject = 0;
  const centerObject = Math.floor((params.objectSensorCount - 1) / 2);
  const rightObject = params.objectSensorCount - 1;
  return [
    {
      label: "NUTRIENT_LEFT",
      world: { agent: 0.5, object: 0.18, nutrient: 0.18, toxin: 0.82, valence: "nutrient" },
      active: new Set([GRID_IDS.nutrientSense, gridObjectId(leftObject)]),
      targetOutputId: GRID_IDS.leftMotor
    },
    {
      label: "NUTRIENT_RIGHT",
      world: { agent: 0.5, object: 0.82, nutrient: 0.82, toxin: 0.18, valence: "nutrient" },
      active: new Set([GRID_IDS.nutrientSense, gridObjectId(rightObject)]),
      targetOutputId: GRID_IDS.rightMotor
    },
    {
      label: "TOXIN_LEFT",
      world: { agent: 0.5, object: 0.18, nutrient: 0.82, toxin: 0.18, valence: "toxin" },
      active: new Set([GRID_IDS.toxinSense, gridObjectId(leftObject)]),
      targetOutputId: GRID_IDS.rightMotor
    },
    {
      label: "TOXIN_RIGHT",
      world: { agent: 0.5, object: 0.82, nutrient: 0.18, toxin: 0.82, valence: "toxin" },
      active: new Set([GRID_IDS.toxinSense, gridObjectId(rightObject)]),
      targetOutputId: GRID_IDS.leftMotor
    },
    {
      label: "MIDPOINT_BOTH",
      world: { agent: 0.5, object: 0.5, nutrient: 0.18, toxin: 0.82, valence: "both" },
      active: new Set([GRID_IDS.nutrientSense, GRID_IDS.toxinSense, gridObjectId(centerObject)]),
      targetOutputId: null
    }
  ];
}

function markGridToxinTag(network, activeIds, params, config) {
  if (!activeIds.has(GRID_IDS.toxinSense)) return;

  for (const neuron of network.neurons) {
    if (neuron.role !== "sensory") continue;
    if (
      neuron.id === GRID_IDS.toxinSense ||
      (params.tagObjectStimWithToxin && neuron.id.startsWith("object") && activeIds.has(neuron.id))
    ) {
      neuron.tagLoad = 1;
    }
  }

  if (config.taggedDepotentiationMode === "specificFactor") {
    network.globalAversiveLoad += config.globalAversiveLoadIncrement;
  }
}

function runGridWorldStep(network, activeIds, params, config) {
  resetNetworkRuntime(network);
  setSensoryOutputs(network, activeIds);
  markGridToxinTag(network, activeIds, params, config);
  propagateAndIntegrateRole(network, "interneuron", config);
  clearSensoryOutputs(network);
  propagateAndIntegrateRole(network, "motor", config);
  return activeMotorIds(network);
}

function classifyGridWorld(activeOutputs, targetOutputId) {
  const active = Array.from(new Set(activeOutputs)).sort();
  const left = active.includes(GRID_IDS.leftMotor);
  const right = active.includes(GRID_IDS.rightMotor);
  if (left && right) return { action: "conflict", activeOutputs: active };
  if (!left && !right) return { action: "noop", activeOutputs: active };
  const outputId = left ? GRID_IDS.leftMotor : GRID_IDS.rightMotor;
  if (!targetOutputId) return { action: outputId === GRID_IDS.leftMotor ? "left" : "right", activeOutputs: active };
  return { action: outputId === targetOutputId ? "correct" : "wrong", activeOutputs: active };
}

function gridReward(action) {
  if (action === "correct") return 1;
  if (action === "wrong" || action === "conflict") return -1;
  return -0.1;
}

function trainGridWorldTrial(network, item, config, params, rng, baselineState) {
  let activeOutputs = runGridWorldStep(network, item.active, params, config);
  let classification = classifyGridWorld(activeOutputs, item.targetOutputId);

  if (classification.action === "noop" || classification.action === "conflict") {
    const forced = rng.next() < 0.5 ? GRID_IDS.leftMotor : GRID_IDS.rightMotor;
    activeOutputs = forceOutput(network, forced);
    classification = classifyGridWorld(activeOutputs, item.targetOutputId);
  }

  updateNetworkEligibility(network, config);
  const reward = gridReward(classification.action);
  const rewardAdvantage = reward - baselineState.baseline;
  applyRewardOutcomeLearning(network, rewardAdvantage, config);
  baselineState.baseline =
    baselineState.baseline * (1 - config.rewardAdvantageBaselineAlpha) +
    reward * config.rewardAdvantageBaselineAlpha;
  applyMaintenanceDecayAndCapture(network, config);
}

function gridWorldCaseSnapshot(network, item, params, config) {
  const clone = structuredClone(network);
  const activeOutputs = runGridWorldStep(clone, item.active, params, config);
  const classified = classifyGridWorld(activeOutputs, item.targetOutputId);
  const roles = new Map(clone.neurons.map((neuron) => [neuron.id, neuron.role]));
  const firingMediumIds = new Set(
    clone.neurons
      .filter((neuron) => neuron.role === "interneuron" && neuron.spike)
      .map((neuron) => neuron.id)
  );
  let liveStemEff = 0;
  let activeStemEff = 0;
  let leftDrive = 0;
  let rightDrive = 0;

  for (const synapse of clone.synapses) {
    if (synapse.state === "pruned" || synapse.state === "dormant") continue;
    const preRole = roles.get(synapse.preNeuronId);
    const postRole = roles.get(synapse.postNeuronId);

    if (preRole === "sensory" && postRole === "interneuron" && item.active.has(synapse.preNeuronId)) {
      liveStemEff += synapse.effectiveWeight;
      if (firingMediumIds.has(synapse.postNeuronId)) activeStemEff += synapse.effectiveWeight;
    }

    if (preRole === "interneuron" && postRole === "motor" && firingMediumIds.has(synapse.preNeuronId)) {
      if (synapse.postNeuronId === GRID_IDS.leftMotor) leftDrive += synapse.effectiveWeight;
      if (synapse.postNeuronId === GRID_IDS.rightMotor) rightDrive += synapse.effectiveWeight;
    }
  }

  return {
    label: item.label,
    world: item.world,
    targetOutputId: item.targetOutputId,
    activeInputs: [...item.active].sort(),
    action: classified.action,
    activeOutputs: classified.activeOutputs,
    firingMediumIds: [...firingMediumIds].sort(),
    firingMediumCount: firingMediumIds.size,
    liveStemEff,
    activeStemEff,
    leftDrive,
    rightDrive
  };
}

function evaluateGridWorld(network, cases, params, config) {
  const rows = cases
    .filter((item) => item.targetOutputId)
    .map((item) => gridWorldCaseSnapshot(network, item, params, config));
  const denom = Math.max(1, rows.length);
  return {
    sr: rows.filter((row) => row.action === "correct").length / denom,
    noop: rows.filter((row) => row.action === "noop").length / denom,
    conflict: rows.filter((row) => row.action === "conflict").length / denom,
    wrong: rows.filter((row) => row.action === "wrong").length / denom,
    left: rows.filter((row) => row.activeOutputs.includes(GRID_IDS.leftMotor)).length / denom,
    right: rows.filter((row) => row.activeOutputs.includes(GRID_IDS.rightMotor)).length / denom
  };
}

function gridWorldWeightTotals(network) {
  const roles = new Map(network.neurons.map((neuron) => [neuron.id, neuron.role]));
  const totals = [
    { outputId: GRID_IDS.leftMotor, live: 0, eff: 0, fast: 0, stable: 0 },
    { outputId: GRID_IDS.rightMotor, live: 0, eff: 0, fast: 0, stable: 0 }
  ];
  const byOutput = new Map(totals.map((item) => [item.outputId, item]));

  for (const synapse of network.synapses) {
    if (synapse.state === "pruned" || synapse.state === "dormant") continue;
    if (roles.get(synapse.preNeuronId) !== "interneuron" || roles.get(synapse.postNeuronId) !== "motor") continue;
    const total = byOutput.get(synapse.postNeuronId);
    if (!total) continue;
    total.live += 1;
    total.eff += synapse.effectiveWeight;
    total.fast += synapse.fastWeight;
    total.stable += synapse.stableWeight;
  }

  return totals;
}

function gridWorldCheckpoint(network, cases, params, config, cumulative, includeGraph) {
  return {
    metrics: evaluateGridWorld(network, cases, params, config),
    stats: stateCounts(network),
    cumulative: { ...cumulative },
    graph: includeGraph ? graphSnapshot(network) : null,
    cases: includeGraph ? cases.map((item) => gridWorldCaseSnapshot(network, item, params, config)) : [],
    weights: includeGraph ? gridWorldWeightTotals(network) : []
  };
}

function runGridWorldSeed(seed, params, includeGraph) {
  const config = createGridWorldConfig(params);
  const network = createLearningNetworkFromBlueprint(createGridWorldTopology(params), config);
  const cases = gridWorldCases(params);
  const trainCases = cases.filter((item) => item.targetOutputId);
  const rng = new SeededRandom(seed);
  const checkpoints = checkpointSet(params.epochs);
  const baselineState = { baseline: 0 };
  const cumulative = { formed: 0, activated: 0, dormant: 0, pruned: 0, tombstoneHit: 0 };
  const rows = [];

  for (let epoch = 1; epoch <= params.epochs; epoch += 1) {
    const formed = tryFormNearestLayeredConnections(
      network.neurons,
      network.synapses,
      network.pairMemory,
      network.tick,
      config,
      params.maxNewConnections
    );
    cumulative.formed += formed.formed;
    cumulative.tombstoneHit += formed.tombstoneHit;

    for (const item of shuffle(trainCases, rng)) {
      trainGridWorldTrial(network, item, config, params, rng, baselineState);
    }

    const state = updateConnectionStates(network.neurons, network.synapses, network.pairMemory, network.tick, config);
    cumulative.activated += state.activated;
    cumulative.dormant += state.dormant;
    cumulative.pruned += state.pruned;
    network.tick += 1;

    if (checkpoints.has(epoch)) {
      rows.push({
        epoch,
        ...gridWorldCheckpoint(network, cases, params, config, cumulative, includeGraph)
      });
    }
  }

  return {
    seed,
    checkpoints: rows,
    final: gridWorldCheckpoint(network, cases, params, config, cumulative, includeGraph)
  };
}

function aggregateGridWorld(seedRuns, params) {
  const first = seedRuns[0];
  const checkpoints = first.checkpoints.map((base, index) => {
    const rows = seedRuns.map((run) => run.checkpoints[index]).filter(Boolean);
    return {
      epoch: base.epoch,
      metrics: {
        sr: mean(rows, (row) => row.metrics.sr),
        noop: mean(rows, (row) => row.metrics.noop),
        conflict: mean(rows, (row) => row.metrics.conflict),
        wrong: mean(rows, (row) => row.metrics.wrong),
        left: mean(rows, (row) => row.metrics.left),
        right: mean(rows, (row) => row.metrics.right)
      },
      stats: aggregateStats(rows),
      cumulative: {
        formed: mean(rows, (row) => row.cumulative.formed),
        activated: mean(rows, (row) => row.cumulative.activated),
        dormant: mean(rows, (row) => row.cumulative.dormant),
        pruned: mean(rows, (row) => row.cumulative.pruned),
        tombstoneHit: mean(rows, (row) => row.cumulative.tombstoneHit)
      },
      graph: base.graph,
      cases: base.cases,
      weights: base.weights
    };
  });
  const finalRows = seedRuns.map((run) => run.final);
  return {
    mode: "gridWorld",
    params,
    seeds: seedRuns.map((run) => run.seed),
    checkpoints,
    final: {
      metrics: {
        sr: mean(finalRows, (row) => row.metrics.sr),
        noop: mean(finalRows, (row) => row.metrics.noop),
        conflict: mean(finalRows, (row) => row.metrics.conflict),
        wrong: mean(finalRows, (row) => row.metrics.wrong),
        left: mean(finalRows, (row) => row.metrics.left),
        right: mean(finalRows, (row) => row.metrics.right)
      },
      stats: aggregateStats(finalRows),
      solved: seedRuns.filter((run) => run.final.metrics.sr >= 0.99).length,
      stuck: seedRuns.filter((run) => run.final.metrics.noop >= 0.8).length
    }
  };
}

function simulate(payload) {
  const mode = payload.mode === "valence" || payload.mode === "gridWorld" ? payload.mode : "natural";
  if (mode === "valence") {
    const params = valenceParams(payload);
    const seeds = DEFAULT_SEEDS.slice(0, params.seedCount);
    return aggregateValence(seeds.map((seed, index) => runValenceSeed(seed, params, index === 0)), params);
  }
  if (mode === "gridWorld") {
    const params = gridWorldParams(payload);
    const seeds = DEFAULT_SEEDS.slice(0, params.seedCount);
    return aggregateGridWorld(seeds.map((seed, index) => runGridWorldSeed(seed, params, index === 0)), params);
  }
  const params = naturalParams(payload);
  const seeds = DEFAULT_SEEDS.slice(0, params.seedCount);
  return aggregateNatural(seeds.map((seed, index) => runNaturalSeed(seed, params, index === 0)), params);
}

async function readJson(req) {
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(chunk);
    if (Buffer.concat(chunks).length > 2_000_000) {
      throw new Error("Request body too large.");
    }
  }
  const text = Buffer.concat(chunks).toString("utf8");
  return text ? JSON.parse(text) : {};
}

function sendJson(res, status, body) {
  const json = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store"
  });
  res.end(json);
}

async function serveStatic(req, res) {
  const url = new URL(req.url, "http://localhost");
  const requestPath = url.pathname === "/" ? "/index.html" : url.pathname;
  const normalized = path.normalize(requestPath).replace(/^(\.\.[/\\])+/, "");
  const filePath = path.join(WEB_ROOT, normalized);
  if (!filePath.startsWith(WEB_ROOT)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  try {
    const data = await fs.readFile(filePath);
    const ext = path.extname(filePath);
    const type = ext === ".js"
      ? "text/javascript; charset=utf-8"
      : ext === ".css"
        ? "text/css; charset=utf-8"
        : "text/html; charset=utf-8";
    res.writeHead(200, { "content-type": type, "cache-control": "no-store" });
    res.end(data);
  } catch (error) {
    res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
    res.end("Not found");
  }
}

const server = http.createServer(async (req, res) => {
  try {
    if (req.method === "POST" && req.url === "/api/simulate") {
      const payload = await readJson(req);
      sendJson(res, 200, simulate(payload));
      return;
    }
    if (req.method === "GET" && req.url === "/api/health") {
      sendJson(res, 200, { ok: true });
      return;
    }
    if (req.method === "GET") {
      await serveStatic(req, res);
      return;
    }
    res.writeHead(405);
    res.end("Method not allowed");
  } catch (error) {
    sendJson(res, 500, {
      error: error instanceof Error ? error.message : String(error),
      stack: process.env.NODE_ENV === "production" ? undefined : error.stack
    });
  }
});

const port = parsePort(process.argv.slice(2));
server.listen(port, "127.0.0.1", () => {
  console.log(`DG-SNN WebUI listening on http://127.0.0.1:${port}`);
});
