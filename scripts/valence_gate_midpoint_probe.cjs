"use strict";
/*
 * Valence-gate midpoint probe.
 *
 * Integrates the EVAL_TODO E9 design conclusion into a small isolated test:
 * separate value gates from pure stimulus gates, then test midpoint behavior.
 *
 * Inputs:
 *   nutrientGate  — nutrient value gate, no tag
 *   toxinGate     — toxin value gate, emits toxin tag + optional hormone load
 *   objectStimA   — pure object/material stimulus, no side and no valence
 *   objectStimB   — pure object/material stimulus, no side and no valence
 *
 * Midpoint stimulus = toxinGate + nutrientGate + both pure object stimuli.
 * Toxin/nutrient motors are semantic choices, not left/right positions. Default
 * reward is zero for both contacts: this keeps the old negative-value channel A
 * silenced, as in E9.
 *
 * Run:
 *   npm run audit:valence-gate-midpoint
 *   TAGGED_MODE=off|taggedImpulse|specificFactor npm run audit:valence-gate-midpoint
 *   TAG_OBJECT_STIM_WITH_TOXIN=1 TAGGED_MODE=specificFactor npm run audit:valence-gate-midpoint
 *   ANALYZE=1 SEED_LIMIT=4 npm run audit:valence-gate-midpoint
 */
const path = require("node:path");

const ROOT = path.join(__dirname, "..");
const { defaultConfig, withConfig } = require(path.join(ROOT, "dist/src/config/newModelConfig"));
const { createLearningNetworkFromBlueprint } = require(path.join(ROOT, "dist/src/core/topologyBlueprint"));
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

function numberEnv(name, fallback) {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function signedNumberEnv(name, fallback) {
  if (process.env[name] === undefined || process.env[name] === "") return fallback;
  const value = Number(process.env[name]);
  return Number.isFinite(value) ? value : fallback;
}

const EPOCHS = numberEnv("EPOCHS", 300);
const TRIALS_PER_EPOCH = numberEnv("TRIALS_PER_EPOCH", 10);
const SEED_LIMIT = process.env.SEED_LIMIT ? Number(process.env.SEED_LIMIT) : null;
const DEFAULT_SEEDS = [21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171];
const SEEDS = SEED_LIMIT ? DEFAULT_SEEDS.slice(0, SEED_LIMIT) : DEFAULT_SEEDS;
const CHECKPOINTS = new Set(
  (process.env.CHECKPOINTS ? process.env.CHECKPOINTS.split(",").map(Number) : [1, 20, 50, 100, 200, 300])
    .filter(Number.isFinite)
    .concat(EPOCHS)
);
const TAGGED_MODE = process.env.TAGGED_MODE || "specificFactor";
const TAG_OBJECT_STIM_WITH_TOXIN =
  process.env.TAG_OBJECT_STIM_WITH_TOXIN === "1" || process.env.TAG_PURE_TOXIN_STIM === "1";
const TOXIN_REWARD = signedNumberEnv("TOXIN_REWARD", 0);
const NUTRIENT_REWARD = signedNumberEnv("NUTRIENT_REWARD", 0);
const BASELINE_ALPHA = numberEnv("BASELINE_ALPHA", 0);
const STEM_FANOUT = numberEnv("STEM_FANOUT", 5);
const MEDIUM_COUNT = numberEnv("MEDIUM_COUNT", 12);
const READOUT_FAST = numberEnv("READOUT_FAST", 0.35);
const GLOBAL_INCREMENT = signedNumberEnv("GLOBAL_INCREMENT", 1.0);
const GLOBAL_DECAY = numberEnv("GLOBAL_DECAY", 0.9);
const GLOBAL_THRESHOLD = signedNumberEnv("GLOBAL_THRESHOLD", 0.5);

const IDS = {
  nutrientGate: "nutrientGate",
  toxinGate: "toxinGate",
  objectStimA: "objectStimA",
  objectStimB: "objectStimB",
  toxinMotor: "toxinMotor",
  nutrientMotor: "nutrientMotor"
};

function createProbeConfig() {
  return withConfig({
    ...defaultConfig,
    leak: 1,
    branchLocalThreshold: 0.1,
    dendriteGateThreshold: 0.1,
    axonThreshold: 1,
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
    taggedDepotentiationMode: TAGGED_MODE,
    taggedCaptureGain: 1.0,
    globalAversiveLoadIncrement: GLOBAL_INCREMENT,
    globalAversiveLoadDecay: GLOBAL_DECAY,
    globalSensitizationThreshold: GLOBAL_THRESHOLD,
    rewardAdvantageBaselineAlpha: BASELINE_ALPHA
  });
}

function createGateTopology() {
  const sensoryNodes = [
    { id: IDS.toxinGate, role: "sensory", position: { x: 0, y: 0.0 }, maxOutputSlots: STEM_FANOUT },
    { id: IDS.objectStimA, role: "sensory", position: { x: 0, y: 0.35 }, maxOutputSlots: STEM_FANOUT },
    { id: IDS.objectStimB, role: "sensory", position: { x: 0, y: 0.65 }, maxOutputSlots: STEM_FANOUT },
    { id: IDS.nutrientGate, role: "sensory", position: { x: 0, y: 1.0 }, maxOutputSlots: STEM_FANOUT }
  ];
  const interneuronNodes = Array.from({ length: MEDIUM_COUNT }, (_, index) => ({
    id: `medium${index}`,
    role: "interneuron",
    position: { x: 1, y: MEDIUM_COUNT === 1 ? 0.5 : index / (MEDIUM_COUNT - 1) },
    branchCount: 1,
    maxInputSlots: 2,
    maxOutputSlots: 1
  }));
  const motorNodes = [
    { id: IDS.toxinMotor, role: "motor", position: { x: 2, y: 0 }, branchCount: 1, maxInputSlots: MEDIUM_COUNT },
    { id: IDS.nutrientMotor, role: "motor", position: { x: 2, y: 1 }, branchCount: 1, maxInputSlots: MEDIUM_COUNT }
  ];
  const synapses = [];

  for (const sensory of sensoryNodes) {
    for (const medium of nearestNodes(sensory, interneuronNodes, Math.min(STEM_FANOUT, interneuronNodes.length))) {
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
      fastWeight: READOUT_FAST,
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

function nearestNodes(source, targets, limit) {
  return [...targets]
    .sort((a, b) => {
      const distance = Math.abs(source.position.y - a.position.y) - Math.abs(source.position.y - b.position.y);
      return distance === 0 ? a.id.localeCompare(b.id) : distance;
    })
    .slice(0, limit);
}

function midpointInputs() {
  return new Set([IDS.toxinGate, IDS.objectStimA, IDS.objectStimB, IDS.nutrientGate]);
}

function objectStimInputs() {
  return new Set([IDS.objectStimA, IDS.objectStimB]);
}

function toxinObjectInputs() {
  return new Set([IDS.toxinGate, IDS.objectStimA, IDS.objectStimB]);
}

function nutrientObjectInputs() {
  return new Set([IDS.nutrientGate, IDS.objectStimA, IDS.objectStimB]);
}

function stimulationCases() {
  return [
    { label: "MIDPOINT", active: midpointInputs() },
    { label: "TOXIN_OBJECT", active: toxinObjectInputs() },
    { label: "NUTRI_OBJECT", active: nutrientObjectInputs() },
    { label: "PURE_OBJECT", active: objectStimInputs() },
    { label: "TOXIN_GATE", active: new Set([IDS.toxinGate]) },
    { label: "NUTR_GATE", active: new Set([IDS.nutrientGate]) },
    { label: "OBJECT_A", active: new Set([IDS.objectStimA]) },
    { label: "OBJECT_B", active: new Set([IDS.objectStimB]) }
  ];
}

function markToxinTag(network, activeIds, config) {
  const toxinPresent = activeIds.has(IDS.toxinGate);
  if (!toxinPresent) return;
  const taggedObjectStimuli = new Set([IDS.objectStimA, IDS.objectStimB]);

  for (const neuron of network.neurons) {
    if (neuron.role !== "sensory") continue;
    if (
      neuron.id === IDS.toxinGate ||
      (TAG_OBJECT_STIM_WITH_TOXIN && taggedObjectStimuli.has(neuron.id) && activeIds.has(neuron.id))
    ) {
      neuron.tagLoad = 1;
    }
  }

  if (config.taggedDepotentiationMode === "specificFactor") {
    network.globalAversiveLoad += config.globalAversiveLoadIncrement;
  }
}

function runNetworkStep(network, activeIds, config) {
  resetNetworkRuntime(network);
  setSensoryOutputs(network, activeIds);
  markToxinTag(network, activeIds, config);
  propagateAndIntegrateRole(network, "interneuron", config);
  clearSensoryOutputs(network);
  propagateAndIntegrateRole(network, "motor", config);
  return activeMotorIds(network);
}

function classify(activeOutputs) {
  const active = Array.from(new Set(activeOutputs)).sort();
  const toxin = active.includes(IDS.toxinMotor);
  const nutrient = active.includes(IDS.nutrientMotor);
  if (toxin && nutrient) return { action: "conflict", activeOutputs: active, contact: null };
  if (toxin) return { action: "toxin", activeOutputs: active, contact: "toxin" };
  if (nutrient) return { action: "nutrient", activeOutputs: active, contact: "nutrient" };
  return { action: "noop", activeOutputs: active, contact: null };
}

function forceOutput(network, motorId) {
  for (const neuron of network.neurons) {
    if (neuron.role === "motor") {
      neuron.outputSignal = neuron.id === motorId ? 1 : 0;
      neuron.spike = neuron.id === motorId;
    }
  }
  return activeMotorIds(network);
}

function rewardForContact(contact) {
  if (contact === "toxin") return TOXIN_REWARD;
  if (contact === "nutrient") return NUTRIENT_REWARD;
  return 0;
}

function runTrainingTrial(network, config, rng, baselineState) {
  const activeIds = midpointInputs();
  let activeOutputs = runNetworkStep(network, activeIds, config);
  let choice = classify(activeOutputs);

  if (choice.action === "noop" || choice.action === "conflict") {
    const forced = rng.next() < 0.5 ? IDS.toxinMotor : IDS.nutrientMotor;
    activeOutputs = forceOutput(network, forced);
    choice = classify(activeOutputs);
  }

  updateNetworkEligibility(network, config);
  const reward = rewardForContact(choice.contact);
  const rewardAdvantage = reward - baselineState.baseline;
  applyRewardOutcomeLearning(network, rewardAdvantage, config);
  baselineState.baseline =
    baselineState.baseline * (1 - config.rewardAdvantageBaselineAlpha) +
    reward * config.rewardAdvantageBaselineAlpha;
  applyMaintenanceDecayAndCapture(network, config);

  return choice;
}

function snapshotCase(network, activeIds, config) {
  const clone = structuredClone(network);
  const activeOutputs = runNetworkStep(clone, activeIds, config);
  const choice = classify(activeOutputs);
  const roles = new Map(clone.neurons.map((neuron) => [neuron.id, neuron.role]));
  const firingMediumIds = new Set(
    clone.neurons
      .filter((neuron) => neuron.role === "interneuron" && neuron.spike)
      .map((neuron) => neuron.id)
  );
  const activeInputIds = new Set(activeIds);
  let toxinDrive = 0;
  let nutrientDrive = 0;
  let activeStemEff = 0;
  let liveStemEff = 0;
  const firingFromInput = new Map();

  for (const synapse of clone.synapses) {
    if (synapse.state === "pruned" || synapse.state === "dormant") continue;
    const preRole = roles.get(synapse.preNeuronId);
    const postRole = roles.get(synapse.postNeuronId);

    if (preRole === "sensory" && postRole === "interneuron" && activeInputIds.has(synapse.preNeuronId)) {
      liveStemEff += synapse.effectiveWeight;
      if (firingMediumIds.has(synapse.postNeuronId)) {
        activeStemEff += synapse.effectiveWeight;
        firingFromInput.set(synapse.preNeuronId, (firingFromInput.get(synapse.preNeuronId) ?? 0) + 1);
      }
    }

    if (preRole === "interneuron" && postRole === "motor" && firingMediumIds.has(synapse.preNeuronId)) {
      if (synapse.postNeuronId === IDS.toxinMotor) toxinDrive += synapse.effectiveWeight;
      if (synapse.postNeuronId === IDS.nutrientMotor) nutrientDrive += synapse.effectiveWeight;
    }
  }

  return {
    action: choice.action,
    activeOutputs: choice.activeOutputs,
    firingMediumCount: firingMediumIds.size,
    liveStemEff,
    activeStemEff,
    toxinDrive,
    nutrientDrive,
    firingFromInput
  };
}

function runSeed(seed) {
  const config = createProbeConfig();
  const network = createLearningNetworkFromBlueprint(createGateTopology(), config);
  const rng = new SeededRandom(seed);
  const baselineState = { baseline: 0 };
  const checkpoints = [];

  for (let epoch = 1; epoch <= EPOCHS; epoch += 1) {
    for (let trial = 0; trial < TRIALS_PER_EPOCH; trial += 1) {
      runTrainingTrial(network, config, rng, baselineState);
    }

    if (CHECKPOINTS.has(epoch)) {
      const mid = snapshotCase(network, midpointInputs(), config);
      checkpoints.push({
        epoch,
        midAction: mid.action,
        midToxinDrive: mid.toxinDrive,
        midNutrientDrive: mid.nutrientDrive
      });
    }
  }

  const snapshots = stimulationCases().map((item) => ({
    label: item.label,
    ...snapshotCase(network, item.active, config)
  }));
  return { seed, checkpoints, snapshots, network };
}

function mean(items, selector) {
  return items.length === 0 ? 0 : items.reduce((sum, item) => sum + selector(item), 0) / items.length;
}

function fmt(value, digits = 3) {
  return Number.isFinite(value) ? value.toFixed(digits) : String(value);
}

function aggregate(results) {
  const epochs = [...CHECKPOINTS].sort((a, b) => a - b);
  const rows = epochs.map((epoch) => {
    const pts = results.map((result) => result.checkpoints.find((item) => item.epoch === epoch)).filter(Boolean);
    const n = Math.max(1, pts.length);
    return {
      epoch,
      n: pts.length,
      toxin: pts.filter((item) => item.midAction === "toxin").length / n,
      nutrient: pts.filter((item) => item.midAction === "nutrient").length / n,
      noop: pts.filter((item) => item.midAction === "noop").length / n,
      conflict: pts.filter((item) => item.midAction === "conflict").length / n,
      toxinDrive: mean(pts, (item) => item.midToxinDrive),
      nutrientDrive: mean(pts, (item) => item.midNutrientDrive)
    };
  }).filter((row) => row.n > 0);

  const labels = stimulationCases().map((item) => item.label);
  const snapshotRows = labels.map((label) => {
    const pts = results.map((result) => result.snapshots.find((item) => item.label === label)).filter(Boolean);
    const n = Math.max(1, pts.length);
    return {
      label,
      toxin: pts.filter((item) => item.action === "toxin").length / n,
      nutrient: pts.filter((item) => item.action === "nutrient").length / n,
      noop: pts.filter((item) => item.action === "noop").length / n,
      conflict: pts.filter((item) => item.action === "conflict").length / n,
      firingMediumCount: mean(pts, (item) => item.firingMediumCount),
      toxinDrive: mean(pts, (item) => item.toxinDrive),
      nutrientDrive: mean(pts, (item) => item.nutrientDrive),
      liveStemEff: mean(pts, (item) => item.liveStemEff),
      activeStemEff: mean(pts, (item) => item.activeStemEff)
    };
  });

  return { rows, snapshotRows, results };
}

function printReport(report) {
  console.log("=== valence-gate midpoint probe ===");
  console.log(
    `seeds=${SEEDS.join(",")} epochs=${EPOCHS} trialsPerEpoch=${TRIALS_PER_EPOCH}` +
    ` taggedMode=${TAGGED_MODE} tagObjectStimWithToxin=${TAG_OBJECT_STIM_WITH_TOXIN ? "1" : "0"}`
  );
  console.log(
    `toxinReward=${TOXIN_REWARD} nutrientReward=${NUTRIENT_REWARD} baselineAlpha=${BASELINE_ALPHA}` +
    ` stemFanout=${STEM_FANOUT} medium=${MEDIUM_COUNT} readoutFast=${READOUT_FAST}`
  );
  console.log("Midpoint active inputs: toxinGate + objectStimA + objectStimB + nutrientGate.");
  console.log("epoch   toxin   nutrient noop   confl  toxinD nutrD");
  for (const row of report.rows) {
    console.log(
      `${String(row.epoch).padStart(5)}  ${fmt(row.toxin)}   ${fmt(row.nutrient)}     ${fmt(row.noop)}  ` +
      `${fmt(row.conflict)}  ${fmt(row.toxinDrive)}  ${fmt(row.nutrientDrive)}`
    );
  }

  console.log("\nfinal frozen stimulation snapshots:");
  console.log("case           toxin nutr  noop  confl fireN stemEff actStem toxinD nutrD");
  for (const row of report.snapshotRows) {
    console.log(
      `${row.label.padEnd(14)} ${fmt(row.toxin)} ${fmt(row.nutrient)} ${fmt(row.noop)} ${fmt(row.conflict)} ` +
      `${fmt(row.firingMediumCount, 1)}   ${fmt(row.liveStemEff, 2)}   ${fmt(row.activeStemEff, 2)}    ` +
      `${fmt(row.toxinDrive, 2)}  ${fmt(row.nutrientDrive, 2)}`
    );
  }

  if (process.env.ANALYZE === "1") {
    console.log("\nper-seed midpoint and stimulation details:");
    for (const result of report.results) {
      const midpoint = result.snapshots.find((item) => item.label === "MIDPOINT");
      console.log(
        `  ${String(result.seed).padStart(3)} midpoint=${midpoint.action}` +
        ` outputs=${midpoint.activeOutputs.length ? midpoint.activeOutputs.join("|") : "-"}` +
        ` toxinD=${fmt(midpoint.toxinDrive, 2)} nutrD=${fmt(midpoint.nutrientDrive, 2)}`
      );
      for (const snap of result.snapshots) {
        const firing = [...snap.firingFromInput.entries()]
          .map(([id, count]) => `${id}:${count}`)
          .join(",");
        console.log(
          `      ${snap.label.padEnd(14)} ${snap.action.padEnd(8)}` +
          ` out=${snap.activeOutputs.length ? snap.activeOutputs.join("|") : "-"}` +
          ` fireN=${snap.firingMediumCount} toxinD=${fmt(snap.toxinDrive, 2)} nutrD=${fmt(snap.nutrientDrive, 2)}` +
          ` firingByInput=${firing || "-"}`
        );
      }
    }
  }
}

printReport(aggregate(SEEDS.map(runSeed)));
