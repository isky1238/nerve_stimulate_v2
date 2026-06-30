"use strict";
/*
 * Developmental topology probe — spontaneous connection + disconnect marking.
 *
 * Complement to topology_family_probe.cjs. That probe asks "can a GIVEN graph
 * learn an input->output mapping?". This probe asks: "can neurons placed in 2D
 * space form a stable readout trunk via proximity-based spontaneous connection
 * (tryFormConnections) + disconnect marking (updateConnectionStates), starting
 * from a graph with NO pre-built readout?"
 *
 * Two initial-network variants, run as contrast:
 *   - stem:  sensory->inter stem pre-wired (decayProtected), inter->motor
 *            readout EMPTY, grown spontaneously. Avoids the bootstrap deadlock
 *            (stem provides sensory drive so reward/eligibility can flow).
 *   - empty: no edges at all; both stem and readout must grow. Likely
 *            bootstrap-deadlocks (no initial readout -> no reward -> no
 *            eligibility -> can't learn); reported as EXPECTED, not a failure.
 *
 * Each epoch: run shuffled reward-only trials (same machinery as
 * topology_family_probe), then one developmental step (tryFormConnections +
 * updateConnectionStates), then optional checkpoint. Records the usual
 * SR/noop/conflict/wrong + readout weight map, PLUS developmental metrics
 * (formed/activated/dormant/pruned/tombstoneHit/meanConnDist/readoutCount).
 *
 * Run:
 *   npm run audit:topology-development
 *   EPOCHS=200 SEED_LIMIT=8 npm run audit:topology-development
 *   VARIANT=stem  EPOCHS=200 SEED_LIMIT=8 npm run audit:topology-development
 *   VARIANT=empty EPOCHS=200 SEED_LIMIT=8 npm run audit:topology-development
 *   FULL=1 CASES=dev_1_5_1_stem npm run audit:topology-development
 */
const path = require("node:path");

const ROOT = path.join(__dirname, "..");
const { defaultConfig, withConfig } = require(path.join(ROOT, "dist/src/config/newModelConfig"));
const {
  createNearestLayeredTopologyBlueprint
} = require(path.join(ROOT, "dist/src/core/layeredTopologyBlueprint"));
const { createLearningNetworkFromBlueprint } = require(path.join(ROOT, "dist/src/core/topologyBlueprint"));
const {
  tryFormConnections,
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

const VARIANTS = ["stem", "empty"];

const DEFAULT_CASES = [
  {
    id: "dev_1_5_1",
    description: "1/5/1 minimal proportional family",
    inputCount: 1,
    mediumCount: 5,
    outputCount: 1,
    synapsesPerInput: 5,
    synapsesPerMedium: 1
  },
  {
    id: "dev_2_10_2",
    description: "2/10/2 = 2x of 1/5/1",
    inputCount: 2,
    mediumCount: 10,
    outputCount: 2,
    synapsesPerInput: 5,
    synapsesPerMedium: 1
  }
];

const DEFAULT_SEEDS = [21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171];

function numberEnv(name, fallback) {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function listEnv(name, fallback) {
  if (!process.env[name]) return fallback;
  const values = process.env[name]
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  return values.length ? values : fallback;
}

const EPOCHS = numberEnv("EPOCHS", 200);
const SEED_LIMIT = process.env.SEED_LIMIT ? Number(process.env.SEED_LIMIT) : null;
const SEEDS = SEED_LIMIT ? DEFAULT_SEEDS.slice(0, SEED_LIMIT) : DEFAULT_SEEDS;
const CHECKPOINTS = new Set(
  listEnv("CHECKPOINTS", ["1", "20", "50", "100", "150", "200"])
    .map(Number)
    .filter(Number.isFinite)
    .concat(EPOCHS)
);
const MAX_NEW_CONN = numberEnv("MAX_NEW_CONN", 8);
const VARIANT_FILTER = process.env.VARIANT ? new Set([process.env.VARIANT]) : new Set(VARIANTS);
const CASE_FILTER = listEnv("CASES", null);

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
    depotentiationRate: 0.64
  });
}

function patternsForCase(testCase) {
  return Array.from({ length: testCase.inputCount }, (_, index) => ({
    inputId: `input${index}`,
    targetOutputId: `output${Math.min(index, testCase.outputCount - 1)}`
  }));
}

function runNetworkStep(network, inputId, config) {
  resetNetworkRuntime(network);
  setSensoryOutputs(network, new Set([inputId]));
  propagateAndIntegrateRole(network, "interneuron", config);
  clearSensoryOutputs(network);
  propagateAndIntegrateRole(network, "motor", config);
  return activeMotorIds(network);
}

function classifyOutputs(activeOutputs, targetOutputId) {
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

function rewardForClassification(classification) {
  if (classification.action === "correct") return 1;
  if (classification.action === "wrong" || classification.action === "conflict") return -1;
  return -0.1;
}

function chooseExplorationOutput(network, rng) {
  const outputs = network.neurons.filter((neuron) => neuron.role === "motor").map((neuron) => neuron.id).sort();
  return outputs[rng.nextInt(outputs.length)];
}

function runTrainingTrial(network, pattern, config, rng, baselineState) {
  let activeOutputs = runNetworkStep(network, pattern.inputId, config);
  let classification = classifyOutputs(activeOutputs, pattern.targetOutputId);
  let explorationOutput = null;

  if (classification.action === "noop" || classification.action === "conflict") {
    explorationOutput = chooseExplorationOutput(network, rng);
    activeOutputs = forceOutput(network, explorationOutput);
    classification = classifyOutputs(activeOutputs, pattern.targetOutputId);
  }

  updateNetworkEligibility(network, config);
  const reward = rewardForClassification(classification);
  const rewardAdvantage = reward - baselineState.baseline;
  const rewardUpdates = applyRewardOutcomeLearning(network, rewardAdvantage, config);
  baselineState.baseline =
    baselineState.baseline * (1 - config.rewardAdvantageBaselineAlpha) +
    reward * config.rewardAdvantageBaselineAlpha;
  const maintenance = applyMaintenanceDecayAndCapture(network, config);

  return {
    reward,
    rewardAdvantage,
    classification: classification.action,
    explorationOutput,
    rewardUpdates,
    captureUpdates: maintenance.captureUpdates,
    decayUpdates: maintenance.decayUpdates
  };
}

function evaluateNetwork(network, patterns, config) {
  const rows = [];

  for (const pattern of patterns) {
    const clone = structuredClone(network);
    const activeOutputs = runNetworkStep(clone, pattern.inputId, config);
    const classification = classifyOutputs(activeOutputs, pattern.targetOutputId);
    rows.push({
      ...pattern,
      classification: classification.action,
      activeOutputs: classification.activeOutputs
    });
  }

  const n = Math.max(1, rows.length);
  return {
    successRate: rows.filter((row) => row.classification === "correct").length / n,
    noopRate: rows.filter((row) => row.classification === "noop").length / n,
    conflictRate: rows.filter((row) => row.classification === "conflict").length / n,
    wrongOnlyRate: rows.filter((row) => row.classification === "wrong").length / n,
    rows
  };
}

function readoutWeightMap(network, patterns) {
  const roles = new Map(network.neurons.map((neuron) => [neuron.id, neuron.role]));
  const byId = new Map(network.neurons.map((neuron) => [neuron.id, neuron]));
  const stems = network.synapses.filter((synapse) =>
    roles.get(synapse.preNeuronId) === "sensory" && roles.get(synapse.postNeuronId) === "interneuron"
  );
  const readouts = network.synapses.filter((synapse) =>
    roles.get(synapse.preNeuronId) === "interneuron" && roles.get(synapse.postNeuronId) === "motor"
  );

  return patterns.map((pattern) => {
    const activeMediumIds = new Set(
      stems
        .filter((synapse) => synapse.preNeuronId === pattern.inputId)
        .map((synapse) => synapse.postNeuronId)
    );
    let correctEff = 0;
    const wrongByOutput = new Map();

    for (const synapse of readouts) {
      if (!activeMediumIds.has(synapse.preNeuronId)) continue;
      if (synapse.postNeuronId === pattern.targetOutputId) {
        correctEff += synapse.effectiveWeight;
      } else {
        wrongByOutput.set(
          synapse.postNeuronId,
          (wrongByOutput.get(synapse.postNeuronId) ?? 0) + Math.abs(synapse.effectiveWeight)
        );
      }
    }

    return {
      ...pattern,
      activeMediumCount: activeMediumIds.size,
      correctEff,
      wrongMaxEff: Math.max(0, ...wrongByOutput.values())
    };
  });
}

// Distance between a synapse's pre/post neurons (Euclidean on position).
function synapseDistance(synapse, byId) {
  const pre = byId.get(synapse.preNeuronId);
  const post = byId.get(synapse.postNeuronId);
  if (!pre || !post) return 0;
  const dx = pre.position.x - post.position.x;
  const dy = pre.position.y - post.position.y;
  return Math.sqrt(dx * dx + dy * dy);
}

// Count live (non-pruned) inter->motor readout synapses.
function countReadoutSynapses(network) {
  const roles = new Map(network.neurons.map((neuron) => [neuron.id, neuron.role]));
  return network.synapses.filter(
    (synapse) =>
      roles.get(synapse.preNeuronId) === "interneuron" &&
      roles.get(synapse.postNeuronId) === "motor" &&
      synapse.state !== "pruned"
  ).length;
}

function emptyDevMetrics() {
  return { formed: 0, activated: 0, dormant: 0, pruned: 0, tombstoneHit: 0, meanConnDist: 0, readoutCount: 0 };
}

function runCaseSeed(testCase, variant, seed) {
  const config = createProbeConfig();
  const topology = createNearestLayeredTopologyBlueprint({
    inputCount: testCase.inputCount,
    mediumCount: testCase.mediumCount,
    outputCount: testCase.outputCount,
    synapsesPerInput: testCase.synapsesPerInput,
    synapsesPerMedium: testCase.synapsesPerMedium,
    readoutMode: variant
  });
  const network = createLearningNetworkFromBlueprint(topology, config);
  const patterns = patternsForCase(testCase);
  const rng = new SeededRandom(seed);
  // Separate RNG for developmental stochasticity so trial outcomes and wiring
  // stay independently reproducible.
  const devRng = new SeededRandom(seed + 7919);
  const baselineState = { baseline: 0 };
  const checkpoints = [];

  for (let epoch = 1; epoch <= EPOCHS; epoch += 1) {
    const shuffled = shufflePatterns(patterns, rng);

    for (const pattern of shuffled) {
      runTrainingTrial(network, pattern, config, rng, baselineState);
    }

    // Developmental step: grow new connections, then update connection states
    // (candidate->active, weak->dormant, dormant->prune). Snapshot synapse
    // count before/after to measure newly formed ones and their distances.
    const beforeIds = new Set(network.synapses.map((synapse) => synapse.id));
    const formedMetrics = tryFormConnections(
      network.neurons,
      network.synapses,
      network.pairMemory,
      network.tick,
      config,
      devRng,
      MAX_NEW_CONN
    );
    const newSynapses = network.synapses.filter((synapse) => !beforeIds.has(synapse.id));
    const stateMetrics = updateConnectionStates(
      network.neurons,
      network.synapses,
      network.pairMemory,
      network.tick,
      config
    );
    network.tick += 1;

    const byId = new Map(network.neurons.map((neuron) => [neuron.id, neuron]));
    const meanConnDist = newSynapses.length === 0
      ? 0
      : newSynapses.reduce((sum, synapse) => sum + synapseDistance(synapse, byId), 0) / newSynapses.length;
    const dev = {
      formed: formedMetrics.formed,
      activated: stateMetrics.activated,
      dormant: stateMetrics.dormant,
      pruned: stateMetrics.pruned,
      tombstoneHit: formedMetrics.tombstoneHit,
      meanConnDist,
      readoutCount: countReadoutSynapses(network)
    };

    if (CHECKPOINTS.has(epoch)) {
      checkpoints.push({
        epoch,
        eval: evaluateNetwork(network, patterns, config),
        weights: readoutWeightMap(network, patterns),
        dev
      });
    } else {
      // Keep last dev metrics attached for final reporting even if not a checkpoint.
      checkpoints._lastDev = dev;
    }
  }

  return {
    seed,
    variant,
    final: evaluateNetwork(network, patterns, config),
    weights: readoutWeightMap(network, patterns),
    finalDev: countReadoutSynapses(network),
    checkpoints
  };
}

function shufflePatterns(patterns, rng) {
  const shuffled = [...patterns];
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = rng.nextInt(index + 1);
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }
  return shuffled;
}

function mean(items, selector) {
  return items.length === 0 ? 0 : items.reduce((sum, item) => sum + selector(item), 0) / items.length;
}

function fmt(value, digits = 3) {
  return Number.isFinite(value) ? value.toFixed(digits) : String(value);
}

function aggregateCase(testCase, variant, results) {
  const checkpointEpochs = [...CHECKPOINTS].sort((a, b) => a - b);
  const rows = checkpointEpochs.map((epoch) => {
    const pts = results
      .map((result) => result.checkpoints.find((checkpoint) => checkpoint.epoch === epoch))
      .filter(Boolean);
    return {
      epoch,
      n: pts.length,
      sr: mean(pts, (item) => item.eval.successRate),
      noop: mean(pts, (item) => item.eval.noopRate),
      conflict: mean(pts, (item) => item.eval.conflictRate),
      wrongOnly: mean(pts, (item) => item.eval.wrongOnlyRate),
      correctEff: mean(pts.flatMap((item) => item.weights), (item) => item.correctEff),
      wrongMaxEff: mean(pts.flatMap((item) => item.weights), (item) => item.wrongMaxEff),
      formed: mean(pts, (item) => item.dev.formed),
      activated: mean(pts, (item) => item.dev.activated),
      dormant: mean(pts, (item) => item.dev.dormant),
      pruned: mean(pts, (item) => item.dev.pruned),
      tombstone: mean(pts, (item) => item.dev.tombstoneHit),
      meanConnDist: mean(pts, (item) => item.dev.meanConnDist),
      readoutN: mean(pts, (item) => item.dev.readoutCount)
    };
  }).filter((row) => row.n > 0);

  const finalNoop = mean(results, (item) => item.final.noopRate);
  const finalReadoutN = mean(results, (item) => item.finalDev);
  // Bootstrap-deadlock signature for the empty variant: noop ~1 across the
  // whole run AND no readout ever formed. This is an EXPECTED outcome, not a
  // failure (user principle #4: don't mistake "pathway not yet formed" for a
  // bug). We label it so the report can't be misread as a regression.
  const lastRow = rows[rows.length - 1];
  const deadlock = variant === "empty"
    && finalNoop >= 0.95
    && finalReadoutN === 0
    && (lastRow ? lastRow.readoutN === 0 : true);

  return {
    testCase,
    variant,
    rows,
    final: {
      sr: mean(results, (item) => item.final.successRate),
      noop: finalNoop,
      conflict: mean(results, (item) => item.final.conflictRate),
      wrongOnly: mean(results, (item) => item.final.wrongOnlyRate),
      correctEff: mean(results.flatMap((item) => item.weights), (item) => item.correctEff),
      wrongMaxEff: mean(results.flatMap((item) => item.weights), (item) => item.wrongMaxEff),
      readoutN: finalReadoutN,
      solved: results.filter((item) => item.final.successRate >= 0.99).length,
      stuck: results.filter((item) => item.final.noopRate >= 0.8).length
    },
    deadlock,
    results
  };
}

function printCaseReport(report) {
  const { testCase, variant } = report;
  console.log(`\n=== ${testCase.id}_${variant} ===`);
  console.log(`${testCase.description} variant=${variant} counts=${testCase.inputCount}/${testCase.mediumCount}/${testCase.outputCount}`);
  console.log(
    "epoch   SR     noop   confl  wrong  formed activ dorm  pruned tomb  meanDist readoutN"
  );
  for (const row of report.rows) {
    console.log(
      `${String(row.epoch).padStart(5)}  ${fmt(row.sr)}  ${fmt(row.noop)}  ${fmt(row.conflict)}  ` +
      `${fmt(row.wrongOnly)}  ${fmt(row.formed)}  ${fmt(row.activated)}  ${fmt(row.dormant)}  ` +
      `${fmt(row.pruned)}  ${fmt(row.tombstone)}  ${fmt(row.meanConnDist)}  ${fmt(row.readoutN)}`
    );
  }
  console.log(
    `final seeds=${report.results.length} SR=${fmt(report.final.sr)} noop=${fmt(report.final.noop)}` +
    ` conflict=${fmt(report.final.conflict)} wrong=${fmt(report.final.wrongOnly)}` +
    ` solved=${report.final.solved}/${report.results.length} stuck=${report.final.stuck}/${report.results.length}` +
    ` readoutN=${fmt(report.final.readoutN)} correctEff=${fmt(report.final.correctEff)} wrongEff=${fmt(report.final.wrongMaxEff)}`
  );
  if (report.deadlock) {
    console.log("* expected: empty variant bootstrap-deadlock (no seed -> no readout -> no reward -> no learning)");
  }

  if (process.env.FULL === "1") {
    console.log("single-stim signatures per seed (C=correct,N=noop,X=conflict,W=wrong):");
    for (const result of report.results) {
      const sig = result.final.rows.map((row) => {
        if (row.classification === "correct") return "C";
        if (row.classification === "noop") return "N";
        if (row.classification === "conflict") return "X";
        return "W";
      }).join("");
      const weights = result.weights
        .map((row) => `${row.inputId}->${row.targetOutputId}:c=${fmt(row.correctEff, 2)},w=${fmt(row.wrongMaxEff, 2)}`)
        .join(" ");
      console.log(`  ${String(result.seed).padStart(3)} ${sig} ${weights}`);
    }
  }
}

function main() {
  console.log("=== developmental topology probe ===");
  console.log(`seeds=${SEEDS.join(",")} epochs=${EPOCHS} checkpoints=${[...CHECKPOINTS].sort((a, b) => a - b).join(",")} maxNewConn=${MAX_NEW_CONN}`);
  console.log("Each epoch: reward-only trials -> tryFormConnections -> updateConnectionStates.");
  console.log("Stem variant tests readout growth from a sensory stem; empty variant tests full spontaneous wiring (likely bootstrap-deadlock).");

  for (const testCase of DEFAULT_CASES) {
    for (const variant of VARIANTS) {
      if (!VARIANT_FILTER.has(variant)) continue;
      const caseId = `${testCase.id}_${variant}`;
      if (CASE_FILTER && !CASE_FILTER.includes(caseId)) continue;
      const results = SEEDS.map((seed) => runCaseSeed(testCase, variant, seed));
      printCaseReport(aggregateCase(testCase, variant, results));
    }
  }
}

main();
