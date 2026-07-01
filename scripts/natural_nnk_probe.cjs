"use strict";
/*
 * Natural m/n/o/k topology probe.
 *
 * This is the morphology baseline the graded valence probe is NOT:
 *   - m sensory / n medium / o motor neurons
 *   - every neuron has k input slots and k output slots
 *   - no prewired stem/readout
 *   - fixed 2D layer positions
 *   - nearest adjacent-layer connections form naturally
 *   - weak unused non-stable links can become dormant/pruned
 *
 * Run:
 *   npm run audit:natural-nnk
 *   N=5 K=2 EPOCHS=200 SEED_LIMIT=8 npm run audit:natural-nnk
 *   M=2 N=10 O=2 K=5 EPOCHS=100 SEED_LIMIT=4 npm run audit:natural-nnk
 *   M=2 N=10 O=5 K=5 EPOCHS=100 SEED_LIMIT=4 FULL=1 npm run audit:natural-nnk
 *   M=2 N=10 O=2 K=5 L=2 EPOCHS=100 SEED_LIMIT=8 npm run audit:natural-nnk
 *   FAST_INIT=1.05 AXON_THRESHOLD=1 N=5 K=2 npm run audit:natural-nnk
 *   FULL=1 N=5 K=2 npm run audit:natural-nnk
 */
const path = require("node:path");

const ROOT = path.join(__dirname, "..");
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

function numberEnv(name, fallback) {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function optionalNumberEnv(name) {
  if (process.env[name] === undefined || process.env[name] === "") return null;
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value > 0 ? value : null;
}

function listEnv(name, fallback) {
  if (!process.env[name]) return fallback;
  const values = process.env[name].split(",").map((item) => item.trim()).filter(Boolean);
  return values.length ? values : fallback;
}

const LEGACY_LAYER_SIZE = (process.env.M === undefined && process.env.O === undefined)
  ? numberEnv("N", 5)
  : numberEnv("LAYER_SIZE", 5);
const LAYER_MULTIPLIER = numberEnv("L", numberEnv("SCALE", 1));
const BASE_INPUT_COUNT = optionalNumberEnv("INPUT_COUNT") ?? optionalNumberEnv("M") ?? LEGACY_LAYER_SIZE;
const BASE_MEDIUM_COUNT = optionalNumberEnv("MEDIUM_COUNT") ?? ((process.env.M !== undefined || process.env.O !== undefined) ? optionalNumberEnv("N") : null) ?? LEGACY_LAYER_SIZE;
const BASE_OUTPUT_COUNT = optionalNumberEnv("OUTPUT_COUNT") ?? optionalNumberEnv("O") ?? LEGACY_LAYER_SIZE;
const INPUT_COUNT = BASE_INPUT_COUNT * LAYER_MULTIPLIER;
const MEDIUM_COUNT = BASE_MEDIUM_COUNT * LAYER_MULTIPLIER;
const OUTPUT_COUNT = BASE_OUTPUT_COUNT * LAYER_MULTIPLIER;
const K = numberEnv("K", 2);
const EPOCHS = numberEnv("EPOCHS", 200);
const SEED_LIMIT = process.env.SEED_LIMIT ? Number(process.env.SEED_LIMIT) : null;
const DEFAULT_SEEDS = [21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171];
const SEEDS = SEED_LIMIT ? DEFAULT_SEEDS.slice(0, SEED_LIMIT) : DEFAULT_SEEDS;
const CHECKPOINTS = new Set(
  listEnv("CHECKPOINTS", ["1", "20", "50", "100", "150", "200"])
    .map(Number)
    .filter(Number.isFinite)
    .concat(EPOCHS)
);
const MAX_NEW_CONN = numberEnv("MAX_NEW_CONN", (INPUT_COUNT + MEDIUM_COUNT) * K);
const FAST_INIT = numberEnv("FAST_INIT", defaultConfig.fastWeightInit);
const AXON_THRESHOLD = numberEnv("AXON_THRESHOLD", 1);
const BRANCH_THRESHOLD = numberEnv("BRANCH_THRESHOLD", 0.1);
const CANDIDATE_MAX_AGE = numberEnv("CANDIDATE_MAX_AGE", 20);
const DORMANT_LIMIT = numberEnv("DORMANT_LIMIT", 50);

function createProbeConfig() {
  return withConfig({
    ...defaultConfig,
    fastWeightInit: FAST_INIT,
    leak: 1,
    branchLocalThreshold: BRANCH_THRESHOLD,
    dendriteGateThreshold: BRANCH_THRESHOLD,
    axonThreshold: AXON_THRESHOLD,
    thresholdAdaptRate: 0,
    refractorySteps: 0,
    fastDecay: 0.9995,
    stableThreshold: 0.12,
    useThreshold: 0.08,
    contributionThreshold: 0.05,
    candidateMaxAge: CANDIDATE_MAX_AGE,
    minConnectionAge: 10,
    dormantLimit: DORMANT_LIMIT,
    connectionDistanceLambda: 1.5,
    connectionThreshold: 0.1,
    rewardAdvantageBaselineAlpha: 0.1
  });
}

function patterns() {
  return Array.from({ length: INPUT_COUNT }, (_, index) => ({
    inputId: `input${index}`,
    targetOutputId: targetOutputIdForInput(index)
  }));
}

function normalizedY(index, count) {
  return count === 1 ? 0.5 : index / (count - 1);
}

function targetOutputIdForInput(inputIndex) {
  if (OUTPUT_COUNT === 1) return "output0";
  const y = normalizedY(inputIndex, INPUT_COUNT);
  const outputIndex = Math.max(0, Math.min(OUTPUT_COUNT - 1, Math.round(y * (OUTPUT_COUNT - 1))));
  return `output${outputIndex}`;
}

function runNetworkStep(network, inputId, config) {
  resetNetworkRuntime(network);
  setSensoryOutputs(network, new Set([inputId]));
  propagateAndIntegrateRole(network, "interneuron", config);
  clearSensoryOutputs(network);
  propagateAndIntegrateRole(network, "motor", config);
  return activeMotorIds(network);
}

function classify(activeOutputs, targetOutputId) {
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

function chooseExplorationOutput(rng) {
  return `output${rng.nextInt(OUTPUT_COUNT)}`;
}

function rewardFor(action) {
  if (action === "correct") return 1;
  if (action === "wrong" || action === "conflict") return -1;
  return -0.1;
}

function runTrainingTrial(network, pattern, config, rng, baselineState) {
  let activeOutputs = runNetworkStep(network, pattern.inputId, config);
  let classification = classify(activeOutputs, pattern.targetOutputId);

  if (classification.action === "noop" || classification.action === "conflict") {
    const exploration = chooseExplorationOutput(rng);
    activeOutputs = forceOutput(network, exploration);
    classification = classify(activeOutputs, pattern.targetOutputId);
  }

  updateNetworkEligibility(network, config);
  const reward = rewardFor(classification.action);
  const rewardAdvantage = reward - baselineState.baseline;
  applyRewardOutcomeLearning(network, rewardAdvantage, config);
  baselineState.baseline =
    baselineState.baseline * (1 - config.rewardAdvantageBaselineAlpha) +
    reward * config.rewardAdvantageBaselineAlpha;
  applyMaintenanceDecayAndCapture(network, config);
}

function evaluate(network, pats, config) {
  const rows = pats.map((pattern) => {
    const clone = structuredClone(network);
    const activeOutputs = runNetworkStep(clone, pattern.inputId, config);
    return { ...pattern, ...classify(activeOutputs, pattern.targetOutputId) };
  });
  const denom = Math.max(1, rows.length);
  return {
    successRate: rows.filter((row) => row.action === "correct").length / denom,
    noopRate: rows.filter((row) => row.action === "noop").length / denom,
    conflictRate: rows.filter((row) => row.action === "conflict").length / denom,
    wrongRate: rows.filter((row) => row.action === "wrong").length / denom,
    rows
  };
}

function pathSnapshot(network, pattern, config) {
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
  let liveStemEff = 0;
  let activeStemEff = 0;
  const driveByOutput = new Map();

  for (const synapse of clone.synapses) {
    if (synapse.state === "pruned" || synapse.state === "dormant") continue;
    const preRole = roles.get(synapse.preNeuronId);
    const postRole = roles.get(synapse.postNeuronId);

    if (preRole === "sensory" && postRole === "interneuron" && synapse.preNeuronId === pattern.inputId) {
      liveStemIds.add(synapse.postNeuronId);
      liveStemEff += synapse.effectiveWeight;
      if (firingMediumIds.has(synapse.postNeuronId)) {
        activeStemEff += synapse.effectiveWeight;
      }
    }

    if (preRole === "interneuron" && postRole === "motor" && firingMediumIds.has(synapse.preNeuronId)) {
      driveByOutput.set(synapse.postNeuronId, (driveByOutput.get(synapse.postNeuronId) ?? 0) + synapse.effectiveWeight);
    }
  }

  clearSensoryOutputs(clone);
  propagateAndIntegrateRole(clone, "motor", config);
  const activeOutputs = activeMotorIds(clone);
  const classified = classify(activeOutputs, pattern.targetOutputId);
  const correctDrive = driveByOutput.get(pattern.targetOutputId) ?? 0;
  const wrongEntries = [...driveByOutput.entries()].filter(([outputId]) => outputId !== pattern.targetOutputId);
  wrongEntries.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]) || a[0].localeCompare(b[0]));
  const [topWrongOutputId, topWrongDrive] = wrongEntries[0] ?? [null, 0];
  const outputDrives = Array.from({ length: OUTPUT_COUNT }, (_, index) => {
    const outputId = `output${index}`;
    return [outputId, driveByOutput.get(outputId) ?? 0];
  });

  return {
    ...pattern,
    action: classified.action,
    activeOutputs: classified.activeOutputs,
    liveStemCount: liveStemIds.size,
    firingMediumCount: firingMediumIds.size,
    liveStemEff,
    activeStemEff,
    correctDrive,
    wrongMaxDrive: Math.abs(topWrongDrive),
    wrongOutputId: topWrongOutputId,
    outputDrives
  };
}

function pathSnapshots(network, pats, config) {
  return pats.map((pattern) => pathSnapshot(network, pattern, config));
}

function connectionStats(network) {
  const roles = new Map(network.neurons.map((neuron) => [neuron.id, neuron.role]));
  const stats = {
    stem: 0,
    readout: 0,
    candidate: 0,
    active: 0,
    stable: 0,
    dormant: 0,
    pruned: 0
  };

  for (const synapse of network.synapses) {
    if (synapse.state === "candidate") stats.candidate += 1;
    else if (synapse.state === "active") stats.active += 1;
    else if (synapse.state === "stable") stats.stable += 1;
    else if (synapse.state === "dormant") stats.dormant += 1;
    else if (synapse.state === "pruned") stats.pruned += 1;

    if (synapse.state === "pruned") continue;
    const pre = roles.get(synapse.preNeuronId);
    const post = roles.get(synapse.postNeuronId);
    if (pre === "sensory" && post === "interneuron") stats.stem += 1;
    if (pre === "interneuron" && post === "motor") stats.readout += 1;
  }

  return stats;
}

function readoutWeightTotals(network) {
  const roles = new Map(network.neurons.map((neuron) => [neuron.id, neuron.role]));
  const totals = Array.from({ length: OUTPUT_COUNT }, (_, index) => ({
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

function shuffle(items, rng) {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = rng.nextInt(i + 1);
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function runSeed(seed) {
  const config = createProbeConfig();
  const topology = createUniformNaturalLayeredTopologyBlueprint({
    inputCount: INPUT_COUNT,
    mediumCount: MEDIUM_COUNT,
    outputCount: OUTPUT_COUNT,
    slotsPerNeuron: K
  });
  const network = createLearningNetworkFromBlueprint(topology, config);
  const pats = patterns();
  const rng = new SeededRandom(seed);
  const baselineState = { baseline: 0 };
  const checkpoints = [];
  const cumulative = { formed: 0, activated: 0, dormant: 0, pruned: 0, tombstoneHit: 0 };

  for (let epoch = 1; epoch <= EPOCHS; epoch += 1) {
    const formed = tryFormNearestLayeredConnections(
      network.neurons,
      network.synapses,
      network.pairMemory,
      network.tick,
      config,
      MAX_NEW_CONN
    );
    cumulative.formed += formed.formed;
    cumulative.tombstoneHit += formed.tombstoneHit;

    for (const pattern of shuffle(pats, rng)) {
      runTrainingTrial(network, pattern, config, rng, baselineState);
    }

    const state = updateConnectionStates(network.neurons, network.synapses, network.pairMemory, network.tick, config);
    cumulative.activated += state.activated;
    cumulative.dormant += state.dormant;
    cumulative.pruned += state.pruned;
    network.tick += 1;

    if (CHECKPOINTS.has(epoch)) {
      checkpoints.push({
        epoch,
        eval: evaluate(network, pats, config),
        stats: connectionStats(network),
        cumulative: { ...cumulative }
      });
    }
  }

  return {
    seed,
    final: evaluate(network, pats, config),
    snapshots: pathSnapshots(network, pats, config),
    weightTotals: readoutWeightTotals(network),
    stats: connectionStats(network),
    cumulative,
    checkpoints,
    network
  };
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
    const pts = results.map((result) => result.checkpoints.find((checkpoint) => checkpoint.epoch === epoch)).filter(Boolean);
    return {
      epoch,
      n: pts.length,
      sr: mean(pts, (item) => item.eval.successRate),
      noop: mean(pts, (item) => item.eval.noopRate),
      conflict: mean(pts, (item) => item.eval.conflictRate),
      wrong: mean(pts, (item) => item.eval.wrongRate),
      stem: mean(pts, (item) => item.stats.stem),
      readout: mean(pts, (item) => item.stats.readout),
      candidate: mean(pts, (item) => item.stats.candidate),
      active: mean(pts, (item) => item.stats.active),
      stable: mean(pts, (item) => item.stats.stable),
      dormant: mean(pts, (item) => item.stats.dormant),
      pruned: mean(pts, (item) => item.stats.pruned),
      formed: mean(pts, (item) => item.cumulative.formed),
      prunedCum: mean(pts, (item) => item.cumulative.pruned)
    };
  }).filter((row) => row.n > 0);

  return {
    rows,
    snapshotRows: aggregateSnapshots(results),
    weightRows: aggregateWeightTotals(results),
    final: {
      sr: mean(results, (item) => item.final.successRate),
      noop: mean(results, (item) => item.final.noopRate),
      conflict: mean(results, (item) => item.final.conflictRate),
      wrong: mean(results, (item) => item.final.wrongRate),
      stem: mean(results, (item) => item.stats.stem),
      readout: mean(results, (item) => item.stats.readout),
      stable: mean(results, (item) => item.stats.stable),
      pruned: mean(results, (item) => item.stats.pruned),
      formed: mean(results, (item) => item.cumulative.formed),
      prunedCum: mean(results, (item) => item.cumulative.pruned),
      solved: results.filter((item) => item.final.successRate >= 0.99).length,
      stuck: results.filter((item) => item.final.noopRate >= 0.8).length
    },
    results
  };
}

function outputDriveMeans(rows) {
  return Array.from({ length: OUTPUT_COUNT }, (_, index) => {
    const outputId = `output${index}`;
    return {
      outputId,
      drive: mean(rows, (row) => {
        const entry = row.outputDrives.find(([id]) => id === outputId);
        return entry ? entry[1] : 0;
      })
    };
  });
}

function aggregateSnapshots(results) {
  const first = results[0];
  if (!first) return [];

  return first.snapshots.map((_, index) => {
    const rows = results.map((result) => result.snapshots[index]).filter(Boolean);
    const denom = Math.max(1, rows.length);
    return {
      inputId: rows[0].inputId,
      targetOutputId: rows[0].targetOutputId,
      correct: rows.filter((row) => row.action === "correct").length / denom,
      noop: rows.filter((row) => row.action === "noop").length / denom,
      conflict: rows.filter((row) => row.action === "conflict").length / denom,
      wrong: rows.filter((row) => row.action === "wrong").length / denom,
      liveStemCount: mean(rows, (row) => row.liveStemCount),
      firingMediumCount: mean(rows, (row) => row.firingMediumCount),
      liveStemEff: mean(rows, (row) => row.liveStemEff),
      activeStemEff: mean(rows, (row) => row.activeStemEff),
      correctDrive: mean(rows, (row) => row.correctDrive),
      wrongMaxDrive: mean(rows, (row) => row.wrongMaxDrive),
      outputDrives: outputDriveMeans(rows)
    };
  });
}

function aggregateWeightTotals(results) {
  return Array.from({ length: OUTPUT_COUNT }, (_, index) => {
    const outputId = `output${index}`;
    const rows = results.map((result) => result.weightTotals.find((item) => item.outputId === outputId)).filter(Boolean);
    return {
      outputId,
      live: mean(rows, (row) => row.live),
      eff: mean(rows, (row) => row.eff),
      fast: mean(rows, (row) => row.fast),
      stable: mean(rows, (row) => row.stable)
    };
  });
}

function printReport(report) {
  console.log("=== natural m/n/o/k probe ===");
  console.log(
    `base m/n/o/k=${BASE_INPUT_COUNT}/${BASE_MEDIUM_COUNT}/${BASE_OUTPUT_COUNT}/${K}` +
    ` l=${LAYER_MULTIPLIER} actual=${INPUT_COUNT}/${MEDIUM_COUNT}/${OUTPUT_COUNT}/${K}` +
    ` seeds=${SEEDS.join(",")} epochs=${EPOCHS} checkpoints=${[...CHECKPOINTS].sort((a, b) => a - b).join(",")} maxNewConn=${MAX_NEW_CONN}`
  );
  console.log(`fastInit=${FAST_INIT} axonThreshold=${AXON_THRESHOLD} branchThreshold=${BRANCH_THRESHOLD} candidateMaxAge=${CANDIDATE_MAX_AGE} dormantLimit=${DORMANT_LIMIT}`);
  console.log("Each epoch: nearest natural formation -> rewardOnly mapping trials -> passive state update.");
  console.log("epoch   SR     noop   confl  wrong  stem  read  cand  act   stab  dorm  prun  formed prunCum");
  for (const row of report.rows) {
    console.log(
      `${String(row.epoch).padStart(5)}  ${fmt(row.sr)}  ${fmt(row.noop)}  ${fmt(row.conflict)}  ${fmt(row.wrong)}  ` +
      `${fmt(row.stem, 1)}  ${fmt(row.readout, 1)}  ${fmt(row.candidate, 1)}  ${fmt(row.active, 1)}  ` +
      `${fmt(row.stable, 1)}  ${fmt(row.dormant, 1)}  ${fmt(row.pruned, 1)}  ${fmt(row.formed, 1)}  ${fmt(row.prunedCum, 1)}`
    );
  }

  const f = report.final;
  console.log(
    `final seeds=${report.results.length} SR=${fmt(f.sr)} noop=${fmt(f.noop)} conflict=${fmt(f.conflict)} wrong=${fmt(f.wrong)}` +
    ` solved=${f.solved}/${report.results.length} stuck=${f.stuck}/${report.results.length}` +
    ` stem=${fmt(f.stem, 1)} readout=${fmt(f.readout, 1)} stable=${fmt(f.stable, 1)} pruned=${fmt(f.pruned, 1)}` +
    ` formed=${fmt(f.formed, 1)} prunedCum=${fmt(f.prunedCum, 1)}`
  );

  console.log("\nfinal single-point path snapshot by input:");
  console.log("input->target  C     N     X     W     stemN fireN stemEff actStem correctD wrongD");
  for (const row of report.snapshotRows) {
    console.log(
      `${row.inputId}->${row.targetOutputId}`.padEnd(14) +
      `${fmt(row.correct)}  ${fmt(row.noop)}  ${fmt(row.conflict)}  ${fmt(row.wrong)}  ` +
      `${fmt(row.liveStemCount, 1)}   ${fmt(row.firingMediumCount, 1)}   ${fmt(row.liveStemEff, 2)}    ` +
      `${fmt(row.activeStemEff, 2)}    ${fmt(row.correctDrive, 2)}     ${fmt(row.wrongMaxDrive, 2)}`
    );
  }

  console.log("\nfinal readout weight totals by output:");
  console.log("output    live  eff    fast   stable");
  for (const row of report.weightRows) {
    console.log(
      `${row.outputId.padEnd(9)}${fmt(row.live, 1)}  ${fmt(row.eff, 2)}  ${fmt(row.fast, 2)}  ${fmt(row.stable, 2)}`
    );
  }

  if (process.env.FULL === "1") {
    console.log("single-stim signatures per seed (C=correct,N=noop,X=conflict,W=wrong):");
    for (const result of report.results) {
      const sig = result.final.rows.map((row) => {
        if (row.action === "correct") return "C";
        if (row.action === "noop") return "N";
        if (row.action === "conflict") return "X";
        return "W";
      }).join("");
      console.log(`  ${String(result.seed).padStart(3)} ${sig}`);
      for (const row of result.snapshots) {
        const outputs = row.activeOutputs.length ? row.activeOutputs.join("|") : "-";
        const driveMap = row.outputDrives.map(([id, drive]) => `${id}:${fmt(drive, 2)}`).join(",");
        console.log(
          `      ${row.inputId}->${row.targetOutputId} ${row.action} outputs=${outputs}` +
          ` stemN=${row.liveStemCount} fireN=${row.firingMediumCount}` +
          ` stemEff=${fmt(row.liveStemEff, 2)} actStem=${fmt(row.activeStemEff, 2)}` +
          ` correctD=${fmt(row.correctDrive, 2)} wrongD=${fmt(row.wrongMaxDrive, 2)}` +
          `${row.wrongOutputId ? ` wrongOut=${row.wrongOutputId}` : ""}` +
          ` drives=[${driveMap}]`
        );
      }
    }
  }
}

printReport(aggregate(SEEDS.map(runSeed)));
