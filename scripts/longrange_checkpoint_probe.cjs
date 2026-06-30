"use strict";
/*
 * Read-only checkpoint diagnostics for longrange_sweep outputs.
 *
 * Modes:
 *   SUBDIR=lr_scale2 node scripts/longrange_checkpoint_probe.cjs weights
 *   SUBDIR=lr_scale2 EPOCH=300 FULL=1 SEED=21 node scripts/longrange_checkpoint_probe.cjs weights
 *   SUBDIR=lr_scale2 EPOCH=300 node scripts/longrange_checkpoint_probe.cjs path
 *
 * weights: prints sensory->inter stems plus grouped inter->motor readout maps.
 * path: reconstructs the checkpoint network and runs real single-object
 *       runChallengeEpisode cases with learning disabled.
 */
const fs = require("node:fs");
const path = require("node:path");

const ROOT = path.join(__dirname, "..");
const {
  createChallengeConfig,
  createChallengeScenarios,
  runChallengeEpisode,
  DEFAULT_CHALLENGE_MAX_STEPS
} = require(path.join(ROOT, "dist/src/world/challenge2d"));
const { defaultConfig, withConfig } = require(path.join(ROOT, "dist/src/config/newModelConfig"));
const {
  createLearningNetworkFromBlueprint,
  createScaledOfflineLearningTopologyBlueprint,
  offlineLearningTopologyBlueprint
} = require(path.join(ROOT, "dist/src/core/topologyBlueprint"));
const { refreshSynapseWeight } = require(path.join(ROOT, "dist/src/core/synapse"));

const DEFAULT_DIR = "/tmp/lr_sweep";
const dir = process.env.LR_DIR || (process.env.SUBDIR ? path.join("/tmp", process.env.SUBDIR) : DEFAULT_DIR);
const requestedEpoch = process.env.EPOCH ? Number(process.env.EPOCH) : null;
const motorThreshold = 1;

const CHANNELS = [
  { inter: "iFoodLeft", sensory: "foodLeft", label: "food-left", correctPost: "leftMotor", correctAction: "left" },
  { inter: "iFoodRight", sensory: "foodRight", label: "food-right", correctPost: "rightMotor", correctAction: "right" },
  { inter: "iToxinLeft", sensory: "toxinLeft", label: "toxin-left", correctPost: "rightMotor", correctAction: "right" },
  { inter: "iToxinRight", sensory: "toxinRight", label: "toxin-right", correctPost: "leftMotor", correctAction: "left" }
];

function fmt(value, digits = 3) {
  return Number.isFinite(value) ? value.toFixed(digits) : String(value);
}

function baseInterId(interId) {
  return interId.replace(/_copy\d+$/, "");
}

function readResults() {
  if (!fs.existsSync(dir)) {
    throw new Error(`longrange dir not found: ${dir}`);
  }

  const seedFilter = process.env.SEED
    ? new Set(process.env.SEED.split(",").map((value) => Number(value.trim())).filter(Number.isFinite))
    : null;

  return fs.readdirSync(dir)
    .filter((name) => /^lr_\d+\.json$/.test(name))
    .sort((a, b) => Number(a.match(/lr_(\d+)/)[1]) - Number(b.match(/lr_(\d+)/)[1]))
    .map((name) => JSON.parse(fs.readFileSync(path.join(dir, name), "utf8")))
    .filter((result) => !seedFilter || seedFilter.has(result.seed));
}

function checkpointFor(result) {
  if (!result.checkpoints || result.checkpoints.length === 0) {
    return null;
  }

  if (requestedEpoch !== null) {
    return result.checkpoints.find((checkpoint) => checkpoint.epoch === requestedEpoch) ?? null;
  }

  return result.checkpoints[result.checkpoints.length - 1];
}

function checkpointEpochLabel(results) {
  if (requestedEpoch !== null) return requestedEpoch;
  const first = results.find((result) => checkpointFor(result));
  return first ? checkpointFor(first).epoch : "none";
}

function synapsesForBase(checkpoint, baseInter) {
  return checkpoint.interMotor.filter((synapse) => baseInterId(synapse.pre) === baseInter);
}

function groupReadout(checkpoint, channel) {
  const synapses = synapsesForBase(checkpoint, channel.inter);
  const left = synapses
    .filter((synapse) => synapse.post === "leftMotor")
    .reduce((sum, synapse) => sum + (synapse.eff ?? 0), 0);
  const right = synapses
    .filter((synapse) => synapse.post === "rightMotor")
    .reduce((sum, synapse) => sum + (synapse.eff ?? 0), 0);
  const correct = channel.correctPost === "leftMotor" ? left : right;
  const wrong = channel.correctPost === "leftMotor" ? right : left;
  const leftFires = Math.abs(left) >= motorThreshold;
  const rightFires = Math.abs(right) >= motorThreshold;
  let action = "noop";

  if (leftFires && rightFires) action = "conflict";
  else if (leftFires) action = "left";
  else if (rightFires) action = "right";

  return {
    label: channel.label,
    copyCount: synapses.filter((synapse) => synapse.post === "leftMotor").length,
    left,
    right,
    correct,
    wrong,
    action,
    clean: action === channel.correctAction
  };
}

function mean(rows, selector) {
  return rows.length === 0 ? 0 : rows.reduce((sum, row) => sum + selector(row), 0) / rows.length;
}

function printWeights(results) {
  const epoch = checkpointEpochLabel(results);
  console.log("=== longrange checkpoint weight map ===");
  console.log(`dir=${dir}`);
  console.log(`epoch=${epoch} seeds=${results.map((result) => result.seed).join(",")}`);
  console.log("");

  const grouped = [];
  for (const result of results) {
    const checkpoint = checkpointFor(result);
    if (!checkpoint) continue;
    for (const channel of CHANNELS) {
      grouped.push({ seed: result.seed, ...groupReadout(checkpoint, channel) });
    }
  }

  console.log("Grouped inter->motor map (copy eff sums; motor fires when |sumEff|>=1):");
  for (const channel of CHANNELS) {
    const rows = grouped.filter((row) => row.label === channel.label);
    const clean = rows.filter((row) => row.clean).length;
    const conflict = rows.filter((row) => row.action === "conflict").length;
    const noop = rows.filter((row) => row.action === "noop").length;
    const wrongOnly = rows.length - clean - conflict - noop;
    console.log(
      `  ${channel.label.padEnd(11)} clean=${clean}/${rows.length}` +
      ` conflict=${conflict}/${rows.length} noop=${noop}/${rows.length} wrongOnly=${wrongOnly}/${rows.length}` +
      ` copies=${fmt(mean(rows, (row) => row.copyCount), 1)}` +
      ` meanCorrect=${fmt(mean(rows, (row) => row.correct))}` +
      ` meanWrong=${fmt(mean(rows, (row) => Math.abs(row.wrong)))}`
    );
  }

  if (process.env.FULL !== "1") {
    console.log("\nSet FULL=1 and optionally SEED=21 to print every checkpoint synapse.");
    return;
  }

  for (const result of results) {
    const checkpoint = checkpointFor(result);
    if (!checkpoint) continue;
    console.log(`\n--- seed ${result.seed} sensory->inter stems ---`);
    for (const synapse of [...checkpoint.sensoryInter].sort((a, b) => `${a.pre}->${a.post}`.localeCompare(`${b.pre}->${b.post}`))) {
      console.log(
        `  ${synapse.pre}->${synapse.post}` +
        ` eff=${fmt(synapse.eff)} fast=${fmt(synapse.fast)} stable=${fmt(synapse.stable)}` +
        ` state=${synapse.state} connected=${synapse.connected}`
      );
    }

    console.log(`--- seed ${result.seed} inter->motor readout ---`);
    for (const synapse of [...checkpoint.interMotor].sort((a, b) => `${a.pre}->${a.post}`.localeCompare(`${b.pre}->${b.post}`))) {
      const channel = CHANNELS.find((item) => item.inter === baseInterId(synapse.pre));
      const direction = channel?.correctPost === synapse.post ? "CORRECT" : "WRONG";
      const drives = Math.abs(synapse.eff ?? 0) >= motorThreshold ? ">=THR" : "     ";
      console.log(
        `  [${direction}] ${synapse.pre}->${synapse.post}` +
        ` eff=${fmt(synapse.eff)} ${drives} fast=${fmt(synapse.fast)}` +
        ` stable=${fmt(synapse.stable)} state=${synapse.state} connected=${synapse.connected}`
      );
    }
  }
}

function createNetworkFromResult(result, checkpoint) {
  const envOverride = result.config?.envOverride ?? {};
  const config = createChallengeConfig(Object.keys(envOverride).length ? withConfig(envOverride) : defaultConfig);
  const topologyScale = result.config?.topologyScale ?? 1;
  const topologyReadoutMode = result.config?.topologyReadoutMode ?? "normalized";
  const blueprint = topologyScale > 1
    ? createScaledOfflineLearningTopologyBlueprint({
      interneuronCopiesPerSensor: topologyScale,
      normalizeReadoutByCopies: topologyReadoutMode !== "raw"
    })
    : offlineLearningTopologyBlueprint;
  const network = createLearningNetworkFromBlueprint(blueprint, config);
  const snapshots = new Map(
    [...checkpoint.sensoryInter, ...checkpoint.interMotor].map((synapse) => [`${synapse.pre}->${synapse.post}`, synapse])
  );

  for (const synapse of network.synapses) {
    const snap = snapshots.get(`${synapse.preNeuronId}->${synapse.postNeuronId}`);
    if (!snap) continue;
    synapse.fastWeight = snap.fast;
    synapse.stableWeight = snap.stable;
    synapse.state = snap.state;
    synapse.connected = snap.connected;
    refreshSynapseWeight(synapse, config);
  }

  return { network, config };
}

function buildSingleStimScenario(kind, side, maxSteps) {
  const scenarios = createChallengeScenarios([101], maxSteps);
  const want = `${kind}-${side}`;
  const scenario = scenarios.find((item) => item.id.includes(want));
  if (!scenario) {
    throw new Error(`no scenario matching ${want}`);
  }
  return scenario;
}

function classifyPath(channel, episode) {
  const first = episode.steps[0];
  const firstAction = first?.executedAction ?? "noop";
  if (firstAction === channel.correctAction) return "C";
  if (firstAction === "conflict") return "X";
  if (firstAction === "noop") return "N";
  return "W";
}

function printPathProbe(results) {
  const epoch = checkpointEpochLabel(results);
  const signatures = [];
  const rows = [];

  console.log("=== longrange checkpoint single-stim path probe ===");
  console.log(`dir=${dir}`);
  console.log(`epoch=${epoch} seeds=${results.map((result) => result.seed).join(",")}`);
  console.log("C=first action correct, X=conflict, N=noop, W=wrong-only");
  console.log("");

  for (const result of results) {
    const checkpoint = checkpointFor(result);
    if (!checkpoint) continue;
    const maxSteps = result.config?.maxSteps ?? DEFAULT_CHALLENGE_MAX_STEPS;
    const parts = [];

    for (const channel of CHANNELS) {
      const { network, config } = createNetworkFromResult(result, checkpoint);
      const [kind, side] = channel.label.split("-");
      const episode = runChallengeEpisode(network, buildSingleStimScenario(kind, side, maxSteps), config, {
        phase: "eval",
        learningMode: "rewardOnly",
        learningEnabled: false,
        seed: 9999,
        observationDropout: 0,
        reverseMapping: false
      });
      const code = classifyPath(channel, episode);
      rows.push({
        seed: result.seed,
        label: channel.label,
        code,
        success: episode.success,
        firstAction: episode.steps[0]?.executedAction ?? "noop",
        actions: episode.steps.map((step) => step.executedAction).join("/")
      });
      parts.push(code);
    }

    signatures.push({ seed: result.seed, signature: parts.join("") });
  }

  for (const channel of CHANNELS) {
    const channelRows = rows.filter((row) => row.label === channel.label);
    console.log(
      `  ${channel.label.padEnd(11)} firstCorrect=${channelRows.filter((row) => row.code === "C").length}/${channelRows.length}` +
      ` conflict=${channelRows.filter((row) => row.code === "X").length}/${channelRows.length}` +
      ` noop=${channelRows.filter((row) => row.code === "N").length}/${channelRows.length}` +
      ` wrongOnly=${channelRows.filter((row) => row.code === "W").length}/${channelRows.length}` +
      ` episodeSuccess=${channelRows.filter((row) => row.success).length}/${channelRows.length}`
    );
  }

  console.log("\nSeed signatures (order foodL foodR toxinL toxinR):");
  for (const item of signatures) {
    console.log(`  ${String(item.seed).padStart(3)} ${item.signature}`);
  }

  if (process.env.FULL === "1") {
    console.log("\nPer-case action traces:");
    for (const row of rows) {
      console.log(`  seed=${row.seed} ${row.label} first=${row.firstAction} success=${row.success} actions=${row.actions}`);
    }
  } else {
    console.log("\nSet FULL=1 and optionally SEED=21 to print per-case action traces.");
  }
}

function main() {
  const mode = process.argv[2] || "weights";
  const results = readResults();

  if (results.length === 0) {
    throw new Error(`no lr_*.json results found in ${dir}`);
  }

  if (mode === "weights") {
    printWeights(results);
  } else if (mode === "path") {
    printPathProbe(results);
  } else {
    console.error("usage: node scripts/longrange_checkpoint_probe.cjs weights|path");
    process.exit(1);
  }
}

main();
