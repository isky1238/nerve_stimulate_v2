"use strict";
/*
 * READ-ONLY diagnostic: snapshot analysis — two modes.
 *
 * The user asked to split snapshot inspection into two concerns:
 *   1. WEIGHT SNAPSHOT  — full-network state analysis: every sensory->inter and
 *      inter->motor synapse's eff/fast/stable/elig/state, grouped CORRECT/WRONG,
 *      to see which pathways developed and which didn't. No simulation, just the
 *      learned weight state.
 *   2. PATH SNAPSHOT    — single-stimulus trajectory SIMULATION via the REAL
 *      runChallengeEpisode (not manual propagation): build a one-object scenario
 *      (food/toxin on left/right at distance 2), run the full multi-step episode
 *      with learning off, and record per-step trajectory (active sensors, inter
 *      firing, motor firing, executed action, reward) PLUS a per-synapse
 *      propagation detail dump on step 0. This rigorously reproduces "food 2-step
 *      success / toxin 12-step noop" as real episode outcomes, not manual-tick
 *      inference.
 *
 * The earlier single_stim_trace.cjs did manual two-tick propagation. It was
 * equivalent to runChallengeNetworkStep under frozen mode (same reset/propagate
 * order) but did NOT walk the real multi-step world loop. This tool replaces it
 * for rigor: path mode uses runChallengeEpisode end-to-end.
 *
 * Run:
 *   SNAPSHOT=supervised|rewardOnly node scripts/snapshot_analyze.cjs weights
 *      -> full-network weight analysis
 *   SNAPSHOT=supervised|rewardOnly node scripts/snapshot_analyze.cjs path <case>
 *      -> single-stimulus trajectory; <case> in food-left|food-right|toxin-left|toxin-right
 *   SNAPSHOT=supervised|rewardOnly node scripts/snapshot_analyze.cjs path all
 *      -> run all 4 single-stim cases, summary
 * Env: SNAPSHOT (default supervised), EPOCHS (for fresh build, default 40)
 */
const fs = require("node:fs");
const path = require("node:path");

const ROOT = path.join(__dirname, "..");
const {
  createChallengeConfig,
  createChallengeScenarios,
  runChallengeEpisode,
  runChallengeExperiment,
  DEFAULT_TRAIN_SEEDS,
  DEFAULT_EVAL_SEEDS,
  DEFAULT_CHALLENGE_MAX_STEPS
} = require(path.join(ROOT, "dist/src/world/challenge2d"));
const { defaultConfig } = require(path.join(ROOT, "dist/src/config/newModelConfig"));
const { readNetworkExport } = require(path.join(ROOT, "dist/src/export/networkExport"));
const { loadNetworkFromExport } = require(path.join(ROOT, "dist/src/export/networkLoader"));
const {
  resetNetworkRuntime,
  setSensoryOutputs,
  clearSensoryOutputs,
  propagateAndIntegrateRole,
  updateNetworkEligibility
} = require(path.join(ROOT, "dist/src/core/mechanism"));

const INTER = ["iFoodLeft", "iFoodRight", "iToxinLeft", "iToxinRight"];
const CORRECT_MOTOR_FOR_INTER = {
  iFoodLeft: "leftMotor",
  iFoodRight: "rightMotor",
  iToxinLeft: "rightMotor",
  iToxinRight: "leftMotor"
};
const MOTOR_AXON_THRESHOLD = 1.0;

function fmt(v, d = 3) { return Number.isFinite(v) ? v.toFixed(d) : String(v); }

function loadNetwork(snapshot) {
  const snapPath = path.join(ROOT, "exports", "pretrained", `2d-challenge-${snapshot}-pretrained.json`);
  if (fs.existsSync(snapPath)) {
    const { network } = loadNetworkFromExport(readNetworkExport(snapPath));
    return network;
  }
  const epochs = Number(process.env.EPOCHS) || 40;
  const config = createChallengeConfig(defaultConfig);
  const mode = snapshot === "rewardOnly" ? "rewardOnly" : "supervised";
  const result = runChallengeExperiment(config, {
    seed: 101, trainSeeds: DEFAULT_TRAIN_SEEDS, evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs, learningMode: mode
  });
  return result.network;
}

// ---------- WEIGHT SNAPSHOT MODE ----------
function weightsMode(network) {
  console.log("=== WEIGHT SNAPSHOT: full-network state ===");
  console.log(`(motor axonThreshold=${MOTOR_AXON_THRESHOLD}; CORRECT = inter->motor in the behaviorally-correct direction)\n`);

  // sensory -> inter (structural stem)
  console.log("--- sensory -> inter (structural stem, decayProtected) ---");
  const sToI = network.synapses.filter((s) => INTER.includes(s.postNeuronId));
  for (const s of sToI) {
    console.log(`  ${s.preNeuronId}->${s.postNeuronId}: eff=${fmt(s.effectiveWeight)} fast=${fmt(s.fastWeight)} stable=${fmt(s.stableWeight)} state=${s.state} dp=${s.decayProtected}`);
  }

  // inter -> motor, grouped CORRECT / WRONG
  console.log("\n--- inter -> motor (learned readout) ---");
  const iToM = network.synapses
    .filter((s) => INTER.includes(s.preNeuronId) && (s.postNeuronId === "leftMotor" || s.postNeuronId === "rightMotor"))
    .map((s) => ({ s, direction: s.postNeuronId !== CORRECT_MOTOR_FOR_INTER[s.preNeuronId] ? "WRONG" : "CORRECT" }));

  for (const dir of ["CORRECT", "WRONG"]) {
    console.log(`\n  [${dir}]`);
    for (const { s, direction } of iToM.filter((x) => x.direction === dir)) {
      const drives = Math.abs(s.effectiveWeight) >= MOTOR_AXON_THRESHOLD ? ">=THR" : "     ";
      console.log(`    ${s.preNeuronId}->${s.postNeuronId}: eff=${fmt(s.effectiveWeight)} ${drives} fast=${fmt(s.fastWeight)} stable=${fmt(s.stableWeight)} elig=${fmt(s.eligibilityTrace, 4)} state=${s.state}`);
    }
  }

  // summary: per inter, does the CORRECT pathway reach threshold?
  console.log("\n--- pathway development summary ---");
  console.log("  (a pathway 'drives' if its |eff| >= motor threshold; 'developed' if stable captured >= stableThreshold)");
  for (const pre of INTER) {
    const correct = CORRECT_MOTOR_FOR_INTER[pre];
    const syn = iToM.find((x) => x.s.preNeuronId === pre && x.s.postNeuronId === correct)?.s;
    const wrong = iToM.filter((x) => x.s.preNeuronId === pre && x.direction === "WRONG").map((x) => x.s);
    const wrongMaxEff = wrong.reduce((m, s) => Math.max(m, Math.abs(s.effectiveWeight)), 0);
    if (syn) {
      const drives = Math.abs(syn.effectiveWeight) >= MOTOR_AXON_THRESHOLD ? "DRIVES" : "weak  ";
      console.log(`  ${pre} -> ${correct}: eff=${fmt(syn.effectiveWeight)} stable=${fmt(syn.stableWeight)} [${drives}]  | wrong-direction max|eff|=${fmt(wrongMaxEff)}`);
    }
  }
}

// ---------- PATH SNAPSHOT MODE ----------
function buildSingleStimScenario(kind, side) {
  // Reuse createChallengeScenarios machinery: find a matching eval scenario.
  // Eval scenarios are: per seed, food-left/right (distance 2-3), toxin-left/right (distance 2).
  // We pick seed 101's matching scenario to get a real one-object world.
  const all = createChallengeScenarios([101], DEFAULT_CHALLENGE_MAX_STEPS);
  const want = `${kind}-${side}`;
  const found = all.find((s) => s.id.includes(want));
  if (!found) throw new Error(`no scenario matching ${want}`);
  return found;
}

function step0Detail(network, config, activeSensorIds) {
  // Per-synapse propagation detail on step 0, using the SAME order as
  // runChallengeNetworkStep (verified equivalent under frozen mode).
  resetNetworkRuntime(network);
  setSensoryOutputs(network, new Set(activeSensorIds));
  propagateAndIntegrateRole(network, "interneuron", config);
  const inters = network.neurons.filter((n) => n.role === "interneuron");
  console.log("    inter firing:");
  for (const n of inters) {
    const f = n.spike ? "FIRE" : "    ";
    console.log(`      ${f} ${n.id}: axonDrive=${fmt(n.axonDrive)} (thr ${MOTOR_AXON_THRESHOLD})`);
  }
  clearSensoryOutputs(network);
  propagateAndIntegrateRole(network, "motor", config);
  const motors = network.neurons.filter((n) => n.role === "motor");
  console.log("    motor firing:");
  for (const n of motors) {
    const f = n.spike ? "FIRE" : "    ";
    console.log(`      ${f} ${n.id}: axonDrive=${fmt(n.axonDrive)} (thr ${MOTOR_AXON_THRESHOLD})`);
  }
  // contributing synapses onto firing motors
  const firingMotors = new Set(motors.filter((n) => n.spike).map((n) => n.id));
  if (firingMotors.size > 0) {
    console.log("    contributing inter->motor synapses onto firing motors:");
    for (const s of network.synapses) {
      if (!INTER.includes(s.preNeuronId) || !firingMotors.has(s.postNeuronId)) continue;
      const dir = s.postNeuronId !== CORRECT_MOTOR_FOR_INTER[s.preNeuronId] ? "WRONG" : "CORRECT";
      console.log(`      [${dir}] ${s.preNeuronId}->${s.postNeuronId}: eff=${fmt(s.effectiveWeight)}`);
    }
  } else {
    console.log("    no motor fired. inter->motor |eff| onto the correct motor for the active inter:");
    for (const s of network.synapses) {
      if (!INTER.includes(s.preNeuronId)) continue;
      if (!inters.find((i) => i.id === s.preNeuronId && i.spike)) continue;
      const dir = s.postNeuronId !== CORRECT_MOTOR_FOR_INTER[s.preNeuronId] ? "WRONG" : "CORRECT";
      console.log(`      [${dir}] ${s.preNeuronId}->${s.postNeuronId}: |eff|=${fmt(Math.abs(s.effectiveWeight))} (needs >= ${MOTOR_AXON_THRESHOLD})`);
    }
  }
  updateNetworkEligibility(network, config);
}

function pathMode(network, caseName) {
  const config = createChallengeConfig(defaultConfig);
  const cases = caseName === "all"
    ? [["food", "left"], ["food", "right"], ["toxin", "left"], ["toxin", "right"]]
    : [caseName.split("-")];

  console.log("=== PATH SNAPSHOT: single-stimulus trajectory (real runChallengeEpisode) ===\n");
  for (const [kind, side] of cases) {
    // fresh clone per case so step0Detail mutations don't leak
    const net = structuredClone(network);
    const sc = buildSingleStimScenario(kind, side);
    const correctMotor = kind === "food" ? (side === "left" ? "leftMotor" : "rightMotor") : (side === "left" ? "rightMotor" : "leftMotor");
    console.log(`--- ${kind}-${side} (correct motor: ${correctMotor}) ---`);
    const ep = runChallengeEpisode(net, sc, config, {
      phase: "eval", learningMode: "rewardOnly", learningEnabled: false,
      seed: 9999, observationDropout: 0, reverseMapping: false
    });
    console.log(`  success=${ep.success}  steps=${ep.steps.length}  reward=${ep.totalReward.toFixed(2)}`);
    console.log(`  actions: ${ep.steps.map((s) => s.executedAction).join(" / ")}`);
    // step 0 detail (re-run on a fresh clone to inspect propagation without altering the episode-run network)
    console.log("  step-0 propagation detail:");
    const detailNet = structuredClone(network);
    // determine active sensors at step 0 from the episode trace
    const step0 = ep.steps[0];
    const activeSensors = step0?.sensoryMapping?.activeSensorIds ?? [];
    step0Detail(detailNet, config, activeSensors);
    console.log("");
  }
}

// ---------- main ----------
const mode = process.argv[2] || "weights";
const snapshot = process.env.SNAPSHOT || "supervised";
const network = loadNetwork(snapshot);
console.log(`[snapshot=${snapshot}]\n`);
if (mode === "weights") {
  weightsMode(network);
} else if (mode === "path") {
  pathMode(network, process.argv[3] || "all");
} else {
  console.error("usage: node snapshot_analyze.cjs weights | path [food-left|food-right|toxin-left|toxin-right|all]");
  process.exit(1);
}
