"use strict";
/*
 * READ-ONLY diagnostic: single-stimulus trajectory trace.
 *
 * Activate ONE sensor (e.g. foodLeft) and trace how that stimulus propagates
 * through sensory -> interneuron -> motor, recording at each layer which neurons
 * fire and the full synapse state (eff/fast/stable/elig/preTrace/postTrace) of
 * every inter->motor connection. The stimulus does branch/randomize in the
 * network, but we observe the single realized trajectory for one step.
 *
 * Purpose: see WHY a stimulus does (or does not) drive the correct motor — is the
 * sensory->inter stem conducting? does the inter fire? does the inter->motor
 * effectiveWeight reach the motor threshold? which wrong-direction synapses
 * compete? This is orthogonal to multi-seed audits: it gives a per-step,
 * per-synapse mechanistic view rather than aggregate success rates.
 *
 * Run:  node scripts/single_stim_trace.cjs [sensorId] [seed]
 *       sensorId in {foodLeft, foodRight, toxinLeft, toxinRight} (default foodLeft)
 *       seed selects which pretrained snapshot to load (default 101 supervised)
 * Env:  SNAPSHOT=supervised|rewardOnly|fresh  (default supervised)
 *       EPOCHS=<n> pretrain epochs if fresh (default 40)
 *
 * Loads exports/pretrained/2d-challenge-<SNAPSHOT>-pretrained.json (or builds fresh).
 */
const fs = require("node:fs");
const path = require("node:path");

const ROOT = path.join(__dirname, "..");
const {
  createChallengeConfig,
  createChallengeScenarios,
  runChallengeExperiment,
  DEFAULT_TRAIN_SEEDS,
  DEFAULT_EVAL_SEEDS,
  DEFAULT_CHALLENGE_MAX_STEPS
} = require(path.join(ROOT, "dist/src/world/challenge2d"));
const { defaultConfig } = require(path.join(ROOT, "dist/src/config/newModelConfig"));
const { createOfflineLearningNetwork } = require(path.join(ROOT, "dist/src/core/evaluation"));
const {
  resetNetworkRuntime,
  setSensoryOutputs,
  clearSensoryOutputs,
  propagateAndIntegrateRole,
  updateNetworkEligibility
} = require(path.join(ROOT, "dist/src/core/mechanism"));
const { readNetworkExport } = require(path.join(ROOT, "dist/src/export/networkExport"));
const { loadNetworkFromExport } = require(path.join(ROOT, "dist/src/export/networkLoader"));

const INTER = ["iFoodLeft", "iFoodRight", "iToxinLeft", "iToxinRight"];
const CORRECT_MOTOR_FOR_INTER = {
  iFoodLeft: "leftMotor",
  iFoodRight: "rightMotor",
  iToxinLeft: "rightMotor",
  iToxinRight: "leftMotor"
};
const MOTOR_AXON_THRESHOLD = 1.0; // challenge config axonThreshold

function fmt(v, d = 3) { return Number.isFinite(v) ? v.toFixed(d) : String(v); }

function neuronState(n) {
  return {
    id: n.id,
    role: n.role,
    spike: n.spike,
    outputSignal: n.outputSignal,
    somaPotential: n.somaPotential,
    axonDrive: n.axonDrive,
    branchInputs: n.branches.map((b) => ({
      id: b.id,
      inputSum: b.inputSum,
      inhibitionLoad: b.inhibitionLoad,
      gain: b.gain,
      active: b.active,
      output: b.output,
      plasticityGate: b.plasticityGate
    }))
  };
}

function interMotorSynapseState(network) {
  return network.synapses
    .filter((s) => INTER.includes(s.preNeuronId) && (s.postNeuronId === "leftMotor" || s.postNeuronId === "rightMotor"))
    .map((s) => {
      const correct = CORRECT_MOTOR_FOR_INTER[s.preNeuronId];
      return {
        pre: s.preNeuronId,
        post: s.postNeuronId,
        direction: s.postNeuronId !== correct ? "WRONG" : "CORRECT",
        effectSign: s.effectSign,
        fast: s.fastWeight,
        stable: s.stableWeight,
        eff: s.effectiveWeight,
        elig: s.eligibilityTrace,
        preTrace: s.preTrace,
        postTrace: s.postTrace,
        state: s.state,
        decayProtected: s.decayProtected,
        recentUse: s.recentUse,
        recentContribution: s.recentContribution
      };
    });
}

function sensoryInterSynapseState(network) {
  return network.synapses
    .filter((s) => INTER.includes(s.postNeuronId))
    .map((s) => ({
      pre: s.preNeuronId,
      post: s.postNeuronId,
      effectSign: s.effectSign,
      fast: s.fastWeight,
      stable: s.stableWeight,
      eff: s.effectiveWeight,
      state: s.state,
      decayProtected: s.decayProtected
    }));
}

function trace(network, config, sensorId) {
  console.log(`\n=== single-stimulus trajectory: sensor=${sensorId} ===`);
  console.log(`(motor axonThreshold=${MOTOR_AXON_THRESHOLD})`);

  resetNetworkRuntime(network);
  setSensoryOutputs(network, new Set([sensorId]));

  // sensory -> interneuron tick
  propagateAndIntegrateRole(network, "interneuron", config);

  const sensors = network.neurons.filter((n) => n.role === "sensory");
  const inters = network.neurons.filter((n) => n.role === "interneuron");
  console.log("\n--- after sensory->inter tick ---");
  console.log("sensory outputs:", sensors.map((n) => `${n.id}=${n.outputSignal}`).join(" "));
  console.log("interneuron firing:");
  for (const n of inters) {
    const firing = n.spike ? "FIRE" : "    ";
    const branchIn = n.branches.map((b) => `in=${fmt(b.inputSum)}/inh=${fmt(b.inhibitionLoad)}/g=${fmt(b.gain)}/out=${fmt(b.output)}`).join("  ");
    console.log(`  ${firing} ${n.id}: soma=${fmt(n.somaPotential)} axonDrive=${fmt(n.axonDrive)} spike=${n.spike}  [${branchIn}]`);
  }

  // sensory->inter synapse states (the structural stem)
  console.log("sensory->inter synapses:");
  for (const s of sensoryInterSynapseState(network)) {
    console.log(`  ${s.pre}->${s.post}: eff=${fmt(s.eff)} fast=${fmt(s.fast)} stable=${fmt(s.stable)} state=${s.state} dp=${s.decayProtected}`);
  }

  clearSensoryOutputs(network);

  // interneuron -> motor tick
  propagateAndIntegrateRole(network, "motor", config);

  const motors = network.neurons.filter((n) => n.role === "motor");
  console.log("\n--- after inter->motor tick ---");
  console.log("interneuron outputs (held from tick1):", inters.map((n) => `${n.id}=${n.outputSignal}`).join(" "));
  console.log("motor firing:");
  for (const n of motors) {
    const firing = n.spike ? "FIRE" : "    ";
    const branchIn = n.branches.map((b) => `in=${fmt(b.inputSum)}/inh=${fmt(b.inhibitionLoad)}/g=${fmt(b.gain)}/out=${fmt(b.output)}`).join("  ");
    console.log(`  ${firing} ${n.id}: soma=${fmt(n.somaPotential)} axonDrive=${fmt(n.axonDrive)} spike=${n.spike} thresh=${MOTOR_AXON_THRESHOLD}  [${branchIn}]`);
  }

  // inter->motor synapse states
  console.log("inter->motor synapses (sorted: WRONG first, then CORRECT):");
  const syns = interMotorSynapseState(network).sort((a, b) => {
    if (a.direction !== b.direction) return a.direction === "WRONG" ? -1 : 1;
    return a.pre.localeCompare(b.pre);
  });
  for (const s of syns) {
    const drives = Math.abs(s.eff) >= MOTOR_AXON_THRESHOLD ? ">=THR" : "     ";
    console.log(`  [${s.direction}] ${s.pre}->${s.post}: eff=${fmt(s.eff)} ${drives} fast=${fmt(s.fast)} stable=${fmt(s.stable)} elig=${fmt(s.elig, 5)} preTr=${fmt(s.preTrace)} postTr=${fmt(s.postTrace)} state=${s.state}`);
  }

  // update eligibility once (step-end semantics, matching challenge2d)
  updateNetworkEligibility(network, config);
  console.log("\n--- after step-end eligibility update ---");
  console.log("inter->motor eligibility (post-update):");
  for (const s of interMotorSynapseState(network).sort((a, b) => a.pre.localeCompare(b.pre))) {
    console.log(`  ${s.pre}->${s.post} [${s.direction}]: elig=${fmt(s.elig, 5)} preTr=${fmt(s.preTrace)} postTr=${fmt(s.postTrace)}`);
  }

  // verdict
  const firingMotors = motors.filter((n) => n.spike).map((n) => n.id);
  const correctMotorForStim = sensorId.startsWith("food")
    ? (sensorId.endsWith("Left") ? "leftMotor" : "rightMotor")
    : (sensorId.endsWith("Left") ? "rightMotor" : "leftMotor"); // toxin: move away
  console.log(`\n=== verdict ===`);
  console.log(`stimulus=${sensorId}  correct motor=${correctMotorForStim}`);
  console.log(`firing motors=[${firingMotors.join(",")}]`);
  console.log(`correct motor fired? ${firingMotors.includes(correctMotorForStim)}`);
  // why: which inter->motor synapses could have driven it
  const correctSyns = interMotorSynapseState(network).filter((s) => s.postNeuronId === correctMotorForStim && s.post === correctMotorForStim);
  console.log(`synapses onto correct motor ${correctMotorForStim}:`);
  for (const s of syns.filter((s) => s.post === correctMotorForStim)) {
    console.log(`  ${s.pre}->${s.post} [${s.direction}] eff=${fmt(s.eff)} (needs >= ${MOTOR_AXON_THRESHOLD} to drive)`);
  }
}

function loadNetwork() {
  const snapshot = process.env.SNAPSHOT || "supervised";
  const snapPath = path.join(ROOT, "exports", "pretrained", `2d-challenge-${snapshot}-pretrained.json`);
  if (fs.existsSync(snapPath)) {
    const snap = readNetworkExport(snapPath);
    const { network } = loadNetworkFromExport(snap);
    console.log(`loaded ${snapshot} pretrained snapshot from ${snapPath}`);
    return network;
  }
  // build fresh
  const epochs = Number(process.env.EPOCHS) || 40;
  const config = createChallengeConfig(defaultConfig);
  const mode = snapshot === "rewardOnly" ? "rewardOnly" : "supervised";
  const result = runChallengeExperiment(config, {
    seed: 101,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs,
    learningMode: mode
  });
  console.log(`built fresh ${mode} network (${epochs}ep, SR=${result.successRate})`);
  return result.network;
}

const sensorId = process.argv[2] || "foodLeft";
const config = createChallengeConfig(defaultConfig);
const network = loadNetwork();
trace(network, config, sensorId);
