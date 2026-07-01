"use strict";
/*
 * Graded-concentration valence probe — the tagged-impulse mechanism under a
 * NATURAL (non-binary) sensory gradient, per user direction:
 *   "物质的感受信号应该是由远近决定的浓度梯度普通冲动刺激信号"
 *
 * Differs from asymmetric_valence_probe (single-step, binary: all toxin channels
 * fire at 1 at center) in two ways:
 *   1. MULTI-STEP approach. Toxin at x=0, nutrient at x=L, agent starts at
 *      center. Each step the agent moves left (->toxin) or right (->nutrient).
 *      Episode ends on contact (x=0 toxin / x=L nutrient) or MAX_STEPS.
 *   2. GRADED concentration. k = clamp(round(n/(1+dist)), 0, n) channels fire —
 *      near = many channels (strong drive, more inters recruited), far = few.
 *      This is population rate-coding of 1/(1+distance): a single graded analog
 *      value <1 cannot cross the inter axonThreshold (stem eff 1.1, thr 1.0), so
 *      intensity is encoded as HOW MANY channels fire, not their analog value.
 *      The tag marks whichever toxin channels are firing this step (the aversive
 *      impulse), and rides the active path as in the single-step probe.
 *
 * Question this answers (user 1.2): under a graded multi-step approach, does the
 * tagged-impulse mechanism still look "defensive for defensive", or does the
 * gradient let approach emerge as a more natural self-organization? Compare
 * TAGGED_MODE=off vs taggedImpulse vs specificFactor on the same graded world.
 *
 * Run:
 *   npm run build && node scripts/graded_valence_probe.cjs
 *   TAGGED_MODE=off          EPOCHS=300 SEED_LIMIT=8 node scripts/graded_valence_probe.cjs
 *   TAGGED_MODE=taggedImpulse EPOCHS=300 SEED_LIMIT=8 node scripts/graded_valence_probe.cjs
 *   TAGGED_MODE=specificFactor EPOCHS=300 SEED_LIMIT=8 ANALYZE=1 node scripts/graded_valence_probe.cjs
 */
const path = require("node:path");

const ROOT = path.join(__dirname, "..");
const { defaultConfig, withConfig } = require(path.join(ROOT, "dist/src/config/newModelConfig"));
const {
  createNearestLayeredTopologyBlueprint
} = require(path.join(ROOT, "dist/src/core/layeredTopologyBlueprint"));
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
const N = numberEnv("N", 5); // channels per valence; inputCount=2N, mediumCount=5N
const LINE_LEN = numberEnv("LINE_LEN", 10); // toxin at 0, nutrient at LINE_LEN
const MAX_STEPS = numberEnv("MAX_STEPS", LINE_LEN + 2);
const TAGGED_MODE = process.env.TAGGED_MODE || "taggedImpulse";
const TOXIN_REWARD = signedNumberEnv("TOXIN_REWARD", 0);
const BASELINE_ALPHA = numberEnv("BASELINE_ALPHA", 0);
const GLOBAL_INCREMENT = signedNumberEnv("GLOBAL_INCREMENT", 1.0);
const GLOBAL_DECAY = numberEnv("GLOBAL_DECAY", 0.9);
const GLOBAL_THRESHOLD = signedNumberEnv("GLOBAL_THRESHOLD", 0.5);

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

function sensoryGroups() {
  const toxinImpulseIds = Array.from({ length: N }, (_, i) => `input${i}`);
  const nutrientImpulseIds = Array.from({ length: N }, (_, i) => `input${N + i}`);
  return { toxinImpulseIds, nutrientImpulseIds };
}

// Population-coded concentration: k of n channels fire at distance dist.
// k = clamp(round(n/(1+dist)), 0, n). dist 0 -> n (max), grows -> fewer.
function channelCount(dist) {
  const k = Math.round(N / (1 + dist));
  return Math.max(0, Math.min(N, k));
}

function motorSemantics() {
  return { leftMotorId: "output0", rightMotorId: "output1", toxinMotorId: "output0", nutrientMotorId: "output1" };
}

function classifyChoice(activeOutputs, sem) {
  const activeLeft = activeOutputs.includes(sem.leftMotorId);
  const activeRight = activeOutputs.includes(sem.rightMotorId);
  if (activeLeft && activeRight) return { action: "conflict" };
  if (activeLeft) return { action: "left" }; // toward toxin
  if (activeRight) return { action: "right" }; // toward nutrient
  return { action: "noop" };
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

function markToxinTag(network, activeToxinIds) {
  const active = new Set(activeToxinIds);
  for (const neuron of network.neurons) {
    if (neuron.role === "sensory" && active.has(neuron.id)) {
      neuron.tagLoad = 1;
    }
  }
}

function motorReadoutEff(network, motorId) {
  const roles = new Map(network.neurons.map((n) => [n.id, n.role]));
  let sum = 0;
  for (const s of network.synapses) {
    if (roles.get(s.preNeuronId) !== "interneuron") continue;
    if (s.postNeuronId !== motorId) continue;
    if (s.state === "pruned") continue;
    sum += s.effectiveWeight;
  }
  return sum;
}

// One multi-step episode. explore=true allows force-exploration on noop/conflict
// (train); explore=false is frozen eval. Returns outcome + per-step trace.
function runEpisode(network, ctx, config, rng, baselineState, explore) {
  let agentX = Math.floor(LINE_LEN / 2);
  const steps = [];
  let contact = null;
  let reward = 0;

  for (let step = 0; step < MAX_STEPS; step += 1) {
    const distToxin = agentX;
    const distNutrient = LINE_LEN - agentX;
    const kT = channelCount(distToxin);
    const kN = channelCount(distNutrient);
    const activeToxin = ctx.toxinImpulseIds.slice(0, kT);
    const activeNutrient = ctx.nutrientImpulseIds.slice(0, kN);
    const activeAll = new Set([...activeToxin, ...activeNutrient]);

    resetNetworkRuntime(network);
    setSensoryOutputs(network, activeAll);
    markToxinTag(network, activeToxin);
    if (config.taggedDepotentiationMode === "specificFactor") {
      network.globalAversiveLoad += config.globalAversiveLoadIncrement;
    }
    propagateAndIntegrateRole(network, "interneuron", config);
    clearSensoryOutputs(network);
    propagateAndIntegrateRole(network, "motor", config);

    let activeOutputs = activeMotorIds(network);
    let choice = classifyChoice(activeOutputs, ctx.sem);
    let exploration = null;
    if (explore && (choice.action === "noop" || choice.action === "conflict")) {
      exploration = rng.next() < 0.5 ? ctx.sem.leftMotorId : ctx.sem.rightMotorId;
      forceOutput(network, exploration);
      choice = { action: exploration === ctx.sem.leftMotorId ? "left" : "right" };
    }

    // execute move
    if (choice.action === "left") agentX = Math.max(0, agentX - 1);
    else if (choice.action === "right") agentX = Math.min(LINE_LEN, agentX + 1);

    // reward / contact
    if (agentX === 0) { contact = "toxin"; reward = TOXIN_REWARD; }
    else if (agentX === LINE_LEN) { contact = "nutrient"; reward = 0; }
    else { reward = 0; }

    updateNetworkEligibility(network, config);
    if (explore) {
      const rewardAdvantage = reward - baselineState.baseline;
      applyRewardOutcomeLearning(network, rewardAdvantage, config);
      baselineState.baseline =
        baselineState.baseline * (1 - config.rewardAdvantageBaselineAlpha) +
        reward * config.rewardAdvantageBaselineAlpha;
      applyMaintenanceDecayAndCapture(network, config);
    }

    steps.push({ step, agentX, kT, kN, action: choice.action, exploration, contact });
    if (contact) break;
  }

  const outcome = contact === "nutrient" ? "approach" : contact === "toxin" ? "toxin" : "stuck";
  return { outcome, contact, finalX: agentX, steps, totalReward: reward };
}

function runSeed(seed) {
  const config = createProbeConfig();
  const topology = createNearestLayeredTopologyBlueprint({
    inputCount: 2 * N,
    mediumCount: 5 * N,
    outputCount: 2,
    synapsesPerInput: 5,
    synapsesPerMedium: 1,
    readoutMode: "prewired"
  });
  const network = createLearningNetworkFromBlueprint(topology, config);
  const ctx = { ...sensoryGroups(), sem: motorSemantics() };
  const rng = new SeededRandom(seed);
  const baselineState = { baseline: 0 };
  const checkpoints = [];

  for (let epoch = 1; epoch <= EPOCHS; epoch += 1) {
    for (let t = 0; t < TRIALS_PER_EPOCH; t += 1) {
      runEpisode(network, ctx, config, rng, baselineState, true);
    }
    if (CHECKPOINTS.has(epoch)) {
      // frozen eval: 1 episode from center, no explore
      const ev = runEpisode(network, ctx, config, rng, baselineState, false);
      checkpoints.push({
        epoch,
        outcome: ev.outcome,
        finalX: ev.finalX,
        steps: ev.steps.length,
        toxinEff: motorReadoutEff(network, ctx.sem.toxinMotorId),
        nutrientEff: motorReadoutEff(network, ctx.sem.nutrientMotorId)
      });
    }
  }

  const finalEv = runEpisode(network, ctx, config, rng, baselineState, false);
  return {
    seed,
    finalOutcome: finalEv.outcome,
    finalX: finalEv.finalX,
    finalSteps: finalEv.steps.length,
    finalToxinEff: motorReadoutEff(network, ctx.sem.toxinMotorId),
    finalNutrientEff: motorReadoutEff(network, ctx.sem.nutrientMotorId),
    checkpoints,
    network
  };
}

function mean(items, selector) {
  return items.length === 0 ? 0 : items.reduce((s, x) => s + selector(x), 0) / items.length;
}
function fmt(v, d = 3) { return Number.isFinite(v) ? v.toFixed(d) : String(v); }

function aggregate(results) {
  const epochs = [...CHECKPOINTS].sort((a, b) => a - b);
  const rows = epochs.map((epoch) => {
    const pts = results.map((r) => r.checkpoints.find((c) => c.epoch === epoch)).filter(Boolean);
    const n = Math.max(1, pts.length);
    return {
      epoch,
      n: pts.length,
      approach: pts.filter((p) => p.outcome === "approach").length / n,
      toxin: pts.filter((p) => p.outcome === "toxin").length / n,
      stuck: pts.filter((p) => p.outcome === "stuck").length / n,
      toxinEff: mean(pts, (p) => p.toxinEff),
      nutrientEff: mean(pts, (p) => p.nutrientEff),
      meanSteps: mean(pts, (p) => p.steps)
    };
  }).filter((r) => r.n > 0);

  const n = Math.max(1, results.length);
  return {
    rows,
    final: {
      approach: results.filter((r) => r.finalOutcome === "approach").length,
      toxin: results.filter((r) => r.finalOutcome === "toxin").length,
      stuck: results.filter((r) => r.finalOutcome === "stuck").length,
      toxinEff: mean(results, (r) => r.finalToxinEff),
      nutrientEff: mean(results, (r) => r.finalNutrientEff)
    },
    n: results.length
  };
}

function printReport(report) {
  console.log("\n=== graded valence probe ===");
  console.log(`N=${N} channels/valence  line=0..${LINE_LEN}  maxSteps=${MAX_STEPS}  taggedMode=${TAGGED_MODE}  toxinReward=${TOXIN_REWARD}`);
  console.log(`concentration k = clamp(round(${N}/(1+dist)),0,${N})  | toxin@0 (left)  nutrient@${LINE_LEN} (right)`);
  console.log("epoch   approach  toxin  stuck  toxinEff  nutrEff  meanSteps");
  for (const r of report.rows) {
    console.log(
      `${String(r.epoch).padStart(5)}  ${fmt(r.approach)}   ${fmt(r.toxin)}  ${fmt(r.stuck)}  ${fmt(r.toxinEff)}   ${fmt(r.nutrientEff)}  ${fmt(r.meanSteps, 1)}`
    );
  }
  const f = report.final;
  console.log(
    `final seeds=${report.n} approach=${f.approach}/${report.n} toxin=${f.toxin}/${report.n} stuck=${f.stuck}/${report.n}` +
    ` toxinEff=${fmt(f.toxinEff)} nutrEff=${fmt(f.nutrientEff)}`
  );
}

function analyzeNetwork(network, ctx, seed) {
  const roles = new Map(network.neurons.map((n) => [n.id, n.role]));
  console.log(`\n>>> ANALYZE seed=${seed} taggedMode=${TAGGED_MODE}`);
  console.log("--- inter -> motor readout (sample: nonzero eff or stable) ---");
  let shown = 0;
  for (const s of network.synapses) {
    if (roles.get(s.preNeuronId) !== "interneuron" || roles.get(s.postNeuronId) !== "motor") continue;
    if (Math.abs(s.effectiveWeight) < 0.01 && s.stableWeight < 0.01) continue;
    const side = s.postNeuronId === ctx.sem.toxinMotorId ? "TOXIN" : "NUTR";
    console.log(`  ${s.preNeuronId}->${s.postNeuronId}(${side}): eff=${fmt(s.effectiveWeight)} fast=${fmt(s.fastWeight)} stable=${fmt(s.stableWeight)} state=${s.state} tag=${fmt(s.tagLoad, 3)}`);
    shown += 1;
  }
  if (shown === 0) console.log("  (all readouts ~0: nothing consolidated)");
}

function main() {
  const results = SEEDS.map((seed) => runSeed(seed));
  printReport(aggregate(results));
  if (process.env.ANALYZE === "1") {
    const ctx = { ...sensoryGroups(), sem: motorSemantics() };
    analyzeNetwork(results[0].network, ctx, results[0].seed);
  }
  // per-seed final outcome
  console.log("\nper-seed final outcome (A=approach nutrient, T=toxin contact, S=stuck):");
  for (const r of results) {
    const ch = { approach: "A", toxin: "T", stuck: "S" }[r.finalOutcome] || "?";
    console.log(`  ${String(r.seed).padStart(3)} ${ch}  x=${r.finalX} steps=${r.finalSteps} toxinEff=${fmt(r.finalToxinEff, 2)} nutrEff=${fmt(r.finalNutrientEff, 2)}`);
  }
}

main();
