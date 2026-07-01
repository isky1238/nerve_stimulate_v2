"use strict";
/*
 * Asymmetric valence 1D probe — toxin (tagged negative reward) vs nutrient (no reward).
 *
 * A 1D line: center start, left = toxin, right = nutrient (fixed). motor1 = move
 * left (-> toxin contact), motor2 = move right (-> nutrient contact). After each
 * contact the agent resets to center. Scientific question: the toxin side carries
 * a credit channel (negative reward + aversive badOutcome tag), the nutrient side
 * carries NO credit channel (positive reward path switched off, pure sensory only).
 * Can the network still learn to APPROACH nutrient?
 *
 * Predicted (prewired): toxin contact -> LTD on motor1 readout -> motor1 weakens ->
 * motor2 relatively stronger -> nutrient preference emerges THROUGH avoidance, not
 * through nutrient-side credit. Predicted (stem/developmental): candidate readouts
 * form but never consolidate (recentContribution>0 unreachable without credit) ->
 * no stable preference, noop/stuck (same lesion as topology_development_probe).
 *
 * Sensory layout (inputCount = 2n): toxin-impulse channels input0..input(n-1) and
 * nutrient-impulse channels inputn..input(2n-1). At center BOTH groups fire = the
 * agent perceives "toxin left, nutrient right" (direction sensing) and must pick a
 * motor. The "receiving-stimulation" 2n pathway is realized as REWARD SEMANTICS,
 * not extra spiking sensory: A = negative reward + aversive tag on toxin contact,
 * B = positive reward on nutrient contact (switched OFF by default). See plan for
 * the 2-step spiking-sensory variant (out of scope here).
 *
 * Single-step trial: set center sensory -> propagate -> read motor -> classify
 * (left/toxin, right/nutrient, noop, conflict) -> force-exploration on noop/conflict
 * -> updateNetworkEligibility -> applyRewardOutcomeLearning(rewardAdvantage, tag)
 * -> applyMaintenanceDecayAndCapture -> reset. Eval (frozen) = one forward pass,
 * no exploration, no learning; record the native choice.
 *
 * Run:
 *   npm run audit:asymmetric-valence
 *   EPOCHS=300 SEED_LIMIT=8 npm run audit:asymmetric-valence
 *   VARIANT=prewired EPOCHS=300 SEED_LIMIT=8 npm run audit:asymmetric-valence
 *   VARIANT=stem     EPOCHS=300 SEED_LIMIT=8 npm run audit:asymmetric-valence
 *   AVERSIVE_STRATEGY=off EPOCHS=300 SEED_LIMIT=8 npm run audit:asymmetric-valence
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

const VARIANTS = ["prewired", "stem"];

// Each case: n scales the sensory groups. inputCount = 2n (n toxin-impulse + n
// nutrient-impulse), outputCount = 2 (left/right motor), mediumCount = 5n.
const DEFAULT_CASES = [
  {
    id: "av_2_5_2",
    description: "n=1 minimal: 2 sensory / 5 inter / 2 motor",
    n: 1,
    mediumCount: 5,
    synapsesPerInput: 5,
    synapsesPerMedium: 1
  },
  {
    id: "av_4_10_2",
    description: "n=2: 4 sensory / 10 inter / 2 motor",
    n: 2,
    mediumCount: 10,
    synapsesPerInput: 5,
    synapsesPerMedium: 1
  }
];

const DEFAULT_SEEDS = [21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171];

function numberEnv(name, fallback) {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

// Like numberEnv but allows any finite value (including negative and zero),
// for params like TOXIN_REWARD where -1 is a legitimate signal value. Only
// falls back when the env var is absent or non-numeric.
function signedNumberEnv(name, fallback) {
  if (process.env[name] === undefined || process.env[name] === "") return fallback;
  const value = Number(process.env[name]);
  return Number.isFinite(value) ? value : fallback;
}

function listEnv(name, fallback) {
  if (!process.env[name]) return fallback;
  const values = process.env[name]
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  return values.length ? values : fallback;
}

const EPOCHS = numberEnv("EPOCHS", 300);
const TRIALS_PER_EPOCH = numberEnv("TRIALS_PER_EPOCH", 20);
const SEED_LIMIT = process.env.SEED_LIMIT ? Number(process.env.SEED_LIMIT) : null;
const SEEDS = SEED_LIMIT ? DEFAULT_SEEDS.slice(0, SEED_LIMIT) : DEFAULT_SEEDS;
const CHECKPOINTS = new Set(
  listEnv("CHECKPOINTS", ["1", "20", "50", "100", "200", "300"])
    .map(Number)
    .filter(Number.isFinite)
    .concat(EPOCHS)
);
const MAX_NEW_CONN = numberEnv("MAX_NEW_CONN", 8);
const VARIANT_FILTER = process.env.VARIANT ? new Set([process.env.VARIANT]) : new Set(VARIANTS);
const CASE_FILTER = listEnv("CASES", null);
const AVERSIVE_STRATEGY = process.env.AVERSIVE_STRATEGY || "badOutcomeDepotentiation";
// TOXIN_REWARD isolates the value channel (A) from the tag channel (B). The 2×2
// experiment proved A is a complete no-op in this config (reward=-1 vs reward=0
// are bit-identical whether tag is on or off). Default 0 — the negative-value
// channel is removed by default per user direction. Set -1 to reproduce the old
// signed-rewardSignal path (kept only as a reproducibility对照).
const TOXIN_REWARD = signedNumberEnv("TOXIN_REWARD", 0);
const SWAP = process.env.SWAP === "1";
// baseline alpha: 0 keeps nutrient at exactly 0 reward signal (no hidden credit via
// baseline drift). Set BASELINE_ALPHA=0.1 to compare with standard advantage-baseline.
const BASELINE_ALPHA = numberEnv("BASELINE_ALPHA", 0);

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
    // Asymmetric valence: toxin tagged negative reward, nutrient no reward.
    // Keep aversiveAvoidanceBonus=0 so goodAvoidance never slips nutrient a positive
    // bonus (would violate "nutrient = no credit"). badOutcomeDepotentiation drives
    // stable-weight depotentiation on motor1 input when toxin is contacted.
    aversiveTagStrategy: AVERSIVE_STRATEGY,
    aversiveTagGain: 0,
    aversiveAvoidanceBonus: 0,
    aversiveDepotentiationRate: 0.5,
    aversiveBadOutcomeThreshold: 0,
    rewardAdvantageBaselineAlpha: BASELINE_ALPHA
  });
}

// Sensory grouping for a case. inputCount = 2n: first n = toxin-impulse, next n =
// nutrient-impulse. At center both groups fire (direction sensing).
function sensoryGroups(testCase) {
  const n = testCase.n;
  const toxinImpulseIds = Array.from({ length: n }, (_, i) => `input${i}`);
  const nutrientImpulseIds = Array.from({ length: n }, (_, i) => `input${n + i}`);
  return { toxinImpulseIds, nutrientImpulseIds, centerIds: new Set([...toxinImpulseIds, ...nutrientImpulseIds]) };
}

// Motor semantics. output0 = left, output1 = right. SWAP flips. Contact: left -> toxin,
// right -> nutrient.
function motorSemantics() {
  const leftMotorId = SWAP ? "output1" : "output0";
  const rightMotorId = SWAP ? "output0" : "output1";
  return { leftMotorId, rightMotorId, toxinMotorId: leftMotorId, nutrientMotorId: rightMotorId };
}

function classifyChoice(activeOutputs, sem) {
  const activeLeft = activeOutputs.includes(sem.leftMotorId);
  const activeRight = activeOutputs.includes(sem.rightMotorId);
  if (activeLeft && activeRight) return { action: "conflict", contact: null };
  if (activeLeft) return { action: "left", contact: "toxin" };
  if (activeRight) return { action: "right", contact: "nutrient" };
  return { action: "noop", contact: null };
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

function chooseExplorationMotor(sem, rng) {
  return rng.next() < 0.5 ? sem.leftMotorId : sem.rightMotorId;
}

// Pure compute: total inter->motor effective drive (sum of effectiveWeight on live
// readout synapses). Used as the per-motor readout strength snapshot.
function motorReadoutEff(network, motorId) {
  const roles = new Map(network.neurons.map((neuron) => [neuron.id, neuron.role]));
  let sum = 0;
  let count = 0;
  for (const synapse of network.synapses) {
    if (roles.get(synapse.preNeuronId) !== "interneuron") continue;
    if (synapse.postNeuronId !== motorId) continue;
    if (synapse.state === "pruned") continue;
    sum += synapse.effectiveWeight;
    count += 1;
  }
  return { eff: sum, count };
}

// One training trial. explore=true allows force-exploration on noop/conflict.
function runTrial(network, ctx, config, rng, baselineState, explore) {
  resetNetworkRuntime(network);
  setSensoryOutputs(network, ctx.centerIds);
  propagateAndIntegrateRole(network, "interneuron", config);
  clearSensoryOutputs(network);
  propagateAndIntegrateRole(network, "motor", config);

  let activeOutputs = activeMotorIds(network);
  let choice = classifyChoice(activeOutputs, ctx.sem);
  let exploration = null;

  if (explore && (choice.action === "noop" || choice.action === "conflict")) {
    exploration = chooseExplorationMotor(ctx.sem, rng);
    forceOutput(network, exploration);
    choice = classifyChoice([exploration], ctx.sem);
  }

  updateNetworkEligibility(network, config);

  // Reward: toxin contact = -1 (tagged negative), nutrient contact = 0 (no reward).
  let reward;
  let aversiveTag;
  if (choice.contact === "toxin") {
    reward = TOXIN_REWARD;
    aversiveTag = { present: true, badOutcome: true, goodAvoidance: false, intensity: 1 };
  } else if (choice.contact === "nutrient") {
    reward = 0;
    aversiveTag = undefined;
  } else {
    // conflict unresolved (eval mode with no explore) — no contact, no reward.
    reward = 0;
    aversiveTag = undefined;
  }

  const rewardAdvantage = reward - baselineState.baseline;
  if (explore) {
    applyRewardOutcomeLearning(network, rewardAdvantage, config, aversiveTag);
    baselineState.baseline =
      baselineState.baseline * (1 - config.rewardAdvantageBaselineAlpha) +
      reward * config.rewardAdvantageBaselineAlpha;
    applyMaintenanceDecayAndCapture(network, config);
  }

  return { action: choice.action, contact: choice.contact, reward, rewardAdvantage, exploration };
}

// Frozen eval: one forward pass, no explore, no learning. Returns native choice.
function evalChoice(network, ctx, config) {
  const clone = structuredClone(network);
  resetNetworkRuntime(clone);
  setSensoryOutputs(clone, ctx.centerIds);
  propagateAndIntegrateRole(clone, "interneuron", config);
  clearSensoryOutputs(clone);
  propagateAndIntegrateRole(clone, "motor", config);
  return classifyChoice(activeMotorIds(clone), ctx.sem);
}

function runCaseSeed(testCase, variant, seed) {
  const config = createProbeConfig();
  const n = testCase.n;
  const topology = createNearestLayeredTopologyBlueprint({
    inputCount: 2 * n,
    mediumCount: testCase.mediumCount,
    outputCount: 2,
    synapsesPerInput: testCase.synapsesPerInput,
    synapsesPerMedium: testCase.synapsesPerMedium,
    readoutMode: variant
  });
  const network = createLearningNetworkFromBlueprint(topology, config);
  const ctx = { ...sensoryGroups(testCase), sem: motorSemantics() };
  const rng = new SeededRandom(seed);
  const devRng = new SeededRandom(seed + 7919);
  const baselineState = { baseline: 0 };
  const checkpoints = [];
  const devEnabled = variant === "stem";

  for (let epoch = 1; epoch <= EPOCHS; epoch += 1) {
    for (let t = 0; t < TRIALS_PER_EPOCH; t += 1) {
      runTrial(network, ctx, config, rng, baselineState, true);
    }

    let dev = null;
    if (devEnabled) {
      const formedMetrics = tryFormConnections(
        network.neurons,
        network.synapses,
        network.pairMemory,
        network.tick,
        config,
        devRng,
        MAX_NEW_CONN
      );
      const stateMetrics = updateConnectionStates(
        network.neurons,
        network.synapses,
        network.pairMemory,
        network.tick,
        config
      );
      network.tick += 1;
      dev = {
        formed: formedMetrics.formed,
        activated: stateMetrics.activated,
        dormant: stateMetrics.dormant,
        pruned: stateMetrics.pruned,
        tombstone: formedMetrics.tombstoneHit
      };
    }

    if (CHECKPOINTS.has(epoch)) {
      const choice = evalChoice(network, ctx, config);
      const m1 = motorReadoutEff(network, ctx.sem.toxinMotorId);
      const m2 = motorReadoutEff(network, ctx.sem.nutrientMotorId);
      checkpoints.push({
        epoch,
        choice: choice.action,
        contact: choice.contact,
        toxinEff: m1.eff,
        toxinReadoutN: m1.count,
        nutrientEff: m2.eff,
        nutrientReadoutN: m2.count,
        dev
      });
    }
  }

  return {
    seed,
    variant,
    finalChoice: evalChoice(network, ctx, config),
    finalToxinEff: motorReadoutEff(network, ctx.sem.toxinMotorId),
    finalNutrientEff: motorReadoutEff(network, ctx.sem.nutrientMotorId),
    checkpoints
  };
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
    const n = Math.max(1, pts.length);
    return {
      epoch,
      n: pts.length,
      leftPct: pts.filter((p) => p.choice === "left").length / n,
      rightPct: pts.filter((p) => p.choice === "right").length / n,
      noopPct: pts.filter((p) => p.choice === "noop").length / n,
      conflictPct: pts.filter((p) => p.choice === "conflict").length / n,
      toxinEff: mean(pts, (p) => p.toxinEff),
      nutrientEff: mean(pts, (p) => p.nutrientEff),
      formed: mean(pts, (p) => p.dev?.formed ?? 0),
      activated: mean(pts, (p) => p.dev?.activated ?? 0)
    };
  }).filter((row) => row.n > 0);

  const finalN = Math.max(1, results.length);
  return {
    testCase,
    variant,
    rows,
    final: {
      leftPct: results.filter((r) => r.finalChoice.action === "left").length / finalN,
      rightPct: results.filter((r) => r.finalChoice.action === "right").length / finalN,
      noopPct: results.filter((r) => r.finalChoice.action === "noop").length / finalN,
      conflictPct: results.filter((r) => r.finalChoice.action === "conflict").length / finalN,
      toxinEff: mean(results, (r) => r.finalToxinEff.eff),
      nutrientEff: mean(results, (r) => r.finalNutrientEff.eff),
      approach: results.filter((r) => r.finalChoice.action === "right").length,
      avoid: results.filter((r) => r.finalChoice.action === "left").length,
      stuck: results.filter((r) => r.finalChoice.action === "noop" || r.finalChoice.action === "conflict").length
    },
    results
  };
}

function printCaseReport(report) {
  const { testCase, variant } = report;
  console.log(`\n=== ${testCase.id}_${variant} ===`);
  console.log(`${testCase.description} variant=${variant} n=${testCase.n} counts=${2 * testCase.n}/${testCase.mediumCount}/2`);
  console.log(`aversive=${AVERSIVE_STRATEGY} baselineAlpha=${BASELINE_ALPHA} swap=${SWAP}`);
  console.log("left=toxin contact(neg reward)  right=nutrient contact(no reward)");
  console.log("epoch   left%  right%  noop%  confl%  toxinEff  nutrEff  formed activ");
  for (const row of report.rows) {
    console.log(
      `${String(row.epoch).padStart(5)}  ${fmt(row.leftPct)}  ${fmt(row.rightPct)}  ${fmt(row.noopPct)}  ` +
      `${fmt(row.conflictPct)}  ${fmt(row.toxinEff)}   ${fmt(row.nutrientEff)}  ${fmt(row.formed)}  ${fmt(row.activated)}`
    );
  }
  console.log(
    `final seeds=${report.results.length} left(toxin)=${fmt(report.final.leftPct)} right(nutr)=${fmt(report.final.rightPct)}` +
    ` noop=${fmt(report.final.noopPct)} confl=${fmt(report.final.conflictPct)}` +
    ` approach=${report.final.approach}/${report.results.length} avoid=${report.final.avoid}/${report.results.length}` +
    ` stuck=${report.final.stuck}/${report.results.length}` +
    ` toxinEff=${fmt(report.final.toxinEff)} nutrEff=${fmt(report.final.nutrientEff)}`
  );
  if (variant === "prewired") {
    if (report.final.rightPct > report.final.leftPct) {
      console.log("* avoidance-driven nutrient approach: motor1(toxin) depressed, motor2(nutrient) relatively stronger.");
    } else if (report.final.stuck === report.results.length) {
      console.log("* no preference emerged (all stuck) — avoidance insufficient to express approach in this config.");
    }
  } else if (variant === "stem") {
    const lastRow = report.rows[report.rows.length - 1];
    if (lastRow && lastRow.activated === 0) {
      console.log("* stem lesion: candidate readouts form (formed>0) but never activate (activ=0) — no credit to consolidate, same as developmental probe.");
    }
  }

  if (process.env.FULL === "1") {
    console.log("per-seed final choice (L=left/toxin, R=right/nutrient, N=noop, X=conflict):");
    for (const result of report.results) {
      const ch = { left: "L", right: "R", noop: "N", conflict: "X" }[result.finalChoice.action] || "?";
      console.log(
        `  ${String(result.seed).padStart(3)} ${ch}  toxinEff=${fmt(result.finalToxinEff.eff, 2)}(n=${result.finalToxinEff.count})` +
        `  nutrEff=${fmt(result.finalNutrientEff.eff, 2)}(n=${result.finalNutrientEff.count})`
      );
    }
  }
}

function main() {
  console.log("=== asymmetric valence 1D probe ===");
  console.log(`seeds=${SEEDS.join(",")} epochs=${EPOCHS} trials/epoch=${TRIALS_PER_EPOCH} checkpoints=${[...CHECKPOINTS].sort((a, b) => a - b).join(",")}`);
  console.log(`aversive=${AVERSIVE_STRATEGY} baselineAlpha=${BASELINE_ALPHA} swap=${SWAP} maxNewConn=${MAX_NEW_CONN} toxinReward=${TOXIN_REWARD}`);
  console.log(`toxin contact = ${TOXIN_REWARD} reward + badOutcome tag; nutrient contact = 0 reward (no credit channel).`);

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
