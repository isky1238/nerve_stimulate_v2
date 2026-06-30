"use strict";
/*
 * READ-ONLY long-range rewardOnly sweep. Does not modify any source file.
 * Run: node scripts/longrange_sweep.cjs            -> coordinator (spawns 8 workers)
 *      node scripts/longrange_sweep.cjs worker <seed> <outpath>
 *
 * Per seed: 300-epoch rewardOnly 2D-challenge training. epochProbe records
 * bilateral motor fast/stable sums every epoch, and at sparse checkpoints
 * structuredClone-s the network and runs a frozen eval to get SR/noop/conflict
 * + per-motor fire rates. Writes one JSON per seed; coordinator aggregates.
 *
 * Env knobs (optional, for confirmation/ablation runs): STABLE_DECAY, FAST_DECAY,
 * SUBDIR (output dir under /tmp, default lr_sweep), SEED_LIMIT (use first N seeds).
 */
const { fork } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const ROOT = path.join(__dirname, "..");

const {
  createChallengeConfig,
  createChallengeScenarios,
  runChallengeExperiment,
  runChallengeEpisode,
  DEFAULT_TRAIN_SEEDS,
  DEFAULT_EVAL_SEEDS,
  DEFAULT_CHALLENGE_MAX_STEPS
} = require(path.join(ROOT, "dist/src/world/challenge2d"));
const { defaultConfig, withConfig } = require(path.join(ROOT, "dist/src/config/newModelConfig"));

// Optional config overrides via env for confirmation runs (no source edit):
//   STABLE_DECAY=1.0   -> disable stable-weight passive decay (tests bottleneck-erosion hypothesis)
//   FAST_DECAY=...     -> override fastDecay
const ENV_OVERRIDE = {};
if (process.env.STABLE_DECAY !== undefined) ENV_OVERRIDE.stableDecay = Number(process.env.STABLE_DECAY);
if (process.env.FAST_DECAY !== undefined) ENV_OVERRIDE.fastDecay = Number(process.env.FAST_DECAY);

const TOTAL_EPOCHS = 300;
const SEEDS = [
  21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131,
  141, 151, 161, 171, 181, 191, 201, 211, 221, 231, 241, 251
];
const CHECKPOINTS = new Set([1, 2, 3, 5, 8, 12, 20, 30, 40, 60, 80, 100, 150, 200, 250, 300]);
const INTERNEURON_IDS = ["iFoodLeft", "iFoodRight", "iToxinLeft", "iToxinRight"];

function motorSums(network) {
  let lf = 0, rf = 0, ls = 0, rs = 0;
  for (const s of network.synapses) {
    if (!INTERNEURON_IDS.includes(s.preNeuronId)) continue;
    if (s.postNeuronId === "leftMotor") { lf += s.fastWeight; ls += s.stableWeight; }
    else if (s.postNeuronId === "rightMotor") { rf += s.fastWeight; rs += s.stableWeight; }
  }
  return { lf, rf, ls, rs };
}

function trainRatesFromEpisodes(episodes) {
  let conflict = 0, noop = 0, total = 0;
  for (const ep of episodes) {
    for (const st of ep.steps) {
      total += 1;
      if (st.executedAction === "conflict") conflict += 1;
      else if (st.executedAction === "noop") noop += 1;
    }
  }
  const d = Math.max(1, total);
  return { conflictRate: conflict / d, noopRate: noop / d };
}

function interMotorSynapseStates(network) {
  const out = [];
  for (const s of network.synapses) {
    if (!INTERNEURON_IDS.includes(s.preNeuronId)) continue;
    if (s.postNeuronId !== "leftMotor" && s.postNeuronId !== "rightMotor") continue;
    out.push({
      pre: s.preNeuronId, post: s.postNeuronId,
      fast: s.fastWeight, stable: s.stableWeight,
      eff: s.effectiveWeight, state: s.state, connected: s.connected
    });
  }
  return out;
}

function sensoryInterSynapseStates(network) {
  const out = [];
  for (const s of network.synapses) {
    if (s.postNeuronId !== "iFoodLeft" && s.postNeuronId !== "iFoodRight" &&
        s.postNeuronId !== "iToxinLeft" && s.postNeuronId !== "iToxinRight") continue;
    out.push({
      pre: s.preNeuronId, post: s.postNeuronId,
      fast: s.fastWeight, stable: s.stableWeight,
      eff: s.effectiveWeight, state: s.state, connected: s.connected
    });
  }
  return out;
}

function frozenEval(network, config, evalScenarios) {
  // eval on a clone so the training network is untouched.
  const clone = structuredClone(network);
  let success = 0, noop = 0, conflict = 0, total = 0, leftFire = 0, rightFire = 0;
  for (let i = 0; i < evalScenarios.length; i += 1) {
    const ep = runChallengeEpisode(clone, evalScenarios[i], config, {
      phase: "eval",
      learningMode: "rewardOnly",
      learningEnabled: false,
      seed: 9999,
      observationDropout: 0,
      reverseMapping: false
    });
    if (ep.success) success += 1;
    for (const st of ep.steps) {
      total += 1;
      if (st.executedAction === "noop") noop += 1;
      else if (st.executedAction === "conflict") conflict += 1;
      if (st.learning && st.learning.activeMotors) {
        if (st.learning.activeMotors.includes("leftMotor")) leftFire += 1;
        if (st.learning.activeMotors.includes("rightMotor")) rightFire += 1;
      }
    }
  }
  const n = Math.max(1, evalScenarios.length);
  const d = Math.max(1, total);
  return {
    evalSR: success / n,
    evalNoop: noop / d,
    evalConflict: conflict / d,
    evalLeftFire: leftFire / d,
    evalRightFire: rightFire / d
  };
}

function runWorker(seed, outPath) {
  const baseConfig = Object.keys(ENV_OVERRIDE).length ? withConfig(ENV_OVERRIDE) : defaultConfig;
  const config = createChallengeConfig(baseConfig);
  const evalScenarios = createChallengeScenarios(DEFAULT_EVAL_SEEDS, DEFAULT_CHALLENGE_MAX_STEPS);
  const weightSeries = [];
  const checkpoints = [];

  const result = runChallengeExperiment(config, {
    seed,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: TOTAL_EPOCHS,
    learningMode: "rewardOnly",
    epochProbe: (epoch, network, epochEpisodes) => {
      const { lf, rf, ls, rs } = motorSums(network);
      weightSeries.push({ epoch, lf, rf, ls, rs });
      if (CHECKPOINTS.has(epoch + 1) || CHECKPOINTS.has(epoch)) {
        const targetEpoch = CHECKPOINTS.has(epoch) ? epoch : epoch + 1;
        if (targetEpoch === epoch || (epoch + 1 === targetEpoch && epoch === TOTAL_EPOCHS - 1)) {
          // record at this epoch
        }
      }
      // Record checkpoint at the epoch index that matches (epoch is 0-based; checkpoint labels are 1-based counts).
      const epochCount = epoch + 1;
      if (CHECKPOINTS.has(epochCount)) {
        const tr = trainRatesFromEpisodes(epochEpisodes);
        const ev = frozenEval(network, config, evalScenarios);
        checkpoints.push({
          epoch: epochCount,
          lf, rf, ls, rs,
          fastSum: lf + rf,
          stableSum: ls + rs,
          effSum: lf + rf + ls + rs,
          trainNoop: tr.noopRate,
          trainConflict: tr.conflictRate,
          interMotor: interMotorSynapseStates(network),
          sensoryInter: sensoryInterSynapseStates(network),
          ...ev
        });
      }
    }
  });

  // final eval (post-training, held-out) for sanity
  const finalEval = {
    evalSR: result.successRate,
    evalNoop: result.noopRate,
    evalConflict: result.conflictRate
  };

  fs.writeFileSync(outPath, JSON.stringify({ seed, weightSeries, checkpoints, finalEval }));
}

function aggregate(seedResults) {
  const cpList = CHECKPOINTS.has(1) ? [...CHECKPOINTS] : [...CHECKPOINTS];
  cpList.sort((a, b) => a - b);
  const rows = [];
  for (const cp of cpList) {
    const pts = seedResults.map((r) => r.checkpoints.find((c) => c.epoch === cp)).filter(Boolean);
    if (pts.length === 0) continue;
    const n = pts.length;
    const mean = (sel) => pts.reduce((s, p) => s + sel(p), 0) / n;
    const sr = mean((p) => p.evalSR);
    const noop = mean((p) => p.evalNoop);
    const confl = mean((p) => p.evalConflict);
    const fast = mean((p) => p.fastSum);
    const stable = mean((p) => p.stableSum);
    const eff = mean((p) => p.effSum);
    const leftFire = mean((p) => p.evalLeftFire);
    const rightFire = mean((p) => p.evalRightFire);
    const solved = pts.filter((p) => p.evalSR >= 0.8).length / n;
    const noopStuck = pts.filter((p) => p.evalNoop >= 0.8).length / n;
    const partial = pts.filter((p) => p.evalSR > 0 && p.evalSR < 0.8).length / n;
    rows.push({
      epoch: cp, n, sr, noop, confl, fast, stable, eff,
      leftFire, rightFire, solved, partial, noopStuck
    });
  }

  // weight-drop timing per seed
  const drop15 = [], drop10 = [];
  for (const r of seedResults) {
    const w = r.weightSeries;
    const e15 = w.find((x) => (x.lf + x.rf + x.ls + x.rs) < 1.5);
    const e10 = w.find((x) => (x.lf + x.rf + x.ls + x.rs) < 1.0);
    if (e15) drop15.push(e15.epoch + 1);
    if (e10) drop10.push(e10.epoch + 1);
  }
  const meanArr = (a) => a.length ? a.reduce((s, x) => s + x, 0) / a.length : NaN;

  return { rows, dropTiming: { drop15: meanArr(drop15), drop10: meanArr(drop10), n15: drop15.length, n10: drop10.length } };
}

function fmt(v, d = 3) { return Number.isFinite(v) ? v.toFixed(d) : "  nan"; }

function main() {
  const subdir = process.env.SUBDIR || "lr_sweep";
  const tmpDir = path.join("/tmp", subdir);
  fs.mkdirSync(tmpDir, { recursive: true });
  const CONCURRENCY = 8;
  const seedsUsed = process.env.SEED_LIMIT ? SEEDS.slice(0, Number(process.env.SEED_LIMIT)) : SEEDS;
  const queue = seedsUsed.map((seed) => ({ seed, outPath: path.join(tmpDir, `lr_${seed}.json`) }));
  const running = [];

  function spawnNext() {
    if (queue.length === 0 && running.length === 0) return finish();
    while (running.length < CONCURRENCY && queue.length > 0) {
      const job = queue.shift();
      const child = fork(__filename, ["worker", String(job.seed), job.outPath], { silent: true });
      running.push(child);
      child.on("exit", (code) => {
        if (code !== 0) console.error(`seed ${job.seed} worker exited ${code}`);
        running.splice(running.indexOf(child), 1);
        spawnNext();
      });
    }
  }

  function finish() {
    const results = [];
    for (const seed of SEEDS) {
      const p = path.join(tmpDir, `lr_${seed}.json`);
      if (fs.existsSync(p)) results.push(JSON.parse(fs.readFileSync(p, "utf8")));
    }
    const agg = aggregate(results);
    printReport(agg, results);
  }

  function printReport(agg, results) {
    const lines = [];
    lines.push("=== rewardOnly 2D-challenge long-range sweep ===");
    lines.push(`seeds=${SEEDS.length} epochs=${TOTAL_EPOCHS} concurrency=8 (per-epoch probe + sparse frozen-eval checkpoints)`);
    lines.push("");
    lines.push("Per-checkpoint aggregate (mean across seeds):");
    lines.push("  epoch   SR     noop   confl  fastSum stableSum effSum  leftFire rightFire %solved %partial %noopStuck");
    for (const r of agg.rows) {
      lines.push(
        `  ${String(r.epoch).padStart(5)}  ${fmt(r.sr)}  ${fmt(r.noop)}  ${fmt(r.confl)}  ` +
        `${fmt(r.fast)}   ${fmt(r.stable)}    ${fmt(r.eff)}   ${fmt(r.leftFire)}   ${fmt(r.rightFire)}  ` +
        `${(r.solved * 100).toFixed(0).padStart(3)}%    ${(r.partial * 100).toFixed(0).padStart(3)}%    ${(r.noopStuck * 100).toFixed(0).padStart(3)}%`
      );
    }
    lines.push("");
    lines.push("Weight-drop timing (effSum = left+right fast+stable):");
    lines.push(`  first epoch effSum<1.5: mean=${fmt(agg.dropTiming.drop15, 1)} (${agg.dropTiming.n15}/${SEEDS.length} seeds ever dropped)`);
    lines.push(`  first epoch effSum<1.0: mean=${fmt(agg.dropTiming.drop10, 1)} (${agg.dropTiming.n10}/${SEEDS.length} seeds ever dropped)`);
    lines.push("");
    // final distribution
    const finalRows = results.map((r) => {
      const last = r.checkpoints[r.checkpoints.length - 1];
      return last ? last : { evalSR: r.finalEval.evalSR, evalNoop: r.finalEval.evalNoop };
    });
    const n = finalRows.length;
    const perfect = finalRows.filter((r) => r.evalSR >= 0.99).length;
    const half = finalRows.filter((r) => r.evalSR > 0.4 && r.evalSR < 0.99).length;
    const stuck = finalRows.filter((r) => r.evalNoop >= 0.8).length;
    lines.push(`Final (epoch ${TOTAL_EPOCHS}) distribution across ${n} seeds:`);
    lines.push(`  SR>=0.99 (solved):   ${perfect}/${n}`);
    lines.push(`  0.4<SR<0.99 (partial): ${half}/${n}`);
    lines.push(`  noop>=0.8 (stuck):   ${stuck}/${n}`);
    lines.push(`  final mean SR=${fmt(finalRows.reduce((s, r) => s + r.evalSR, 0) / n)}  noop=${fmt(finalRows.reduce((s, r) => s + r.evalNoop, 0) / n)}`);
    console.log(lines.join("\n"));
  }

  spawnNext();
}

if (process.argv[2] === "worker") {
  runWorker(parseInt(process.argv[3], 10), process.argv[4]);
} else {
  main();
}
