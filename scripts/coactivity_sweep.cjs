"use strict";
/*
 * READ-ONLY diagnostic: observe the "整体效应" of the BAP/STDP/credit-assignment
 * plasticity across seeds. Does NOT modify any source file.
 *
 * What it measures (per seed, at sparse checkpoints during rewardOnly 2D-challenge
 * training): the eligibility distribution and coactivity structure, to verify the
 * new carrier does not cause extreme behavior on some seeds (the user's
 * "normal-summation risk / 看整体效应" concern).
 *
 * Per checkpoint records, over inter->motor synapses:
 *   - coactive count: how many incoming synapses per firing motor have nonzero elig
 *     (split LTP>0 / LTD<0) — min/mean/max
 *   - eligibility distribution: mean / stddev / fracPositive / fracNegative / maxAbs
 *   - LTP/LTD balance: sum(+elig) / (sum(|-elig|) + eps)
 *   - eligibility-vs-|effWeight| correlation (is BAP weighting causing runaway?)
 *   - extreme flag: any |elig| > 10x mean, or >50% synapses same sign (collapse)
 *
 * Run:  node scripts/coactivity_sweep.cjs            -> coordinator (6 workers)
 *       node scripts/coactivity_sweep.cjs worker <seed> <outpath>
 * Env:  EPOCHS (default 40), SEED_LIMIT (default 6), CONCURRENCY (default 6)
 */
const { fork } = require("node:child_process");
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

const EPOCHS = Number(process.env.EPOCHS) || 40;
const SEEDS = [21, 31, 41, 51, 61, 71, 81, 91];
const CHECKPOINTS = new Set([1, 5, 10, 20, 40].filter((e) => e <= EPOCHS));
const INTER = ["iFoodLeft", "iFoodRight", "iToxinLeft", "iToxinRight"];

function interMotorSynapses(network) {
  const out = [];
  for (const s of network.synapses) {
    if (!INTER.includes(s.preNeuronId)) continue;
    if (s.postNeuronId !== "leftMotor" && s.postNeuronId !== "rightMotor") continue;
    out.push(s);
  }
  return out;
}

function motorSpikeState(network) {
  const firing = new Set();
  for (const n of network.neurons) {
    if (n.role === "motor" && n.spike) firing.add(n.id);
  }
  return firing;
}

function stats(arr) {
  if (arr.length === 0) return { mean: 0, sd: 0, min: 0, max: 0 };
  const mean = arr.reduce((s, x) => s + x, 0) / arr.length;
  const variance = arr.reduce((s, x) => s + (x - mean) ** 2, 0) / arr.length;
  return { mean, sd: Math.sqrt(variance), min: Math.min(...arr), max: Math.max(...arr) };
}

function probe(network) {
  const syns = interMotorSynapses(network);
  const firingMotors = motorSpikeState(network);
  const eligs = syns.map((s) => s.eligibilityTrace);
  const absEffs = syns.map((s) => Math.abs(s.effectiveWeight));

  // coactive count per firing motor
  const coactiveByMotor = {};
  for (const m of firingMotors) {
    const incoming = syns.filter((s) => s.postNeuronId === m && s.eligibilityTrace !== 0);
    const ltp = incoming.filter((s) => s.eligibilityTrace > 0).length;
    const ltd = incoming.filter((s) => s.eligibilityTrace < 0).length;
    coactiveByMotor[m] = { ltp, ltd };
  }
  const coactiveCounts = Object.values(coactiveByMotor).map((c) => c.ltp + c.ltd);

  const eligStats = stats(eligs.map((e) => Math.abs(e)));
  const fracPos = eligs.filter((e) => e > 1e-12).length / Math.max(1, eligs.length);
  const fracNeg = eligs.filter((e) => e < -1e-12).length / Math.max(1, eligs.length);
  const sumPos = eligs.filter((e) => e > 0).reduce((s, e) => s + e, 0);
  const sumNegAbs = eligs.filter((e) => e < 0).reduce((s, e) => s + Math.abs(e), 0);
  const balance = sumPos / (sumNegAbs + 1e-9);

  // correlation elig vs |eff|
  let corr = 0;
  if (syns.length > 2) {
    const eMean = eligs.reduce((s, x) => s + x, 0) / eligs.length;
    const fMean = absEffs.reduce((s, x) => s + x, 0) / absEffs.length;
    let num = 0, de = 0, df = 0;
    for (let i = 0; i < syns.length; i += 1) {
      num += (eligs[i] - eMean) * (absEffs[i] - fMean);
      de += (eligs[i] - eMean) ** 2;
      df += (absEffs[i] - fMean) ** 2;
    }
    corr = de > 0 && df > 0 ? num / Math.sqrt(de * df) : 0;
  }

  // extreme flags
  const meanAbs = eligStats.mean;
  const maxAbs = eligStats.max;
  const extremeMax = meanAbs > 0 && maxAbs > 10 * meanAbs;
  const collapse = fracPos > 0.5 || fracNeg > 0.5;

  return {
    firingMotors: [...firingMotors],
    coactive: {
      mean: coactiveCounts.length ? coactiveCounts.reduce((s, x) => s + x, 0) / coactiveCounts.length : 0,
      max: coactiveCounts.length ? Math.max(...coactiveCounts) : 0
    },
    eligMeanAbs: meanAbs, eligSdAbs: eligStats.sd, eligMaxAbs: maxAbs,
    fracPos, fracNeg, balance, corr,
    extremeMax, collapse
  };
}

function runWorker(seed, outPath) {
  const config = createChallengeConfig(defaultConfig);
  const checkpoints = [];
  runChallengeExperiment(config, {
    seed,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: EPOCHS,
    learningMode: "rewardOnly",
    epochProbe: (epoch, network) => {
      const ec = epoch + 1;
      if (!CHECKPOINTS.has(ec)) return;
      checkpoints.push({ epoch: ec, ...probe(network) });
    }
  });
  fs.writeFileSync(outPath, JSON.stringify({ seed, checkpoints }));
}

function aggregate(seedResults) {
  const cps = [...CHECKPOINTS].sort((a, b) => a - b);
  const rows = [];
  for (const cp of cps) {
    const pts = seedResults.map((r) => r.checkpoints.find((c) => c.epoch === cp)).filter(Boolean);
    if (pts.length === 0) continue;
    const n = pts.length;
    const mean = (sel) => pts.reduce((s, p) => s + sel(p), 0) / n;
    rows.push({
      epoch: cp, n,
      coactMean: mean((p) => p.coactive.mean),
      coactMax: mean((p) => p.coactive.max),
      eligMeanAbs: mean((p) => p.eligMeanAbs),
      eligMaxAbs: mean((p) => p.eligMaxAbs),
      fracPos: mean((p) => p.fracPos),
      fracNeg: mean((p) => p.fracNeg),
      balance: mean((p) => p.balance),
      corr: mean((p) => p.corr),
      extremeFrac: pts.filter((p) => p.extremeMax).length / n,
      collapseFrac: pts.filter((p) => p.collapse).length / n
    });
  }
  return rows;
}

function fmt(v, d = 3) { return Number.isFinite(v) ? v.toFixed(d) : "  nan"; }

function printReport(rows, results) {
  const lines = [];
  const n = results.length;
  lines.push("=== BAP/STDP coactivity & eligibility 整体效应 sweep ===");
  lines.push(`seeds=${n} epochs=${EPOCHS} mode=rewardOnly (inter->motor synapses)`);
  lines.push("");
  lines.push("Per-checkpoint aggregate (mean across seeds):");
  lines.push("  epoch  coactMean coactMax eligMean  eligMax  fracPos fracNeg balance  corr  %extremeMax %collapse");
  for (const r of rows) {
    lines.push(
      `  ${String(r.epoch).padStart(5)}  ${fmt(r.coactMean, 2)}     ${fmt(r.coactMax, 2)}    ${fmt(r.eligMeanAbs)}  ${fmt(r.eligMaxAbs)}  ${fmt(r.fracPos)}  ${fmt(r.fracNeg)}  ${fmt(r.balance)}  ${fmt(r.corr)}  ${(r.extremeFrac * 100).toFixed(0).padStart(3)}%       ${(r.collapseFrac * 100).toFixed(0).padStart(3)}%`
    );
  }
  lines.push("");
  lines.push("Per-seed final (epoch " + EPOCHS + ") extreme flags:");
  lines.push("  seed  eligMean  eligMax  fracPos fracNeg extremeMax collapse");
  let healthy = 0, extreme = 0;
  for (const r of results) {
    const last = r.checkpoints[r.checkpoints.length - 1];
    if (!last) continue;
    const isExtreme = last.extremeMax || last.collapse;
    if (isExtreme) extreme += 1; else healthy += 1;
    lines.push(
      `  ${String(r.seed).padStart(4)}  ${fmt(last.eligMeanAbs)}  ${fmt(last.eligMaxAbs)}  ${fmt(last.fracPos)}  ${fmt(last.fracNeg)}  ${String(last.extremeMax).padStart(5)}   ${String(last.collapse).padStart(5)}`
    );
  }
  lines.push("");
  lines.push(`Final seed health: ${healthy}/${n} healthy, ${extreme}/${n} extreme (extremeMax | collapse)`);
  console.log(lines.join("\n"));
}

function main() {
  const tmpDir = path.join("/tmp", "coact_sweep");
  fs.mkdirSync(tmpDir, { recursive: true });
  const CONCURRENCY = Number(process.env.CONCURRENCY) || 6;
  const seedsUsed = process.env.SEED_LIMIT ? SEEDS.slice(0, Number(process.env.SEED_LIMIT)) : SEEDS.slice(0, 6);
  const queue = seedsUsed.map((seed) => ({ seed, outPath: path.join(tmpDir, `cs_${seed}.json`) }));
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
    for (const seed of seedsUsed) {
      const p = path.join(tmpDir, `cs_${seed}.json`);
      if (fs.existsSync(p)) results.push(JSON.parse(fs.readFileSync(p, "utf8")));
    }
    printReport(aggregate(results), results);
  }

  spawnNext();
}

if (process.argv[2] === "worker") {
  runWorker(parseInt(process.argv[3], 10), process.argv[4]);
} else {
  main();
}
