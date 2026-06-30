"use strict";
/*
 * READ-ONLY wrong-prior × rewardOnly long-range unlearning experiment.
 * Does not modify any source file.
 *
 * Hypothesis under test (EVAL_TODO C-bucket, "待测"):
 *   supervised has an active stableWeight depotentiation path (wasWronglyActive,
 *   plasticity.ts:112). rewardOnly's applyRewardLearning only touches fastWeight
 *   (deltaStable:0) and skips eligibilityTrace===0 synapses — so it has NO active
 *   stable depotentiation. Therefore a wrong-prior (stable dual-lock on
 *   wrong-direction inter->motor synapses) should unlearn FAST under supervised
 *   (known: 1-3 epochs) but get STUCK / decay-glacially under rewardOnly, which
 *   can only rely on passive stableDecay=0.99999 (0.99999^300 ≈ 0.9997, ~negligible).
 *
 * Design (isolates the unlearning-mode variable, prior held constant):
 *   Phase 1 — inject wrong-prior: supervised + reverseMapping=true, 40 ep (matches
 *             transferAudit PRETRAIN_EPOCHS). Produces a strong stable dual-lock.
 *   Phase 2a — control arm:  continue-learn supervised + correct mapping, 300 ep.
 *   Phase 2b — test arm:     continue-learn rewardOnly + correct mapping, 300 ep.
 *   Both arms start from a structuredClone of the SAME wrong-prior network.
 *
 * Per-epoch probe: frozen eval SR (correct mapping, learning off) + wrong-direction
 *   stableCount / maxStable / maxFast + correct-direction maxFast + dualLock.
 *
 * Run:  node scripts/wrongprior_rewardonly.cjs            -> coordinator (6 workers)
 *       node scripts/wrongprior_rewardonly.cjs worker <seed> <outpath>
 * Env:  EPOCHS (arm epochs, default 300), SEED_LIMIT (default 6), CONCURRENCY (default 6)
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
const { defaultConfig } = require(path.join(ROOT, "dist/src/config/newModelConfig"));
const { dumpWrongPriorSynapseState } = require(path.join(ROOT, "dist/src/world/diagnostics"));

const PRETRAIN_EPOCHS = 40;
const ARM_EPOCHS = Number(process.env.EPOCHS) || 300;
const SEEDS = [21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131];
// Dense early (supervised recovers in 1-3 ep), sparse later (catch stuck/cliff).
const CHECKPOINTS = new Set(
  [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 40, 60, 80, 100, 150, 200, 250, 300]
    .filter((e) => e <= ARM_EPOCHS)
);

function frozenEval(network, config, evalScenarios) {
  const clone = structuredClone(network);
  let success = 0, noop = 0, conflict = 0, total = 0;
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
    }
  }
  const n = Math.max(1, evalScenarios.length);
  const d = Math.max(1, total);
  return { evalSR: success / n, evalNoop: noop / d, evalConflict: conflict / d };
}

function probeNetwork(network, config, evalScenarios) {
  const dump = dumpWrongPriorSynapseState(network, config);
  const ev = frozenEval(network, config, evalScenarios);
  return {
    wrongStableCount: dump.wrongDirectionStableCount,
    wrongMaxStable: dump.wrongDirectionMaxStableWeight,
    wrongMaxFast: dump.wrongDirectionMaxFastWeight,
    correctMaxFast: dump.correctDirectionMaxFastWeight,
    dualLock: dump.wrongDirectionStableCount > 0 ? 1 : 0,
    ...ev
  };
}

function runArm(config, wrongPriorNetwork, evalScenarios, learningMode, seed) {
  const checkpoints = [];
  runChallengeExperiment(config, {
    seed,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: ARM_EPOCHS,
    learningMode,
    reverseMapping: false, // correct mapping during continued learning
    initialNetwork: structuredClone(wrongPriorNetwork),
    epochProbe: (epoch, network) => {
      const epochCount = epoch + 1;
      if (!CHECKPOINTS.has(epochCount)) return;
      checkpoints.push({ epoch: epochCount, learningMode, ...probeNetwork(network, config, evalScenarios) });
    }
  });
  return checkpoints;
}

function runWorker(seed, outPath) {
  const config = createChallengeConfig(defaultConfig);
  const evalScenarios = createChallengeScenarios(DEFAULT_EVAL_SEEDS, DEFAULT_CHALLENGE_MAX_STEPS);

  // Phase 1: inject wrong-prior via supervised + reverseMapping.
  const wrongPriorResult = runChallengeExperiment(config, {
    seed,
    trainSeeds: DEFAULT_TRAIN_SEEDS,
    evalSeeds: DEFAULT_EVAL_SEEDS,
    epochs: PRETRAIN_EPOCHS,
    learningMode: "supervised",
    reverseMapping: true
  });
  const wrongPriorNetwork = wrongPriorResult.network;
  const preTrain = probeNetwork(wrongPriorNetwork, config, evalScenarios);

  // Phase 2: two arms from identical wrong-prior clone.
  const supervisedArm = runArm(config, wrongPriorNetwork, evalScenarios, "supervised", seed);
  const rewardOnlyArm = runArm(config, wrongPriorNetwork, evalScenarios, "rewardOnly", seed);

  // time-to-recover: first checkpoint epoch where evalSR >= 0.8.
  const ttr = (arm) => {
    const hit = arm.find((c) => c.evalSR >= 0.8);
    return hit ? hit.epoch : null;
  };
  // dual-lock clear: first checkpoint epoch where dualLock === 0.
  const dlc = (arm) => {
    const hit = arm.find((c) => c.dualLock === 0);
    return hit ? hit.epoch : null;
  };

  fs.writeFileSync(outPath, JSON.stringify({
    seed,
    preTrain,
    supervisedArm,
    rewardOnlyArm,
    timeToRecover: { supervised: ttr(supervisedArm), rewardOnly: ttr(rewardOnlyArm) },
    dualLockClear: { supervised: dlc(supervisedArm), rewardOnly: dlc(rewardOnlyArm) }
  }));
}

function aggregate(seedResults) {
  const cps = [...CHECKPOINTS].sort((a, b) => a - b);
  const arms = ["supervised", "rewardOnly"];
  const out = {};
  for (const arm of arms) {
    const rows = [];
    for (const cp of cps) {
      const pts = seedResults.map((r) => r[`${arm}Arm`].find((c) => c.epoch === cp)).filter(Boolean);
      if (pts.length === 0) continue;
      const n = pts.length;
      const mean = (sel) => pts.reduce((s, p) => s + sel(p), 0) / n;
      rows.push({
        epoch: cp, n,
        sr: mean((p) => p.evalSR),
        noop: mean((p) => p.evalNoop),
        wrongStable: mean((p) => p.wrongStableCount),
        wrongMaxStable: mean((p) => p.wrongMaxStable),
        wrongMaxFast: mean((p) => p.wrongMaxFast),
        correctMaxFast: mean((p) => p.correctMaxFast),
        dualLockFrac: mean((p) => p.dualLock)
      });
    }
    out[arm] = rows;
  }
  return out;
}

function fmt(v, d = 3) { return Number.isFinite(v) ? v.toFixed(d) : "  nan"; }

function printReport(agg, results) {
  const lines = [];
  const n = results.length;
  lines.push("=== wrong-prior × rewardOnly long-range unlearning experiment ===");
  lines.push(`seeds=${n} pretrainEpochs=${PRETRAIN_EPOCHS} (supervised+reverseMapping) armEpochs=${ARM_EPOCHS}`);
  lines.push("prior held constant; only continued-learning mode differs between arms.");
  lines.push("");

  // preTrain sanity
  const pt = results.map((r) => r.preTrain);
  const pm = (sel) => pt.reduce((s, p) => s + sel(p), 0) / n;
  lines.push("Pre-train wrong-prior state (after Phase 1, before continued learning):");
  lines.push(`  evalSR=${fmt(pm((p) => p.evalSR))}  wrongStableCount=${fmt(pm((p) => p.wrongStableCount), 2)}/4  ` +
    `wrongMaxStable=${fmt(pm((p) => p.wrongMaxStable))}  wrongMaxFast=${fmt(pm((p) => p.wrongMaxFast))}  ` +
    `dualLockFrac=${fmt(pm((p) => p.dualLock))}`);
  lines.push("");

  for (const arm of ["supervised", "rewardOnly"]) {
    lines.push(`--- ${arm} arm (continue-learn, correct mapping) ---`);
    lines.push("  epoch   SR    noop  wStable  wMaxStable wMaxFast cMaxFast dualLock%");
    for (const r of agg[arm]) {
      lines.push(
        `  ${String(r.epoch).padStart(5)}  ${fmt(r.sr)}  ${fmt(r.noop)}  ` +
        `${fmt(r.wrongStable, 2)}    ${fmt(r.wrongMaxStable)}   ${fmt(r.wrongMaxFast)}  ${fmt(r.correctMaxFast)}  ` +
        `${(r.dualLockFrac * 100).toFixed(0).padStart(3)}%`
      );
    }
    lines.push("");
  }

  // time-to-recover + dual-lock-clear per seed
  lines.push("Per-seed time-to-recover (first epoch evalSR>=0.8) and dual-lock-clear (first epoch dualLock=0):");
  lines.push("  seed   supTTR  rewTTR   supDLC  rewDLC");
  for (const r of results) {
    const f = (v) => (v === null ? "  -" : String(v).padStart(3));
    lines.push(`  ${String(r.seed).padStart(4)}   ${f(r.timeToRecover.supervised)}    ${f(r.timeToRecover.rewardOnly)}     ${f(r.dualLockClear.supervised)}    ${f(r.dualLockClear.rewardOnly)}`);
  }
  const ttrSup = results.filter((r) => r.timeToRecover.supervised !== null).map((r) => r.timeToRecover.supervised);
  const ttrRew = results.filter((r) => r.timeToRecover.rewardOnly !== null).map((r) => r.timeToRecover.rewardOnly);
  const dlcSup = results.filter((r) => r.dualLockClear.supervised !== null).map((r) => r.dualLockClear.supervised);
  const dlcRew = results.filter((r) => r.dualLockClear.rewardOnly !== null).map((r) => r.dualLockClear.rewardOnly);
  const mean = (a) => a.length ? (a.reduce((s, x) => s + x, 0) / a.length).toFixed(1) : "never";
  lines.push(`  mean TTR:  supervised=${mean(ttrSup)} (n=${ttrSup.length}/${n})   rewardOnly=${mean(ttrRew)} (n=${ttrRew.length}/${n})`);
  lines.push(`  mean DLC:  supervised=${mean(dlcSup)} (n=${dlcSup.length}/${n})   rewardOnly=${mean(dlcRew)} (n=${dlcRew.length}/${n})`);

  console.log(lines.join("\n"));
}

function main() {
  const subdir = process.env.SUBDIR || "wp_rew";
  const tmpDir = path.join("/tmp", subdir);
  fs.mkdirSync(tmpDir, { recursive: true });
  const CONCURRENCY = Number(process.env.CONCURRENCY) || 6;
  const seedsUsed = process.env.SEED_LIMIT ? SEEDS.slice(0, Number(process.env.SEED_LIMIT)) : SEEDS.slice(0, 6);
  const queue = seedsUsed.map((seed) => ({ seed, outPath: path.join(tmpDir, `wp_${seed}.json`) }));
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
      const p = path.join(tmpDir, `wp_${seed}.json`);
      if (fs.existsSync(p)) results.push(JSON.parse(fs.readFileSync(p, "utf8")));
    }
    const agg = aggregate(results);
    printReport(agg, results);
  }

  spawnNext();
}

if (process.argv[2] === "worker") {
  runWorker(parseInt(process.argv[3], 10), process.argv[4]);
} else {
  main();
}
