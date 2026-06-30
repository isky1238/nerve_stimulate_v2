"use strict";
/*
 * Read-only diagnostic for longrange_sweep JSON outputs.
 *
 * Run:
 *   npm run audit:rewardonly:natural-stim
 *   SUBDIR=lr_stdp_bap npm run audit:rewardonly:natural-stim
 *   LR_DIR=/tmp/lr_stdp_bap CHECKPOINTS=150,200,250,300 node scripts/natural_stim_probe.cjs
 */
const fs = require("node:fs");
const path = require("node:path");

const DEFAULT_DIR = "/tmp/lr_stdp_bap";
const dir = process.env.LR_DIR || (process.env.SUBDIR ? path.join("/tmp", process.env.SUBDIR) : DEFAULT_DIR);
const checkpoints = (process.env.CHECKPOINTS || "150,200,250,300")
  .split(",")
  .map((value) => Number(value.trim()))
  .filter((value) => Number.isFinite(value));

const INTER = ["iFoodLeft", "iFoodRight", "iToxinLeft", "iToxinRight"];
const LABEL = {
  iFoodLeft: "food-left",
  iFoodRight: "food-right",
  iToxinLeft: "toxin-left",
  iToxinRight: "toxin-right"
};
const CORRECT_POST = {
  iFoodLeft: "leftMotor",
  iFoodRight: "rightMotor",
  iToxinLeft: "rightMotor",
  iToxinRight: "leftMotor"
};
const THRESHOLD = 1;

function baseInterId(interId) {
  return interId.replace(/_copy\d+$/, "");
}

function readResults() {
  if (!fs.existsSync(dir)) {
    throw new Error(`longrange dir not found: ${dir}`);
  }
  return fs.readdirSync(dir)
    .filter((name) => /^lr_\d+\.json$/.test(name))
    .sort((a, b) => Number(a.match(/lr_(\d+)/)[1]) - Number(b.match(/lr_(\d+)/)[1]))
    .map((name) => JSON.parse(fs.readFileSync(path.join(dir, name), "utf8")));
}

function classify(checkpoint, pre) {
  const synapses = checkpoint.interMotor.filter((synapse) => baseInterId(synapse.pre) === pre);
  const left = synapses
    .filter((synapse) => synapse.post === "leftMotor")
    .reduce((sum, synapse) => sum + (synapse.eff ?? 0), 0);
  const right = synapses
    .filter((synapse) => synapse.post === "rightMotor")
    .reduce((sum, synapse) => sum + (synapse.eff ?? 0), 0);
  const correctPost = CORRECT_POST[pre];
  const correctEff = correctPost === "leftMotor" ? left : right;
  const wrongEff = correctPost === "leftMotor" ? right : left;
  const wrongMaxEff = Math.abs(wrongEff);
  const leftFires = Math.abs(left) >= THRESHOLD;
  const rightFires = Math.abs(right) >= THRESHOLD;
  const correctAction = correctPost === "leftMotor" ? "left" : "right";
  let action = "noop";

  if (leftFires && rightFires) {
    action = "conflict";
  } else if (leftFires) {
    action = "left";
  } else if (rightFires) {
    action = "right";
  }

  const response =
    action === correctAction
      ? "cleanCorrect"
      : action === "conflict"
        ? "conflict"
        : action === "noop"
          ? "noop"
          : "wrongOnly";

  return {
    response,
    correctEff,
    wrongMaxEff,
    copyCount: synapses.filter((synapse) => synapse.post === "leftMotor").length,
    correctPass: Math.abs(correctEff) >= THRESHOLD,
    wrongPass: wrongMaxEff >= THRESHOLD
  };
}

function count(items, predicate) {
  return items.filter(predicate).length;
}

function mean(items, selector) {
  return items.length === 0 ? 0 : items.reduce((sum, item) => sum + selector(item), 0) / items.length;
}

function checkpointFor(result, epoch) {
  return result.checkpoints.find((checkpoint) => checkpoint.epoch === epoch);
}

function printCheckpoint(results, epoch) {
  const checkpointsAtEpoch = results
    .map((result) => ({ seed: result.seed, checkpoint: checkpointFor(result, epoch) }))
    .filter((item) => item.checkpoint);

  if (checkpointsAtEpoch.length === 0) {
    return;
  }

  const all = [];
  const byStim = new Map();

  for (const pre of INTER) {
    const rows = checkpointsAtEpoch.map((item) => ({
      seed: item.seed,
      ...classify(item.checkpoint, pre)
    }));
    byStim.set(pre, rows);
    all.push(...rows);
  }

  const finalRows = checkpointsAtEpoch.map((item) => item.checkpoint);
  console.log(`\nEpoch ${epoch} across ${checkpointsAtEpoch.length} seeds`);
  console.log(`  eval: SR=${mean(finalRows, (row) => row.evalSR).toFixed(3)} noop=${mean(finalRows, (row) => row.evalNoop).toFixed(3)} conflict=${mean(finalRows, (row) => row.evalConflict).toFixed(3)}`);
  console.log(`  all: cleanCorrect=${count(all, (row) => row.response === "cleanCorrect")}/${all.length} correctEff>=1=${count(all, (row) => row.correctPass)}/${all.length} wrongEff>=1=${count(all, (row) => row.wrongPass)}/${all.length} conflict=${count(all, (row) => row.response === "conflict")}/${all.length} noop=${count(all, (row) => row.response === "noop")}/${all.length} wrongOnly=${count(all, (row) => row.response === "wrongOnly")}/${all.length}`);

  for (const pre of INTER) {
    const rows = byStim.get(pre);
    console.log(
      `  ${LABEL[pre].padEnd(11)} clean=${count(rows, (row) => row.response === "cleanCorrect")}/${rows.length}` +
      ` correctPass=${count(rows, (row) => row.correctPass)}/${rows.length}` +
      ` wrongPass=${count(rows, (row) => row.wrongPass)}/${rows.length}` +
      ` conflict=${count(rows, (row) => row.response === "conflict")}/${rows.length}` +
      ` noop=${count(rows, (row) => row.response === "noop")}/${rows.length}` +
      ` copies=${mean(rows, (row) => row.copyCount).toFixed(1)}` +
      ` meanCorrect=${mean(rows, (row) => row.correctEff).toFixed(3)}` +
      ` meanWrongMax=${mean(rows, (row) => row.wrongMaxEff).toFixed(3)}`
    );
  }
}

function printSeedLateSignature(results) {
  console.log("\nLate seed signatures (C=clean, X=conflict, N=noop, W=wrong-only; order foodL foodR toxinL toxinR):");
  for (const result of results) {
    const parts = checkpoints.map((epoch) => {
      const checkpoint = checkpointFor(result, epoch);
      if (!checkpoint) {
        return `${epoch}:----`;
      }
      const signature = INTER.map((pre) => {
        const response = classify(checkpoint, pre).response;
        if (response === "cleanCorrect") return "C";
        if (response === "conflict") return "X";
        if (response === "noop") return "N";
        return "W";
      }).join("");
      return `${epoch}:${signature}`;
    });
    console.log(`  ${String(result.seed).padStart(3)} ${parts.join(" ")}`);
  }
}

function main() {
  const results = readResults();
  console.log("=== natural single-stim response probe ===");
  console.log(`dir=${dir}`);
  console.log(`seeds=${results.map((result) => result.seed).join(",")}`);
  for (const checkpoint of checkpoints) {
    printCheckpoint(results, checkpoint);
  }
  printSeedLateSignature(results);
}

main();
