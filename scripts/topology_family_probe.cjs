"use strict";
/*
 * Reward-only probe for arbitrary layered topology families.
 *
 * This is intentionally separate from 2D challenge semantics. It tests whether
 * a layered topology shape can develop input->output mappings while reporting
 * the same diagnostic family we use elsewhere: SR/noop/conflict, weight map,
 * and single-stim responses.
 *
 * Run:
 *   npm run audit:topology-family
 *   EPOCHS=200 SEED_LIMIT=8 CASES=ratio_2_10_2_fan5x1 npm run audit:topology-family
 *   FULL=1 CASES=ratio_2_10_2_fan5x2 npm run audit:topology-family
 */
const path = require("node:path");

const ROOT = path.join(__dirname, "..");
const { defaultConfig, withConfig } = require(path.join(ROOT, "dist/src/config/newModelConfig"));
const {
  createNearestLayeredTopologyBlueprint,
  reduceLayerCounts,
  sameLayerRatio
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

const DEFAULT_CASES = [
  {
    id: "ratio_1_5_1_fan5x1",
    family: "ratio",
    description: "1/5/1 minimal proportional family, one output",
    inputCount: 1,
    mediumCount: 5,
    outputCount: 1,
    synapsesPerInput: 5,
    synapsesPerMedium: 1
  },
  {
    id: "ratio_2_10_2_fan5x1",
    family: "ratio",
    description: "2/10/2 = 2x of 1/5/1, nearest single-output readout",
    inputCount: 2,
    mediumCount: 10,
    outputCount: 2,
    synapsesPerInput: 5,
    synapsesPerMedium: 1
  },
  {
    id: "ratio_2_10_2_fan5x2",
    family: "ratio",
    description: "2/10/2 proportional counts, medium fanout=2 stress",
    inputCount: 2,
    mediumCount: 10,
    outputCount: 2,
    synapsesPerInput: 5,
    synapsesPerMedium: 2
  },
  {
    id: "arb_1_10_2_fan5x1",
    family: "arbitrary",
    description: "1/10/2 arbitrary/non-proportional family",
    inputCount: 1,
    mediumCount: 10,
    outputCount: 2,
    synapsesPerInput: 5,
    synapsesPerMedium: 1
  },
  {
    id: "arb_2_5_2_fan3x1",
    family: "arbitrary",
    description: "2/5/2 arbitrary/non-proportional family",
    inputCount: 2,
    mediumCount: 5,
    outputCount: 2,
    synapsesPerInput: 3,
    synapsesPerMedium: 1
  }
];

const DEFAULT_SEEDS = [21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171];

function numberEnv(name, fallback) {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function listEnv(name, fallback) {
  if (!process.env[name]) return fallback;
  const values = process.env[name]
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  return values.length ? values : fallback;
}

const EPOCHS = numberEnv("EPOCHS", 200);
const SEED_LIMIT = process.env.SEED_LIMIT ? Number(process.env.SEED_LIMIT) : null;
const SEEDS = SEED_LIMIT ? DEFAULT_SEEDS.slice(0, SEED_LIMIT) : DEFAULT_SEEDS;
const CHECKPOINTS = new Set(
  listEnv("CHECKPOINTS", ["20", "50", "100", "150", "200"])
    .map(Number)
    .filter(Number.isFinite)
    .concat(EPOCHS)
);
const CASE_IDS = new Set(listEnv("CASES", DEFAULT_CASES.map((item) => item.id)));
const CASES = DEFAULT_CASES.filter((item) => CASE_IDS.has(item.id));

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
    depotentiationRate: 0.64
  });
}

function patternsForCase(testCase) {
  return Array.from({ length: testCase.inputCount }, (_, index) => ({
    inputId: `input${index}`,
    targetOutputId: `output${Math.min(index, testCase.outputCount - 1)}`
  }));
}

function runNetworkStep(network, inputId, config) {
  resetNetworkRuntime(network);
  setSensoryOutputs(network, new Set([inputId]));
  propagateAndIntegrateRole(network, "interneuron", config);
  clearSensoryOutputs(network);
  propagateAndIntegrateRole(network, "motor", config);
  return activeMotorIds(network);
}

function classifyOutputs(activeOutputs, targetOutputId) {
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

function rewardForClassification(classification) {
  if (classification.action === "correct") return 1;
  if (classification.action === "wrong" || classification.action === "conflict") return -1;
  return -0.1;
}

function chooseExplorationOutput(network, rng) {
  const outputs = network.neurons.filter((neuron) => neuron.role === "motor").map((neuron) => neuron.id).sort();
  return outputs[rng.nextInt(outputs.length)];
}

function runTrainingTrial(network, pattern, config, rng, baselineState) {
  let activeOutputs = runNetworkStep(network, pattern.inputId, config);
  let classification = classifyOutputs(activeOutputs, pattern.targetOutputId);
  let explorationOutput = null;

  if (classification.action === "noop" || classification.action === "conflict") {
    explorationOutput = chooseExplorationOutput(network, rng);
    activeOutputs = forceOutput(network, explorationOutput);
    classification = classifyOutputs(activeOutputs, pattern.targetOutputId);
  }

  updateNetworkEligibility(network, config);
  const reward = rewardForClassification(classification);
  const rewardAdvantage = reward - baselineState.baseline;
  const rewardUpdates = applyRewardOutcomeLearning(network, rewardAdvantage, config);
  baselineState.baseline =
    baselineState.baseline * (1 - config.rewardAdvantageBaselineAlpha) +
    reward * config.rewardAdvantageBaselineAlpha;
  const maintenance = applyMaintenanceDecayAndCapture(network, config);

  return {
    reward,
    rewardAdvantage,
    classification: classification.action,
    explorationOutput,
    rewardUpdates,
    captureUpdates: maintenance.captureUpdates,
    decayUpdates: maintenance.decayUpdates
  };
}

function evaluateNetwork(network, patterns, config) {
  const rows = [];

  for (const pattern of patterns) {
    const clone = structuredClone(network);
    const activeOutputs = runNetworkStep(clone, pattern.inputId, config);
    const classification = classifyOutputs(activeOutputs, pattern.targetOutputId);
    rows.push({
      ...pattern,
      classification: classification.action,
      activeOutputs: classification.activeOutputs
    });
  }

  return summarizeRows(rows);
}

function summarizeRows(rows) {
  const n = Math.max(1, rows.length);
  return {
    successRate: rows.filter((row) => row.classification === "correct").length / n,
    noopRate: rows.filter((row) => row.classification === "noop").length / n,
    conflictRate: rows.filter((row) => row.classification === "conflict").length / n,
    wrongOnlyRate: rows.filter((row) => row.classification === "wrong").length / n,
    rows
  };
}

function readoutWeightMap(network, patterns) {
  const roles = new Map(network.neurons.map((neuron) => [neuron.id, neuron.role]));
  const stems = network.synapses.filter((synapse) =>
    roles.get(synapse.preNeuronId) === "sensory" && roles.get(synapse.postNeuronId) === "interneuron"
  );
  const readouts = network.synapses.filter((synapse) =>
    roles.get(synapse.preNeuronId) === "interneuron" && roles.get(synapse.postNeuronId) === "motor"
  );

  return patterns.map((pattern) => {
    const activeMediumIds = new Set(
      stems
        .filter((synapse) => synapse.preNeuronId === pattern.inputId)
        .map((synapse) => synapse.postNeuronId)
    );
    let correctEff = 0;
    const wrongByOutput = new Map();

    for (const synapse of readouts) {
      if (!activeMediumIds.has(synapse.preNeuronId)) continue;
      if (synapse.postNeuronId === pattern.targetOutputId) {
        correctEff += synapse.effectiveWeight;
      } else {
        wrongByOutput.set(
          synapse.postNeuronId,
          (wrongByOutput.get(synapse.postNeuronId) ?? 0) + Math.abs(synapse.effectiveWeight)
        );
      }
    }

    return {
      ...pattern,
      activeMediumCount: activeMediumIds.size,
      correctEff,
      wrongMaxEff: Math.max(0, ...wrongByOutput.values())
    };
  });
}

function runCaseSeed(testCase, seed) {
  const config = createProbeConfig();
  const topology = createNearestLayeredTopologyBlueprint(testCase);
  const network = createLearningNetworkFromBlueprint(topology, config);
  const patterns = patternsForCase(testCase);
  const rng = new SeededRandom(seed);
  const baselineState = { baseline: 0 };
  const checkpoints = [];

  for (let epoch = 1; epoch <= EPOCHS; epoch += 1) {
    const shuffled = shufflePatterns(patterns, rng);

    for (const pattern of shuffled) {
      runTrainingTrial(network, pattern, config, rng, baselineState);
    }

    if (CHECKPOINTS.has(epoch)) {
      checkpoints.push({
        epoch,
        eval: evaluateNetwork(network, patterns, config),
        weights: readoutWeightMap(network, patterns)
      });
    }
  }

  return {
    seed,
    final: evaluateNetwork(network, patterns, config),
    weights: readoutWeightMap(network, patterns),
    checkpoints
  };
}

function shufflePatterns(patterns, rng) {
  const shuffled = [...patterns];
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = rng.nextInt(index + 1);
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }
  return shuffled;
}

function mean(items, selector) {
  return items.length === 0 ? 0 : items.reduce((sum, item) => sum + selector(item), 0) / items.length;
}

function fmt(value, digits = 3) {
  return Number.isFinite(value) ? value.toFixed(digits) : String(value);
}

function aggregateCase(testCase, results) {
  const checkpointEpochs = [...CHECKPOINTS].sort((a, b) => a - b);
  const rows = checkpointEpochs.map((epoch) => {
    const pts = results
      .map((result) => result.checkpoints.find((checkpoint) => checkpoint.epoch === epoch))
      .filter(Boolean);
    return {
      epoch,
      n: pts.length,
      sr: mean(pts, (item) => item.eval.successRate),
      noop: mean(pts, (item) => item.eval.noopRate),
      conflict: mean(pts, (item) => item.eval.conflictRate),
      wrongOnly: mean(pts, (item) => item.eval.wrongOnlyRate),
      correctEff: mean(pts.flatMap((item) => item.weights), (item) => item.correctEff),
      wrongMaxEff: mean(pts.flatMap((item) => item.weights), (item) => item.wrongMaxEff)
    };
  }).filter((row) => row.n > 0);

  return {
    testCase,
    rows,
    final: {
      sr: mean(results, (item) => item.final.successRate),
      noop: mean(results, (item) => item.final.noopRate),
      conflict: mean(results, (item) => item.final.conflictRate),
      wrongOnly: mean(results, (item) => item.final.wrongOnlyRate),
      correctEff: mean(results.flatMap((item) => item.weights), (item) => item.correctEff),
      wrongMaxEff: mean(results.flatMap((item) => item.weights), (item) => item.wrongMaxEff),
      solved: results.filter((item) => item.final.successRate >= 0.99).length,
      stuck: results.filter((item) => item.final.noopRate >= 0.8).length
    },
    results
  };
}

function printCaseReport(report) {
  const testCase = report.testCase;
  const reduced = reduceLayerCounts(testCase);
  const baselineRatio = { inputCount: 1, mediumCount: 5, outputCount: 1 };
  const sameAsBaseline = sameLayerRatio(testCase, baselineRatio);

  console.log(`\n=== ${testCase.id} ===`);
  console.log(`${testCase.description}`);
  console.log(
    `family=${testCase.family} counts=${testCase.inputCount}/${testCase.mediumCount}/${testCase.outputCount}` +
    ` reduced=${reduced.inputCount}/${reduced.mediumCount}/${reduced.outputCount} commonScale=${reduced.commonScale}` +
    ` sameAs1/5/1=${sameAsBaseline}`
  );
  console.log(`fanout input->medium=${testCase.synapsesPerInput} medium->output=${testCase.synapsesPerMedium}`);
  console.log("epoch   SR     noop   confl  wrong  correctEff wrongEff");
  for (const row of report.rows) {
    console.log(
      `${String(row.epoch).padStart(5)}  ${fmt(row.sr)}  ${fmt(row.noop)}  ${fmt(row.conflict)}  ` +
      `${fmt(row.wrongOnly)}  ${fmt(row.correctEff)}     ${fmt(row.wrongMaxEff)}`
    );
  }
  console.log(
    `final seeds=${report.results.length} SR=${fmt(report.final.sr)} noop=${fmt(report.final.noop)}` +
    ` conflict=${fmt(report.final.conflict)} wrong=${fmt(report.final.wrongOnly)}` +
    ` solved=${report.final.solved}/${report.results.length} stuck=${report.final.stuck}/${report.results.length}` +
    ` correctEff=${fmt(report.final.correctEff)} wrongEff=${fmt(report.final.wrongMaxEff)}`
  );

  if (process.env.FULL === "1") {
    console.log("single-stim signatures per seed (C=correct,N=noop,X=conflict,W=wrong):");
    for (const result of report.results) {
      const sig = result.final.rows.map((row) => {
        if (row.classification === "correct") return "C";
        if (row.classification === "noop") return "N";
        if (row.classification === "conflict") return "X";
        return "W";
      }).join("");
      const weights = result.weights
        .map((row) => `${row.inputId}->${row.targetOutputId}:c=${fmt(row.correctEff, 2)},w=${fmt(row.wrongMaxEff, 2)}`)
        .join(" ");
      console.log(`  ${String(result.seed).padStart(3)} ${sig} ${weights}`);
    }
  }
}

function main() {
  if (CASES.length === 0) {
    throw new Error(`No matching cases for CASES=${process.env.CASES}`);
  }

  console.log("=== topology family reward-only probe ===");
  console.log(`seeds=${SEEDS.join(",")} epochs=${EPOCHS} checkpoints=${[...CHECKPOINTS].sort((a, b) => a - b).join(",")}`);
  console.log("Metrics: SR/noop/conflict/wrong-only + grouped correct/wrong readout eff + single-stim signatures.");
  console.log("Note: arbitrary cases are legal topology experiments; only sameLayerRatio=true cases are proportional scale comparisons.");

  for (const testCase of CASES) {
    const results = SEEDS.map((seed) => runCaseSeed(testCase, seed));
    printCaseReport(aggregateCase(testCase, results));
  }
}

main();
