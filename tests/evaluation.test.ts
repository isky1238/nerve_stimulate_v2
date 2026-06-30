import test from "node:test";
import assert from "node:assert/strict";
import { configFieldGroups, defaultConfig, withConfig } from "../src/config/newModelConfig";
import { arbitrateMotorAction } from "../src/core/arbitration";
import { formatAuditReport, runPre2DAudit } from "../src/core/audit";
import { runAllEvaluations, runLearningDemo } from "../src/core/evaluation";
import {
  createNearestLayeredTopologyBlueprint,
  reduceLayerCounts,
  sameLayerRatio
} from "../src/core/layeredTopologyBlueprint";
import {
  createLearningNetworkFromBlueprint,
  createScaledOfflineLearningTopologyBlueprint,
  offlineLearningTopologyBlueprint
} from "../src/core/topologyBlueprint";
import {
  computeAversiveModulator,
  computeAversiveRewardSignal,
  computeAversiveStableDepotentiationDelta,
  computeRewardFastDelta,
  computeRewardModulator,
  computeStableCaptureAmount,
  computeStableDepotentiationDelta,
  computeStdpEligibilityDelta,
  computeSupervisedFastDelta,
  nextActivityTrace,
  nextEligibilityTrace,
  positiveEligibilityScale,
  shouldApplyAversiveStableDepotentiation
} from "../src/core/plasticityMechanisms";
import { explainTrace, runLearningTrace } from "../src/core/trace";
import { createChallengePretrainExports } from "../src/export/challengePretrainExport";
import { formatWorld2DAuditReport, runWorld2DAudit } from "../src/world/audit2d";
import { formatWorld2DChallengeAuditReport, runWorld2DChallengeAudit } from "../src/world/audit2dChallenge";
import { formatWorld2DComplexAuditReport, runWorld2DComplexAudit } from "../src/world/audit2dComplex";
import { createChallengeConfig, runChallengeExperiment } from "../src/world/challenge2d";
import { formatArbitrationAuditReport, runArbitrationAudit } from "../src/world/auditArbitration";
import { formatArbitrationMatrixReport, runArbitrationMatrixAudit } from "../src/world/auditArbitrationMatrix";
import { formatTransferAuditReport, runTransferAudit } from "../src/world/transferAudit";
import { formatTransferMatrixReport, runTransferAuditMatrix } from "../src/world/transferMatrix";
import { tryFormConnections, updateConnectionStates } from "../src/core/development";
import { SeededRandom } from "../src/core/random";

function assertClose(actual: number, expected: number, tolerance = 1e-12): void {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} !== ${expected}`);
}

test("offline evaluation suite passes Test A-E", () => {
  const results = runAllEvaluations(defaultConfig);
  assert.deepEqual(
    results.map((result) => ({ name: result.name, passed: result.passed })),
    results.map((result) => ({ name: result.name, passed: true }))
  );
});

test("learning demo exports a trained network snapshot", () => {
  const demo = runLearningDemo(defaultConfig);
  assert.equal(demo.metrics.accuracy, 1);
  assert.ok(demo.network.synapses.length > 0);
  assert.ok(demo.events.length > 0);
});

test("model config field groups classify every public config field once", () => {
  const counts = new Map<string, number>();
  for (const fields of Object.values(configFieldGroups)) {
    for (const field of fields) {
      counts.set(field, (counts.get(field) ?? 0) + 1);
    }
  }

  const configFields = Object.keys(defaultConfig);
  const missing = configFields.filter((field) => !counts.has(field));
  const duplicated = [...counts.entries()].filter(([, count]) => count !== 1).map(([field]) => field);

  assert.deepEqual(missing, []);
  assert.deepEqual(duplicated, []);
});

test("scaled topology expands redundant interneuron stems without changing normalized readout drive", () => {
  const scaled = createScaledOfflineLearningTopologyBlueprint({
    interneuronCopiesPerSensor: 2,
    normalizeReadoutByCopies: true
  });
  const network = createLearningNetworkFromBlueprint(scaled, defaultConfig);
  const iFoodLeftReadouts = network.synapses.filter((synapse) =>
    synapse.preNeuronId === "iFoodLeft" || synapse.preNeuronId === "iFoodLeft_copy2"
  );
  const iFoodLeftToLeft = iFoodLeftReadouts
    .filter((synapse) => synapse.postNeuronId === "leftMotor")
    .reduce((sum, synapse) => sum + synapse.fastWeight, 0);

  assert.equal(offlineLearningTopologyBlueprint.interneuronNodes.length, 4);
  assert.equal(scaled.sensoryNodes.length, 4);
  assert.equal(scaled.interneuronNodes.length, 8);
  assert.equal(scaled.motorNodes.length, 2);
  assert.equal(scaled.synapses.length, 24);
  assert.equal(network.synapses.filter((synapse) => synapse.decayProtected).length, 8);
  assertClose(iFoodLeftToLeft, 0.35);
});

test("layered topology ratios distinguish proportional families from one-layer changes", () => {
  assert.deepEqual(reduceLayerCounts({ inputCount: 2, mediumCount: 10, outputCount: 2 }), {
    inputCount: 1,
    mediumCount: 5,
    outputCount: 1,
    commonScale: 2
  });
  assert.deepEqual(reduceLayerCounts({ inputCount: 4, mediumCount: 20, outputCount: 4 }), {
    inputCount: 1,
    mediumCount: 5,
    outputCount: 1,
    commonScale: 4
  });

  assert.equal(
    sameLayerRatio(
      { inputCount: 1, mediumCount: 5, outputCount: 1 },
      { inputCount: 2, mediumCount: 10, outputCount: 2 }
    ),
    true
  );
  assert.equal(
    sameLayerRatio(
      { inputCount: 1, mediumCount: 5, outputCount: 1 },
      { inputCount: 1, mediumCount: 10, outputCount: 2 }
    ),
    false
  );
  assert.equal(
    sameLayerRatio(
      { inputCount: 1, mediumCount: 5, outputCount: 1 },
      { inputCount: 2, mediumCount: 5, outputCount: 2 }
    ),
    false
  );
});

test("nearest layered topology builds fanout-n local connections and slot limits", () => {
  const topology = createNearestLayeredTopologyBlueprint({
    inputCount: 2,
    mediumCount: 10,
    outputCount: 2,
    synapsesPerInput: 5,
    synapsesPerMedium: 2
  });
  const network = createLearningNetworkFromBlueprint(topology, defaultConfig);
  const input0Targets = topology.synapses
    .filter((synapse) => synapse.preNeuronId === "input0")
    .map((synapse) => synapse.postNeuronId)
    .sort();
  const input1Targets = topology.synapses
    .filter((synapse) => synapse.preNeuronId === "input1")
    .map((synapse) => synapse.postNeuronId)
    .sort();

  assert.equal(topology.sensoryNodes.length, 2);
  assert.equal(topology.interneuronNodes.length, 10);
  assert.equal(topology.motorNodes.length, 2);
  assert.equal(topology.synapses.length, 30);
  assert.deepEqual(input0Targets, ["medium0", "medium1", "medium2", "medium3", "medium4"]);
  assert.deepEqual(input1Targets, ["medium5", "medium6", "medium7", "medium8", "medium9"]);
  assert.equal(topology.sensoryNodes.every((node) => node.maxOutputSlots === 5), true);
  assert.equal(topology.interneuronNodes.every((node) => node.maxInputSlots === 1), true);
  assert.equal(topology.interneuronNodes.every((node) => node.maxOutputSlots === 2), true);
  assert.equal(topology.motorNodes.every((node) => node.maxInputSlots === 10), true);
  assert.equal(network.synapses.filter((synapse) => synapse.decayProtected).length, 10);
});

test("plasticity mechanism calculations stay separated from update flow", () => {
  assertClose(nextActivityTrace(0.2, 1, 0.85), 0.32);

  const stdp = computeStdpEligibilityDelta(
    {
      preTrace: 0.5,
      postTrace: 0.25,
      preActive: 1,
      postActive: 1,
      effectSign: 1,
      effectiveWeight: 2
    },
    defaultConfig
  );
  assertClose(stdp.ltpEligibility, 1);
  assertClose(stdp.ltdEligibility, 0.5);
  assertClose(stdp.eligibilityDelta, 0.5);
  assertClose(nextEligibilityTrace(0.2, stdp.eligibilityDelta, defaultConfig), 0.68);

  assertClose(positiveEligibilityScale(4), 0.25);
  assertClose(computeRewardModulator(2, defaultConfig), Math.tanh(2));
  assertClose(computeRewardFastDelta(0.5, 2, 1, 0.4, defaultConfig), 0.004);
  assertClose(computeSupervisedFastDelta(true, false, 1, 1, defaultConfig), 0.08);
  assertClose(computeSupervisedFastDelta(false, true, 1, 1, defaultConfig), -0.08);
  assertClose(computeStableDepotentiationDelta(-0.5, 1, 1, defaultConfig), -0.01);
  assertClose(computeStableCaptureAmount(0.5, defaultConfig), 0.01);

  const goodTag = { present: true, badOutcome: false, goodAvoidance: true, intensity: 1 };
  const badTag = { present: true, badOutcome: true, goodAvoidance: false, intensity: 1 };
  const markerConfig = withConfig({
    aversiveTagStrategy: "avoidanceMarker",
    aversiveAvoidanceBonus: 0.6
  });
  const modulatorConfig = withConfig({
    aversiveTagStrategy: "modulatorOnly",
    aversiveTagGain: 0.5
  });
  const depotConfig = withConfig({
    aversiveTagStrategy: "badOutcomeDepotentiation",
    aversiveDepotentiationRate: 0.04
  });

  assertClose(computeAversiveRewardSignal(0.2, goodTag, markerConfig), 0.8);
  assertClose(computeAversiveRewardSignal(0.2, goodTag, defaultConfig), 0.2);
  assert.ok(computeAversiveModulator(0.1, goodTag, modulatorConfig) > 0.1);
  assert.equal(shouldApplyAversiveStableDepotentiation(badTag, depotConfig), true);
  assert.equal(shouldApplyAversiveStableDepotentiation(goodTag, depotConfig), false);
  assertClose(computeAversiveStableDepotentiationDelta(-0.5, 1, 1, depotConfig), -0.02);
});

test("learning trace records propagation, gate snapshots, and supervised weight changes", () => {
  const trace = runLearningTrace(defaultConfig, { epochs: 2, learningOn: true });
  const first = trace.episodes[0];
  const supervisedEvent = trace.episodes
    .flatMap((episode) => episode.weightEvents)
    .find((event) => event.kind === "supervised");

  assert.equal(trace.episodes.length, 8);
  assert.equal(first.inputLabel, "foodLeft");
  assert.equal(first.targetMotorId, "leftMotor");
  assert.equal(first.phases.length, 2);
  assert.ok(first.phases[0].propagationEvents.length > 0);
  assert.ok(first.phases[0].neurons.some((neuron) => neuron.branches.some((branch) => branch.active)));
  assert.ok(supervisedEvent);
  assert.notEqual(supervisedEvent.beforeFastWeight, supervisedEvent.afterFastWeight);
  assert.match(supervisedEvent.feedback, /^supervised-/);
});

test("learning-off trace keeps runtime evidence without supervised or capture updates", () => {
  const trace = runLearningTrace(defaultConfig, { epochs: 1, learningOn: false });
  const weightEvents = trace.episodes.flatMap((episode) => episode.weightEvents);

  assert.ok(trace.episodes.some((episode) => episode.phases.some((phase) => phase.propagationEvents.length > 0)));
  assert.equal(weightEvents.some((event) => event.kind === "supervised" || event.kind === "capture"), false);
});

test("trace explanation summarizes path and weight feedback", () => {
  const trace = runLearningTrace(defaultConfig, { epochs: 1, learningOn: true });
  const explanation = explainTrace(trace);

  assert.match(explanation, /Trace dg-snn-trace-v0\.1/);
  assert.match(explanation, /input foodLeft -> target leftMotor/);
  assert.match(explanation, /Active paths:/);
  assert.match(explanation, /supervised-target-reinforce/);
});

test("pre-2D audit passes required suites and keeps diagnostic boundaries visible", () => {
  const report = runPre2DAudit(defaultConfig);
  const requiredSuites = report.suites.filter((suite) => suite.required);
  const diagnosticSuites = report.suites.filter((suite) => !suite.required);
  const formatted = formatAuditReport(report);

  assert.equal(report.requiredPassed, true);
  assert.ok(requiredSuites.length >= 4);
  assert.ok(diagnosticSuites.length >= 1);
  assert.equal(requiredSuites.every((suite) => suite.passed), true);
  assert.match(formatted, /fixed-topology supervised offline learning/);
  assert.match(formatted, /input edge-case diagnostics/);
});

test("motor arbitration records noop, single action, and conflict", () => {
  assert.equal(arbitrateMotorAction([]).action, "noop");
  assert.equal(arbitrateMotorAction(["leftMotor"]).action, "left");
  assert.equal(arbitrateMotorAction(["rightMotor"]).action, "right");
  assert.equal(arbitrateMotorAction(["leftMotor", "rightMotor"]).action, "conflict");
});

test("2D-lite audit passes required suites and records conflict arbitration", () => {
  const report = runWorld2DAudit(defaultConfig);
  const requiredSuites = report.suites.filter((suite) => suite.required);
  const conflictSuite = report.suites.find((suite) => suite.name === "2D-lite composite and conflict arbitration");
  const formatted = formatWorld2DAuditReport(report);

  assert.equal(report.requiredPassed, true);
  assert.ok(requiredSuites.length >= 6);
  assert.equal(requiredSuites.every((suite) => suite.passed), true);
  assert.equal(conflictSuite?.metrics.conflictDecision, "conflict");
  assert.equal(conflictSuite?.metrics.conflictTaskSuccess, false);
  assert.match(formatted, /2D-lite/);
  assert.match(formatted, /fixed-topology supervised world tasks/);
});

test("2D-challenge audit passes required bottleneck suites and reports reward-only feasibility", () => {
  const report = runWorld2DChallengeAudit(defaultConfig);
  const requiredSuites = report.suites.filter((suite) => suite.required);
  const rewardOnlySuite = report.suites.find((suite) => suite.name === "2D-challenge reward-only feasibility");
  const conflictSuite = report.suites.find((suite) => suite.name === "2D-challenge conflict boundary");
  const formatted = formatWorld2DChallengeAuditReport(report);

  assert.equal(report.requiredPassed, true);
  assert.ok(requiredSuites.length >= 7);
  assert.equal(requiredSuites.every((suite) => suite.passed), true);
  assert.ok(Number(rewardOnlySuite?.metrics.rewardUpdateCount ?? 0) > 0);
  assert.equal(conflictSuite?.metrics.firstExecutedAction, "conflict");
  assert.match(formatted, /2D-challenge/);
  assert.match(formatted, /reward-only/);
});

test("reward-only challenge learning records advantage-based reward signals", () => {
  const result = runChallengeExperiment(createChallengeConfig(defaultConfig), {
    seed: 31,
    trainSeeds: [1],
    evalSeeds: [101],
    epochs: 1,
    learningMode: "rewardOnly"
  });
  const trainSteps = result.trace.episodes.flatMap((episode) =>
    episode.phase === "train" ? episode.steps : []
  );

  assert.ok(trainSteps.length > 0);
  assert.ok(trainSteps.every((step) => typeof step.rewardBaseline === "number"));
  assert.ok(trainSteps.every((step) => typeof step.rewardAdvantage === "number"));
  assert.ok(trainSteps.every((step) => typeof step.rewardSignal === "number"));
  assert.ok(trainSteps.every((step) => step.aversiveTag && typeof step.aversiveTag.present === "boolean"));
  assert.ok(trainSteps.some((step) => step.aversiveTag?.present));
  assert.ok(trainSteps.some((step) => step.rewardAdvantage !== step.reward));
  assert.equal(result.trace.config.rewardAdvantageBaselineAlpha, createChallengeConfig(defaultConfig).rewardAdvantageBaselineAlpha);
  assert.equal(result.trace.config.aversiveTagStrategy, "off");
});

test("aversive avoidance marker changes rewardOnly learning signal without changing raw reward", () => {
  const aversiveConfig = createChallengeConfig(
    withConfig({
      aversiveTagStrategy: "avoidanceMarker",
      aversiveAvoidanceBonus: 0.5
    })
  );
  const result = runChallengeExperiment(aversiveConfig, {
    seed: 31,
    trainSeeds: [1],
    evalSeeds: [101],
    epochs: 1,
    learningMode: "rewardOnly"
  });
  const toxinGoodSteps = result.trace.episodes
    .flatMap((episode) => episode.phase === "train" ? episode.steps : [])
    .filter((step) => step.aversiveTag?.goodAvoidance);

  assert.ok(toxinGoodSteps.length > 0);
  assert.ok(toxinGoodSteps.some((step) => step.rewardSignal > step.rewardAdvantage));
  assert.ok(toxinGoodSteps.every((step) => typeof step.reward === "number"));
  assert.equal(result.trace.config.aversiveTagStrategy, "avoidanceMarker");
  assert.equal(result.trace.config.aversiveAvoidanceBonus, 0.5);
});

test("epsilon-greedy exploration makes noop visible during rewardOnly training and stays deterministic", () => {
  const epsConfig = createChallengeConfig(
    withConfig({ explorationStrategy: "epsilonGreedy", explorationEpsilon: 0.2 })
  );
  const result = runChallengeExperiment(epsConfig, {
    seed: 41,
    trainSeeds: [1],
    evalSeeds: [101],
    epochs: 1,
    learningMode: "rewardOnly"
  });
  const trainSteps = result.trace.episodes.flatMap((episode) =>
    episode.phase === "train" ? episode.steps : []
  );

  // epsilon-greedy does not force a motor on every noop step, so the network's
  // own noop decisions must now appear in the training trace (the diagnostic
  // signal that conflict-gated exploration used to mask).
  assert.ok(trainSteps.some((step) => step.executedAction === "noop"));
  // exploration still occurs (some steps carry an explorationAction override).
  assert.ok(trainSteps.some((step) => step.explorationAction !== null));
  // trace records the strategy that produced it.
  assert.equal(result.trace.config.explorationStrategy, "epsilonGreedy");
  assert.equal(result.trace.config.explorationEpsilon, 0.2);

  // determinism: same seed reproduces the stable trace.
  const replay = runChallengeExperiment(epsConfig, {
    seed: 41,
    trainSeeds: [1],
    evalSeeds: [101],
    epochs: 1,
    learningMode: "rewardOnly"
  });
  assert.deepEqual(replay.trace.episodes, result.trace.episodes);
});

test("conflictGated exploration strategy remains available as a toggle", () => {
  const cgConfig = createChallengeConfig(
    withConfig({ explorationStrategy: "conflictGated" })
  );
  const result = runChallengeExperiment(cgConfig, {
    seed: 41,
    trainSeeds: [1],
    evalSeeds: [101],
    epochs: 1,
    learningMode: "rewardOnly"
  });
  assert.equal(result.trace.config.explorationStrategy, "conflictGated");
  const trainSteps = result.trace.episodes.flatMap((episode) =>
    episode.phase === "train" ? episode.steps : []
  );
  // legacy behaviour: a noop decision is always overridden by a forced motor,
  // so executedAction must never be "noop" during rewardOnly training.
  assert.ok(trainSteps.every((step) => step.executedAction !== "noop"));
});

test("2D-challenge pretrained exports preserve learned network snapshots", () => {
  const exports = createChallengePretrainExports(defaultConfig, { outputDir: "exports/test-pretrained" });
  const modes = exports.map((item) => item.mode).sort();
  const rewardOnly = exports.find((item) => item.mode === "rewardOnly");
  const supervised = exports.find((item) => item.mode === "supervised");

  assert.deepEqual(modes, ["rewardOnly", "supervised"]);
  assert.ok(rewardOnly);
  assert.ok(supervised);
  assert.ok(rewardOnly.snapshot.synapses.length > 0);
  assert.ok(supervised.snapshot.synapses.length > 0);
  assert.equal(rewardOnly.snapshot.metrics.learningMode, "rewardOnly");
  assert.equal(supervised.snapshot.metrics.learningMode, "supervised");
  assert.ok(Number(rewardOnly.snapshot.metrics.rewardUpdateCount) > 0);
  assert.ok(Number(supervised.snapshot.metrics.supervisedUpdateCount) > 0);
  assert.equal(rewardOnly.snapshot.events[0] && typeof rewardOnly.snapshot.events[0], "object");
});

test("2D-complex audit passes required bottleneck suites and preserves conflict boundary", () => {
  const report = runWorld2DComplexAudit(defaultConfig);
  const requiredSuites = report.suites.filter((suite) => suite.required);
  const conflictSuite = report.suites.find((suite) => suite.name === "2D-complex conflict boundary (Family E)");
  const baselineSuite = report.suites.find((suite) => suite.name === "2D-complex supervised multi-step baseline (Family A)");
  const formatted = formatWorld2DComplexAuditReport(report);

  assert.equal(report.requiredPassed, true);
  assert.ok(requiredSuites.length >= 9);
  assert.equal(requiredSuites.every((suite) => suite.passed), true);
  assert.equal(conflictSuite?.metrics.firstExecutedAction, "conflict");
  assert.ok(Number(baselineSuite?.metrics.supervisedUpdateCount ?? 0) > 0);
  assert.match(formatted, /2D-complex/);
  assert.match(formatted, /Family/);
});

test("arbitration audit passes required suites and preserves true conflict boundary", () => {
  const report = runArbitrationAudit(defaultConfig);
  const requiredSuites = report.suites.filter((suite) => suite.required);
  const semanticSuite = report.suites.find((suite) => suite.name === "arbitration supervised semantic conflict resolution (Family F train)");
  const conflictSuite = report.suites.find((suite) => suite.name === "arbitration true conflict preservation (Family E)");
  const formatted = formatArbitrationAuditReport(report);

  assert.equal(report.requiredPassed, true);
  assert.ok(requiredSuites.length >= 7);
  assert.equal(requiredSuites.every((suite) => suite.passed), true);
  assert.ok(Number(semanticSuite?.metrics.trainedSuccessRate ?? 0) >= 0.9);
  assert.equal(conflictSuite?.metrics.firstExecutedActions, "conflict,conflict");
  assert.match(formatted, /arbitration/);
});

test("arbitration matrix audit passes required suites and preserves true conflict at default tau", () => {
  const report = runArbitrationMatrixAudit(defaultConfig);
  const requiredSuites = report.suites.filter((suite) => suite.required);
  const disjointSuite = report.suites.find((suite) => suite.name === "arbitration matrix disjoint scenario generalization");
  const conflictSuite = report.suites.find((suite) => suite.name === "arbitration matrix true conflict preservation at default tau");
  const blankSuite = report.suites.find((suite) => suite.name === "arbitration matrix blank world preservation");
  const formatted = formatArbitrationMatrixReport(report);

  assert.equal(report.requiredPassed, true);
  assert.ok(requiredSuites.length >= 5);
  assert.equal(requiredSuites.every((suite) => suite.passed), true);
  assert.ok(Number(disjointSuite?.metrics.trainedSuccessRate ?? 0) >= 0.8);
  assert.ok(Number(disjointSuite?.metrics.freshSuccessRate ?? 1) <= 0.2);
  assert.ok(Number(conflictSuite?.metrics.fallbackRate ?? 0) >= 0.9);
  assert.equal(blankSuite?.metrics.noopRate, 1);
  assert.match(formatted, /arbitration matrix/);
  assert.match(formatted, /tau/);
});

test("transfer audit passes required suites and reports pretrained-vs-fresh separation", () => {
  const report = runTransferAudit(defaultConfig);
  const requiredSuites = report.suites.filter((suite) => suite.required);
  const formatted = formatTransferAuditReport(report);

  assert.equal(report.requiredPassed, true);
  assert.ok(requiredSuites.length >= 5);
  assert.equal(requiredSuites.every((suite) => suite.passed), true);
  assert.match(formatted, /transfer/);
  assert.match(formatted, /pretrained/);
});

test("transfer matrix aggregates stress axes and surfaces cell gate status", async () => {
  const report = await runTransferAuditMatrix({
    pretrainSeeds: [101, 102],
    evalSeedSets: [[201, 202, 203]],
    concurrency: 2
  });
  const formatted = formatTransferMatrixReport(report);

  assert.equal(report.summary.cellsRun, 2);
  assert.equal(report.cells.every((cell) => cell.report !== null), true);
  assert.equal(report.cells.every((cell) => cell.error === null), true);
  assert.ok(report.summary.rewardOnlySuccessSeparation);
  assert.ok(report.summary.dropout02.rewardOnlyDelta);
  assert.ok(report.summary.dropout03.rewardOnlyDelta);
  assert.ok(report.summary.continuedLearning);
  assert.ok(report.summary.wrongPrior.postCLWrongDirectionMaxStableWeight);
  assert.ok(report.summary.wrongPrior.postCLWrongDirectionMaxFastWeight);
  assert.equal(report.summary.continuedLearning.reversals.length, 0);
  assert.equal(report.requiredPassed, report.summary.failedCells.length === 0);
  assert.ok(report.summary.rewardOnlyMeanRewardDelta.min >= 0);
  assert.ok(report.summary.rewardOnlySuccessSeparation.min >= 0);
  assert.ok(report.summary.continuedLearning.separation.min >= 0);
  assert.match(formatted, /Stress axes:/);
  assert.match(formatted, /fresh=noop/);
  assert.match(formatted, /wrong-prior postCL max stable/);
  assert.match(formatted, /wrong-prior postCL dual-lock/);
  assert.match(formatted, /continued-learning/);
});

test("layered topology readoutMode=prewired is the default and keeps stem+readout edges", () => {
  const explicit = createNearestLayeredTopologyBlueprint({
    inputCount: 1,
    mediumCount: 5,
    outputCount: 1,
    synapsesPerInput: 5,
    synapsesPerMedium: 1,
    readoutMode: "prewired"
  });
  const implicit = createNearestLayeredTopologyBlueprint({
    inputCount: 1,
    mediumCount: 5,
    outputCount: 1,
    synapsesPerInput: 5,
    synapsesPerMedium: 1
  });
  // 5 stem (sensory->inter) + 5 readout (inter->motor) = 10 edges.
  assert.equal(explicit.synapses.length, 10);
  assert.equal(implicit.synapses.length, 10);
  assert.equal(explicit.synapses.filter((s) => s.kind === "structuralStem").length, 5);
  assert.equal(explicit.synapses.filter((s) => s.kind === "plasticReadout").length, 5);
});

test("layered topology readoutMode=stem leaves readout empty and reserves growth slots", () => {
  const topology = createNearestLayeredTopologyBlueprint({
    inputCount: 1,
    mediumCount: 5,
    outputCount: 1,
    synapsesPerInput: 5,
    synapsesPerMedium: 1,
    readoutMode: "stem"
  });
  // Only stem edges; no pre-built readout.
  assert.equal(topology.synapses.length, 5);
  assert.equal(topology.synapses.every((s) => s.kind === "structuralStem"), true);
  // Interneurons and motors need free slots for the developmental loop to
  // attach readout synapses (otherwise hasFreeSlot is always false).
  assert.equal(topology.interneuronNodes.every((n) => (n.maxOutputSlots ?? 0) >= 5), true);
  assert.equal(topology.motorNodes.every((n) => (n.maxInputSlots ?? 0) >= 5), true);
});

test("layered topology readoutMode=empty places neurons only and reserves growth slots", () => {
  const topology = createNearestLayeredTopologyBlueprint({
    inputCount: 2,
    mediumCount: 5,
    outputCount: 2,
    synapsesPerInput: 3,
    synapsesPerMedium: 1,
    readoutMode: "empty"
  });
  assert.equal(topology.synapses.length, 0);
  assert.equal(topology.sensoryNodes.length, 2);
  assert.equal(topology.interneuronNodes.length, 5);
  assert.equal(topology.motorNodes.length, 2);
  // Every node needs growth slots so spontaneous wiring can attach on both
  // sides of each layer.
  assert.equal(topology.sensoryNodes.every((n) => (n.maxOutputSlots ?? 0) >= 5), true);
  assert.equal(
    topology.interneuronNodes.every((n) => (n.maxInputSlots ?? 0) >= 2 && (n.maxOutputSlots ?? 0) >= 5),
    true
  );
  assert.equal(topology.motorNodes.every((n) => (n.maxInputSlots ?? 0) >= 5), true);
});

test("developmental step forms readout connections from a stem-only network", () => {
  const topology = createNearestLayeredTopologyBlueprint({
    inputCount: 1,
    mediumCount: 5,
    outputCount: 1,
    synapsesPerInput: 5,
    synapsesPerMedium: 1,
    readoutMode: "stem"
  });
  const network = createLearningNetworkFromBlueprint(topology, defaultConfig);
  const before = network.synapses.length;
  const rng = new SeededRandom(42);
  const metrics = tryFormConnections(
    network.neurons,
    network.synapses,
    network.pairMemory,
    network.tick,
    defaultConfig,
    rng,
    8
  );
  // Readout synapses (inter->motor) should have grown.
  assert.ok(metrics.formed > 0, "expected at least one formed connection");
  assert.ok(network.synapses.length > before);
  const roles = new Map(network.neurons.map((n) => [n.id, n.role]));
  const hasReadout = network.synapses.some(
    (s) => roles.get(s.preNeuronId) === "interneuron" && roles.get(s.postNeuronId) === "motor"
  );
  assert.equal(hasReadout, true);
});

test("developmental pairMemory tombstone blocks immediate re-formation after prune", () => {
  const topology = createNearestLayeredTopologyBlueprint({
    inputCount: 1,
    mediumCount: 5,
    outputCount: 1,
    synapsesPerInput: 5,
    synapsesPerMedium: 1,
    readoutMode: "stem"
  });
  const network = createLearningNetworkFromBlueprint(topology, defaultConfig);
  const rng = new SeededRandom(7);
  // Form connections.
  const formed = tryFormConnections(network.neurons, network.synapses, network.pairMemory, 0, defaultConfig, rng, 8);
  assert.ok(formed.formed > 0);
  // Force a candidate into the prune path: age it past candidateMaxAge with
  // low recentUse so updateConnectionStates prunes it (mirrors evaluation Test D).
  const victim = network.synapses.find((s) => s.state === "candidate");
  assert.ok(victim);
  victim.age = defaultConfig.candidateMaxAge + 1;
  victim.recentUse = 0;
  const pruned = updateConnectionStates(network.neurons, network.synapses, network.pairMemory, 1, defaultConfig);
  assert.ok(pruned.pruned > 0);
  // A tombstone for the pruned pair should now block re-formation within cooldown.
  const blocked = tryFormConnections(network.neurons, network.synapses, network.pairMemory, 2, defaultConfig, rng, 8);
  assert.ok(blocked.tombstoneHit > 0, "expected tombstone to block immediate re-formation");
});
