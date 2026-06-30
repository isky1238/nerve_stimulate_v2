import test from "node:test";
import assert from "node:assert/strict";
import { defaultConfig, withConfig } from "../src/config/newModelConfig";
import { arbitrateMotorAction } from "../src/core/arbitration";
import { formatAuditReport, runPre2DAudit } from "../src/core/audit";
import { runAllEvaluations, runLearningDemo } from "../src/core/evaluation";
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
  assert.ok(trainSteps.some((step) => step.rewardAdvantage !== step.reward));
  assert.equal(result.trace.config.rewardAdvantageBaselineAlpha, createChallengeConfig(defaultConfig).rewardAdvantageBaselineAlpha);
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

test("transfer matrix aggregates stress axes and enforces rewardOnly/continued-learning gates", async () => {
  const report = await runTransferAuditMatrix({
    pretrainSeeds: [101, 102],
    evalSeedSets: [[201, 202, 203]],
    concurrency: 2
  });
  const formatted = formatTransferMatrixReport(report);

  assert.equal(report.requiredPassed, true);
  assert.equal(report.summary.cellsRun, 2);
  assert.ok(report.summary.rewardOnlySuccessSeparation);
  assert.ok(report.summary.dropout02.rewardOnlyDelta);
  assert.ok(report.summary.dropout03.rewardOnlyDelta);
  assert.ok(report.summary.continuedLearning);
  assert.ok(report.summary.wrongPrior.postCLWrongDirectionMaxStableWeight);
  assert.ok(report.summary.wrongPrior.postCLWrongDirectionMaxFastWeight);
  assert.equal(report.summary.continuedLearning.reversals.length, 0);
  assert.ok(report.summary.rewardOnlyMeanRewardDelta.min > 0);
  assert.ok(report.summary.rewardOnlySuccessSeparation.min >= 0);
  assert.ok(report.summary.continuedLearning.separation.min >= 0);
  assert.match(formatted, /Stress axes:/);
  assert.match(formatted, /fresh=noop/);
  assert.match(formatted, /wrong-prior postCL max stable/);
  assert.match(formatted, /wrong-prior postCL dual-lock/);
  assert.match(formatted, /continued-learning/);
});
