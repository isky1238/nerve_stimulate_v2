import { createHash } from "node:crypto";
import { defaultConfig, ModelConfig, withConfig } from "../config/newModelConfig";
import { createOfflineLearningNetwork, LearningNetwork, offlinePatterns, Pattern } from "./evaluation";
import { indexNeurons, integrateNeuron, resetBranchInputs, resetNeuronRuntime, setSensoryOutput } from "./neuron";
import { applySupervisedMotorLearning, captureStableWeights, decayWeights, updateEligibility } from "./plasticity";
import { SeededRandom } from "./random";
import { propagateSynapses } from "./synapse";
import { runLearningTrace } from "./trace";

export interface Pre2DAuditReport {
  version: string;
  generatedAt: string;
  requiredPassed: boolean;
  summary: string;
  suites: AuditSuiteResult[];
}

export interface AuditSuiteResult {
  name: string;
  required: boolean;
  passed: boolean;
  metrics: Record<string, number | string | boolean>;
  conclusion: string;
  notes: string[];
}

interface ExperimentPattern {
  sensorIds: string[];
  expectedMotorId: string | null;
  label: string;
}

interface ExperimentResult {
  accuracy: number;
  finalAccuracy: number;
  stableWeightRatio: number;
  supervisedUpdates: number;
  captureUpdates: number;
  trials: TrialResult[];
}

interface TrialResult {
  label: string;
  expectedMotorId: string | null;
  activeMotors: string[];
  correct: boolean;
}

interface ExperimentOptions {
  epochs: number;
  learningOn: boolean;
  orderSeed?: number;
  evaluationPatterns?: ExperimentPattern[];
}

const REQUIRED_FINAL_ACCURACY = 0.95;

export function runPre2DAudit(config: ModelConfig = defaultConfig): Pre2DAuditReport {
  const auditConfig = createAuditConfig(config);
  const suites = [
    auditDeterminism(auditConfig),
    auditLearningAblation(auditConfig),
    auditOrderRobustness(auditConfig),
    auditMirrorMapping(auditConfig),
    auditBlankInputSilence(auditConfig),
    auditInputDiagnostics(auditConfig)
  ];
  const requiredPassed = suites.filter((suite) => suite.required).every((suite) => suite.passed);

  return {
    version: "dg-snn-pre2d-audit-v0.1",
    generatedAt: new Date().toISOString(),
    requiredPassed,
    summary: requiredPassed
      ? "Required pre-2D checks passed, but conclusions remain limited to fixed-topology supervised offline learning."
      : "At least one required pre-2D check failed; do not enter 2D environment tests yet.",
    suites
  };
}

export function formatAuditReport(report: Pre2DAuditReport): string {
  const lines = [
    `Audit ${report.version}`,
    `requiredPassed=${report.requiredPassed}`,
    report.summary
  ];

  for (const suite of report.suites) {
    lines.push("");
    lines.push(`${suite.passed ? "PASS" : "FAIL"} ${suite.required ? "REQUIRED" : "DIAGNOSTIC"} ${suite.name}`);
    lines.push(`  metrics: ${JSON.stringify(suite.metrics)}`);
    lines.push(`  conclusion: ${suite.conclusion}`);
    for (const note of suite.notes) {
      lines.push(`  note: ${note}`);
    }
  }

  return lines.join("\n");
}

function auditDeterminism(config: ModelConfig): AuditSuiteResult {
  const first = runLearningTrace(config, { epochs: 12, learningOn: true, seed: 1 });
  const second = runLearningTrace(config, { epochs: 12, learningOn: true, seed: 1 });
  const firstStable = stableTraceProjection(first);
  const secondStable = stableTraceProjection(second);
  const passed = firstStable === secondStable;
  const traceDigest = digest(firstStable);

  return {
    name: "reproducibility / deterministic replay",
    required: true,
    passed,
    metrics: {
      sameStableTrace: passed,
      firstFinalEpochAccuracy: first.metrics.finalEpochAccuracy,
      secondFinalEpochAccuracy: second.metrics.finalEpochAccuracy,
      supervisedUpdates: first.metrics.supervisedUpdateCount,
      normalizedTraceDigest: traceDigest
    },
    conclusion: passed
      ? "Same config and seed produce identical trace metrics, episodes, and weight events after removing timestamps."
      : "Same config and seed produced different trace evidence.",
    notes: [
      "Timestamps and default export filenames are intentionally excluded from determinism checks.",
      "This does not yet prove deterministic behavior for future stochastic 2D worlds."
    ]
  };
}

function auditLearningAblation(config: ModelConfig): AuditSuiteResult {
  const learningOn = runExperiment(config, canonicalPatterns(), { epochs: 40, learningOn: true });
  const learningOff = runExperiment(config, canonicalPatterns(), { epochs: 40, learningOn: false });
  const passed =
    learningOn.finalAccuracy >= REQUIRED_FINAL_ACCURACY &&
    learningOn.finalAccuracy > learningOff.finalAccuracy + 0.5 &&
    learningOn.supervisedUpdates > 0 &&
    learningOn.stableWeightRatio > 0.2;

  return {
    name: "actual learnability / learning-off ablation",
    required: true,
    passed,
    metrics: {
      learningOnFinalAccuracy: learningOn.finalAccuracy,
      learningOffFinalAccuracy: learningOff.finalAccuracy,
      learningOnAllTrialAccuracy: learningOn.accuracy,
      learningOffAllTrialAccuracy: learningOff.accuracy,
      supervisedUpdates: learningOn.supervisedUpdates,
      stableWeightRatio: learningOn.stableWeightRatio
    },
    conclusion: passed
      ? "The fixed offline readout learns only when supervised plasticity is enabled."
      : "The learning-on versus learning-off evidence is not strong enough.",
    notes: [
      "This proves supervised readout adaptation on a fixed basis, not autonomous task discovery.",
      "All-trial accuracy includes early training mistakes; final accuracy is the gating metric."
    ]
  };
}

function auditOrderRobustness(config: ModelConfig): AuditSuiteResult {
  const permutations = permutePatterns(canonicalPatterns());
  const results = permutations.map((patterns) =>
    runExperiment(config, patterns, { epochs: 40, learningOn: true })
  );
  const finalAccuracies = results.map((result) => result.finalAccuracy);
  const passed = finalAccuracies.every((accuracy) => accuracy >= REQUIRED_FINAL_ACCURACY);

  return {
    name: "all-permutation order robustness",
    required: true,
    passed,
    metrics: {
      permutations: permutations.length,
      minFinalAccuracy: Math.min(...finalAccuracies),
      maxFinalAccuracy: Math.max(...finalAccuracies),
      meanFinalAccuracy: mean(finalAccuracies)
    },
    conclusion: passed
      ? "The offline mapping is not dependent on one fixed presentation order."
      : "At least one shuffled order failed the final-epoch accuracy threshold.",
    notes: [
      "All 24 permutations of the four offline patterns are covered.",
      "Network initialization is still fixed, so this is an order-robustness check, not a stochastic initialization check."
    ]
  };
}

function auditMirrorMapping(config: ModelConfig): AuditSuiteResult {
  const result = runExperiment(config, mirroredPatterns(), { epochs: 40, learningOn: true, orderSeed: 7 });
  const passed = result.finalAccuracy >= REQUIRED_FINAL_ACCURACY && result.supervisedUpdates > 0;

  return {
    name: "alternate label-set learnability",
    required: true,
    passed,
    metrics: {
      finalAccuracy: result.finalAccuracy,
      allTrialAccuracy: result.accuracy,
      supervisedUpdates: result.supervisedUpdates,
      stableWeightRatio: result.stableWeightRatio
    },
    conclusion: passed
      ? "The same fixed topology can learn a left/right mirrored target mapping."
      : "The model appears overfit to the canonical target mapping.",
    notes: [
      "This is still not broad generalization; it only checks one alternate label set.",
      "Passing this reduces the risk that Test E is only a hardcoded canonical mapping."
    ]
  };
}

function auditBlankInputSilence(config: ModelConfig): AuditSuiteResult {
  const result = runExperiment(config, canonicalPatterns(), {
    epochs: 40,
    learningOn: true,
    orderSeed: 11,
    evaluationPatterns: [{ label: "blank", sensorIds: [], expectedMotorId: null }]
  });
  const blank = result.trials[0];
  const passed = blank.correct && blank.activeMotors.length === 0;

  return {
    name: "blank input silence",
    required: true,
    passed,
    metrics: {
      blankCorrect: blank.correct,
      blankActiveMotors: blank.activeMotors.join(","),
      finalAccuracyOnBlankSet: result.finalAccuracy
    },
    conclusion: passed
      ? "A trained offline network stays silent when no sensory input is active."
      : "A trained offline network fires without sensory evidence; 2D entry is blocked.",
    notes: [
      "Blank input is a hard safety gate because 2D worlds will frequently contain no actionable object.",
      "This does not define behavior for contradictory multi-object observations."
    ]
  };
}

function auditInputDiagnostics(config: ModelConfig): AuditSuiteResult {
  const evaluationPatterns = [
    { label: "same-action-composite", sensorIds: ["foodLeft", "toxinRight"], expectedMotorId: "leftMotor" },
    { label: "opposite-action-conflict", sensorIds: ["foodLeft", "foodRight"], expectedMotorId: null }
  ];
  const result = runExperiment(config, canonicalPatterns(), {
    epochs: 40,
    learningOn: true,
    orderSeed: 11,
    evaluationPatterns
  });
  const sameAction = result.trials.find((trial) => trial.label === "same-action-composite");
  const conflict = result.trials.find((trial) => trial.label === "opposite-action-conflict");
  const passed = sameAction?.correct === true && conflict?.correct === true;

  return {
    name: "input edge-case diagnostics",
    required: false,
    passed,
    metrics: {
      sameActionCompositeCorrect: sameAction?.correct ?? false,
      sameActionCompositeActiveMotors: sameAction?.activeMotors.join(",") ?? "",
      conflictNoSingleMotor: conflict?.correct ?? false,
      conflictActiveMotors: conflict?.activeMotors.join(",") ?? ""
    },
    conclusion: passed
      ? "Composite and contradictory inputs both have defined behavior under this diagnostic set."
      : "Composite input is partly handled, but contradictory simultaneous inputs still lack arbitration.",
    notes: [
      "Conflict input has no learned arbitration rule in the current offline task.",
      "2D entry should not assume robust behavior for simultaneous or contradictory stimuli."
    ]
  };
}

function runExperiment(
  config: ModelConfig,
  trainingPatterns: ExperimentPattern[],
  options: ExperimentOptions
): ExperimentResult {
  const network = createOfflineLearningNetwork(config);
  let correct = 0;
  let total = 0;
  let supervisedUpdates = 0;
  let captureUpdates = 0;

  for (let epoch = 0; epoch < options.epochs; epoch += 1) {
    const epochPatterns =
      options.orderSeed === undefined
        ? trainingPatterns
        : shufflePatterns(trainingPatterns, options.orderSeed + epoch);

    for (const pattern of epochPatterns) {
      const result = runAuditTrial(network, pattern, config, options.learningOn);
      correct += result.correct ? 1 : 0;
      total += 1;
      supervisedUpdates += result.supervisedUpdates;
      captureUpdates += result.captureUpdates;
    }
  }

  const evaluationPatterns = options.evaluationPatterns ?? trainingPatterns;
  const trials = evaluationPatterns.map((pattern) => runAuditTrial(network, pattern, config, false));
  const finalAccuracy = trials.filter((trial) => trial.correct).length / Math.max(1, trials.length);

  return {
    accuracy: correct / Math.max(1, total),
    finalAccuracy,
    stableWeightRatio: stableWeightRatio(network),
    supervisedUpdates,
    captureUpdates,
    trials
  };
}

function runAuditTrial(
  network: LearningNetwork,
  pattern: ExperimentPattern,
  config: ModelConfig,
  learningOn: boolean
): TrialResult & { supervisedUpdates: number; captureUpdates: number } {
  const neuronsById = indexNeurons(network.neurons);

  for (const neuron of network.neurons) {
    resetNeuronRuntime(neuron);
  }

  const activeSensors = new Set(pattern.sensorIds);
  for (const neuron of network.neurons) {
    if (neuron.role === "sensory") {
      setSensoryOutput(neuron, activeSensors.has(neuron.id) ? 1 : 0);
    }
  }

  network.tick += 1;
  resetBranchInputs(network.neurons);
  propagateSynapses(neuronsById, network.synapses, network.tick, config);
  for (const neuron of network.neurons) {
    if (neuron.role === "interneuron") {
      integrateNeuron(neuron, config);
    }
  }

  for (const neuron of network.neurons) {
    if (neuron.role === "sensory") {
      setSensoryOutput(neuron, 0);
    }
  }

  network.tick += 1;
  resetBranchInputs(network.neurons);
  propagateSynapses(neuronsById, network.synapses, network.tick, config);
  for (const neuron of network.neurons) {
    if (neuron.role === "motor") {
      integrateNeuron(neuron, config);
    }
  }

  updateEligibility(network.synapses, neuronsById, config);
  const activeMotors = network.neurons
    .filter((neuron) => neuron.role === "motor" && neuron.outputSignal !== 0)
    .map((neuron) => neuron.id);
  const activeMotorSet = new Set(activeMotors);
  const correct =
    pattern.expectedMotorId === null
      ? activeMotorSet.size === 0
      : activeMotorSet.has(pattern.expectedMotorId) && activeMotorSet.size === 1;

  let supervisedUpdates = 0;
  let captureUpdates = 0;
  if (learningOn && pattern.expectedMotorId !== null) {
    supervisedUpdates = applySupervisedMotorLearning(
      network.synapses,
      neuronsById,
      pattern.expectedMotorId,
      activeMotorSet,
      1,
      config
    ).length;
    captureUpdates = captureStableWeights(network.synapses, config).length;
  }

  decayWeights(network.synapses, config);

  return {
    label: pattern.label,
    expectedMotorId: pattern.expectedMotorId,
    activeMotors,
    correct,
    supervisedUpdates,
    captureUpdates
  };
}

function canonicalPatterns(): ExperimentPattern[] {
  return offlinePatterns().map(fromPattern);
}

function mirroredPatterns(): ExperimentPattern[] {
  return offlinePatterns().map((pattern) => ({
    sensorIds: [pattern.sensorId],
    expectedMotorId: pattern.targetMotorId === "leftMotor" ? "rightMotor" : "leftMotor",
    label: `${pattern.sensorId}->mirror`
  }));
}

function fromPattern(pattern: Pattern): ExperimentPattern {
  return {
    sensorIds: [pattern.sensorId],
    expectedMotorId: pattern.targetMotorId,
    label: `${pattern.sensorId}->${pattern.targetMotorId}`
  };
}

function createAuditConfig(config: ModelConfig): ModelConfig {
  return withConfig({
    ...config,
    leak: 1,
    branchLocalThreshold: 0.1,
    dendriteGateThreshold: 0.1,
    axonThreshold: 1,
    thresholdAdaptRate: 0,
    refractorySteps: 0,
    fastDecay: 0.9995,
    stableThreshold: 0.12,
    useThreshold: 0.08
  });
}

function shufflePatterns(patterns: ExperimentPattern[], seed: number): ExperimentPattern[] {
  const rng = new SeededRandom(seed);
  const shuffled = [...patterns];

  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = rng.nextInt(index + 1);
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }

  return shuffled;
}

function permutePatterns(patterns: ExperimentPattern[]): ExperimentPattern[][] {
  if (patterns.length <= 1) {
    return [patterns];
  }

  const permutations: ExperimentPattern[][] = [];

  for (let index = 0; index < patterns.length; index += 1) {
    const head = patterns[index];
    const tail = [...patterns.slice(0, index), ...patterns.slice(index + 1)];

    for (const permutation of permutePatterns(tail)) {
      permutations.push([head, ...permutation]);
    }
  }

  return permutations;
}

function stableWeightRatio(network: LearningNetwork): number {
  const stable = network.synapses.reduce((sum, synapse) => sum + synapse.stableWeight, 0);
  const total = network.synapses.reduce(
    (sum, synapse) => sum + synapse.fastWeight + synapse.stableWeight,
    0
  );
  return stable / Math.max(1e-9, total);
}

function stableTraceProjection(trace: ReturnType<typeof runLearningTrace>): string {
  return JSON.stringify({
    metrics: trace.metrics,
    episodes: trace.episodes
  });
}

function digest(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}
