import { fork } from "node:child_process";
import { rmSync } from "node:fs";
import { join } from "node:path";
import { TransferAuditReport, TransferAuditSuiteResult } from "./transferAudit";

export const MATRIX_TMP_DIR = join("exports", "pretrained", "transfer-matrix-tmp");

export function cellTmpDir(label: string): string {
  const root = process.env.TRANSFER_MATRIX_TMP_ROOT ?? MATRIX_TMP_DIR;
  return join(root, label);
}

export interface TransferMatrixOptions {
  pretrainSeeds?: number[];
  evalSeedSets?: number[][];
  concurrency?: number;
  tmpRoot?: string;
}

export interface TransferMatrixCell {
  label: string;
  pretrainSeed: number;
  evalSeeds: number[];
  auditSeed: number;
}

export interface TransferMatrixCellResult {
  cell: TransferMatrixCell;
  report: TransferAuditReport | null;
  error: string | null;
  durationMs: number;
}

export interface TransferMatrixReport {
  version: string;
  generatedAt: string;
  requiredPassed: boolean;
  grid: {
    pretrainSeeds: number[];
    evalSeedSets: number[][];
    cellCount: number;
    concurrency: number;
  };
  cells: TransferMatrixCellResult[];
  summary: {
    cellsRun: number;
    cellsAllRequiredPass: number;
    suitePassCounts: Record<string, { passed: number; total: number }>;
    supervisedSeparation: { min: number; mean: number; max: number };
    rewardOnlyMeanRewardDelta: { min: number; mean: number; max: number };
    rewardOnlySuccessSeparation: { min: number; mean: number; max: number };
    dropout02: {
      supervisedSeparation: { min: number; mean: number; max: number };
      rewardOnlyDelta: { min: number; mean: number; max: number };
    };
    dropout03: {
      supervisedSeparation: { min: number; mean: number; max: number };
      rewardOnlyDelta: { min: number; mean: number; max: number };
    };
    continuedLearning: {
      separation: { min: number; mean: number; max: number };
      reversals: string[];
    };
    wrongPrior: {
      separation: { min: number; mean: number; max: number };
      nonVacuousCells: number;
      reversals: string[];
    };
    failedCells: string[];
  };
}

const MATRIX_VERSION = "dg-snn-transfer-matrix-v0.1";

const DEFAULT_PRETRAIN_SEEDS = [101, 102, 103, 104, 105];
const DEFAULT_EVAL_SEED_SETS: number[][] = [
  [201, 202, 203, 204, 205],
  [301, 302, 303, 304, 305],
  [401, 402, 403, 404, 405]
];
const DEFAULT_CONCURRENCY = 4;

const DEFAULT_MULTI_MATRIX_POOLS: { pretrainSeeds: number[]; evalSeedSets: number[][] }[] = [
  {
    pretrainSeeds: [101, 102, 103, 104, 105],
    evalSeedSets: [
      [201, 202, 203, 204, 205],
      [301, 302, 303, 304, 305],
      [401, 402, 403, 404, 405]
    ]
  },
  {
    pretrainSeeds: [501, 502, 503, 504, 505],
    evalSeedSets: [
      [601, 602, 603, 604, 605],
      [701, 702, 703, 704, 705],
      [801, 802, 803, 804, 805]
    ]
  },
  {
    pretrainSeeds: [901, 902, 903, 904, 905],
    evalSeedSets: [
      [1001, 1002, 1003, 1004, 1005],
      [1101, 1102, 1103, 1104, 1105],
      [1201, 1202, 1203, 1204, 1205]
    ]
  }
];
const DEFAULT_MULTI_MATRIX_CONCURRENCY = 2;

const REQUIRED_SUITE_NAMES = [
  "transfer loader preserves frozen-eval behavior (disk round-trip)",
  "transfer frozen-pretrained vs frozen-fresh separation (supervised)",
  "transfer frozen-pretrained vs frozen-fresh separation (rewardOnly)",
  "transfer eval seed isolation",
  "transfer conflict boundary preservation",
  "transfer blank world preservation"
];

export function buildTransferMatrixGrid(options: TransferMatrixOptions = {}): TransferMatrixCell[] {
  const pretrainSeeds = options.pretrainSeeds ?? DEFAULT_PRETRAIN_SEEDS;
  const evalSeedSets = options.evalSeedSets ?? DEFAULT_EVAL_SEED_SETS;
  const cells: TransferMatrixCell[] = [];

  for (const pretrainSeed of pretrainSeeds) {
    for (const evalSeeds of evalSeedSets) {
      const auditSeed = pretrainSeed * 1000 + evalSeeds[0];
      const label = `${pretrainSeed}-${evalSeeds[0]}`;
      cells.push({ label, pretrainSeed, evalSeeds, auditSeed });
    }
  }

  return cells;
}

export async function runTransferAuditMatrix(
  options: TransferMatrixOptions = {}
): Promise<TransferMatrixReport> {
  const cells = buildTransferMatrixGrid(options);
  const concurrency = Math.max(1, options.concurrency ?? DEFAULT_CONCURRENCY);
  const tmpRoot = options.tmpRoot ?? MATRIX_TMP_DIR;

  const results: TransferMatrixCellResult[] = [];
  const queue = [...cells];

  async function worker(): Promise<void> {
    while (queue.length > 0) {
      const cell = queue.shift();
      if (!cell) {
        return;
      }
      results.push(await runCell(cell, tmpRoot));
    }
  }

  await Promise.all(Array.from({ length: Math.min(concurrency, cells.length) }, () => worker()));

  rmSync(tmpRoot, { recursive: true, force: true });

  return summarize(cells, results, concurrency);
}

async function runCell(cell: TransferMatrixCell, tmpRoot: string): Promise<TransferMatrixCellResult> {
  const cliPath = join(__dirname, "..", "cli.js");
  const args = [
    "audit:transfer:cell",
    String(cell.pretrainSeed),
    cell.evalSeeds.join(","),
    String(cell.auditSeed),
    cell.label
  ];
  const start = Date.now();

  return new Promise((resolve) => {
    const child = fork(cliPath, args, {
      silent: true,
      env: { ...process.env, TRANSFER_MATRIX_TMP_ROOT: tmpRoot }
    });
    const stdoutChunks: Buffer[] = [];
    const stderrChunks: Buffer[] = [];

    child.stdout?.on("data", (chunk: Buffer) => stdoutChunks.push(chunk));
    child.stderr?.on("data", (chunk: Buffer) => stderrChunks.push(chunk));

    child.on("error", (error) => {
      resolve({
        cell,
        report: null,
        error: `fork error: ${error.message}`,
        durationMs: Date.now() - start
      });
    });

    child.on("close", (code) => {
      const durationMs = Date.now() - start;
      const stdout = Buffer.concat(stdoutChunks).toString("utf8").trim();
      const stderr = Buffer.concat(stderrChunks).toString("utf8").trim();

      if (code !== 0) {
        resolve({
          cell,
          report: null,
          error: `cell exit ${code}; stderr: ${stderr || "(empty)"}`,
          durationMs
        });
        return;
      }

      try {
        const report = JSON.parse(stdout) as TransferAuditReport;
        resolve({ cell, report, error: null, durationMs });
      } catch (error) {
        resolve({
          cell,
          report: null,
          error: `stdout parse failed: ${error instanceof Error ? error.message : String(error)}; stdout head: ${stdout.slice(0, 200)}`,
          durationMs
        });
      }
    });
  });
}

function summarize(
  cells: TransferMatrixCell[],
  results: TransferMatrixCellResult[],
  concurrency: number
): TransferMatrixReport {
  const suitePassCounts: Record<string, { passed: number; total: number }> = {};
  for (const name of REQUIRED_SUITE_NAMES) {
    suitePassCounts[name] = { passed: 0, total: 0 };
  }

  const supervisedSeparations: number[] = [];
  const rewardOnlyDeltas: number[] = [];
  const rewardOnlySuccessSeparations: number[] = [];
  const dropout02SupervisedSeparations: number[] = [];
  const dropout02RewardOnlyDeltas: number[] = [];
  const dropout03SupervisedSeparations: number[] = [];
  const dropout03RewardOnlyDeltas: number[] = [];
  const continuedLearningSeparations: number[] = [];
  const continuedLearningReversals: string[] = [];
  const wrongPriorSeparations: number[] = [];
  const wrongPriorReversals: string[] = [];
  const failedCells: string[] = [];
  let cellsAllRequiredPass = 0;

  for (const result of results) {
    const report = result.report;
    if (!report) {
      failedCells.push(`${result.cell.label} (run error: ${result.error})`);
      continue;
    }

    if (report.requiredPassed) {
      cellsAllRequiredPass += 1;
    } else {
      failedCells.push(result.cell.label);
    }

    for (const suite of report.suites) {
      if (!REQUIRED_SUITE_NAMES.includes(suite.name)) {
        continue;
      }
      const counter = suitePassCounts[suite.name];
      counter.total += 1;
      if (suite.passed) {
        counter.passed += 1;
      }
    }

    const supSep = findSuite(report, "transfer frozen-pretrained vs frozen-fresh separation (supervised)");
    const rewSep = findSuite(report, "transfer frozen-pretrained vs frozen-fresh separation (rewardOnly)");
    if (supSep) {
      supervisedSeparations.push(Number(supSep.metrics.separation));
    }
    if (rewSep) {
      rewardOnlyDeltas.push(Number(rewSep.metrics.pretrainedMeanReward) - Number(rewSep.metrics.freshMeanReward));
      rewardOnlySuccessSeparations.push(Number(rewSep.metrics.pretrainedSuccessRate) - Number(rewSep.metrics.freshSuccessRate));
    }

    const dropout02 = findSuite(report, "transfer observation dropout 0.2 diagnostic");
    if (dropout02) {
      dropout02SupervisedSeparations.push(Number(dropout02.metrics.separation));
      dropout02RewardOnlyDeltas.push(Number(dropout02.metrics.rewardOnlyDelta));
    }

    const dropout03 = findSuite(report, "transfer observation dropout 0.3 diagnostic");
    if (dropout03) {
      dropout03SupervisedSeparations.push(Number(dropout03.metrics.separation));
      dropout03RewardOnlyDeltas.push(Number(dropout03.metrics.rewardOnlyDelta));
    }

    const continued = findSuite(report, "transfer continued-learning head-start diagnostic");
    if (continued) {
      const separation = Number(continued.metrics.separation);
      continuedLearningSeparations.push(separation);
      if (separation < 0) {
        continuedLearningReversals.push(result.cell.label);
      }
    }

    const wrongPrior = findSuite(report, "transfer wrong-prior continued-learning diagnostic");
    if (wrongPrior) {
      const separation = Number(wrongPrior.metrics.separation);
      wrongPriorSeparations.push(separation);
      if (separation >= 0) {
        wrongPriorReversals.push(result.cell.label);
      }
    }
  }

  const requiredPassed =
    failedCells.length === 0 &&
    results.every((result) => result.report !== null) &&
    rewardOnlyDeltas.length === results.length && Math.min(...rewardOnlyDeltas) > 0 &&
    rewardOnlySuccessSeparations.length === results.length && Math.min(...rewardOnlySuccessSeparations) >= 0 &&
    continuedLearningSeparations.length === results.length && Math.min(...continuedLearningSeparations) >= 0;

  return {
    version: MATRIX_VERSION,
    generatedAt: new Date().toISOString(),
    requiredPassed,
    grid: {
      pretrainSeeds: Array.from(new Set(cells.map((cell) => cell.pretrainSeed))).sort((a, b) => a - b),
      evalSeedSets: Array.from(new Set(cells.map((cell) => cell.evalSeeds.join(","))))
        .map((csv) => csv.split(",").map((value) => Number(value))),
      cellCount: cells.length,
      concurrency
    },
    cells: results,
    summary: {
      cellsRun: results.length,
      cellsAllRequiredPass,
      suitePassCounts,
      supervisedSeparation: stats(supervisedSeparations),
      rewardOnlyMeanRewardDelta: stats(rewardOnlyDeltas),
      rewardOnlySuccessSeparation: stats(rewardOnlySuccessSeparations),
      dropout02: {
        supervisedSeparation: stats(dropout02SupervisedSeparations),
        rewardOnlyDelta: stats(dropout02RewardOnlyDeltas)
      },
      dropout03: {
        supervisedSeparation: stats(dropout03SupervisedSeparations),
        rewardOnlyDelta: stats(dropout03RewardOnlyDeltas)
      },
      continuedLearning: {
        separation: stats(continuedLearningSeparations),
        reversals: continuedLearningReversals
      },
      wrongPrior: {
        separation: stats(wrongPriorSeparations),
        nonVacuousCells: wrongPriorSeparations.filter((value) => value < 0).length,
        reversals: wrongPriorReversals
      },
      failedCells
    }
  };
}

function findSuite(report: TransferAuditReport, name: string): TransferAuditSuiteResult | undefined {
  return report.suites.find((suite) => suite.name === name);
}

function stats(values: number[]): { min: number; mean: number; max: number } {
  if (values.length === 0) {
    return { min: 0, mean: 0, max: 0 };
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return { min, mean, max };
}

export function formatTransferMatrixReport(report: TransferMatrixReport): string {
  const lines: string[] = [
    `Audit ${report.version}`,
    `requiredPassed=${report.requiredPassed}`,
    `Grid: pretrainSeeds=[${report.grid.pretrainSeeds.join(",")}] x evalSets=${report.grid.evalSeedSets.length} = ${report.grid.cellCount} cells`,
    `Concurrency: ${report.grid.concurrency} (forked child processes)`,
    "",
    "cell            pretrain  evalSet        auditSeed  loader  supSep  rewSep  seedIso  conflict  blank  required"
  ];

  for (const result of report.cells) {
    const cell = result.cell;
    const evalSetShort = `${cell.evalSeeds[0]}..${cell.evalSeeds[cell.evalSeeds.length - 1]}`;
    const label = cell.label.padEnd(16);
    const pretrain = String(cell.pretrainSeed).padEnd(8);
    const evalSet = evalSetShort.padEnd(15);
    const audit = String(cell.auditSeed).padEnd(10);

    if (!result.report) {
      lines.push(`${label}${pretrain}${evalSet}${audit}ERROR  -  ${result.error ?? "unknown"}`);
      continue;
    }

    const flags = suiteFlags(result.report);
    const required = result.report.requiredPassed ? "PASS" : "FAIL";
    lines.push(`${label}${pretrain}${evalSet}${audit}${flags.loader}  ${flags.supSep}  ${flags.rewSep}  ${flags.seedIso}  ${flags.conflict}  ${flags.blank}  ${required}`);
  }

  lines.push("");
  lines.push("Summary:");
  lines.push(`  cells run:                ${report.summary.cellsRun}`);
  lines.push(`  cells all-required-pass:  ${report.summary.cellsAllRequiredPass}/${report.summary.cellsRun}`);
  for (const name of REQUIRED_SUITE_NAMES) {
    const counter = report.summary.suitePassCounts[name];
    if (!counter) {
      continue;
    }
    const short = shortSuiteName(name);
    lines.push(`  ${short.padEnd(26)} ${counter.passed}/${counter.total}`);
  }
  const sup = report.summary.supervisedSeparation;
  const rew = report.summary.rewardOnlyMeanRewardDelta;
  lines.push(`  sup separation:           min=${sup.min.toFixed(3)} mean=${sup.mean.toFixed(3)} max=${sup.max.toFixed(3)}`);
  lines.push(`  rew meanReward delta:     min=${rew.min.toFixed(3)} mean=${rew.mean.toFixed(3)} max=${rew.max.toFixed(3)}`);
  lines.push(`  failed cells:             ${report.summary.failedCells.length === 0 ? "(none)" : report.summary.failedCells.join(", ")}`);

  lines.push("");
  lines.push("Stress axes:");
  const rewSuccess = report.summary.rewardOnlySuccessSeparation;
  lines.push(`  rewardOnly success sep (frozen):  min=${rewSuccess.min.toFixed(3)} mean=${rewSuccess.mean.toFixed(3)} max=${rewSuccess.max.toFixed(3)}`);
  const d02sup = report.summary.dropout02.supervisedSeparation;
  const d02rew = report.summary.dropout02.rewardOnlyDelta;
  lines.push(`  dropout 0.2 supervised sep:       min=${d02sup.min.toFixed(3)} mean=${d02sup.mean.toFixed(3)} max=${d02sup.max.toFixed(3)}`);
  lines.push(`  dropout 0.2 rewardOnly delta:     min=${d02rew.min.toFixed(3)} mean=${d02rew.mean.toFixed(3)} max=${d02rew.max.toFixed(3)}`);
  const d03sup = report.summary.dropout03.supervisedSeparation;
  const d03rew = report.summary.dropout03.rewardOnlyDelta;
  lines.push(`  dropout 0.3 supervised sep:       min=${d03sup.min.toFixed(3)} mean=${d03sup.mean.toFixed(3)} max=${d03sup.max.toFixed(3)}`);
  lines.push(`  dropout 0.3 rewardOnly delta:     min=${d03rew.min.toFixed(3)} mean=${d03rew.mean.toFixed(3)} max=${d03rew.max.toFixed(3)}`);
  const cont = report.summary.continuedLearning.separation;
  lines.push(`  continued-learning sep:           min=${cont.min.toFixed(3)} mean=${cont.mean.toFixed(3)} max=${cont.max.toFixed(3)}`);
  lines.push(`  continued-learning reversals:     ${report.summary.continuedLearning.reversals.length === 0 ? "0" : report.summary.continuedLearning.reversals.join(", ")}`);
  const wp = report.summary.wrongPrior.separation;
  lines.push(`  wrong-prior sep:                  min=${wp.min.toFixed(3)} mean=${wp.mean.toFixed(3)} max=${wp.max.toFixed(3)}`);
  lines.push(`  wrong-prior non-vacuous cells:    ${report.summary.wrongPrior.nonVacuousCells}/${report.summary.cellsRun} (separation<0)`);
  lines.push(`  wrong-prior reversals:            ${report.summary.wrongPrior.reversals.length === 0 ? "0" : report.summary.wrongPrior.reversals.join(", ")}`);

  return lines.join("\n");
}

function suiteFlags(report: TransferAuditReport): {
  loader: string;
  supSep: string;
  rewSep: string;
  seedIso: string;
  conflict: string;
  blank: string;
} {
  return {
    loader: flagFor(report, "transfer loader preserves frozen-eval behavior (disk round-trip)"),
    supSep: flagFor(report, "transfer frozen-pretrained vs frozen-fresh separation (supervised)"),
    rewSep: flagFor(report, "transfer frozen-pretrained vs frozen-fresh separation (rewardOnly)"),
    seedIso: flagFor(report, "transfer eval seed isolation"),
    conflict: flagFor(report, "transfer conflict boundary preservation"),
    blank: flagFor(report, "transfer blank world preservation")
  };
}

function flagFor(report: TransferAuditReport, name: string): string {
  const suite = findSuite(report, name);
  if (!suite) {
    return "MISS";
  }
  return suite.passed ? "PASS" : "FAIL";
}

function shortSuiteName(name: string): string {
  if (name.includes("loader")) {
    return "loader round-trip";
  }
  if (name.includes("(supervised)")) {
    return "supervised separation";
  }
  if (name.includes("(rewardOnly)")) {
    return "rewardOnly separation";
  }
  if (name.includes("seed isolation")) {
    return "seed isolation";
  }
  if (name.includes("conflict")) {
    return "conflict boundary";
  }
  if (name.includes("blank")) {
    return "blank world";
  }
  return name;
}

export interface MultiMatrixReport {
  version: string;
  generatedAt: string;
  requiredPassedAll: boolean;
  matrixCount: number;
  totalCells: number;
  matrices: TransferMatrixReport[];
  wrongPriorAggregate: {
    cellsRun: number;
    nonVacuousCells: number;
    separation: { min: number; mean: number; max: number };
    reversals: string[];
  };
  continuedLearningAggregate: {
    cellsRun: number;
    separation: { min: number; mean: number; max: number };
    reversals: string[];
  };
}

export interface MultiMatrixOptions {
  pools?: { pretrainSeeds: number[]; evalSeedSets: number[][] }[];
  concurrency?: number;
}

export async function runTransferAuditMatrixMulti(
  options: MultiMatrixOptions = {}
): Promise<MultiMatrixReport> {
  const pools = options.pools ?? DEFAULT_MULTI_MATRIX_POOLS;
  const concurrency = options.concurrency ?? DEFAULT_MULTI_MATRIX_CONCURRENCY;

  const matrices = await Promise.all(
    pools.map((pool, index) =>
      runTransferAuditMatrix({
        pretrainSeeds: pool.pretrainSeeds,
        evalSeedSets: pool.evalSeedSets,
        concurrency,
        tmpRoot: join(MATRIX_TMP_DIR, `pool-${index}`)
      })
    )
  );

  const wrongPriorSeparations: number[] = [];
  const wrongPriorReversals: string[] = [];
  let wrongPriorNonVacuous = 0;
  const continuedLearningSeparations: number[] = [];
  const continuedLearningReversals: string[] = [];
  let totalCells = 0;

  matrices.forEach((matrix, matrixIndex) => {
    totalCells += matrix.summary.cellsRun;
    matrix.cells.forEach((cellResult) => {
      if (!cellResult.report) {
        return;
      }
      const wrongPrior = findSuite(
        cellResult.report,
        "transfer wrong-prior continued-learning diagnostic"
      );
      if (wrongPrior) {
        const separation = Number(wrongPrior.metrics.separation);
        wrongPriorSeparations.push(separation);
        if (separation < 0) {
          wrongPriorNonVacuous += 1;
        } else {
          wrongPriorReversals.push(`${matrixIndex}:${cellResult.cell.label}`);
        }
      }
      const continued = findSuite(
        cellResult.report,
        "transfer continued-learning head-start diagnostic"
      );
      if (continued) {
        const separation = Number(continued.metrics.separation);
        continuedLearningSeparations.push(separation);
        if (separation < 0) {
          continuedLearningReversals.push(`${matrixIndex}:${cellResult.cell.label}`);
        }
      }
    });
  });

  const requiredPassedAll = matrices.every((matrix) => matrix.requiredPassed);

  return {
    version: MATRIX_VERSION,
    generatedAt: new Date().toISOString(),
    requiredPassedAll,
    matrixCount: matrices.length,
    totalCells,
    matrices,
    wrongPriorAggregate: {
      cellsRun: wrongPriorSeparations.length,
      nonVacuousCells: wrongPriorNonVacuous,
      separation: stats(wrongPriorSeparations),
      reversals: wrongPriorReversals
    },
    continuedLearningAggregate: {
      cellsRun: continuedLearningSeparations.length,
      separation: stats(continuedLearningSeparations),
      reversals: continuedLearningReversals
    }
  };
}

export function formatMultiMatrixReport(report: MultiMatrixReport): string {
  const lines: string[] = [
    `Audit ${report.version} (multi-matrix)`,
    `requiredPassedAll=${report.requiredPassedAll}`,
    `Matrices: ${report.matrixCount}, total cells: ${report.totalCells}`,
    ""
  ];

  report.matrices.forEach((matrix, index) => {
    const wp = matrix.summary.wrongPrior.separation;
    const cont = matrix.summary.continuedLearning.separation;
    lines.push(`Matrix ${index}: pretrain=[${matrix.grid.pretrainSeeds.join(",")}] requiredPassed=${matrix.requiredPassed}`);
    lines.push(`  cells: ${matrix.summary.cellsRun}, all-required-pass: ${matrix.summary.cellsAllRequiredPass}/${matrix.summary.cellsRun}`);
    lines.push(`  wrong-prior sep:    min=${wp.min.toFixed(3)} mean=${wp.mean.toFixed(3)} max=${wp.max.toFixed(3)} non-vacuous=${matrix.summary.wrongPrior.nonVacuousCells}/${matrix.summary.cellsRun}`);
    lines.push(`  continued-learning sep: min=${cont.min.toFixed(3)} mean=${cont.mean.toFixed(3)} max=${cont.max.toFixed(3)}`);
    if (matrix.summary.failedCells.length > 0) {
      lines.push(`  failed cells: ${matrix.summary.failedCells.join(", ")}`);
    }
  });

  lines.push("");
  lines.push("Cross-matrix aggregate:");
  const wpAgg = report.wrongPriorAggregate;
  lines.push(`  wrong-prior cells:           ${wpAgg.cellsRun}`);
  lines.push(`  wrong-prior non-vacuous:     ${wpAgg.nonVacuousCells}/${wpAgg.cellsRun} (separation<0)`);
  lines.push(`  wrong-prior sep:             min=${wpAgg.separation.min.toFixed(3)} mean=${wpAgg.separation.mean.toFixed(3)} max=${wpAgg.separation.max.toFixed(3)}`);
  lines.push(`  wrong-prior reversals:       ${wpAgg.reversals.length === 0 ? "0" : wpAgg.reversals.join(", ")}`);
  const contAgg = report.continuedLearningAggregate;
  lines.push(`  continued-learning cells:    ${contAgg.cellsRun}`);
  lines.push(`  continued-learning sep:      min=${contAgg.separation.min.toFixed(3)} mean=${contAgg.separation.mean.toFixed(3)} max=${contAgg.separation.max.toFixed(3)}`);
  lines.push(`  continued-learning reversals:${contAgg.reversals.length === 0 ? "0" : contAgg.reversals.join(", ")}`);

  return lines.join("\n");
}
