import { existsSync } from "node:fs";
import { join } from "node:path";
import { defaultConfig } from "./config/newModelConfig";
import { formatAuditReport, runPre2DAudit } from "./core/audit";
import { runAllEvaluations, runLearningDemo } from "./core/evaluation";
import { explainTrace, readTraceExport, runLearningTrace, writeTraceExport } from "./core/trace";
import { writeChallengePretrainExports } from "./export/challengePretrainExport";
import { createNetworkExport, writeNetworkExport } from "./export/networkExport";
import { formatWorld2DAuditReport, runWorld2DAudit } from "./world/audit2d";
import { formatWorld2DChallengeAuditReport, runWorld2DChallengeAudit } from "./world/audit2dChallenge";
import { formatWorld2DComplexAuditReport, runWorld2DComplexAudit } from "./world/audit2dComplex";
import { formatArbitrationAuditReport, runArbitrationAudit } from "./world/auditArbitration";
import { formatArbitrationMatrixReport, runArbitrationMatrixAudit } from "./world/auditArbitrationMatrix";
import { formatTransferAuditReport, runTransferAudit } from "./world/transferAudit";
import { cellTmpDir, formatMultiMatrixReport, formatTransferMatrixReport, runTransferAuditMatrix, runTransferAuditMatrixMulti } from "./world/transferMatrix";

function main(): void {
  const command = process.argv[2] ?? "eval";

  if (command === "eval") {
    runEvalCommand();
    return;
  }

  if (command === "export") {
    runExportCommand(process.argv[3]);
    return;
  }

  if (command === "export:2d-challenge") {
    runChallengePretrainExportCommand(process.argv[3]);
    return;
  }

  if (command === "trace") {
    runTraceCommand(process.argv[3]);
    return;
  }

  if (command === "explain") {
    runExplainCommand(process.argv[3]);
    return;
  }

  if (command === "audit") {
    runAuditCommand();
    return;
  }

  if (command === "audit:2d") {
    runWorld2DAuditCommand();
    return;
  }

  if (command === "audit:2d-challenge") {
    runWorld2DChallengeAuditCommand();
    return;
  }

  if (command === "audit:2d-complex") {
    runWorld2DComplexAuditCommand();
    return;
  }

  if (command === "audit:arbitration") {
    runArbitrationAuditCommand();
    return;
  }

  if (command === "audit:arbitration:matrix") {
    runArbitrationMatrixAuditCommand();
    return;
  }

  if (command === "audit:transfer") {
    runTransferAuditCommand();
    return;
  }

  if (command === "audit:transfer:cell") {
    runTransferCellCommand(process.argv.slice(3));
    return;
  }

  if (command === "audit:transfer:matrix") {
    runTransferMatrixCommand();
    return;
  }

  if (command === "audit:transfer:matrix:multi") {
    runTransferMatrixMultiCommand();
    return;
  }

  console.error(`Unknown command: ${command}`);
  console.error("Usage: npm run eval | npm run export | npm run export:2d-challenge | npm run trace | npm run explain -- <trace-file> | npm run audit | npm run audit:2d | npm run audit:2d-challenge | npm run audit:2d-complex | npm run audit:arbitration | npm run audit:arbitration:matrix | npm run audit:transfer | npm run audit:transfer:cell | npm run audit:transfer:matrix | npm run audit:transfer:matrix:multi");
  process.exitCode = 1;
}

function runEvalCommand(): void {
  const results = runAllEvaluations(defaultConfig);

  for (const result of results) {
    const status = result.passed ? "PASS" : "FAIL";
    console.log(`${status} ${result.name}`);
    console.log(`  metrics: ${JSON.stringify(result.metrics)}`);
  }

  if (results.some((result) => !result.passed)) {
    process.exitCode = 1;
  }
}

function runExportCommand(outputPath?: string): void {
  const demo = runLearningDemo(defaultConfig);
  const snapshot = createNetworkExport({
    seed: 1,
    config: demo.config,
    neurons: demo.network.neurons,
    synapses: demo.network.synapses,
    pairMemory: demo.network.pairMemory,
    metrics: demo.metrics,
    events: demo.events
  });
  const filePath = outputPath ?? join("exports", `network-${Date.now()}.json`);
  writeNetworkExport(filePath, snapshot);
  console.log(`Wrote ${filePath}`);
  console.log(`metrics: ${JSON.stringify(demo.metrics)}`);
}

function runChallengePretrainExportCommand(outputDir?: string): void {
  const exports = writeChallengePretrainExports(defaultConfig, { outputDir });

  for (const item of exports) {
    console.log(`Wrote ${item.filePath}`);
    console.log(`mode: ${item.mode}`);
    console.log(`metrics: ${JSON.stringify(item.snapshot.metrics)}`);
  }
}

function runTraceCommand(outputPath?: string): void {
  const trace = runLearningTrace(defaultConfig);
  const filePath = outputPath ?? join("exports", `trace-${Date.now()}.json`);
  writeTraceExport(filePath, trace);
  console.log(`Wrote ${filePath}`);
  console.log(`metrics: ${JSON.stringify(trace.metrics)}`);
}

function runExplainCommand(filePath?: string): void {
  if (!filePath) {
    console.error("Usage: npm run explain -- <trace-file>");
    process.exitCode = 1;
    return;
  }

  if (!existsSync(filePath)) {
    console.error(`Trace file not found: ${filePath}`);
    process.exitCode = 1;
    return;
  }

  try {
    console.log(explainTrace(readTraceExport(filePath)));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`Unable to explain trace ${filePath}: ${message}`);
    process.exitCode = 1;
  }
}

function runAuditCommand(): void {
  const report = runPre2DAudit(defaultConfig);
  console.log(formatAuditReport(report));

  if (!report.requiredPassed) {
    process.exitCode = 1;
  }
}

function runWorld2DAuditCommand(): void {
  const report = runWorld2DAudit(defaultConfig);
  console.log(formatWorld2DAuditReport(report));

  if (!report.requiredPassed) {
    process.exitCode = 1;
  }
}

function runWorld2DChallengeAuditCommand(): void {
  const report = runWorld2DChallengeAudit(defaultConfig);
  console.log(formatWorld2DChallengeAuditReport(report));

  if (!report.requiredPassed) {
    process.exitCode = 1;
  }
}

function runWorld2DComplexAuditCommand(): void {
  const report = runWorld2DComplexAudit(defaultConfig);
  console.log(formatWorld2DComplexAuditReport(report));

  if (!report.requiredPassed) {
    process.exitCode = 1;
  }
}

function runArbitrationAuditCommand(): void {
  const report = runArbitrationAudit(defaultConfig);
  console.log(formatArbitrationAuditReport(report));

  if (!report.requiredPassed) {
    process.exitCode = 1;
  }
}

function runArbitrationMatrixAuditCommand(): void {
  const report = runArbitrationMatrixAudit(defaultConfig);
  console.log(formatArbitrationMatrixReport(report));

  if (!report.requiredPassed) {
    process.exitCode = 1;
  }
}

function runTransferAuditCommand(): void {
  const report = runTransferAudit(defaultConfig);
  console.log(formatTransferAuditReport(report));

  if (!report.requiredPassed) {
    process.exitCode = 1;
  }
}

function runTransferCellCommand(args: string[]): void {
  const [pretrainSeedRaw, evalSeedsRaw, auditSeedRaw, label] = args;
  const pretrainSeed = Number(pretrainSeedRaw);
  const evalSeeds = (evalSeedsRaw ?? "").split(",").map((value) => Number(value)).filter((value) => Number.isFinite(value));
  const auditSeed = Number(auditSeedRaw);

  if (!Number.isFinite(pretrainSeed) || evalSeeds.length === 0 || !Number.isFinite(auditSeed)) {
    console.error("Usage: npm run audit:transfer:cell -- <pretrainSeed> <evalSeedsCsv> <auditSeed> <label>");
    process.exitCode = 1;
    return;
  }

  const tmpDir = cellTmpDir(label ?? `${pretrainSeed}-${evalSeeds[0]}`);
  const report = runTransferAudit(defaultConfig, {
    cell: {
      pretrainSeed,
      evalSeeds,
      auditSeed,
      tmpDir,
      label: label ?? `${pretrainSeed}-${evalSeeds[0]}`
    }
  });
  console.log(JSON.stringify(report));

  if (!report.requiredPassed) {
    process.exitCode = 1;
  }
}

async function runTransferMatrixCommand(): Promise<void> {
  const report = await runTransferAuditMatrix();
  console.log(formatTransferMatrixReport(report));

  if (!report.requiredPassed) {
    process.exitCode = 1;
  }
}

async function runTransferMatrixMultiCommand(): Promise<void> {
  const report = await runTransferAuditMatrixMulti();
  console.log(formatMultiMatrixReport(report));

  if (!report.requiredPassedAll) {
    process.exitCode = 1;
  }
}

main();
