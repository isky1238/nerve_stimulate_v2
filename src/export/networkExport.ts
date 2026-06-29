import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { ModelConfig } from "../config/newModelConfig";
import { Neuron } from "../core/neuron";
import { PairMemory } from "../core/development";
import { Synapse } from "../core/synapse";

export interface NetworkExport {
  version: string;
  generatedAt: string;
  seed: number;
  config: ModelConfig;
  neurons: unknown[];
  branches: unknown[];
  synapses: Synapse[];
  pairMemory: PairMemory[];
  metrics: Record<string, number | string | boolean>;
  events: unknown[];
}

export function createNetworkExport(params: {
  seed: number;
  config: ModelConfig;
  neurons: Neuron[];
  synapses: Synapse[];
  pairMemory: PairMemory[];
  metrics: Record<string, number | string | boolean>;
  events?: unknown[];
}): NetworkExport {
  return {
    version: "dg-snn-v0.1",
    generatedAt: new Date().toISOString(),
    seed: params.seed,
    config: params.config,
    neurons: params.neurons.map((neuron) => ({
      id: neuron.id,
      role: neuron.role,
      subtype: neuron.subtype,
      threshold: neuron.baseThreshold,
      dynamicThreshold: neuron.dynamicThreshold,
      recentSpikeRate: neuron.recentSpikeRate,
      dormantTime: neuron.dormantTime,
      position: neuron.position,
      inputSlots: neuron.inputSlots,
      outputSlots: neuron.outputSlots
    })),
    branches: params.neurons.flatMap((neuron) =>
      neuron.branches.map((branch) => ({
        neuronId: neuron.id,
        branchId: branch.id,
        inputSum: branch.inputSum,
        inhibitionLoad: branch.inhibitionLoad,
        gain: branch.gain,
        plasticityGate: branch.plasticityGate,
        activeRate: branch.recentActiveRate
      }))
    ),
    synapses: params.synapses,
    pairMemory: params.pairMemory,
    metrics: params.metrics,
    events: params.events ?? []
  };
}

export function writeNetworkExport(filePath: string, snapshot: NetworkExport): void {
  mkdirSync(dirname(filePath), { recursive: true });
  writeFileSync(filePath, `${JSON.stringify(snapshot, null, 2)}\n`, "utf8");
}

export function readNetworkExport(filePath: string): NetworkExport {
  const raw = readFileSync(filePath, "utf8");
  return JSON.parse(raw) as NetworkExport;
}
