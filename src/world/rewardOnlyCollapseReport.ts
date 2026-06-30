import type { ChallengeExperimentResult } from "./challenge2d";
import type { CollapseEpochSample } from "./diagnostics";

const COLLAPSE_SEED = 101;
const CHALLENGE_COLLAPSE_SEED = 21;
const COLLAPSE_EPOCHS = 40;

function fmt(value: number, digits = 3): string {
  return value.toFixed(digits);
}

export function formatRewardOnlyChallengeCollapseReport(
  result: ChallengeExperimentResult,
  samples: CollapseEpochSample[]
): string {
  const lines: string[] = [];
  lines.push("Audit dg-snn-rewardonly-challenge-collapse-v0.1");
  lines.push(`seed=${CHALLENGE_COLLAPSE_SEED} epochs=${COLLAPSE_EPOCHS} learningMode=rewardOnly (2D-challenge Family A)`);
  lines.push("");
  lines.push("Per-epoch bilateral fastWeight symmetry + training conflict/noop:");
  lines.push("  epoch  leftFast  rightFast  asymmetry  trainConflict  trainNoop");
  for (const s of samples) {
    lines.push(
      `  ${String(s.epoch).padStart(5)}  ${fmt(s.leftFastSum).padStart(8)}  ${fmt(s.rightFastSum).padStart(9)}  ` +
        `${fmt(s.fastAsymmetry).padStart(9)}  ${fmt(s.trainConflictRate).padStart(13)}  ${fmt(s.trainNoopRate).padStart(9)}`
    );
  }
  lines.push("");
  lines.push("Final eval (frozen):");
  lines.push(
    `  2D-challenge Family A:   SR=${fmt(result.successRate)}  conflict=${fmt(result.conflictRate)}  noop=${fmt(result.noopRate)}  meanReward=${fmt(result.meanReward)}`
  );
  lines.push("");
  const first = samples[0];
  const last = samples[samples.length - 1];
  if (first && last) {
    lines.push("Reading guide:");
    lines.push(
      `  fastWeight drift: left ${fmt(first.leftFastSum)} -> ${fmt(last.leftFastSum)}, ` +
        `right ${fmt(first.rightFastSum)} -> ${fmt(last.rightFastSum)}, ` +
        `asymmetry ${fmt(first.fastAsymmetry)} -> ${fmt(last.fastAsymmetry)}`
    );
    lines.push("  - High evalNoopRate (~0.85) with trainNoopRate~0: NOT bilateral mutual-cancellation.");
    lines.push("    Real cause (A/B-verified): fastWeight decays below motor threshold under advantage's");
    lines.push("    net-negative delta, while conflict-gated exploration forces a motor during training and");
    lines.push("    MASKS the noop (trainNoop=0); frozen eval exposes it. Low asymmetry is expected, not the bug.");
    lines.push("  - If asymmetry grows and noop drops: advantage selected one side -> noop resolved.");
  }
  return lines.join("\n");
}

export function formatRewardOnlyComplexCollapseReport(
  familyA: ChallengeExperimentResult,
  composite: ChallengeExperimentResult,
  samples: CollapseEpochSample[]
): string {
  const lines: string[] = [];
  lines.push("Audit dg-snn-rewardonly-collapse-v0.1");
  lines.push(`seed=${COLLAPSE_SEED} epochs=${COLLAPSE_EPOCHS} learningMode=rewardOnly (complex Family A pretrain)`);
  lines.push("");
  lines.push("Per-epoch bilateral fastWeight symmetry + training conflict/noop:");
  lines.push("  epoch  leftFast  rightFast  asymmetry  trainConflict  trainNoop");
  for (const s of samples) {
    lines.push(
      `  ${String(s.epoch).padStart(5)}  ${fmt(s.leftFastSum).padStart(8)}  ${fmt(s.rightFastSum).padStart(9)}  ` +
        `${fmt(s.fastAsymmetry).padStart(9)}  ${fmt(s.trainConflictRate).padStart(13)}  ${fmt(s.trainNoopRate).padStart(9)}`
    );
  }
  lines.push("");
  lines.push("Final eval (frozen):");
  lines.push(
    `  Family A   (4-pattern single-object):    SR=${fmt(familyA.successRate)}  conflict=${fmt(familyA.conflictRate)}  noop=${fmt(familyA.noopRate)}  meanReward=${fmt(familyA.meanReward)}`
  );
  lines.push(
    `  Multi-obj  (Families B/C/D composite):   SR=${fmt(composite.successRate)}  conflict=${fmt(composite.conflictRate)}  noop=${fmt(composite.noopRate)}  meanReward=${fmt(composite.meanReward)}`
  );
  lines.push("");
  const first = samples[0];
  const last = samples[samples.length - 1];
  if (first && last) {
    lines.push("Reading guide:");
    lines.push(
      `  fastWeight drift: left ${fmt(first.leftFastSum)} -> ${fmt(last.leftFastSum)}, ` +
        `right ${fmt(first.rightFastSum)} -> ${fmt(last.rightFastSum)}, ` +
        `asymmetry ${fmt(first.fastAsymmetry)} -> ${fmt(last.fastAsymmetry)}`
    );
    lines.push("  - If leftFast/rightFast track together (low asymmetry) and Family A conflict=0: advantage broke");
    lines.push("    collapse via spike-count arbitration, NOT via asymmetric weights -> structural bilateral wiring persists.");
    lines.push("  - If asymmetry grows: advantage's fastWeight depotentiation selected one side -> bilateral co-enhancement broken.");
    lines.push("  - Multi-object conflict>0 with low asymmetry: residual conflict is compositional vote-tie, not bilateral co-enhancement.");
  }
  return lines.join("\n");
}
