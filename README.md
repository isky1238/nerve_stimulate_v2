# DG-SNN V2 Model

This directory contains a TypeScript CLI prototype for the dual-gated spiking neural network described in the design notes.

## Commands

```bash
npm install
npm run test
npm run eval
npm run export
npm run export:2d-challenge
npm run trace
npm run explain -- exports/trace-xxx.json
npm run audit
npm run audit:2d
npm run audit:2d-challenge
```

`npm run eval` runs the offline Test A-E suite before any 2D world integration. `npm run export` writes a JSON network snapshot under `exports/`. `npm run export:2d-challenge` writes supervised and reward-only pretrained challenge snapshots under `exports/pretrained/` for future transfer attempts and baseline comparisons.
`npm run trace` writes an explainable training trace with propagation, gate snapshots, and weight changes. `npm run explain -- <trace-file>` prints a compact explanation for a recorded trace.
`npm run audit` runs the pre-2D audit suite: deterministic replay, learning-off ablation, all-permutation order robustness, mirrored mapping, blank-input silence, and input edge-case diagnostics. `requiredPassed=true` means the fixed-topology supervised offline task is reproducible and learnable; `FAIL DIAGNOSTIC` entries are known boundaries, not proof of 2D readiness. See `PRE_2D_AUDIT.md`.
`npm run audit:2d` runs the deterministic 2D-lite audit suite: environment-level replay, multi-seed object placement, mirrored world positions, blank world silence, learning-on/off ablation, supervised-plasticity ablation, and composite/conflict arbitration. Passing it only supports the controlled fixed-topology supervised 2D-lite claim; it does not prove autonomous reward learning or broad 2D generalization.
`npm run audit:2d-challenge` runs the multi-step bottleneck challenge: supervised upper bound, frozen lower bound, reward-only feasibility, train/eval seed isolation, blank/conflict boundaries, and observation-dropout diagnostics. Passing it means the next bottleneck is reward credit assignment, not 2D-lite reproducibility.
