# 2D-Complex 审计（Level 4 前置验证）

## 阶段目标

Level 3 transfer 矩阵已稳定（45 格跨 3 seed 池，`requiredPassedAll=true`），但暴露上限：
4 patterns / 8 synapses / lr 0.08，fresh 在 4 episodes 饱和，continued-learning sep=0.000 跨 45 格零方差。
继续堆同构 seed 信息增量低。本阶段提升任务复杂度，验证模型在多物体、距离优先级、
distractor 干扰下的组合泛化能力。

不在本阶段证明：

- 真实环境迁移能力。
- 可学习动作仲裁（步骤 2，conflict → 决策策略）。
- 新物体种类 / 新 sensor 拓扑（步骤 1 后续子阶段）。

## 复测命令

```powershell
npm test
npm run audit:2d-complex
npm run audit:2d-challenge
npm run audit:transfer
```

## 核心架构：spike-count 仲裁

现有 4 sensory neuron 是二进制 spiking（`Signal = -1 | 0 | 1`），不编码连续 magnitude。
为引入距离信息，2D-complex 使用 **多 tick 感觉发放 + motor spike-count 仲裁**：

- 每步运行 3 个 sub-tick 循环（`COMPLEX_SPIKE_TICKS = 3`）。
- 距离 d 的物体让对应 sensor 发放 `4 - d` 个 tick（dist 1→3 ticks，dist 2→2，dist 3→1）。
- Motor 神经元在 3 个 sub-tick 中累计 spike 次数（`leak=1` 保证 somaPotential 累积）。
- 仲裁按 spike count：left > right → "left"，right > left → "right"，相等非零 → "conflict"，皆零 → "noop"。

**物理验证**（supervised 单物体学完后 inter→motor 权重 ~1.5）：

- food-left dist 1 + food-right dist 3：iFoodLeft 发放 3 tick → leftMotor 累计 3×spike；
  iFoodRight 发放 1 tick → rightMotor 累计 1×spike。leftMotor(3) > rightMotor(1) → "left"。正确。
- food-left dist 3 + toxin-left dist 2：iFoodLeft 1 tick，iToxinLeft 2 tick。
  leftMotor 1×spike，rightMotor 2×spike → "right"（避近毒）。正确。

网络通过 spike-count 求和自然组合多物体策略，**无需新学习**即可在多物体场景泛化——
这是 compositional generalization 的强测试。

**关键实现细节**：3-tick 循环结束后，最后一个 tick 的 interneuron `outputSignal` 可能为 0
（该 tick 无感觉输入），导致 `applySupervisedMotorLearning` 的 `isActiveSignal(pre.outputSignal)`
判 false，不产生更新。修复：循环中追踪 `interSpikedThisStep` 与 `motorSpikeCounts`，
循环后恢复 `outputSignal=1 / spike=true` 给本步曾 spike 的神经元。这是**步内聚合学习信号**
（spike-count summation 的 eligibility 表达），不是严格意义上的最后一个物理 tick 状态——
目的是让 supervised 学习在步边界正确触发，而非还原物理 spike 序列。

## 场景 family

全部 y=center，1D 距离（agent 只左右移动）。

**Family A — 单物体（pretrain + baseline，复用 2D-challenge 4 pattern）**
food-left/right, toxin-left/right，distance 2-3。

**Family B — 多物体同向（composite，可组合）**
- food-left + toxin-right（都投 left）
- food-right + toxin-left（都投 right）

**Family C — distractor（近远同类，应忽略远物）**
- food-left dist 1 + food-right dist 3 → "left"
- food-right dist 1 + food-left dist 3 → "right"
- toxin-left dist 1 + toxin-right dist 3 → "right"（避近毒）
- toxin-right dist 1 + toxin-left dist 3 → "left"

**Family D — priority（food+toxin 同侧，距离判定）**
- food-left dist 2 + toxin-left dist 3 → "left"（趋近食）
- food-left dist 3 + toxin-left dist 2 → "right"（避近毒）
- food-right dist 2 + toxin-right dist 3 → "right"
- food-right dist 3 + toxin-right dist 2 → "left"

**Family E — true conflict（等距异侧同类，期望 "conflict"）**
- food-left dist 2 + food-right dist 2 → "conflict"
- toxin-left dist 2 + toxin-right dist 2 → "conflict"

## Priority 规则（`expectedActionForComplexState`）

确定性 supervised 策略（非学习得来）：

1. 收集所有 dx≠0 的物体，按 kind 分 food/toxin，各组按 distance 升序。
2. nearestFood → foodVote（dx<0→"left"，dx>0→"right"，趋食）。
3. nearestToxin → toxinVote（dx>0→"left"，dx<0→"right"，避毒）。
4. 两者同向 → 返回该向。
5. 两者冲突（food+toxin 同侧）→ toxin 距离 <= food 距离时避毒，否则趋食。
6. 仅 food：异侧等距 → "conflict"；否则最近 food 方向。
7. 仅 toxin：异侧等距 → "conflict"；否则避最近毒方向。
8. 无物体 → "noop"。

## Audit suites

### Required（gate `requiredPassed`）

| Suite | 阈值 | 说明 |
|-------|------|------|
| Determinism | sameStableTrace + SR >= 0.8 | 同 seed supervised 40 epoch trace 一致 |
| Supervised baseline (Family A) | SR >= 0.8 + supervisedUpdateCount > 0 | supervised 上限 |
| Frozen separation | sep >= 0.3 | supervised vs frozen 隔离 |
| Multi-object same-direction (Family B) | SR >= 0.5 | 组合泛化 |
| Distractor priority (Family C) | SR >= 0.5 | 距离优先级，部分可测（瓶颈项） |
| Priority resolution (Family D) | SR >= 0.5 | food+toxin 同侧距离判定 |
| Conflict boundary (Family E) | firstAction == "conflict" + SR == 0 | 等距真冲突边界 |
| Dropout 0.2 robustness | SR >= 0.5 | 20% dropout 容忍 |
| Blank preservation | noopRate == 1 + meanReward == 0 | 空世界静止 |

### Diagnostic（记录，不 gate）

- Dropout 0.3 robustness — SR 记录，未来可能升 required。
- rewardOnly feasibility (Family A) — sameStableTrace + rewardUpdateCount > 0。
- rewardOnly multi-object (Families B/C/D) — SR 记录，"能学"preview。
- Tighter maxSteps stress (maxSteps=4) — SR 记录，complex 默认 6。

## Config

`createComplexConfig` 复用 `createChallengeConfig`（leak=1, branchLocalThreshold=0.1,
dendriteGateThreshold=0.1, axonThreshold=1, thresholdAdaptRate=0, refractorySteps=0,
fastDecay=0.9995, stableThreshold=0.12, useThreshold=0.08, depotentiationRate=0.64）。
`maxSteps` 默认 6（`DEFAULT_COMPLEX_MAX_STEPS`），distance-3 遍历 + 1 步余量。

## 已知边界

1. **无拓扑改动**：4 sensory / 4 interneuron / 2 motor 不变。距离编码通过 spike-count
   仲裁实现，不改 `Signal` 类型，不改神经元数量或连接。2D-challenge / transfer 的二进制
   sensory 不受影响。
2. **Priority 是确定性 supervised**：`expectedActionForComplexState` 是规则函数，不是学习得来。
   learnable arbitration 是步骤 2。
3. **rewardOnly on multi-object 是 diagnostic**：不 gate。当前数据 rewardOnly Family A SR=0.5、
   多物体 SR=0.2，远低于 supervised SR=1.0——rewardOnly 有更新与部分信号，但未自主掌握组合策略。
   说明组合泛化需要 supervised bootstrapping，是预期 Level 4 发现。
4. **新 family gate 保守（>= 0.5）**：前置验证阶段，目标是"非空真"而非"高性能"。
   Family C 当前 SR=0.5、conflictRate=0.333——距离 spike-count 产生部分可测优先级但仍是瓶颈项，
   不是"已忽略远物"。多轮矩阵稳定后再收紧。
5. **maxSteps=6 是 complex 专属**：2D-challenge 仍用 12（dropout diagnostic 用 4）。不混淆。
6. **distractor 物种延后**：本阶段 distractor = 距离优先级（同类近远）。新增 distractor kind
   + sensor 是步骤 1 后续子阶段，需拓扑改动，独立计划。
7. **不修 transfer matrix**：transfer 作为固定回归 gate 保留。wrong-prior diagnostic 保持现状。
8. **Family E conflict 是边界记录**：equidistant 同类异侧无 priority 解，返回 "conflict"。
   验证网络仍尊重 conflict 边界（不强行仲裁）。
9. **Spike-count 仲裁是 complex 专属**：`arbitrateComplexMotorAction` 不替换
   `arbitrateMotorAction`。2D-challenge / transfer 仍用 binary `arbitrateMotorAction`。

## 结论表述

可以说：

> 当前 DG-SNN V2 在 2D-complex 审计下，pretrained 网络在多物体同向组合（Family B SR=1.0）、
> 距离优先级（Family D SR=1.0）上通过保守阈值；distractor 距离优先级（Family C SR=0.5，
> conflictRate=0.333）仅部分可测，仍为瓶颈项；等距真冲突（Family E）正确触发 conflict 边界；
> dropout 0.2 required 通过。这是 **Level 4 前置验证（非 Level 4 完成）**——任务复杂度从
> 4 patterns / 8 synapses 提升到多物体组合 + 距离优先级，compositional generalization 在
> supervised bootstrapping 下成立。rewardOnly 多物体 SR=0.2、Family A rewardOnly SR=0.5
> 表明 rewardOnly 有更新与部分信号，但**未自主掌握组合策略**。

不要说：

> 已经证明真实环境迁移稳定成立；rewardOnly 自主学会了多物体组合策略；或 2D-complex 已完成 Level 4。
> （2D-complex 只证明 supervised bootstrapping 下的组合泛化可行；可学习仲裁、新 sensor/物种泛化、
> 真实感知迁移均未涉及。）
