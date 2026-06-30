# Transfer 瓶颈突破验证计划

## 当前阶段目标

`2D-challenge` 已经验证了 supervised/frozen/reward-only 可行性，并保存了预训练快照。
本阶段改为验证迁移问题：

- 预训练快照能否被正确加载回网络（经过真实磁盘读写路径）。
- 加载后的预训练网络在 held-out eval 场景中是否提供可测量的初始优势，优于 fresh model。
- supervised 与 rewardOnly 两种预训练快照的迁移效果是否可分离。
- frozen-pretrained vs frozen-fresh 是否构成最干净的知识保留测试。
- 预训练网络是否仍尊重 blank、conflict 边界。
- 高 dropout (0.2, 0.3) 与多物体场景下的迁移鲁棒性是否可记录。

本阶段不证明：

- 真实环境迁移能力。
- 跨拓扑泛化能力。
- 预训练快照的部署可用性。

## 复测命令

```powershell
npm test
npm run eval
npm run audit
npm run audit:2d
npm run audit:2d-challenge
npm run audit:transfer
npm run audit:transfer:matrix
npm run export:2d-challenge
```

## 学习模式

- `supervised`：预训练快照之一，作为上限 baseline。
- `rewardOnly`：预训练快照之一，作为奖励学习候选。
- `frozen`：禁用学习、探索、capture 和 decay，用于纯净的预训练知识测试。

## 预训练快照与加载路径

`npm run export:2d-challenge` 生成磁盘快照：

- `exports/pretrained/2d-challenge-supervised-pretrained.json`
- `exports/pretrained/2d-challenge-rewardOnly-pretrained.json`

`audit:transfer` 在运行时通过 `runChallengeExperiment` 重新生成预训练网络（确定性，seed=101，epochs=40），
然后经过 **真实磁盘读写路径** 验证加载：

```
createNetworkExport  ->  writeNetworkExport  ->  readNetworkExport  ->  loadNetworkFromExport
```

`loadNetworkFromExport` 用 `createOfflineLearningNetwork(snapshot.config)` 重建拓扑骨架，
再用 `snapshot.synapses`（完整 `Synapse[]`）覆盖学习状态；
neuron/branch 的部分未导出字段（`overactiveTime`、branch `dormantTime`）保持 fresh-init。

临时磁盘文件写入 `exports/pretrained/transfer-audit-tmp/`，审计结束后自动清理。

## Required suites

`npm run audit:transfer` 当前 required 覆盖：

- transfer loader preserves frozen-eval behavior (disk round-trip)：对 supervised 与 rewardOnly 各自，
  比较 load-before（内存训练网络）与 load-after（write->read->load）在**同一 frozen 条件**
  （evalSeeds=`[201..205]`，epochs=0，learningEnabled=false）下的 `successRate / meanReward / noopRate / conflictRate`
  四项完全相等；supervised load-after `successRate >= 0.8`。
  **不**比较 `snapshot.metrics` 的训练期累计计数（`rewardUpdateCount` 等），frozen eval 不产生这些。
- frozen-pretrained vs frozen-fresh separation (supervised)：预训练 supervised 在 held-out eval seeds
  `[201,202,203,204,205]` 上的 `successRate >= 0.5`，且比 fresh-fresh 至少高 `0.3`，`fresh.noopRate === 1`。
  **定位为"加载与知识保留门"，不是主要迁移证据**：fresh-frozen 恒 noop 使 separation=1.000 几乎是 smoke test；
  它确认 loader 保真 + 预训练知识没丢，但不证明迁移强度。主要迁移证据是 rewardOnly。
- frozen-pretrained vs frozen-fresh separation (rewardOnly)：`pretrained.meanReward > fresh.meanReward`、
  `fresh.noopRate === 1`、**`pretrained.successRate > 0`**（避免零成功只是均值稍高的假阳性）。
  这个门仍应解读为"rewardOnly 预训练产生了非零可加载行为"，不是强迁移证据：fresh-frozen
  是结构性 noop baseline，rewardOnly separation 主要是非零 vs 零。
- transfer eval seed isolation：迁移 eval seeds `[201..205]` 与预训练 train seeds `[1..5]`、
  eval seeds `[101..105]` 无重叠。
- transfer conflict boundary preservation：预训练网络在 `conflictChallengeScenario` 上首步
  `executedAction === "conflict"` 且 `successRate === 0`。
- transfer blank world preservation：预训练网络在 `blankChallengeScenario` 上 `noopRate === 1` 且 `meanReward === 0`。

观察 dropout `0.2`、`0.3`、多物体 composite（`sameActionCompositeChallengeScenario`）、continued-learning head-start
与 wrong-prior continued-learning 目前是 diagnostic，不阻断 `requiredPassed`。

wrong-prior diagnostic（`auditWrongPriorDiagnostic`）在 suite 内部用 `reverseMapping=true` 重新预训练一份 supervised
快照（food-left→right, toxin-left→right 等），经过磁盘 round-trip 加载，再用 1 epoch 正确映射继续训练，
与 fresh + 1 epoch 对比。`passed=separation<0`：语义是"测到预期退化"——wrong-prior 预训练应让网络必须先
unlearn 错误突触，performance 低于 fresh。但这不是自动的健康信号：若 separation 在矩阵中统一为 `-1.000`，
说明 fresh+1ep 已饱和而 wrong-prior+1ep 尚未恢复，暴露的是 **1 epoch 压力预算太短**。必须同时看
`postCLWrongDirectionMaxFastWeight`、`postCLWrongDirectionMaxStableWeight`、`postCLDualLockConfirmed`
以及 2/3/5/10 epoch 恢复曲线。

## 矩阵聚合门槛（`npm run audit:transfer:matrix`）

15 格 (5 pretrain × 3 evalSet) 受控矩阵的 `requiredPassed` 在每格 required suite 全通过之外，额外 gate：

- `rewardOnlyMeanRewardDelta.min > 0`：每格 rewardOnly pretrained meanReward 严格大于 fresh。
- `rewardOnlySuccessSeparation.min >= 0`：每格 rewardOnly successRate 不低于 fresh（无反转）。
- `continuedLearningSeparation.min >= 0`：每格 pretrained+1ep 不差于 fresh+1ep（无 reversal）。

dropout `0.2`/`0.3` 的 rewardOnly delta 仅记录（observational），不 gate。用户明确"开始看"——
若未来 dropout 0.3 rewardOnly delta min 翻负，应升级为 gate。

dropout diagnostic 使用 `maxSteps=4`（默认 12 步会吸收 dropout 噪声：frozen 网络只需 2-3
可见步即成功，dropout 0.3 期望可见步 ~8.4 无影响）。continued-learning diagnostic 使用
`1 epoch`（`5 epoch` 会让 fresh 在 1-2 epoch 内饱和到 successRate=1.0，separation=0.000 永真）。
wrong-prior diagnostic 使用 `reverseMapping=true` 预训练 40 epoch + `1 epoch` 正确映射继续训练，
separation<0 表示 wrong-prior hurts（gate 真正可失败）。

矩阵报告新增 "Stress axes" 段，聚合跨格统计：
- rewardOnly success separation (frozen)
- dropout 0.2/0.3 supervised separation 与 rewardOnly delta
- continued-learning separation 与 reversals 计数
- wrong-prior separation、non-vacuous cells（separation<0 的格子数）、postCL wrong-direction stable/fast 权重、
  postCL dual-lock cells 与 reversals 计数

wrong-prior axis 当前是 observational，不 gate。目标是验证 non-vacuity（跨格 separation<0 稳定成立）。
若多轮矩阵只是稳定 `separation=-1.000` 且 postCL wrong-direction fast/stable 权重仍高，不能升级为"健康 gate"；
应记录为 unlearning 失败证据。只有出现可测的 postCL 错误权重下降/部分恢复，才考虑升级。

## 已知边界

1. **Config 耦合加载**：loader 依赖 `createOfflineLearningNetwork` 的硬编码拓扑。若未来拓扑动态化，loader 需重写或扩展 `createNetworkExport` 为完整快照。
2. **`overactiveTime` 与 branch `dormantTime` 未导出**：loader 无法恢复，保持 fresh-init。对短 eval（12 步）影响可忽略。
3. **`plasticityGate / inputSum / inhibitionLoad / gain` 是 runtime-only**：每 tick 由 `resetNeuronRuntime` + `integrateNeuron` 重算，快照里的值陈旧，loader 不恢复。
4. **Fresh-frozen 恒 noop**：初始 interneuron→motor fastWeight=0.35 双发，均不达阈值 1.0。这是 feature 不是 bug —— 让 separation 测试干净。但 0.3 阈值是"学没学到东西"的宽松测试，不是"学得多好"的紧测试。
5. **Transfer eval seeds 必须 held-out**：`[201..205]` 与 pretrain 的 `[1..5]`/`[101..105]` 完全不重叠，否则测的是记忆不是迁移。Suite 4 显式校验。
6. **每个 suite 独立重新 load**：实验就地 mutate network，共享会导致 suite 间互相污染。
7. **磁盘 round-trip 而非内存 round-trip**：Suite 1 必须经过 `writeNetworkExport → readNetworkExport → loadNetworkFromExport` 真实文件路径。内存 `JSON.parse(JSON.stringify())` 只测序列化形状，测不到磁盘读写。
8. **dropout delta 不 gate 是刻意的**：用户明确"开始看"。若未来 dropout 0.3 rewardOnly delta min 翻负，应升级为 gate；当前保持 observational。
9. **continued-learning gate 目前退化为"预训练不退化"**：`min >= 0` 意味着"预训练没有比 fresh 更差"。当前 15 格 separation 恒 0、reversal=0，不能解释为正向 head-start。
10. **rewardOnly success separation 被 fresh-frozen=0 抬高**：frozen 下 fresh 恒 noop（successRate=0），suite 3 已要求 `pretrained.successRate > 0`。因此 rewardOnly frozen separation 主要测 loader 保真 + rewardOnly 非零，不是强迁移证据。
11. **dropout maxSteps=4 是刻意的，已让 axis 非空真**：12 步预算下 frozen 网络只需 2-3 可见步即成功，dropout 0.3 期望可见步 ~8.4，无影响——15 格 dropout delta 与 frozen baseline 完全相同。4 步预算下 dropout 0.3 期望可见步 ~2.8，产生真实退化。advantage 更新后当前矩阵为：`dropout 0.2 rewardOnly delta min=0.445 mean=0.529`，`dropout 0.3 rewardOnly delta min=0.345 mean=0.502`，梯度可见但低于旧 raw-reward 基线。若 maxSteps=4 下 dropout delta 跨多轮矩阵稳定不翻负，可考虑升级为 gate。
12. **continued-learning 1 epoch + 1 trainSeed 是刻意的，但不足以让 gate 非空真**：5 epoch on `[1..5]` 让 fresh 饱和到 successRate=1.0，separation=0.000 永真。收紧到 1 epoch on `[1]`（4 episodes，~8-12 updates/synapse vs ~9 needed for threshold）后，fresh 仍在 15 格全部饱和——任务（4 patterns / 8 synapses / lr 0.08）过于平凡地可学。**此 gate 的价值是捕获 reversal（pretrained 在继续训练下退化，separation < 0），不是测量正向 head-start**。要把 separation 推到正数，需要 wrong-prior（pretrained 必须先 unlearn）或扩展任务复杂度——前者已在 wrong-prior diagnostic 中实现（见 boundary 13），后者 out of scope。当前 15 格 `continued-learning sep: min=0.000 mean=0.000 max=0.000`，reversals=0，gate 通过的含义是"预训练不退化"，不是"预训练有正向 head-start"。
13. **wrong-prior 仅对 supervised 有意义**：rewardOnly 不读 `expectedAction`（仅 reward-driven），`reverseMapping` 对 rewardOnly 是 no-op。本阶段 wrong-prior diagnostic 只测 supervised pretrained。
14. **wrong-prior `passed=separation<0` 只是"有伤害"信号**：与其他 diagnostic 的 `passed` 语义不同——这里 `passed=true` 意味"测到了预期退化"。但 `-1.000` uniform 现在应读成 **1 epoch 压力预算失败**，不是健康非空真，也不是系统无法 unlearn。矩阵 summary 必须打印 postCL wrong-direction fast/stable 权重与 dual-lock。
15. **wrong-prior 的磁盘 round-trip 是 suite 内联的**：不像主 supervised/rewardOnly pretrain 在 `runTransferAudit` 顶部统一做，wrong-prior pretrain 在 suite 内部做。这是为了让主 pretrain 流程不受 wrong-prior 影响，保持现有 suite 行为不变。代价是每个 cell 多一次 pretrain（~40 epoch supervised），矩阵总耗时增加约 1/3。
16. **wrong-prior epoch curve 已把"卡死 vs 慢恢复"分开**：15-cell 曲线（pretrain seeds 101-105 × eval sets 201/301/401，continued-learning epochs 0/1/2/3/5/10）显示：0ep wrongFast≈1.959/wrongStable≈2.000；1ep preSR=0.000、freshSR=1.000、sep=-1.000、wrongFast≈1.066、wrongStable≈0.055；2ep preSR≈0.850、sep≈-0.150、wrongFast≈0.799；3ep preSR=1.000、sep=0.000、wrongFast≈0.529；10ep wrongFast≈0 且 SR=1.000。结论：supervised stable depotentiation 很快，非 stable dual-lock；1ep 失败主要是 wrong-direction fast path 尚未压下；2ep 部分恢复，3ep 恢复到 fresh-level。1ep gate 应继续作为压力诊断，但不能说系统无法 unlearn。

17. **rewardOnly credit assignment 是结构性未解问题**：当前 rewardOnly 已从 raw reward 改为 Hebbian × advantage(`reward - runningBaseline`) 更新。advantage 的负 delta **已能压低活跃错误通路的 fastWeight**(去增强),但**沉默错误通路(eligibility=0)仍免疫**,且 fastWeight 下限仍钳 0、无符号翻转;无 target 信号时仍没有非 target motor 压制;`applyRewardLearning` 不做 stableWeight depotentiation;探索只在 noop/conflict 时触发。advantage 更新打破了 complex Family A 的旧 conflict 表型,但 2D-challenge rewardOnly 仍 SR=0.5 且 noopRate 高,multi-object 仍 SR=0.5。不能把 rewardOnly 读成已解决。

## 结论表述

可以说：

> 当前 DG-SNN V2 的 L0-L3 受控审计通过；但若干通过项是 vacuous 或被 fresh-frozen=0 抬高。transfer matrix 仍可作为 loader 保真、边界保持、rewardOnly 非零行为和无反转回归测试；continued-learning gate 当前只支持"预训练不退化"，不支持正向 head-start；rewardOnly separation 主要是非零 vs 结构性零 baseline。advantage 更新改善了部分 rewardOnly 表型，但不构成强迁移证据。wrong-prior 是当前唯一真正可失败的 unlearning 压力诊断；uniform `-1.000` 表示 1-epoch 预算失败，而 epoch curve 已显示 2ep 部分恢复、3ep 恢复，不能解读为 stable dual-lock 或无法 unlearn。

不要说：

> 已经证明真实环境迁移稳定成立；rewardOnly credit assignment 已解决；wrong-prior `-1.000` 是健康非空真或系统无法 unlearn；或 dropout 下迁移优势保持。
