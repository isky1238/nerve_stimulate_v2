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
unlearn 错误突触，performance 低于 fresh。这是 continued-learning gate 在当前任务复杂度下**真正可失败**的路径。

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
- wrong-prior separation、non-vacuous cells（separation<0 的格子数）与 reversals 计数

wrong-prior axis 当前是 observational，不 gate。目标是验证 non-vacuity（跨格 separation<0 稳定成立）。
若多轮矩阵稳定 separation<0，可升级为 gate（`wrongPriorSeparation.max < 0`）。

## 已知边界

1. **Config 耦合加载**：loader 依赖 `createOfflineLearningNetwork` 的硬编码拓扑。若未来拓扑动态化，loader 需重写或扩展 `createNetworkExport` 为完整快照。
2. **`overactiveTime` 与 branch `dormantTime` 未导出**：loader 无法恢复，保持 fresh-init。对短 eval（12 步）影响可忽略。
3. **`plasticityGate / inputSum / inhibitionLoad / gain` 是 runtime-only**：每 tick 由 `resetNeuronRuntime` + `integrateNeuron` 重算，快照里的值陈旧，loader 不恢复。
4. **Fresh-frozen 恒 noop**：初始 interneuron→motor fastWeight=0.35 双发，均不达阈值 1.0。这是 feature 不是 bug —— 让 separation 测试干净。但 0.3 阈值是"学没学到东西"的宽松测试，不是"学得多好"的紧测试。
5. **Transfer eval seeds 必须 held-out**：`[201..205]` 与 pretrain 的 `[1..5]`/`[101..105]` 完全不重叠，否则测的是记忆不是迁移。Suite 4 显式校验。
6. **每个 suite 独立重新 load**：实验就地 mutate network，共享会导致 suite 间互相污染。
7. **磁盘 round-trip 而非内存 round-trip**：Suite 1 必须经过 `writeNetworkExport → readNetworkExport → loadNetworkFromExport` 真实文件路径。内存 `JSON.parse(JSON.stringify())` 只测序列化形状，测不到磁盘读写。
8. **dropout delta 不 gate 是刻意的**：用户明确"开始看"。若未来 dropout 0.3 rewardOnly delta min 翻负，应升级为 gate；当前保持 observational。
9. **continued-learning gate 是强声明**：`min >= 0` 意味着"预训练永不吃亏，即便给 fresh 等量继续训练预算"。当前 15 格支持此声明；若未来某格 fresh 追上，gate 会失败，这是设计意图。
10. **rewardOnly success separation gate 在 frozen 下与 suite 3 冗余**：frozen 下 fresh 恒 noop（successRate=0），suite 3 已要求 `pretrained.successRate > 0`。此 gate 在 continued-learning 矩阵中才独立起作用，但写在 frozen 矩阵里是冗余防御——若未来 fresh-frozen 不再恒 noop（例如初始权重改了），此 gate 会独立 catch。
11. **dropout maxSteps=4 是刻意的，已让 axis 非空真**：12 步预算下 frozen 网络只需 2-3 可见步即成功，dropout 0.3 期望可见步 ~8.4，无影响——15 格 dropout delta 与 frozen baseline 完全相同。4 步预算下 dropout 0.3 期望可见步 ~2.8，产生真实退化：`dropout 0.2 rewardOnly delta min=0.725 mean=0.964`，`dropout 0.3 rewardOnly delta min=0.670 mean=0.947`，梯度可见。若 maxSteps=4 下 dropout delta 跨多轮矩阵稳定不翻负，可考虑升级为 gate。
12. **continued-learning 1 epoch + 1 trainSeed 是刻意的，但不足以让 gate 非空真**：5 epoch on `[1..5]` 让 fresh 饱和到 successRate=1.0，separation=0.000 永真。收紧到 1 epoch on `[1]`（4 episodes，~8-12 updates/synapse vs ~9 needed for threshold）后，fresh 仍在 15 格全部饱和——任务（4 patterns / 8 synapses / lr 0.08）过于平凡地可学。**此 gate 的价值是捕获 reversal（pretrained 在继续训练下退化，separation < 0），不是测量正向 head-start**。要把 separation 推到正数，需要 wrong-prior（pretrained 必须先 unlearn）或扩展任务复杂度——两者均 out of scope（见 boundary 13）。当前 15 格 `continued-learning sep: min=0.000 mean=0.000 max=0.000`，reversals=0，gate 通过的含义是"预训练不退化"，不是"预训练有正向 head-start"。
13. **本阶段不引入 wrong-prior 测试**：wrong-prior（预训练用相反映射）能让 continued-learning gate 真正可失败（pretrained 必须先 unlearn 再 relearn），但需要新代码（`expectedActionForChallengeState` 硬编码，需加 `reverseMapping` flag）。本阶段保持最小范围，wrong-prior 留待下一阶段。

> 已过时：本阶段（wrong-prior 引入后）已实现 `reverseMapping` flag 与 `auditWrongPriorDiagnostic`。下面 14-17 是 wrong-prior 阶段的边界。

14. **wrong-prior 仅对 supervised 有意义**：rewardOnly 不读 `expectedAction`（仅 reward-driven），`reverseMapping` 对 rewardOnly 是 no-op。本阶段 wrong-prior diagnostic 只测 supervised pretrained。
15. **wrong-prior `passed=separation<0` 是非空真信号，不是错误**：与其他 diagnostic 的 `passed` 语义不同——这里 `passed=true` 意味"测到了预期退化"，`passed=false` 意味"wrong-prior 被 1 epoch 完全 unlearn，任务太平凡"。报告读者需注意此语义反转。
16. **wrong-prior 的磁盘 round-trip 是 suite 内联的**：不像主 supervised/rewardOnly pretrain 在 `runTransferAudit` 顶部统一做，wrong-prior pretrain 在 suite 内部做。这是为了让主 pretrain 流程不受 wrong-prior 影响，保持现有 suite 行为不变。代价是每个 cell 多一次 pretrain（~40 epoch supervised），矩阵总耗时增加约 1/3。
17. **wrong-prior 仍受任务复杂度限制**：若 1 epoch continued-learning 足以把 wrong-prior fastWeight 从 1.0 压到 threshold 以下（错误 motor 不再激活），separation 趋 0。supervised 的 -lr*0.7 vs +lr 不对称（衰减比增强慢 30%），wrong-prior 应该能产生信号，但需矩阵数据验证。若跨格 separation 仍为 0，则任务复杂度限制确认触及天花板，下一阶段需任务复杂度扩展（Level 4）。

## 结论表述

可以说：

> 当前 DG-SNN V2 在 15 格受控迁移矩阵下，预训练快照对 held-out eval 场景表现出稳定、可测量的初始优势；rewardOnly 存在跨种子波动但无失败格、无反转；dropout stress axis 在 maxSteps=4 收紧后产生真实退化梯度（0.2 → 0.3 delta 下降）；continued-learning gate 在 1 epoch + 1 trainSeed 收紧后仍受任务复杂度限制，separation 跨格为 0，gate 的含义是"预训练不退化"而非"正向 head-start"；wrong-prior diagnostic 引入 reverseMapping 预训练 + 正确映射继续训练，若跨格 separation<0，则 continued-learning gate 真正可失败——这是 Level 3 受控迁移证据中首次拥有可失败的 unlearning 测试。这是 Level 3 受控迁移证据。

不要说：

> 已经证明真实环境迁移稳定成立；或 dropout 下迁移优势保持。
