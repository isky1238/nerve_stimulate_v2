# Arbitration 审计（步骤 2：可学习动作仲裁）

## 阶段目标

2D-complex 已冻结为回归 gate。当前 raw spike-count 仲裁能处理非冲突和距离加权，
但在 semantic conflict 上会打平：

- food + toxin 同侧等距时，food 通路和 toxin 通路分别推动相反 motor。
- raw `arbitrateComplexMotorAction` 返回 `"conflict"`。
- `expectedActionForComplexState` 已有 priority 规则：同侧 food/toxin 冲突时，避毒优先。

本阶段新增 post-hoc learnable resolver，只在 raw decision 为 `"conflict"` 时介入，
把可解 semantic conflict 转为 left/right；真对称 conflict 仍通过 confidence threshold 回落为 `"conflict"`。

不在本阶段证明：

- reward-only 自主仲裁已经解决。
- SNN 拓扑内生仲裁已经完成。
- 真实/半真实输入迁移。
- 新 sensor / 新物种 distractor。

## 复测命令

```powershell
npm test
npm run audit:arbitration
npm run audit:2d-complex
npm run audit:2d-challenge
npm run audit:transfer
```

## Resolver 架构

Arbitrator 是独立的线性分类器，不加入 SNN synapse，不改 4 sensory / 4 interneuron / 2 motor 拓扑。

Evidence 只来自 spike counts：

```text
[iFoodLeft, iFoodRight, iToxinLeft, iToxinRight, leftMotor, rightMotor]
```

推理规则：

```text
if raw_decision != "conflict": pass-through
else:
  delta = logit_left - logit_right
  if abs(delta) < tau: return "conflict"
  return delta > 0 ? "left" : "right"
```

默认 `tau = 0.1`。Family E 上的 symmetric evidence 应产生接近 0 的 delta，从而保留 conflict。

## 训练协议

Required 使用 supervised label，先证明表达能力：

1. 先训练 2D-complex supervised SNN，得到稳定 spike evidence。
2. Family F dist-2 semantic conflict 提供主训练样本。
3. Family A 单物体 evidence 提供校准样本，避免线性模型只从 Family F 学成“同侧反向”。
4. 训练线性 resolver，SGD cross-entropy。
5. Eval 在 Family F held-out distance dist-1/dist-3。

Reward-only arbitration 只作为 diagnostic，不 gate。

## Family F：Semantic Conflict

| Scenario | Objects | Raw | Expected |
|---|---|---|---|
| F1 | food-left d2 + toxin-left d2 | conflict | right |
| F2 | food-right d2 + toxin-right d2 | conflict | left |
| F3 | food-left d1 + toxin-left d1 | conflict | right |
| F4 | food-right d1 + toxin-right d1 | conflict | left |
| F5 | food-left d3 + toxin-left d3 | conflict | right |
| F6 | food-right d3 + toxin-right d3 | conflict | left |

F1/F2 是 train split，F3-F6 是 held-out distance split。

Family E 不变：food-left + food-right 等距、toxin-left + toxin-right 等距仍期望 `"conflict"`。

## Required Suites

| Suite | Gate | 说明 |
|---|---|---|
| Determinism | same trace + held-out SR >= 0.8 | 训练可复现 |
| Semantic raw gate | Family F raw 全部 conflict | 验证问题真实存在 |
| Supervised arbitration | trained SR >= 0.9, fresh SR <= 0.1, sep >= 0.8 | 学到可解冲突 |
| Held-out distance | dist-1/dist-3 SR >= 0.8 | 距离泛化 |
| Non-degradation | A >= 0.8, B/C/D >= 0.5 | 不破坏 2D-complex |
| True conflict preservation | Family E 仍 conflict | 不强行仲裁 |
| Blank preservation | noopRate == 1, meanReward == 0 | 空世界静止 |

## Diagnostic Suites

- Motor-only ablation：只给 leftMotor/rightMotor counts，预期无法解 semantic tie。
- Reverse-prior control：反向 semantic label 训练，检查 resolver 是否跟随标签。
- Wrong-prior normal eval：反向训练后按正常标签评估，应明显变差。
- Reward-only feasibility：REINFORCE-style reward update，记录 SR，不 gate。**当前 recordCount 极低（~2），SR 数值不构成可行性证据，仅证明 reward update 路径不抛错。需扩到 ≥30 条记录后再讨论判别力。**
- Frozen baseline：未训练 resolver 应回落 conflict。
- Multi-seed matrix：3x3 seed 矩阵，**当前 9 cell 全使用受控 Family F scenario geometry（6 个固定场景换 seed），证明的是训练确定性，不是 scenario 泛化稳定性**。暂不 gate。

## 已知边界

1. Resolver 是 post-hoc linear classifier，不是 SNN 内生神经元。
2. `arbitrateComplexMotorAction` 仍是 raw 决策来源，未被替换。
3. Evidence 不含 raw world object、scenario id、expectedAction 或 distance。
4. Family A 校准样本是必要的：仅用 Family F 会欠定，线性模型可能学成“同侧反向”而不是 food/toxin 语义。
5. Family E 保留 conflict 是 threshold 结果，不是单独读取场景 ID。
6. Reward-only 仍是 diagnostic；失败不阻塞 requiredPassed。**当前 recordCount≈2 的 SR=1 是 empty-gate 风险——任何返回常量的模型都能拿满分。判别力需 ≥30 条记录才可讨论。**
7. 本阶段不修改 network export / loader；arbitrator state 不进入 synapse 数组。
8. Multi-seed matrix 的 successMean=1 是训练确定性结果，不是 scenario 泛化——9 cell 共用同一组 6 个 Family F 场景几何，只换 seed。

## 结论表述

可以说：

> 当前 DG-SNN V2 通过 arbitration 审计；在 supervised spike evidence 下，线性 post-hoc resolver
> 可将 semantic raw conflict 转为 priority-correct action，并在 held-out distance 上保持成功，
> 同时保留 Family E true conflict 和 blank/noop 边界。这证明“post-hoc 线性 resolver 在 supervised
> spike evidence 下可解 semantic conflict”在受控 2D-complex 范围内可行。
> **本阶段证明的是 post-hoc 线性 resolver 的可行性，不是 SNN 内生仲裁——拓扑未增加仲裁神经元，
> 也不是 reward-only 自主仲裁——reward-only diagnostic 当前样本量不足以判别（recordCount≈2，
> SR=1 不构成可行性证据）。**

不要说：

> SNN 已经内生学会动作仲裁；reward-only 自主仲裁已经解决；该 resolver 已具备真实环境部署价值；
> 或 multi-seed matrix 的 successMean=1 代表 scenario 泛化稳定性（当前仅证明训练确定性）。
