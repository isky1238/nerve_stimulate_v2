# 2D-challenge 瓶颈突破验证计划

## 当前阶段目标

`2D-lite` 已经稳定收敛，继续增加同构 seed 的边际价值较低。本阶段改为验证瓶颈：

- 多步 episode 是否可复现。
- supervised 是否能作为多步上限 baseline。
- frozen baseline 是否明显更差。
- reward-only 是否至少产生可复现 reward-driven updates。
- train/eval seed 是否隔离。
- blank、conflict、低噪声场景是否被明确记录。

## 复测命令

```powershell
npm test
npm run eval
npm run audit
npm run audit:2d
npm run audit:2d-challenge
npm run export:2d-challenge
```

## 学习模式

- `supervised`：目标 motor 直达学习，只作为上限 baseline。
- `rewardOnly`：禁用监督更新，只使用 reward learning；训练时允许确定性探索以产生 eligibility。
- `frozen`：禁用学习、探索、capture 和 decay，作为下限 baseline。

## Required suites

`npm run audit:2d-challenge` 当前 required 覆盖：

- multi-step deterministic replay：同 seed 多步 trace 必须一致。
- supervised multi-step baseline：supervised successRate 至少 `0.8`。
- frozen baseline separation：supervised 比 frozen 至少高 `0.3`。
- reward-only feasibility：不要求高成功率，但必须有非零 reward updates，且 replay 稳定。
- train/eval seed isolation：训练 seeds `[1,2,3,4,5]`，评估 seeds `[101,102,103,104,105]`。
- blank sparse world：空场景应保持 noop、无 reward。
- conflict boundary：矛盾输入必须记录为 conflict，不能计为成功。

observation dropout `0.1` 目前是 diagnostic，不阻断 requiredPassed。

## 预训练导出

`npm run export:2d-challenge` 默认保存：

- `exports/pretrained/2d-challenge-supervised-pretrained.json`
- `exports/pretrained/2d-challenge-rewardOnly-pretrained.json`

保存内容包括：

- 训练后 neuron / branch / synapse 状态。
- challenge config。
- successRate、meanReward、rewardUpdateCount、supervisedUpdateCount 等指标。
- train/eval seed 元数据。
- 评估 episode 摘要。

使用边界：

- `supervised` 导出只作为上限 baseline。
- `rewardOnly` 导出可作为未来真实测试的预学习候选。
- 两者都不是部署模型，也不是泛化证明。

## 结论表述

可以说：

> 当前 DG-SNN V2 通过了多步 2D-challenge 的 supervised/frozen/reward-only 可行性审计；reward-only 已产生可复现 reward updates，但是否能自主解决任务仍需进一步验证。

不要说：

> 当前模型已经完成 reward-only 自主二维学习。
