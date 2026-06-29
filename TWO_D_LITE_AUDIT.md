# 2D-lite 多轮复测计划与结论边界

## 当前阶段目标

本阶段只验证最小二维闭环：

- world state -> observation -> sensory mapping -> motor output -> action arbitration -> reward。
- 固定拓扑网络在受控 2D-lite 场景中的可复现表现。
- 多 seed、多场景、消融和冲突输入下的审计结果。

本阶段不证明：

- 自主奖励学习。
- 拓扑自动发现。
- 带噪声、遮挡、复杂路径规划的真实二维环境能力。
- 未定义输入组合上的泛化能力。

## 复测命令

```powershell
npm test
npm run eval
npm run audit
npm run audit:2d
```

`npm run audit` 仍然是 pre-2D 离线门槛。`npm run audit:2d` 是最小二维闭环门槛。

## 2D-lite required suites

`npm run audit:2d` 当前覆盖：

- deterministic replay：同 seed 运行两次，normalized trace digest 必须一致。
- multi-seed object placement：至少 `seed=1,2,3,4,5` 的 food/toxin 左右位置复测。
- mirrored world positions：镜像对象坐标后仍需满足同一 food/toxin 策略。
- blank world silence：空世界必须输出 `noop`。
- learning-on versus learning-off：learningOn 必须显著优于 learningOff。
- supervised-plasticity ablation：`supervisedLearningRate=0` 时必须阻断学习表现。
- composite and conflict arbitration：同向组合可输出动作；矛盾组合必须记录为 `conflict`，不能计作成功动作。

## Trace 要求

2D-lite trace 每步记录：

- world state。
- agent state。
- raw observation。
- sensory mapping。
- active motors。
- arbitrated action。
- reward。
- terminal reason。

这保证后续可以解释一次完整环境闭环，而不是只看最终准确率。

## 结论表述

可以说：

> 当前 DG-SNN V2 在受控 2D-lite 固定拓扑监督任务上，通过了多轮可复现审计，并能记录环境级 trace 与动作冲突边界。

不要说：

> 当前模型已经证明具备真实二维环境泛化能力、自主奖励学习能力或实际部署可用性。
