# 二维环境前审计结论

## 当前结论

当前模型可以进入“二维环境设计准备”，但不应直接宣称具备真实环境学习能力。

更严格地说，现有结果只证明：

- 固定拓扑、固定 one-hot 输入、监督标签直达的离线任务可以复现。
- learningOn 明显优于 learningOff。
- 24 种四样本训练顺序全排列均可收敛。
- 左右镜像标签集也可学习，降低了完全硬编码映射的风险。
- 空输入时训练后的网络保持静默。

当前结果不能证明：

- 自主奖励学习。
- 拓扑自动发现任务结构。
- 多物体、冲突输入、噪声输入下的稳定行为。
- 二维世界中的观察、动作、奖励闭环可用。
- 泛化到未定义输入组合。

因此 Test E 应理解为 **fixed-topology supervised smoke test**，不是泛化学习证明。

## 必需门槛

进入二维环境测试前，至少运行：

```powershell
npm ci
npm test
npm run eval
npm run audit
npm run trace -- exports\trace-pre2d.json
npm run explain -- exports\trace-pre2d.json
npm run export -- exports\network-pre2d.json
```

通过标准：

- `npm test` 全部通过。
- `npm run eval` 中 Test A-E 全部 PASS。
- `npm run audit` 输出 `requiredPassed=true`。
- `npm run audit` 允许出现 `FAIL DIAGNOSTIC`，但它必须被当作结论边界记录，不能忽略。
- trace 能解释一次输入、传播路径、输出和权重变化。

## 已知边界

当前 `npm run audit` 会显示：

- `PASS REQUIRED blank input silence`
- `FAIL DIAGNOSTIC input edge-case diagnostics`

这个诊断失败来自冲突输入：

- `foodLeft + foodRight` 会同时激活 `leftMotor,rightMotor`。

这不是 bug 修复完成的信号，而是进入二维环境前必须承认的边界：当前网络没有动作仲裁规则。二维世界如果同时看到左右目标，行为可能抖动、双输出或无效。

## 下一步收紧项

二维环境前优先补：

- 全局 seed 合约：world、感知噪声、动作裁决、连接形成、奖励事件都必须由可记录 seed 派生。
- 环境级 trace：记录 world state、agent position、raw observation、sensory mapping、motor arbitration、reward、terminal reason。
- 冲突输入策略：明确双 motor 时是 abstain、winner-take-all、保持上一步，还是交给环境动作层仲裁。
- 训练/评估隔离：训练结束后冻结学习，再单独评估，避免计分窗口继续学习造成误判。
- 更多消融：`supervisedLearningRate=0`、`stableCaptureRate=0`、阈值扰动、错误初始权重、连接缺失。

## 推荐表述

可以说：

> 当前 DG-SNN V2 在固定拓扑监督离线任务上具备可复现的学习表现，并通过了进入二维环境前的 required 审计门槛。

不要说：

> 当前模型已经证明具备泛化学习能力或真实二维环境可用性。
