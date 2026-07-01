# 评估阶段方向与工程记忆

Type: project
Date: 2026-06-30
Scope: `/root/research/nerve_stimulate_v2`

> 本文件是当前唯一的评估指导文档。原 `nerve-stimulate-v2-eval-direction.md` 的项目记忆已合并到这里,避免 TODO 与 memory 状态分裂。

## 当前实证基线

### 分层重构行为基线(2026-06-30)
- 当前分支:`master`(decayProtected feature + 分层 refactor 已 ff-merge 进 master 并删除原分支);本轮分层只移动边界,不引入新学习机制。
- 冻结门:`npm test` 18/18;`audit:transfer:matrix` 15/15;`audit:2d-challenge` / `audit:2d-complex` requiredPassed=true。
- rewardOnly collapse 基线保持:
  - `audit:rewardonly:challenge-collapse`:40ep frozen SR=0.500,conflict=0.000,noop=0.857,meanReward=0.550。
  - `audit:rewardonly:collapse`:Family A SR=1.000/conflict=0/noop=0;multi-object SR=0.500,conflict≈0.333。
- 长程 spot:`SEED_LIMIT=4 SUBDIR=lr_spot_refactor npm run audit:rewardonly:longrange` → epoch300 mean SR=0.875,noop=0.353,noopStuck=0/4;stem cliff 未复发。
- 结构拆分后的职责边界:
  - config:只分类参数,保留 `defaultConfig` / `withConfig` 对外兼容。
  - topology blueprint:声明 sensory / interneuron / motor 节点;`sensory→inter` 是 structural stem(`stableWeight=1.1`,`decayProtected=true`),`inter→motor` 是 plastic readout(`fastWeight=0.35`,`decayProtected=false`)。
  - mechanism:传播/整合、eligibility、reward/supervised learning、capture/decay、exploration selection;`decayProtected` 只跳过 stableDecay,不跳过 fastDecay/learning/effectiveWeight。
  - task/environment:scenario、observation、expectedAction、world step、reward/terminal;不 import synapse/plasticity/export/audit。
  - runner:train/eval loop、epochProbe、trace/result aggregation。
  - diagnostics/report:audit 读取 metrics/diagnostics,报告只格式化事实,不反向驱动机制。
  - export/IO:snapshot shape/write/read;loader 从 topology blueprint skeleton 恢复,旧 snapshot 缺 `decayProtected` 时保留 blueprint 默认。

### 参数/机制分类表
| 类别 | 当前归属 | 说明 |
| --- | --- | --- |
| 阈值(thresholds) | `branchLocalThreshold`,`dendriteGateThreshold`,`axonThreshold`,`stableThreshold`,`useThreshold`,`contributionThreshold`,`weakWeightThreshold`,`negativeThreshold`,`inhibitionFreezeThreshold` | 判定门槛,不是结构线本身。 |
| 固定结构属性(structural) | topology blueprint 节点/边、slot limits、growth/cooldown 字段、`Synapse.decayProtected` | `decayProtected` 是固定结构 stem 属性,不是学习率或 reward 参数。 |
| 学习时间尺度(learningDynamics) | `fastDecay`,`stableDecay`,`eligibilityDecay`,`traceDecay`,`fastLearningRate`,`stableCaptureRate`,`supervisedLearningRate`,`rewardAdvantageBaselineAlpha`,`depotentiationRate` | 控制 fast/stable/trace/baseline 变化速度。 |
| 信息因子(signal/exploration) | `explorationStrategy`,`explorationEpsilon`,observation dropout,spike-count duration | 决定 learner 看见什么动作/观测证据。 |
| 训练协议(experimentDefaults/runner) | seeds、epochs、maxSteps、learningMode、reverseMapping、epochProbe | 由 task/runner/audit 提供,不混入机制。 |
| audit gate/diagnostics | required/diagnostic 标记、success/noop/conflict、wrong-prior、collapse、long-range cliff | 只读解释层;不能把 report 文案当机制事实。 |

### 2D-challenge(`requiredPassed=true`)
- supervised: SR=1.0,meanReward=1.05,conflictRate=0
- rewardOnly(advantage 更新后): **SR=0.5,meanReward=0.55,conflictRate=0,noopRate=0.857**
  → advantage 避免了旧版双侧 conflict,但 binary challenge 下仍大量 noop,不是自主学习已解决。

### 2D-complex(`requiredPassed=true`)
- Family A supervised SR=1.0;Family B(同向组合)=1.0;Family D(距离优先级)=1.0;Family E(等距冲突)正确触发 conflict
- **Family C(distractor)SR=0.5,conflictRate=0.333** — 瓶颈项,远物没被真正忽略
- rewardOnly(advantage 更新后):Family A **SR=1.0,conflictRate=0**;多物体 **SR=0.5,conflictRate=0.3125**
  → advantage 打破了 complex Family A 的旧 conflict 塌缩,但多物体仍未达 supervised 水平。

### Transfer matrix(15 格全过)
- sup separation: 1.000 / 1.000 / 1.000
- rew meanReward delta: min=0.550 mean=0.550 max=0.550
- rewardOnly success sep (frozen;fresh=noop): min=0.500 mean=0.500 max=0.500
  → 仍被 fresh-frozen=0 抬高,是 loader 保真 + rewardOnly 非零行为,不是强迁移证据。
- dropout 0.2 rewardOnly delta: min=0.445 mean=0.529 max=0.550
- dropout 0.3 rewardOnly delta: min=0.345 mean=0.502 max=0.550
- **continued-learning sep: 0.000 / 0.000 / 0.000**(15 格零方差,gate 退化为"预训练不退化")
- **wrong-prior sep: -1.000 / -1.000 / -1.000**(uniform 极端值)
- wrong-prior postCL:stable count=0/0/0,max stable=0.055(均值),max fast=1.066,**dual-lock=0/15**
- wrong-prior epoch curve(15-cell,continued-learning epochs 0/1/2/3/5/10):
  1ep fail(preSR=0.000,freshSR=1.000,sep=-1.000,wrongFast=1.066,wrongStable=0.055);
  2ep partial recovery(preSR=0.850,sep=-0.150,wrongFast=0.799);
  3ep recovered(preSR=1.000,sep=0.000,wrongFast=0.529);
  10ep wrongFast=0.000,SR=1.000。
  → 不是 stable dual-lock,也不是零去固化;stable depotentiation 很快,1 epoch 失败主要是 wrong-direction **fast** path 需要 2-3 epoch 才不妨碍行为。

## 已完成

- [x] transferMatrix.ts 增加 wrong-prior postCL 聚合指标(stableCount / maxStable / maxFast / dualLockCells)
- [x] 矩阵报告 rewardOnly success sep 明确标 frozen + fresh=noop,防误读为强迁移
- [x] transferAudit.ts wrong-prior 文案收紧:separation<0 只表"有伤害",不再说成"健康非空真";uniform -1.000 按 unlearning 失败读
- [x] TRANSFER_AUDIT.md / COMPLEX_AUDIT.md 把 rewardOnly credit assignment 写成结构性未解问题,而非"学得慢"
- [x] rewardOnly 改为 advantage 更新:`rewardAdvantage = reward - runningBaseline`,baseline 由 `rewardAdvantageBaselineAlpha=0.1` 更新
- [x] trace 记录 `rewardBaseline` / `rewardAdvantage`;预训练 metadata 记录 `rewardAdvantageBaselineAlpha`
- [x] 新增测试确认 rewardOnly trace 中 advantage 信号与 raw reward 可分离;`npm test` 18/18 通过
- [x] wrong-prior epoch curve 已跑通:1ep 压力失败,2ep 部分恢复,3ep 恢复;结论从"无法 unlearn"收紧为"fast-path unlearning 慢半拍"

## 待办(按优先级)

### A. 诊断探针 — 让 wrong-prior 解读从框架性收紧到实证
- [x] 在 wrong-prior suite 打印 pretrain 期(postCL 之前)的 wrong-direction stable / fast 权重,与 postCL 配对
      → transferMatrix summary 已加 preTrain 配对行。读数(15 格均值):**preTrain max fast=1.959, stable count=4, dual-lock=15/15** → **postCL(1ep) max fast=1.066, stable count=0, dual-lock=0/15**。
      → 结论:**stable depotentiation 快**(1 epoch supervised `wasWronglyActive` 把 stable dual-lock 15/15 清空),**fast-path depotentiation 慢**(1.959→1.066,需 2-3 epoch)。1ep 的 -1.000 纯属 fast-path 滞后,非 stable lock-in。
- [x] 把 postCL max fast(当前 1.066)纳入 matrix summary 打印,不止 stable count
      → fast 不归零是当前 -1.000 的主因,需在 summary 可见
- [x] 跑 wrong-prior 在 2/3/5/10 epoch continued-learning 的 sep 曲线,找 sep 从 -1 上升的拐点
      → 结论:**慢恢复**;2 epoch 部分恢复(sep=-0.15, wrongFast=0.799),3 epoch 恢复(sep=0, wrongFast=0.529),10ep wrongFast=0。非卡死、非 stable lock-in。1 epoch 保留为压力诊断。

### B. 诊断探针 — 分清 rewardOnly 剩余失败类型(已完成,见下方"A/B 诊断结论")
- [x] 加 rewardOnly 诊断:记录训练中 left/right motor 入突触 fastWeight 对称性随 epoch 演化
      → 新增 `audit:rewardonly:collapse`(complex)+ `audit:rewardonly:challenge-collapse`(challenge)。读数见下。
- [x] 记录 rewardOnly 训练期 successRate / conflictRate / noopRate 随 epoch 曲线
      → 关键发现:complex 与 challenge 训练期 trainConflictRate/trainNoopRate **全 0**(exploration 在掩盖),eval 才暴露 noop。
- [x] 将 Family A(binary challenge)、Family A(complex spike-count)、多物体(Families B/C/D)分开读
      → 三者失败机制不同:complex Family A 已解决(1.0/0);challenge 是 fastWeight 衰减 + exploration 掩盖 noop;多物体是 compositional vote-tie。

### A/B 诊断结论(2026-06-30,阶段完成)

**wrong-prior = 慢恢复。** stable depotentiation 1 epoch 清空 dual-lock,fast-path depotentiation 需 2-3 epoch。非"无法 unlearn"、非 stable lock-in。supervised 的 `wasWronglyActive` stable 去固化有效;rewardOnly 缺这个路径(见 C)。

**rewardOnly "双侧共增强→conflict 塌缩" 假说 — 不成立(refuted)。**
- complex Family A 训练 40 epoch:**trainConflictRate=0、trainNoopRate=0 全程**;leftFastSum/rightFastSum 同步从 ~1.3 衰减到 ~0.4,**asymmetry 全程低(~0.04)**;最终 eval SR=1.0/conflict=0/noop=0。Family A 是靠 **spike-count 仲裁**(非对称权重)解决的,advantage 没有选边,通路保持对称。
- 残留多物体 conflict=0.333 是 **compositional vote-tie**(两物体投相反 motor、spike 数相等),低 asymmetry 下发生,非双侧共增强。
- **challenge rewardOnly noop=0.857 的真因**:训练期 trainNoop=0 因为 `selectExplorationAction` 在 noop 时强制选 motor **掩盖了 noop**;但 fastWeight 衰减到 ~0.5(< motor 阈值 ~1.0),eval(关 exploration)就 noop 86%。advantage 的 net 负 delta(baseline EMA 超过 mean reward)让 fastWeight 慢慢衰减低于阈值,而 conflict-gated exploration 让网络在训练期看不到自己的 noop,拿不到"必须行动"的学习信号。

**对 C 档的重定向(关键):** 原计划的 bilateral-targeted 机制(loser-suppression、互斥/winner-take-all)**不是首选**——双侧共增强已被证伪。真正该先加的是:
1. **fastWeight 衰减/阈值问题**:advantage net 负导致 fastWeight 跌破 motor 阈值。候选:advantage 归一化、fastWeight 下限、或让 baseline 不超过 success reward。
2. **ε-greedy 替换 conflict-gated 探索**:让网络即便有弱偏好也探索,且训练期不掩盖 noop(给"该行动却没行动"一个学习信号)。
3. **多物体 compositional vote-tie**:distance-weighted 仲裁在 rewardOnly 下失效(因 fastWeight 太低),修 1 后可能自愈,否则需独立处理。

### 并发复测收紧(2026-06-30,6 audit 并发跑)

复跑 `audit:2d-challenge` / `audit:2d-complex` / `audit:rewardonly:collapse` / `audit:rewardonly:challenge-collapse` / `audit:transfer:matrix` / `audit:rewardonly:longrange`(SEED_LIMIT=8),全部 requiredPassed=true,数字与基线**完全复现**(无回归):

- 2d-challenge rewardOnly:SR=0.5/noop=0.857/conflict=0/meanReward=0.55 ✓
- 2d-complex:Family A rewardOnly SR=1.0/0/0;多物体 SR=0.5/conflict=0.3125 ✓
- transfer:matrix 15/15;rew delta=0.550;success sep=0.500;dropout 0.2=0.529/0.3=0.502;continued-learning sep=0.000;wrong-prior sep=-1.000,preTrain max fast=1.959/dual-lock=15/15 → postCL max fast=1.066/dual-lock=0/15 ✓
- longrange 8seed×300ep:mean SR=0.906,noop=0.265,5/8 solved+3/8 partial+0/8 stuck ✓ 悬崖未复发

**新观察(收紧结论):**

1. **#5 challenge noop 有瞬态不对称抖动,非纯对称衰减。** `challenge-collapse`(seed 21)asymmetry 全程不平稳:epoch 10=0.345、epoch 29=0.338、epoch 32=0.317(rightFast 爬到 ~0.62、leftFast 跌到 ~0.27),但**无法锁定**,epoch 39 回收到 0.038。对照 `collapse`(complex)asymmetry 全程平稳低(~0.04,仅 epoch 26 一次 0.29 抖动)。→ challenge 网络会**短暂选边**但守不住,两侧随后在 advantage net 负 delta 下一起衰减跌破阈。这比旧表述"低对称衰减"更精确:challenge 的失败含一个**不稳定的瞬态 commit**。complex Family A 无此抖动(靠 spike-count 仲裁,不需要 commit 一侧)。
2. **#4 多物体 conflict 模式与 supervised Family C 共享**(见上 C 档 #4 收紧):supervised 也 fail Family C 0.5/0.333,故修 rewardOnly stable 去固化不会自愈 multi-object conflict。
3. **#3 rewardOnly wrong-prior 不可测**(见上 C 档 stable 去固化收紧):reverseMapping 对 rewardOnly 是 no-op,无注入路径。
4. **longrange `dropTiming` 指标已失效(轻微)。** decayProtected 修复后 effSum(left+right fast+stable)单调增长(stable 持续 capture),`first epoch effSum<1.5/1.0` 恒为 0/24。该指标是为修复前的悬崖设计的,现在读数恒空。可保留作"悬崖未复发"的反向证据(0/24 dropped = 无坠崖),但不再能作坠崖时序读数。

### C. 结构性修复 — rewardOnly credit assignment(依赖 A/B 结论,**已据 A/B 重定向**)
- [x] advantage baseline:用相对回报替代 raw reward,减少正奖励轮流共增强
- [~] **ε-greedy 替 conflict-gated 探索 — 试了,REGRESSED,已回退默认**
      → 实现:`explorationStrategy: "conflictGated" | "epsilonGreedy"` + `explorationEpsilon`(newModelConfig.ts),`selectExplorationAction` 分派(challenge2d.ts)。默认仍 conflictGated,ε-greedy 保留为 toggle + 测试。
      → 结果(ε=0.2 常数,非退火):**全面退化**。2D-challenge rewardOnly SR 0.5→0、noopRate 0.857→**1.0**;transfer rewardOnly pretrained noopRate=1、separation=0 → 15/15 cell requiredPassed=false(gate 翻负)。`rewardUpdateCount` 反而升高(55717)但全是把权重往下推。
      → 根因:常数 20% 随机 forcing + advantage net 负 delta → fastWeight 持续衰减,网络永不 commit,frozen eval 100% noop。**conflictGated 的"commit 后停 forcing"对收敛是必要的**,ε-greedy 把它去掉了。
      → 教训:诊断说"exploration 掩盖 noop"是对的,但"让 noop 可见"不等于"能学会行动"——网络没有 cold-start 信号就永不 commit。ε-greedy 单独加反而是净负。
      → 候选复活方向(未做):退火 ε(高→低,先 cold-start 后收敛);或 ε-greedy 只在 commit 后探另一侧(不在 noop 时停 forcing)。但都属第二变体,等先试 fastWeight 阈值再说。
- [x] **(已修)长程 rewardOnly noop 悬崖 — 根因不是 advantage/credit,是 sensory→inter 干线被 stableDecay 侵蚀过阈**
      → 长程验证(24 seed × 300ep,只读 scratch 脚本,不改源):rewardOnly 2D-challenge 不是单调恢复,是"爬升到峰(epoch~200 SR 0.865)→ 灾难性悬崖(250ep SR 0.083 → 300ep 0/24 全死)"。
      → 决定性证据:`runChallengeNetworkStep` 是两 tick 结构,tick1 传导 sensory→inter 后整合 inter,**inter 的 somaPotential = sensory→inter eff(单次传导,无跨 tick 累积)** → inter 发放当且仅当 `eff_sensory→inter ≥ axonThreshold(1.0)`。而 sensory→inter init stable=1.1,余量只有 0.1,被 `stableDecay=0.99999` 慢慢侵蚀,~200-250ep 跌破 1.0 → inter 停发 → 整条 motor 链瞬间静默(下游 inter→motor 的 6.8 stable 完全无关,瓶颈一闭下游无输入)。
      → **不可逆吸收态**:inter 不发 → inter→motor eligibility=0 → reward 学习拿不到梯度 → 无法重建 sensory→inter(且 reward 学习只动 fast,不动 stable)。
      → **反证**:全局 `stableDecay=1.0` 跑 8seed×300ep,悬崖消失(300ep SR 0.938 vs 基线 0.000)。坐实根因。
      → **修复(已落地)**:不是全局关 stableDecay(那会让 learned stable 永不遗忘、wrong-prior 难读),而是给 `Synapse` 加 `decayProtected: boolean` 语义——结构性硬件干线(sensory→inter fixed stem)标 true,`decayWeights` 跳过其 stableDecay,fastDecay/学习/传导照常;learned/plastic 突触仍 false。`createOfflineLearningNetwork` 对 sensory→inter(stable=1.1)传 true,inter→motor 传 false;loader 字段拷贝白名单加 `decayProtected`。
      → 验证:`npm test` 18/18;transfer matrix 15/15 不翻负(rew delta=0.550、success sep=0.500、wrong-prior postCL max fast=1.066/dual-lock=0/15 全持平基线);长程 24seed×300ep 悬崖消失——250ep SR 0.083→**0.927**、300ep 0/24→**19/24 solved + 5/24 partial + 0/24 stuck**、mean SR 0.000→**0.948**。曲线在 epoch≤200 与基线重合(茎还没跌破阈),之后基线坠崖、修复继续爬升。
      → 注意:40ep 的 `audit:rewardonly:challenge-collapse` noopRate 仍 0.857(未变)——悬崖在 ~200ep 才发生,40ep 时 stem 还没跌破阈,故短期行为不变。40ep noop 是另一回事(bootstrap/commit 问题,ε-greedy 试过净负),与长程悬崖不同层。
- [ ] 给 `applyRewardLearning` 加 loser-suppression(原计划,降级:双侧共增强已证伪,但仍可补沉默通路免疫问题)
      (当前 `applyRewardLearning` 对 eligibility=0 的沉默错误通路仍免疫;负 advantage 已能压活跃错误通路)
- [x] 把 supervised 的 `wasWronglyActive` 去固化逻辑以 reward-driven 形式补进 rewardOnly
      (当前 rewardOnly 仍无 stableWeight depotentiation 路径,只能靠 stableDecay=0.99999 被动遗忘)
      **【并发复测收紧 2026-06-30】** "wrong-prior 在 rewardOnly 下会卡更久" 此前判为**不可测**:`reverseMapping` 只翻 `expectedAction`(`challenge2d.ts:250-252`),reward 由 `scoreChallengeStep(state,after,executedAction)` 算、不读 expectedAction,rewardOnly 路径(`:270-275`)用 `rewardAdvantage=reward-baseline` 也不读 expectedAction → **rewardOnly+reverseMapping 注入是 pure no-op**。transfer:matrix wrong-prior 只跑 supervised。
      **【已测 2026-06-30,`scripts/wrongprior_rewardonly.cjs`,6seed×300ep,只读脚本不改源】** 用 **bypass 设计**绕开上述 no-op:**Phase1 用 supervised+reverseMapping 注入 wrong-prior**(supervised 读 expectedAction,有效;preTrain dualLock=6/6、wrongMaxStable=2.0=maxWeight),**Phase2 两臂都用 reverseMapping=false(正确映射)继续学**,只变 learningMode。这样 reward 是真实正确映射 reward,wrong-prior 网络被 stable 逼着做错→低 reward→负 advantage→压 fast(实测 rewardOnly 臂 wrongMaxFast 0.68→0.05,主动信号,非 no-op)。
      → 结果:supervised 臂 6/6 在 **1ep** 恢复(SR=1.0)、1ep 清空 dualLock(wrongMaxStable 2.0→0.073,`wasWronglyActive` 一刀砍到阈下);**rewardOnly 臂 0/6 在 300ep 内恢复、0/6 清空 dualLock,wrongMaxStable 全程钉死 2.000,SR 全程 0,dualLock 100%**。
      → 结论比原假设更强:不是"卡更久",是"**300ep 内永久卡死**"。根因双层:① `applyRewardLearning` 只动 fastWeight(deltaStable:0)、不碰 stableWeight,stable=2.0 单凭自己持续驱动错误 motor,fast 衰减 irrelevant;② **自维持锁**——错误 stable(2.0)驱动错误 motor 发放→coactivity→eligibility>0→`captureStableWeights` 每步把 fast→stable 回补,stable 钉在 maxWeight=2.0 不被动衰减(对比 supervised 臂砍到 0.073 阈下后停发→无 coactivity→无 capture→只剩被动慢衰减 0.073→0.066)。③ 正确通路 correctMaxFast 全程 0.000——错误 stable 逼网络一直失败→负 advantage→正确突触 eligibilityTrace=0 被 `continue` 跳过,bootstrap 不起来(与第1条 bootstrap/commit 同根)。
      → 公平性边界:wrong-prior 由 supervised 注入(人工强 stable=2.0 锁)。实验回答"**给定 stable dual-lock,rewardOnly 能否解开→不能**",不回答"rewardOnly 是否会自造这种锁"(那归长程悬崖/collapse audit)。
      → 工程含义:reward-driven stable 去固化是**必要的**(能打断自维持锁),但单加它解不了 correct-path bootstrap(correctMaxFast=0),必须与第1条同治。
- [ ] 多物体 compositional vote-tie
      **【并发复测收紧 2026-06-30】** 不能指望"修 rewardOnly stable 去固化后自愈":supervised **本身**就 fail Family C(distractor)SR=0.5/conflict=0.333(`audit:2d-complex` required 项),而 supervised 已有 stable 去固化。rewardOnly 多物体 conflict=0.3125 的签名与 Family C 的 0.333 一致 → conflict 模式是**共享的 distractor 距离仲裁极限**,非 rewardOnly 特有 credit 缺口。修 #1 不会消除 Family-C conflict。本项需独立的距离仲裁修复(与 #1 解耦)。注:rewardOnly 多物体 SR=0.5 仍低于 supervised 多物体(B/D=1.0+C=0.5≈0.83),所以另有 rewardOnly 特有的幅度缺口,但 conflict 模式本身共享。
- [ ] 修复后重跑目标:
      2D-challenge rewardOnly SR > 0.5 且 noopRate 明显下降(< 0.3);
      2D-complex multi-object rewardOnly SR > 0.5 且 conflictRate < 0.3;
      transfer matrix rewardOnly sep 不因修复翻负(ε-greedy 已证会翻负,新机制必须不翻负);
      **新增回归**:`audit:rewardonly:challenge-collapse` 训练期 trainNoopRate > 0(不再被掩盖)、最终 evalNoopRate 下降。

**C 档进度:ε-greedy(常数)净负已回退;长程 noop 悬崖已修(decayProtected sensory stem,24/24 stuck→0/24);rewardOnly stable 去固化缺口已实测坐实(6seed×300ep,supervised 1ep 恢复 vs rewardOnly 0/6 永久卡死,自维持 stable 锁)。剩余:rewardOnly 40ep 短期 noop(bootstrap/commit,非悬崖)、多物体 compositional vote-tie(supervised 也 fail Family C,需独立距离仲裁修复)。**

### E. STDP/BAP 塑性基线大改(2026-06-30,分支 `feat/stdp-bap-baseline`,粗颗粒度跑通)

动机:把"重要性判定"从前向瞬时二值 coactivity 改为后向 BAP 加权 × STDP 时间窗,对治五层缺口(载体/时间因果/信用分配/错误信号/力度)。计划见 `/root/.claude/plans/functional-napping-clarke.md`。两阶段,soft 红线(翻负记录不回退)。

**Phase 1(载体+时间窗,commit 3888c5d):**
- `updateEligibility` 重写为 signed STDP:`ltp=stdpLtpRate×preTrace×postActive×bapWeight`,`ltd=stdpLtdRate×postTrace×preActive×bapWeight`;`bapWeight=effectSign×|effectiveWeight|`(保留抑制性符号,按贡献加权)。preTrace/postTrace 从死代码激活,稳态归一化到 [0,1]。complex2d 的 eligibility 更新移入 micro-tick 循环。supervised stable 去固化触发条件 `elig>0` → `wasWronglyActive + |elig|`(STDP signed 不再让错误突触靠 LTD 负号逃过去固化)。
- 状态:supervised SR=1.0 不变(requiredPassed=true);rewardOnly 退化到 noop(eligibility 幅度 ~0.05,比 baseline ±1 弱 ~20x);2 transfer 测试挂(soft)。

**Phase 2(信用分配+modulator,commit e3eff46):**
- `normalizeEligibility`:per-post-spike 除法归一化正 eligibility(总 LTP credit 每次 post 发放=1.0),`eligibilityNormalization` toggle(默认 on)。`computeModulator=tanh(|advantage|×modulatorGain)` 与二值 inhibition-freeze gate 复合。`applyRewardLearning`/`applySupervisedMotorLearning` 加 modulator 参数(supervised 传 1)。**未补 reward→stable 去固化**(plan 2.3:看新载体能否自愈)。
- 状态:`npm test` 18/18(Phase 2 归一化+modulator 让 rewardOnly 恢复非零行为,解开 Phase 1 挂的 2 transfer 测试)。rewardOnly SR=0.25/noop=0.947(baseline 0.5/0.857,比 Phase 1 的 noop=1.0 恢复但仍低于 baseline)。**红线 requiredPassed=true,rew delta min=0.275(baseline 0.55,降但不翻负)**。

**关键验证(整体效应,`scripts/coactivity_sweep.cjs`,6seed×40ep rewardOnly):**
- **不是"某些 seed 极端化"——是所有 seed 一致塌缩**:`%extremeMax=0`(归一化控住单突触失控),但 `%collapse=100%`(全突触 eligibility 同号正)。
- **STDP 的 LTD 半边基本是死的**:LTP/LTD balance = 5e8(正 eligibility 远大于负)。两 tick 结构下 post-before-pre 时序几乎不发生 → LTD 项不激活。STDP 退化成纯 Hebbian LTP。
- elig-|eff| 相关性 0.83-0.97:BAP 加权生效(强突触 elig 大),但正反馈风险(强→elig 大→更强)。
- eligibility 幅度 0.2(归一化后比 Phase 1 的 0.05 大,但仍比 baseline ±1 小)。

**wrong-prior 自愈测试(`wrongprior_rewardonly.cjs`,6seed×50ep):**
- rewardOnly 仍 **0/6 自愈**,dualLock 100% 全程,wrongMaxStable 2.0→1.942 几乎不动。**新载体(BAP+STDP+信用分配+modulator)没能让 rewardOnly 解开 stable 锁** → 坐实 plan 2.3 "不能→后续单独加 reward→stable 去固化开关" 分支。
- 副作用:supervised 臂 dual-lock 现在**也清不掉**(supDLC=never vs baseline 1ep 清空)——|elig| 太小,stable 去固化砍不到阈下。supTTR 仍在(8.2ep 恢复行为,靠 fastWeight),但 stable 锁残留。

**E 档结论(粗颗粒度,待统一度量衡):**
1. 两阶段语义链已搭通,supervised 不破(18/18,requiredPassed=true),红线不翻负(rew delta 0.275>0)。
2. rewardOnly 行为低于 baseline(SR 0.25 vs 0.5),属计划预期"效果差后期解决"。
3. 诊断坐实两个结构性问题(非 seed 随机性、非参数微调能解):
   - ~~**STDP LTD 半边在两 tick 结构下不激活**(post-before-pre 时序罕见)→ 需统一度量衡时的传导延迟/物理时间轴。~~ **已推翻,见 E2 修订**:LTD 失效根因是 trace 归一化 bug(时间窗被压成 1 步),不是拓扑/时序,修好 trace 后 LTD 半边活了。
   - **eligibility 幅度比 baseline 弱 ~5-20x**(归一化+STDP 稀疏性)→ 依赖 elig 幅度的学习(rewardOnly 建通路、supervised stable 去固化)都变慢/失效。**E2 部分缓解**:trace 修复后 supervised stable 去固化恢复(postCL wrongMaxStable 1.99986→0.766)。
4. **新载体不能自愈 rewardOnly 的 stable 锁** → 印证"reward→stable 去固化开关仍必要"(与 C 档 #3 一致),但单加它(上一轮 C3 结论)不充分,需与新载体同治。
5. ~~下一步优先级:统一度量衡(物理时间轴/传导延迟,让 LTD 半边能激活 + eligibility 幅度重定标)~~ **E2 修订**:LTD 半边已靠 trace 修复激活,传导延迟/物理时间轴降级为可选优化(非必需)。下一步优先级改为:eligibility 幅度重定标 + 评估补 reward→stable 去固化。

### E2. 时间窗 bug 修复 + haiku agent 分发复测(2026-06-30,commit 36ea064 + 9346645)

**方法**:主诊断者卡在"LTD 失效根因"判定上(先判活动率不对称,试 per-tick 更新回归后放弃)。按用户要求分发 4 个 haiku agent,同一提示词、同一证据、彼此盲测,独立找盲点。

**4 agent 共识:**
- `eligibilityDecay=0.9` 是放大器,非根因(慢衰减积分器把每步微小净偏置放大成巨大累积)。
- per-tick 更新放弃**是对的**(agent 都分析:tick1 时 motor 还没发,会产生"未授权 LTD"或两边归零,不解决核心)。
- 活动率不对称存在,但幅度被误判/非主因。
- 5e8 的 balance 部分是**比值假象**(`sumPos/(sumNegAbs+1e-9)`,分母趋 0 时爆炸),非"LTP 极其巨大"。

**agent C 独家发现(主诊断者忽视的真根因):trace 归一化 bug。** 主诊断者写的 `(trace*decay + active) / (1/(1-decay))` 不是"归一化到 [0,1]"——除以 1/(1-decay) 同时把衰减率乘了 (1-decay),有效衰减 = `decay*(1-decay)` = 0.1275(traceDecay=0.85),**trace 时间常数从 ~6.7 步压缩到 ~1.1 步**。跨步 STDP 时序(尤其 post-before-pre LTD)在 1 步记忆下无法表达 → LTD 失效。数学验证:声称稳态 1.0,实际 0.172。

**修复 1(commit 36ea064):trace 用正确 EMA。** `trace = trace*decay + active*(1-decay)`,稳态=1.0,时间常数~1/(1-decay) 步。结果:LTP/LTD balance 5e8→~1.5(fracNeg 0.02→0.19-0.31,LTD 半边活了)。但 rewardOnly 反而退化(SR 0.25→0、noop 0.947→1.0,红线 rew delta 0.275→0)——因为 trace 记忆 6x 更久后,无 sensory 步的 preTrace 残留单边 LTP 更严重。

**修复 2(commit 9346645,agent A/B/C 一致推荐):LTP 门控 preActive。** `ltpElig = stdpLtpRate × preTrace × postActive × preActive × bap`。无 sensory 步(preActive=0)LTP=0,不单边积累。LTD 仍门控 preActive(post-before-pre 需 pre 当前发才能检测 post 先发)。结果:rewardOnly SR 0→0.25/noop 1.0→0.947(恢复 Phase 2 水平);test 16/18→17/18;balance fracPos 1.0→0.5/fracNeg 0→0.5(近对称);**红线 rew delta 0→0.55(回到 baseline 水平,不翻负)**、success sep 0.5、continuedLearning 0;supervised wrong-prior stable 去固化部分恢复(postCL wrongMaxStable 1.99986→0.766、stableCount 4→2)。

**E2 关键结论:**
1. **"先修时间轴"修对的是 trace 时间窗 bug,不是加传导延迟/物理 ms。** 用户的"先修时间轴不然判断不了"判断正确——但根因是主诊断者自己写坏的 trace 归一化,不是 STDP 本身或拓扑。修好后 STDP LTD 半边活了,传导延迟/物理时间轴降级为可选优化。
2. **主诊断者之前把"LTD 失效根因"判为"活动率不对称"是错的**——agent 分发纠正了。教训:单人诊断易陷自己推理闭环;分发独立 agent(同提示词、彼此盲)能有效打破。agent C 的数学验证(声称稳态 1.0 实际 0.172)是决定性证据。
3. requiredPassed=false 只因 rewardOnly pretrain SR=0(效果差,预期),**不是红线翻负、不是崩溃**(matrix 偶发 fork crash 是并发竞态,重跑可复现真实状态)。
4. rewardOnly 仍弱(SR 0.25 vs baseline 0.5)——属"效果差后期解决",但 STDP 语义链现在健康(LTP/LTD 对称、supervised 不破、红线不翻负)。
5. 下一步:eligibility 幅度重定标 + 评估补 reward→stable 去固化。传导延迟/物理时间轴非必需。

### E3. single-stimulus 轨迹追踪 — 推翻 rewardOnly "全局 noop 塌缩" 误判(2026-06-30,`scripts/snapshot_analyze.cjs`)

**工具(两种快照模式,固化)**:
- **权重快照**(`weights` 模式):全量网络状态分析。列所有 sensory→inter stem + inter→motor 突触的 eff/fast/stable/elig/state,按 CORRECT/WRONG 分组,给"哪些通路建够(DRIVES)/没建够(weak)"汇总。不模拟,只读已学权重。
- **路径快照**(`path` 模式):单刺激轨迹**模拟**,用真实 `runChallengeEpisode` 跑完整多步 episode(非手动 tick),逐 case 记录 success/steps/actions/reward + step-0 逐突触传播细节(哪个 inter 发、motor axonDrive 多少、为何没到阈值)。
- 用法:`SNAPSHOT=supervised|rewardOnly node scripts/snapshot_analyze.cjs weights|path [food-left|food-right|toxin-left|toxin-right|all]`。旧 `single_stim_trace.cjs`(手动两 tick)已删除,被本工具替代(path 模式用真实 episode 更严谨)。

**权重快照全量证据(rewardOnly):**
| inter | CORRECT motor | eff | 状态 | wrong-max\|eff\| |
|---|---|---|---|---|
| iFoodLeft | leftMotor | 1.990 | DRIVES | 0.191 |
| iFoodRight | rightMotor | 1.490 | DRIVES | 0.247 |
| iToxinLeft | rightMotor | 0.456 | weak | 0.086 |
| iToxinRight | leftMotor | 0.437 | weak | 0.024 |
→ food 两条通路**对称**建够(1.49-1.99),toxin 两条通路**对称**没建够(0.43-0.46),wrong-direction 全 <0.25(正确削低)。不是随机失败,是 toxin 这一类特异地弱。

**路径快照真实 episode 证据(rewardOnly,4 case):**
- food-right:success=true,2步 right/right。iFoodRight 发→rightMotor axonDrive=1.490≥1→发。✓
- toxin-left:success=false,12步 noop。iToxinLeft 发→rightMotor axonDrive=0.456<1→不发→noop。
- toxin-right:success=false,12步 noop。iToxinRight 发→leftMotor axonDrive=0.437<1→不发→noop。
(food-left 同理 success,见前轮)→ 严谨复现"food success / toxin noop",非手动 tick 推断。

**关键发现(颠覆 E 档 #2 + E2 的 rewardOnly 诊断):**
- rewardOnly 预训练快照**不是全局 noop**。frozen eval 逐场景:food 场景全部 success(left/right 正确,2步拿食物,reward=1.1);toxin 场景全部 noop(12步)。
- 因果(路径快照):food 通路 eff=1.99 发育成功;toxin 通路 eff=0.456 够不到 motor 阈值 1.0→noop。
- audit 报的 "SR=0.25/noop=0.947" 是 food(2步成功)+toxin(12步 noop)混合的**统计假象**,把"toxin 通路弱"误读成"整体 noop 塌缩"。

**对前几轮诊断的修正:**
1. E 档 #2 "rewardOnly SR 0.25 vs baseline 0.5,效果差" + E2 "rewardOnly SR 0→0.25" 的归因(credit assignment/eligibility 幅度/stable 锁)**部分瞄错靶子**。rewardOnly 真问题不是全局塌缩,是 **toxin 回避通路特异地建不夠**(eff 0.456 vs food 1.99)。
2. 500ep wrong-prior 自愈"吸收态"(E 档 #4/E2)结论仍成立(那是 wrong-prior 注入场景),但**不该外推到正常 rewardOnly 训练**——正常训练下 rewardOnly 的 food 通路发育成功。
3. 验证用户第 4 条原则("发育需过程/机遇,别把通路未建成当 bug 忽视"):rewardOnly 确实发育出了一半正确通路(food),之前盯着聚合 SR 判"模型坏"忽视了这点。snapshot_analyze 一个工具就推翻了 3 轮聚合诊断的误判。

**真问题定位(待 E4 解决):toxin 回避的 credit assignment。**
- food 逼近:做动作→拿到食物→正 reward→强化通路。直接,通路建到 eff=1.99。
- toxin 回避:做动作→没碰 toxin→**避免**了负 reward。但"避免"不产生正 reward 信号(拿到的是"没扣分",reward=0 或弱)→ credit 弱→通路只建到 eff=0.456,够不到 motor 阈值→noop。
- 这是"避免负信号如何产生等价强化"的结构缺口,与 food 逼近的"正信号直接强化"不对称。用户已有关于"负面信号产生加强刺激"的新想法(待 E4)。

**E3 结论:snapshot_analyze(weights+path 双模式)固化。下一步从"加 stable 去固化开关"转向"解决 toxin 回避 credit assignment(避免负 reward 的强化信号)",等用户新想法。**

### E4. STDP/BAP 长程复测 — stem cliff 未复发,但 rewardOnly 长程学习退化(2026-06-30)

**长程读数(隔壁 agent 复测,8seed×300ep):**
- 当前 `feat/stdp-bap-baseline`(STDP/BAP 大改后):epoch300 final mean SR=**0.344**;`SR>=0.99` solved=**0/8**;`0.4<SR<0.99` partial=**3/8**;`noop>=0.8` stuck=**2/8**。分类不是完备分桶,其余 seed 属低成功但非 noop-stuck 区间。
- 对照 master(decayProtected 修复后,24seed×300ep):epoch300 mean SR=**0.948**;solved=**19/24**;stuck=**0/24**。
- 曲线签名:没有旧版 ~200ep 后 sensory→inter stem 跌破阈值导致的 0/24 全死悬崖;但也没有继续爬升,200ep 后 conflict 升到约 **0.46**,表现为 partial/乱动而非 solved。

**解释收紧:**
1. **decayProtected 修复仍有效。** 长程没有复发"stem 被 stableDecay 侵蚀→inter 停发→全链路静默"的硬悬崖;structural stem 的 stableSum 仍能保持/增长。
2. **STDP/BAP 当前版本是长程净退化。** 相比 master 修复后的 19/24 solved、SR 0.948,本分支只有 0/8 solved、SR 0.344;不能把"悬崖未复发"误读成"长程行为改进"。
3. **退化与 E3 single-stim 证据一致。** food 通路能部分建成,但 toxin 回避通路 eff 约 0.45,够不到 motor 阈值 1.0;长程下不是全局 noop,而是 toxin credit assignment 弱导致 conflict/noop 混合。
4. **后续优先级不变:** 先解决"避免负 reward 如何产生足够强化"的 toxin 回避 credit assignment;STDP/BAP 语义链可继续保留为实验分支,但在该问题解决前不能替代 master(decayProtected)作为行为基线。

### E5. 激素代理实验前边界清理 — 参数/机制/runner 拆薄(2026-06-30)

在实现 `aversive neuromodulatory tag` 之前先做行为保持清理,避免把方法、参数、机制计算和 audit transport 混在一起:

- 新增 `src/core/plasticityMechanisms.ts`:只放纯计算函数,包括 activity trace、BAP-weighted STDP eligibility delta、eligibility normalization scale、reward modulator、reward/supervised fast delta、stable depotentiation delta、stable capture amount。
- `src/core/plasticity.ts` 保留突触遍历、gate 检查、状态落账、event 记录;不再内联 STDP/modulator/capture/depotentiation 公式。
- `src/core/mechanism.ts` 保留网络级编排;`computeModulator` 仅作为兼容 re-export,公式来源在 `plasticityMechanisms.ts`。
- `src/config/newModelConfig.ts` 删除宽泛 `learningDynamics` 分组,改为一次性主分类:`thresholds`、`neuralDynamics`、`weightBounds`、`plasticityTimescales`、`plasticityRates`、`plasticityMechanisms`、`signalModulation`、`exploration`、`structural`、`experimentDefaults`。新增测试要求 `defaultConfig` 每个字段恰好属于一个主分类。
- `transferMatrix` runner 修正:子进程即使 `requiredPassed=false` 也会输出 JSON;父进程先解析 stdout,再让 matrix summary 判 gate。这样 transport/IO 不再吞掉失败报告,不会把 rewardOnly gate 翻红伪装成 `stderr empty` 的 worker crash。

**验证:**
- `npm run build` 通过。
- `npm test` 20/20 通过(新增 config 分类测试 + plasticity mechanism 纯公式测试)。
- `npm run audit:transfer:matrix` 现在真实显示当前 STDP/BAP 分支行为红项:`requiredPassed=false`;15 cell 中 loader/supervised/seed/conflict/blank 全过,但 rewardOnly separation 仅 **6/15** 过,failed cells 为 102/103/105 相关 9 格;rew meanReward delta min=0.000 mean=0.220 max=0.550。这是机制退化事实,不是 runner 崩溃。

**E 档不做(本分支 out of scope):** ~~物理时间轴(tick→ms)、传导延迟队列~~(E2 后降级为可选)、严格 τ₊/τ₋ 生物数值对齐、reward→stable 去固化开关(E3 后降级:wrong-prior 仍需,但正常 rewardOnly 的主问题是 toxin credit assignment,优先级让位)。

### E6. aversive tag / "激素代理" 实验 — 默认关闭实现 + 长程 spot(2026-06-30)

**实现边界:**
- 新增 `aversiveTagStrategy: "off" | "modulatorOnly" | "avoidanceMarker" | "badOutcomeDepotentiation" | "combined"`。默认 `off`,所以基线行为只多记录 tag/信号,不改变学习。
- `challengeTask` 只负责从任务事实派生 aversive tag:可见 toxin、noop/接近/冲突/碰撞=badOutcome,远离/避开=goodAvoidance。它不 import synapse/plasticity/export/audit。
- `plasticityMechanisms` 只放纯计算:`computeAversiveRewardSignal`、`computeAversiveModulator`、`computeAversiveStableDepotentiationDelta`。
- `plasticity` 只负责落账:`applyAversiveStableDepotentiation` 在 badOutcome 时按 eligibility/gate 削 motor 入突触 stableWeight。
- `mechanism` 负责 rewardOnly 编排:`rewardAdvantage -> rewardSignal -> modulator -> rewardLearning + optional aversive stable depotentiation`。
- trace 增加 `rewardSignal` 和 `aversiveTag`;长程脚本支持 `AVERSIVE_*` 环境变量;新增 `audit:rewardonly:natural-stim` 读取长程 JSON,输出 food/toxin 单刺激正确/错误/冲突/noop 权重画像。

**默认 off 验证:**
- `npm run build` 通过;`npm test` **21/21** 通过。
- `audit:2d-challenge` requiredPassed=true;当前 STDP/BAP 分支 rewardOnly 仍 SR=**0.25**,noop=**0.947**,conflict=0。
- `audit:2d-complex` requiredPassed=true;rewardOnly diagnostic Family A 仍 SR=0/noop=1。
- `audit:transfer:matrix` 仍 requiredPassed=false,但这是机制事实:loader/supervised/seed/conflict/blank 全 15/15,**rewardOnly separation 6/15**。runner transport 已不再伪装成 worker crash。

**术语收紧:**
- 建议把用户说的"长程 toxin 自然刺激回复曲线"在代码/报告里写成 **natural single-stim response** 或 **toxin single-stim response**。意思是"不人工指定正确 motor,只把一个真实 task observation 喂给已训练网络并冻结评估",比"自然刺激"更不容易误解成生物实验。

**当前 STDP/BAP 默认长程画像(`/tmp/lr_stdp_bap`,8seed×300ep):**
- epoch300 eval mean SR=**0.344**,noop=0.200,conflict=**0.460**。
- 单刺激全量:cleanCorrect=**11/32**,correctEff>=1=30/32,wrongEff>=1=**19/32**,conflict=19/32,noop=2/32。
- food-left clean=2/8,food-right clean=0/8;toxin-left clean=4/8,toxin-right clean=5/8。
- 解释:不是 stem cliff 复发,也不是纯 toxin 不发;而是 correct 和 wrong 权重大量同时过阈,后期变成冲突型"星空式"权重场。正确通路常常已建成,但错误侧也被点亮,无法稳定收敛成单一主干。

**4 seed spot 结果(300ep):**
- `modulatorOnly,gain=0.5`:mean SR=**0.125**,noop=0,conflict=**0.875**;cleanCorrect=2/16。全局调制放大了"都增强",净负。
- `badOutcomeDepotentiation,rate=0.04`:mean SR=**0.250**,noop=0.631,conflict=0.211;toxin wrong 被压住,但 toxin 通路仍弱/noop,food wrong 后期冲突上升。单独削 stable 不足。
- `avoidanceMarker,bonus=0.5`:4seed epoch200 SR=**1.000**,epoch300 SR=**0.813**;cleanCorrect 200ep=16/16,300ep=13/16。
- `combined,bonus=0.5,rate=0.04`:epoch300 SR=**0.813**,与 avoidanceMarker 接近;没有看到 depotentiation 明确增益。

**8 seed 复核(best spot: `avoidanceMarker,bonus=0.5`):**
- 长程:epoch150 SR=0.938,epoch200 SR=**1.000**,epoch250 SR=0.938,epoch300 SR=**0.844**;300ep solved=4/8,partial=4/8,stuck=0/8。
- 单刺激:epoch200 cleanCorrect=**32/32**,wrongEff>=1=0/32;epoch300 cleanCorrect=**27/32**,wrongEff>=1=4/32,conflict=4/32,noop=1/32。
- 后期退化主要来自 food-left/food-right 的错误侧权重上升(food-right 300ep clean=5/8),不是 toxin 回避继续缺失。toxin-left 8/8 clean,toxin-right 7/8 clean。

**结论(只按当前 spot 级别):**
1. 用户提出的"负向感觉神经元触发的带 reward 标记冲动"方向有实证信号。`avoidanceMarker` 不使用监督式 correct motor 标签,只把"可见 toxin 后成功远离/避免"转换成正向 rewardSignal,能把 8seed 200ep 从默认 0/8 solved 提到 8/8 solved。
2. 当前有效部分更像"避免负结果的正向固化门",不是"坏结果直接去极化"。直接 badOutcome depotentiation 容易削弱或误伤,目前未带来净增益。
3. 300ep 回落说明机制仍不稳:早期能形成正确网络主干,后期仍可能出现错误侧权重补亮。下一步应先研究 over-training/capture/food wrong-side 抑制,再决定是否引入更精细的 aversive depotentiation。
4. 现阶段不能升为行为基线。需 24seed×300ep、bonus/rate sweep、transfer matrix gate,以及多物体 complex 复核后再定。

### E7. 神经元基数/长步数/并发/权重图谱复测(2026-06-30)

**实验基础设施:**
- `topologyBlueprint` 新增 `createScaledOfflineLearningTopologyBlueprint({ interneuronCopiesPerSensor, normalizeReadoutByCopies })`。默认 `offlineLearningTopologyBlueprint` 仍是 1x,现有模型/测试不变。
- scale 只通过长程实验的 `initialNetwork` 打开;canonical sensory/motor 名称保持不变,每个 sensory 对应的 interneuron 干线复制为 `iFoodLeft_copy2` 等。
- `TOPOLOGY_READOUT_MODE=normalized` 默认把每条 inter→motor 初始 fastWeight 设为 `0.35/scale`,保证同一 sensory 通道的初始总 readout 仍是 0.35;`raw` 保留为放大输入强度的对照模式。
- `scripts/longrange_sweep.cjs` 新增环境参数:`EPOCHS`,`MAX_STEPS`,`SEEDS`,`SEED_LIMIT`,`CHECKPOINTS`,`CONCURRENCY`,`TOPOLOGY_SCALE`,`TOPOLOGY_READOUT_MODE`。
- `audit:rewardonly:natural-stim` 已改为按同一 sensory 通道汇总复制 inter 的 eff,不再把每条复制突触单独当成 motor 阈值判定。
- 新增 `audit:rewardonly:weight-map` 和 `audit:rewardonly:checkpoint-probe`:
  - `weight-map` 读取长程 checkpoint,输出 grouped/full sensory→inter 和 inter→motor 权重图谱。
  - `checkpoint-probe` 从 checkpoint 重建真实网络,运行 `runChallengeEpisode` 单点 food/toxin left/right,不是只靠权重阈值推断。

**验证:**
- `npm run build` 通过。
- `npm test` **22/22** 通过;新增 scale topology 测试锁住 1x 默认不变和 scale=2 normalized 初始总 readout=0.35。
- smoke:`SUBDIR=lr_scale2_smoke SEED_LIMIT=2 EPOCHS=20 CHECKPOINTS=1,5,10,20 TOPOLOGY_SCALE=2 ...` 可落盘并被 natural-stim / weight-map / checkpoint-probe 读回。20ep 仍全 noop,只作为管线验证。

**扩容对照结果(400ep,normalized,8 并发):**

1. `scale=2,aversive=off,8seed×400ep`
   - 命令:`SUBDIR=lr_scale2_off8_e400 SEED_LIMIT=8 EPOCHS=400 TOPOLOGY_SCALE=2 TOPOLOGY_READOUT_MODE=normalized CONCURRENCY=8 npm run audit:rewardonly:longrange`
   - epoch400 mean SR=**0.500**,noop=**0.857**,conflict=0,solved=0/8,stuck=8/8。
   - 单点/权重图:food-left/right 8/8 clean;toxin-left/right 0/8 clean、8/8 noop。结论:神经元复制本身不足,只稳定形成 food 主干,toxin 回避仍无正向固化。

2. `scale=2,avoidanceMarker bonus=0.5,8seed×400ep`
   - 命令:`SUBDIR=lr_scale2_avoid8_e400 SEED_LIMIT=8 EPOCHS=400 TOPOLOGY_SCALE=2 AVERSIVE_TAG_STRATEGY=avoidanceMarker AVERSIVE_AVOIDANCE_BONUS=0.5 ...`
   - epoch250 SR=**0.969**,epoch300 SR=0.969,epoch400 SR=**0.969**;epoch400 solved=7/8,stuck=0/8,noop=0,conflict=0.021。
   - 单点 epoch400:food-left 8/8 clean,food-right 8/8 clean,toxin-left 8/8 clean,toxin-right 7/8 clean + 1/8 conflict。
   - 结论:scale=2 + avoidanceMarker 能形成接近完整的正确主干;残余失败集中在个别 toxin wrong-side stable 也过阈。

3. `scale=4,avoidanceMarker bonus=0.5,8seed×400ep`
   - 命令:`SUBDIR=lr_scale4_avoid8_e400 SEED_LIMIT=8 EPOCHS=400 TOPOLOGY_SCALE=4 ...`
   - epoch400 mean SR=**0.563**,noop=**0.819**,solved=0/8,stuck=6/8。
   - 单点 epoch400:food-left/right 8/8 clean;toxin-left 1/8 clean,toxin-right 1/8 clean。
   - 结论:基数不是单调越大越好。normalized 后每条复制 readout 太薄或学习被分散,toxin 通路多数达不到 motor 阈值。

4. `scale=2,avoidanceMarker bonus=0.5,24seed×400ep`
   - 命令:`SUBDIR=lr_scale2_avoid24_e400 EPOCHS=400 CHECKPOINTS=100,150,200,250,300,350,400 TOPOLOGY_SCALE=2 TOPOLOGY_READOUT_MODE=normalized CONCURRENCY=8 AVERSIVE_TAG_STRATEGY=avoidanceMarker AVERSIVE_AVOIDANCE_BONUS=0.5 npm run audit:rewardonly:longrange`
   - 长程:epoch200 SR=0.844,epoch250 SR=0.948,epoch300 SR=0.969,epoch350 SR=**0.979**,epoch400 SR=**0.969**。
   - epoch400 分布:solved=**21/24**,partial=3/24,stuck=0/24,noop=0,conflict=0.021。
   - 单点 epoch400:
     - food-left **24/24** clean,meanCorrect=1.990,meanWrong=0.028。
     - food-right **24/24** clean,meanCorrect=2.142,meanWrong=0.045。
     - toxin-left **23/24** clean,1/24 conflict,meanCorrect=1.462,meanWrong=0.289。
     - toxin-right **22/24** clean,2/24 conflict,meanCorrect=1.504,meanWrong=0.472。
   - 失败 seed:81/151 为 toxin-right conflict,241 为 toxin-left conflict。full weight map 显示 sensory→inter stems 全部保持 eff=1.1;失败来自 toxin wrong motor 的 stableWeight 也被固化到阈值以上,不是 stem 断裂、也不是 formatter 误判。

**解释收紧:**
1. "模型发育需要神经基数/机遇/长期过程"这个怀疑有必要。1x/300ep spot 会低估 scale=2 + avoidanceMarker 的潜力;24seed×400ep 显示它不是偶然单 seed。
2. 这轮 `TOPOLOGY_SCALE` 不是完整的"等比放大神经基数"实验。它只复制每个 sensory 通道的 interneuron 副本,固定 sensory/motor 数、motor 阈值、capture/use/stable 阈值和学习率;因此 scale=4 退化**不能**证明"增大神经基数无效"。
3. scale=4 normalized 明显退化,只能说明当前这套"只扩中间层 + 每条 readout 除以 scale + 固定单节点阈值/突触规则"不具备规模不变性。该结果是对复杂模拟的警告:扩节点时必须同时设计 slot/局部连接/阈值/归一化/竞争规则,不能只改一个维度。
4. 当前最好解释:avoidanceMarker 提供了 toxin 回避的正向固化信号;scale=2 这类有限冗余提供更多形成机会;两者叠加能解决多数 toxin noop。但残余 conflict 仍是"错误 stable 也被固化"问题,需要后续研究选择性去固化或更局部的 aversive 标记。
5. 不能把短程 audit 当作实验价值否定。短程测试只保证通路/回归,长程发育要看 checkpoint 曲线、权重图谱和真实单点激发。

**与用户预期模型的差距(需下一轮补齐):**
- 用户预期更接近 `2 input / ~10 medium / 2 output`,每个节点 1-5 条局部突触(至少多数节点 >2),按最近/局部连接机制形成通路。
- 当前 challenge blueprint 是语义固定 `4 sensory / 4 inter / 2 motor`,scale 只复制 inter,并不是 `2/10/2` 或其最小公约数族。
- 下一轮应新增真正的 topology family:输入/中间/输出层按比例扩展,节点 slot 数、局部半径、fan-in/fan-out、阈值归一化一起显式参数化,再重新比较 1x/2x/4x。

### E8. topology family 测试集补齐 — 比例族/fanout-n/最近连接(2026-06-30)

**语义修正:**
- `2/10/2` 不应被理解成"只把中间层放大到 10"。它是 `1/5/1` 的 2x 共同倍数族。
- `1/10/2`、`2/5/2` 不是 `1/5/1` 的等比放大,而是不同最小比例族/特殊不均衡模拟。它们仍然是合法 topology 实验,只是报告时不能把它们和 `2/10/2` 混进同一个 scale 结论里。
- 测试集必须允许任意合理的 `input / medium / output / synapses n` 组合;只有当我们声称"等比放大"时,才检查 `sameLayerRatio=true`。
- 用户口语里的"最小公倍数/最小公约数模型"在代码里拆成两件事:
  - `reduceLayerCounts`:用最大公约数把 `2/10/2` 归约成 `1/5/1`,并记录 `commonScale=2`。
  - `sameLayerRatio`:判断两个层数是否属于同一比例族。

**已补测试边界:**
- 新增 `src/core/layeredTopologyBlueprint.ts`:
  - `reduceLayerCounts({2,10,2}) -> {1,5,1, commonScale:2}`。
  - `sameLayerRatio(1/5/1, 2/10/2)=true`。
  - `sameLayerRatio(1/5/1, 1/10/2)=false`。
  - `sameLayerRatio(1/5/1, 2/5/2)=false`。
- 新增 `createNearestLayeredTopologyBlueprint`:
  - 参数是 `inputCount / mediumCount / outputCount / synapsesPerInput / synapsesPerMedium`。
  - fanout n 限制在 1-5。
  - input→medium 和 medium→output 都按 y 轴最近节点连接。
  - slot 数由实际边自动写入,避免默认 slot 上限误伤 scale 实验。
- 新测试固定 `2/10/2, synapsesPerInput=5, synapsesPerMedium=2`:
  - 2 input、10 medium、2 output。
  - input0 最近连接 medium0-4,input1 最近连接 medium5-9。
  - 每个 medium 有 1 条输入、2 条输出;每个 output 有 10 条输入。
  - 可以实例化为真实 `LearningNetwork`。

**仍未完成:**
- 该 topology family 目前只进入结构测试和 standalone rewardOnly probe,还没有接入 challenge/world 训练 runner。
- 下一步需要做的是:把 `layeredTopologyBlueprint` 接入更正式的 topology-family experiment,再跑 `1/5/1`、`2/10/2`、`4/20/4` 等比例族,并与 `1/10/2`、`2/5/2` 这种不均衡族分开报告。

**第一批 standalone topology-family probe(8seed×200ep):**
- 新增 `scripts/topology_family_probe.cjs` / `npm run audit:topology-family`。它不使用 2D challenge 语义,只跑任意 layered topology 的 input→output rewardOnly 映射,并输出旧指标族:SR、noop、conflict、wrong-only、correctEff、wrongEff、single-stim signature。
- 命令:`SEED_LIMIT=8 EPOCHS=200 CHECKPOINTS=20,50,100,150,200 npm run audit:topology-family`。
- `ratio_1_5_1_fan5x1`:SR=**1.000**,noop=0,conflict=0,solved=8/8,correctEff=1.702。
- `ratio_2_10_2_fan5x1`:SR=**1.000**,noop=0,conflict=0,solved=8/8,correctEff=1.685。该结果说明 `2/10/2` 作为 `1/5/1` 的 2x 比例族,在最近单输出 readout 下没有出现仅由等比放大导致的退化。
- `ratio_2_10_2_fan5x2`:SR=**0.000**,conflict=**1.000**,correctEff=1.847,wrongEff=1.732。FULL signature 全部 `XX`:每个 input 的正确和错误 output 都过阈。这是 fanout=2/full readout 带来的结构性 conflict,不是等比基数本身失效。
- `arb_1_10_2_fan5x1`:SR=**1.000**,noop=0,conflict=0,correctEff=1.021,wrongEff=0.633。它不是 `1/5/1` 等比族,但作为任意 topology 是可跑且可解的。
- `arb_2_5_2_fan3x1`:SR=**0.500**,noop=**0.500**,conflict=0,correctEff=0.884。FULL signature 全部 `CN`:input0 过阈,input1 正确 eff 长期低于 1.0。初步判断是最近连接覆盖/中间层容量不足/阈值组合导致的一侧通路弱化。

**当前解释:**
1. 新测试集已经能把三类问题分开:等比比例族、任意不均衡 topology、fanout 造成的结构性 conflict。
2. `fanout n>=2` 不是自动更好;如果 medium 同时连到所有 output,且没有局部竞争/抑制/选择性去固化,就会出现 correct 和 wrong output 同时过阈。
3. `2/5/2` 的 noop 表明"少 medium + 最近连接 + 固定阈值"会让某些 input 覆盖不足;这更接近用户担心的单节点输入/输出固定问题,应继续作为 topology-family 指标。

### E9. tagged-impulse 去固化机制 — 替换 A/B 负值通道 + graded 浓度梯度重跑(2026-07-01)

**背景**:E6 的 aversive tag 有两条去固化通道:通道 A(任务层 `reward=-1` × `computeRewardFastDelta`,作用 fast,符号相关)+ 通道 B(`applyAversiveStableDepotentiation = -rate×|elig|` 反向项,作用 stable,符号无关)。用户要求消 A、用"翻转累积方向"替换 B,落地"激素/带标记冲动"两变体。

**关键更正(numberEnv bug,推翻此前两轮结论)**:首轮 2×2 用 `numberEnv`(要求 `>0`)读 `TOXIN_REWARD`,`-1` 被静默拒成回退值,"reward=0"格实际全是 `-1`。修 `signedNumberEnv` 后真 2×2 证实:**A 不是 no-op,是符号病源头**(`reward=-1`+tag off → nutrEff 失控到 8.094:负 reward × 负 eligibility → spurious LTP 营养侧);`reward=0`+tag off 是对称 conflict(1.581/1.581),无 runaway → "unchecked capture 单向累积到满权"理论不成立。

**新机制(已落地,4 commit c6a1302→df36f97)**:
- tag 搭便车传播:毒物感觉神经元 `tagLoad=1` → `propagateSynapses` 标记突触+累积 post tag → `integrateNeuron` 放电时向前带,**不改前向 effect**。
- 翻转累积方向(替换 B):tag 命中固化 readout → `captureStableWeights` 翻转 `stable -= amount; fast += amount`(回流=de-consolidation),仅 motor-post,保护 sensory 干线。
- 两变体:`taggedImpulse`(tag 命中即翻转)、`specificFactor`(全局 aversive load ∧ tag 与门,load 有衰减=激素窗口)。
- 消 A:探针默认 `TOXIN_REWARD=0`;删 B:`applyAversiveStableDepotentiation` 等三函数删除。
- `npm test` 38/38 green。

**二元探针 3 模式矩阵(av_4_10_2_prewired, 8seed 300ep)**:off→conflict 1.0 对称(证 A 消 B 删无残留);taggedImpulse/specificFactor→8/8 趋营养(toxinEff 0.087 ≪ nutrEff 1.282);specificFactor 门关(GLOBAL_INCREMENT=0)→conflict(证与门)。

**stem/empty 单点+全量权重分析(e07c085,用户点 2/5,细化"noop"结论)**:
- prewired off:toxin 0.320 + nutr 0.313 双侧固化 → conflict。prewired taggedImpulse:toxin 削到 0.017(flip),nutr 0.313 存活 → 8/8 趋营养。**flip 是 prewired 里削 toxin 的 active 机制**。
- **stem off ≡ stem taggedImpulse 逐位相同**:全 readout eff=0.006 stable=0 → noop 塌缩。**flip 在 stem inert**(门要 stable>0,stem readout 从没固化 stable=0)。stem 病灶 = 既有"双侧形成 + 无正信号固化"鸡生蛋,**与 tagged 机制无关**。
- stem epoch20 曾短暂 right=1.0(nutrEff 1.002)再塌缩——**真实瞬态正确通路**(原则 4:别把未持续住的通路当 bug)。
- stem 双侧形成:每个 medium→output0 AND output1,tag 搭便车到两侧,但因 stable=0 flip 不触发,inert。

**graded 浓度梯度重跑(060f0f4,用户点 1/1.1/1.2,自然终点测试)**:
- 新探针 `graded_valence_probe.cjs`:多步 1D(toxin@0,nutrient@L,中心起步),浓度群体编码 k=clamp(round(n/(1+dist)),0,n) 通道发放(单模拟值<1 打不动 inter 阈值 1.0,故强度=发放通道数=招募 inter 数=motor drive 强度),只标正在发放的 toxin 通道带 tag。
- 结果(N=5, line 0..10, 300ep×8seed):
  - off:8/8 到 toxin(x=0,5步)。toxinEff 2.72(固化)。无 erosion → 被 toxin 吸过去。
  - taggedImpulse:epoch20 8/8 toxin → epoch50 6/8 toxin+2/8 stuck → epoch100+ **8/8 stuck 中心**(x=5,12步 noop)。**toxinEff 削 2.72→0.51(flip 生效,避毒成功:0 toxin 接触)**。nutrEff 1.77 ≈ 5×0.35 初始(营养侧没学到东西)。
  - specificFactor:与 taggedImpulse 相同(逼近 toxin 时 load 足够开门)。
- **发现(回答 1.2)**:梯度下机制**不是"为防御而防御"**——真削了 toxin(2.72→0.51)、真避毒(8/8 toxin→0)。但**approach 表达不出来**:中心 x=5 处 kN=round(5/6)=1,只 1 个 nutrient inter 发放,贡献 ~0.35 ≪ motor 阈值 1.0 → 营养 motor 不发 → 卡中心。二元探针的 8/8 趋营养依赖中心全通道强驱动(k=n);梯度暴露了**单靠 avoidance 表达不出 approach**——缺(a)正营养信用(reward=0、A 消 → nutrient readout 停在初始 0.35 不长)和(b)中心感知够强。与 1.3 一致(负 tag 给避毒以意义;正向 approach 需自己的信号),也是 deferred 的 candidate/credit 鸡生蛋。

**群体聚焦假说验证 + 放大实验(f008e19,用户:ANALYZE 单点 / 放大模型 / 不均衡放大感受器)**:
- 单点刺激扫描(每位置 x 只激活该距离的某侧通道,测放电 inter 数 / 各自 eff / 实际驱动 driveSum vs 总 nutrEff):
  - N=5 baseline 中心 x=5:kN=1 → 5 inter 放电,有效 3 个各 0.24 → drive **0.725 < 阈值 1.0** → stuck。总 nutrEff 1.17 散布在距离递减的 9 个 inter(0.24/0.09/0.06);motor 只在 x≥9(已到 nutrient)放电。**鸡生蛋是阈值锁**:权重存在但决策位置不够阈值,要凑够得先靠近目标。nutrEff 是衰减的初始值(无 capture:reward=0、A 消 → 无正信用沉淀)。
- 放大矩阵(300ep×8seed,taggedImpulse):
  - **scale10(N_toxin=N_nutrient=10):8/8 APPROACH** ✅。kN(中心)=2 → drive 1.7 → motor 发 → 右移 → 营养通路激活 → **use-dependent capture 自强化(nutrEff 1.17→3.57,无奖励,capture 是使用门控非奖励门控)**。toxinEff 被 tag 削到 0.29。**确认阈值假说:过阈值一次,Hebbian capture 接管,不需奖励信号**。Caveat:nutrient motor 在每个 x 都发(x=0 也 1.135>1),故 approach 是 valence 区分但非梯度敏感(二元 motor 无法表达梯度强度)。
  - asym5_10 / asym5_15(营养放大):8/8 **TOXIN**(反噬)。不等通道数破坏拓扑/连接平衡 → toxin readout 固化(2.64/3.81,tag 削不过 capture)→ nutrient 塌到 ~0。**放大必须对称**。
- **结论**:approach 缺口的根因是**阈值锁 + 无正信用**,非"无权重"。对称放大过阈值即用 use-dependent 自强化打破鸡生蛋——这把 deferred 自发放电的动机从"破 candidate 鸡生蛋"细化到"提供初始激活让 use-dependent capture 启动正向通路"(放大模型=外生激活的一种;自发放电=内生激活的等价目标)。

**点 8(aversiveTag 死形参)核实**:`applyRewardOutcomeLearning` 的 `aversiveTag` 形参**不是死的**——`computeAversiveRewardSignal`(avoidanceMarker/combined 加 bonus)、`computeAversiveModulator`(modulatorOnly 增益)都用它,是 E6/E7 长程 8/8 的依赖。死的只是探针内部(strategy="off" 下构造的 aversiveTag 被短路),已清(e07c085)。不删核心形参。

### E10. natural n/n/n/k 形态基线 — 固定 slot + 自然最近连接 + 被动断开(2026-07-01)

**用户目标澄清:**
- 这不是 `graded_valence_probe` 的 10/25/2 浓度梯度模型,也不是 `topology_family_probe` 的预接线比例族。
- 理想形态基线是 `n/n/n/k`:n 个感受、n 个中间、n 个效应;所有神经元都有固定 k 个输入槽和 k 个输出槽;二维位置固定;连接自然形成;优先最近连接;弱/未用连接要能被动断开。

**已落地边界:**
- 新增 `createUniformNaturalLayeredTopologyBlueprint({layerSize:n, slotsPerNeuron:k})`:
  - 只创建三层节点,不预铺 stem/readout。
  - 所有节点都声明 `maxInputSlots=k` 且 `maxOutputSlots=k`。注:感受层输入槽/效应层输出槽目前是形态占位,受 `roleCompatible` 限制不会被使用。
- 新增 `tryFormNearestLayeredConnections`:
  - 只允许相邻层 `sensory→interneuron`、`interneuron→motor`,避免通用 development 里的 `inter→inter` 抢占 slot。
  - 按二维距离最近优先形成,仍遵守 slot、threshold、pairMemory/tombstone。
- 新增 `scripts/natural_nnk_probe.cjs` / `npm run audit:natural-nnk`:
  - 每 epoch:nearest 自然形成 → rewardOnly 映射训练 → `updateConnectionStates` 被动状态更新。
  - 报告 SR/noop/conflict/wrong、stem/readout 数、candidate/active/stable/dormant/pruned、累计 formed/pruned。
- 新增回归测试:固定 slot scaffold、nearest adjacent-layer formation、弱未用非 stable 连接 dormant→pruned。`npm test` 当前 41/41。

**首轮 spot-check(4seed×100ep):**
- 默认 `n=5,k=2,fastInit=0.12,axonThreshold=1`:
  - epoch1 已自然形成约 18 条边(stem/readout 各约 9),但 SR=0/noop=1。
  - 到 epoch50/100:stem=0,readout=0,pruned=50,formed=50。**边不是没长,而是初始权重打不过阈值 → recentUse=0 → 被动断开全部剪掉。**
- 外生过阈 `FAST_INIT=1.05,n=5,k=2`:
  - 连接保住并固化(stable≈17.8),但 final SR≈0.10/conflict≈0.80。
  - 解释:初始激活确实启动了 use-dependent capture,但 k=2 的多输出最近 fanout 在无竞争/抑制下产生结构性 conflict。
- 单通路对照 `FAST_INIT=1.05,K=1`:
  - final SR≈0.80/noop≈0.20/conflict=0,stable≈9,pruned≈1。
  - 解释:nearest 单通路能跑通;残余 noop 来自一条必要 stem 被早期被动剪掉。被动断开存在,但门槛/时间尺度过硬。

**当前解释:**
1. `n/n/n/k` 形态基线已能把三件事拆开:自然连接能否长出、初始激活能否跨阈、被动断开是否过早剪枝。
2. 默认失败再次支持 E9 的"阈值锁 + 无初始激活"结论:不是无权重/无连接,而是连接没有被使用,随后被 passive pruning 删除。
3. `FAST_INIT=1.05` 证明"过阈一次→use-dependent capture 接管"在自然形态基线也成立;但 `k=2` 同时暴露另一个短板:多突触 fanout 缺少竞争/抑制/选择性固化,会把正确与邻近错误 output 一起点亮。
4. 下一步不应直接把 k>1 判死。要测试:更慢 passive prune、自发放电/背景噪声作为内生初始激活、局部竞争或 winner-take-one readout、以及 connection formation 是否需要按目标层配额而非单纯最近贪心。

### E11. natural m/n/o/k 扩展 — k=5 单点刺激 + 权重快照(2026-07-01)

**实现补齐:**
- `createUniformNaturalLayeredTopologyBlueprint` 从 `n/n/n/k` 扩展为 `m/n/o/k`;旧 `layerSize` 用法仍兼容。
- `natural_nnk_probe.cjs` 支持 `M/N/O/K`:
  - 若设置 `M` 或 `O`,则 `N` 表示 medium 数;否则 `N` 仍是旧共同层大小。
  - target output 由 input/output 的二维 y 位置最近映射;`2/10/2` 为 output0/output1,`2/10/5` 为 output0/output4。
- 新增 final 单点刺激/路径权重快照:
  - `C/N/X/W` per input。
  - `stemN/fireN/stemEff/actStem/correctD/wrongD`,区分"有 stem 权重"与"实际 firing medium 是否过阈"。

**16seed×100ep,k=5 汇总:**
- `M=2 N=10 O=2 K=5` 默认 `fastInit=0.12`:
  - final SR=0,noop=1,solved=0/16。stem=10,readout=0,pruned=20。
  - 单点:每 input `stemN=5, stemEff≈0.54, fireN=0`。解释:每条 stem 约 0.108,分散到 5 个 medium,单个 medium 不过 axon 阈值 1.0;readout 被剪空。
- `M=2 N=10 O=5 K=5` 默认 `fastInit=0.12`:
  - final SR=0,noop=1,solved=0/16。stem=10,readout=0,pruned=50。
  - 同样是阈值锁;O 更多只让初始 readout 候选更多,随后全剪。
- `FAST_INIT=1.05, M=2 N=10 O=2 K=5`:
  - final SR=**1.0**,solved=16/16。stem=10,readout=10,stable=20,pruned=0。
  - 单点:每 input `fireN=5, correctD≈5.12, wrongD=0`,签名全 `CC`。说明 m/n/m/k 在输出数匹配且最近连接可分区时,只要初始激活跨阈,use-dependent capture 能形成干净主干。
- `FAST_INIT=1.05, M=2 N=10 O=5 K=5`:
  - final SR=0,conflict=1,solved=0/16。stem=10,readout=25,stable=35。
  - 单点:input0 激活 output0|output1|output2,input1 激活 output2|output3|output4;`correctD≈4.9, wrongD≈4.5`。解释:正确和邻近错误 output 都强固化,是结构性多输出 conflict。
- 补充 `FAST_INIT=1.05, M=2 N=10 O=3 K=5`:
  - final SR=0,conflict=1。`correctD≈4.76, wrongD≈1.77-2.63`。即 O≠M 时,即使 wrong drive 小于 correct,仍过 motor 阈值并造成 conflict;不是 O=5 特例。

**当前解释:**
1. k=5 本身不必然失败;`2/10/2/5 + 过阈`能 16/16 解。
2. 默认失败继续是"每条自然形成 stem 太弱,单个 medium 不过阈,readout 无 use 被剪",不是没有自然成边。
3. `m/n/o/k` 中 O≠M 暴露了目标层空间/输出竞争缺口:nearest 让一个 input 的 5 个 medium 横跨多个 output 近邻,只要都过阈就多 motor 同时发放。下一步需要局部 winner-take-one、输出侧侧抑制、或 formation 时限制每个 input basin 只固化一个目标输出。

### E12. valence-gate midpoint 集成 — 价值门/纯刺激门/激素代理分离(2026-07-01)

**目的:**
- 把 E9 的 tagged-impulse/specificFactor 结论接入一个更小的门控输入结构,区别于 `graded_valence_probe` 的群体通道。
- 输入拆成 4 个门:
  - `nutrientGate`:营养价值门,不带 tag。
  - `toxinGate`:毒物价值门,发放 toxin tag;在 `specificFactor` 下释放 global aversive load(激素代理)。
  - `stimLeft`:左侧纯物质刺激门,默认不带 tag。
  - `stimRight`:右侧纯物质刺激门,默认不带 tag。
- 中点刺激 = 四门同时发放;左 motor=toxin 接触,右 motor=nutrient 接触。`TOXIN_REWARD=0,NUTRIENT_REWARD=0,BASELINE_ALPHA=0`,保持 A 负 reward 通道关闭,只测 tag/capture。

**实现:**
- 新增 `scripts/valence_gate_midpoint_probe.cjs` / `npm run audit:valence-gate-midpoint`。
- 固定 prewired 小拓扑:`4 sensory / 12 medium / 2 motor`,每个 sensory→nearest 5 medium,medium→nearest motor。
- 输出 checkpoint 中点行为(left/right/noop/conflict),以及 final frozen 单点/组合刺激快照:
  - `MIDPOINT`
  - `TOXIN_SIDE = toxinGate + stimLeft`
  - `NUTRIENT_SIDE = nutrientGate + stimRight`
  - 单门:`TOXIN_GATE / NUTR_GATE / STIM_LEFT / STIM_RIGHT`
- 对照开关:`TAG_PURE_TOXIN_STIM=1` 可让 toxin 侧纯刺激门也携带 tag,测试"激素/价值门是否需要绑定纯刺激通路"。

**8seed×300ep 结果:**
- `TAGGED_MODE=off`:
  - 中点全程 conflict。epoch300:leftD≈1.953,rightD≈1.955。
  - 单点:TOXIN_SIDE→left,NUTRIENT_SIDE→right,两个纯刺激门也各自能驱动对应 motor。说明结构通路正常,但中点双侧同时过阈。
- `TAGGED_MODE=taggedImpulse`:
  - epoch200 起中点 8/8 right(nutrient),epoch300:leftD≈0.716,rightD≈1.955。
  - 单点:TOXIN_SIDE/TOXIN_GATE/STIM_LEFT 都变 noop;NUTRIENT_SIDE/NUTR_GATE/STIM_RIGHT 仍 right。
  - 解释:toxinGate 的 tag 已足以削左侧 basin 到阈值下,中点转为营养侧。
- `TAGGED_MODE=specificFactor`:
  - 与 taggedImpulse 同级:epoch200 起中点 8/8 right,epoch300:leftD≈0.716,rightD≈1.955。
  - `GLOBAL_INCREMENT=0` 门控关闭后退回 off:中点 conflict,leftD≈1.953,rightD≈1.955。证明 specificFactor 不是死开关,确实依赖激素代理。
- `TAGGED_MODE=specificFactor TAG_PURE_TOXIN_STIM=1`:
  - 中点同样 8/8 right,但左侧削得更深:leftD≈0.468。
  - 解释:把 toxin 侧纯刺激也纳入 tag 会加强去固化,但不是中点转向营养的必要条件。

**当前解释:**
1. 分离"价值门"和"纯刺激门"后,中点行为测试能复现 E9 的关键机制:off=双侧 conflict;tagged/specificFactor=削 toxin-side,left drive 跌破阈值,中点选择 nutrient side。
2. 这说明 tagged impulse 不要求把所有纯刺激都编码成 toxin/nutrient 群体通道;一个 toxin 价值门 + 激素代理已能调制同侧 basin。
3. 风险仍在:当前是 prewired 小拓扑,不是自然 `m/n/o/k` 发育;而且 `stimLeft` 在 tagged 模式下也被削到 noop,说明左侧纯刺激和 toxinGate 在同一 basin 内共享读出,还没有真正解决"纯刺激保持中性、价值门只调制语义"的绑定问题。下一步应把该门控结构接入自然 formation,再观察是否需要局部绑定/侧抑制。

**语义修正(同日复核):**
- 用户指出 `stimLeft/stimRight` 不应表示左右侧,而应是纯粹"接收到物体/物质刺激"。已把脚本改为:
  - `objectStimA/objectStimB`:纯物体刺激门,无左右语义、无 valence、默认不带 tag。
  - `toxinGate`:毒物价值门,释放 toxin tag;`specificFactor` 下释放 global aversive load(激素代理)。
  - `nutrientGate`:营养门,不释放 tag;营养物体 = `nutrientGate + objectStimA + objectStimB`。
  - 输出也改为 `toxinMotor/nutrientMotor`,不再用 `left/right` 描述语义选择。
- `TAG_OBJECT_STIM_WITH_TOXIN=1` 是误伤压力测试:当毒物出现时,连纯物体刺激门也被 tag 标记。旧 `TAG_PURE_TOXIN_STIM=1` 仅作为兼容别名保留。

**语义修正后 8seed×300ep 复跑:**
- `TAGGED_MODE=off`:
  - 中点全程 conflict;epoch300 `toxinD≈1.953,nutrD≈1.955`。
  - `PURE_OBJECT` 本身也 conflict(`toxinD≈1.30,nutrD≈1.30`),说明纯物体刺激没有价值方向,只是结构上同时触发两个 basin。
- `TAGGED_MODE=specificFactor`:
  - epoch200 起中点 8/8 选择 nutrient;epoch300 `toxinD≈0.716,nutrD≈1.955`。
  - `GLOBAL_INCREMENT=0` 后退回 off 的 conflict,证明激素代理仍是必要门控。
  - 单点显示 `TOXIN_GATE→noop`,但 `NUTR_GATE→nutrient`;`PURE_OBJECT→nutrient` 是因为 toxin basin 被削到阈值下,不是纯刺激自己携带营养标记。
- `TAGGED_MODE=specificFactor TAG_OBJECT_STIM_WITH_TOXIN=1`:
  - epoch200 短暂 nutrient,epoch300 变 noop;`toxinD≈0.468,nutrD≈0.964`,两侧都低于阈值。
  - 这正是"误伤"风险:把共享纯物体刺激也纳入毒物 tag 会长期削弱 shared basin,最后连营养侧也无法可靠发放。

**修正后解释:**
1. `objectStimA/objectStimB` 现在确认为纯"有物体刺激"门,不分左右;毒物差异来自 `toxinGate` 的 tag/激素代理,营养侧保持无 tag。
2. 不给纯刺激门打 tag 时,`toxinGate` 已足以把 toxin basin 削到阈值下,中点自判断转向 nutrient。
3. 给纯刺激门也打 tag 会过度去固化,复现用户预判的误伤问题。下一步如果接入自然 `m/n/o/k` 发育,应优先测试"tag 只标价值门"与"tag 扩散到共享纯刺激"的长期差异。

### E13. natural m/n/o/k × L 等比基数放大(2026-07-01)

**实现补齐:**
- `natural_nnk_probe.cjs` 新增 `L`/`SCALE` 倍率参数。
- 语义:`base m/n/o/k × L => actual (mL)/(nL)/(oL)/k`;只放大神经元层数,不改变每节点 slot 数 `k`。

**16seed×100ep,L=2,k=5:**
- `M=2 N=10 O=2 K=5 L=2` → actual `4/20/4/5`,默认 `fastInit=0.12`:
  - final SR=0,noop=1,solved=0/16。stem=20,readout=0,pruned≈80.9。
  - 单点:每 input `stemN=5,fireN=0,stemEff≈0.49`。结论仍是阈值锁;等比放大节点数不增加单 input 的 per-medium drive。
- `M=2 N=10 O=5 K=5 L=2` → actual `4/20/10/5`,默认 `fastInit=0.12`:
  - final SR=0,noop=1,solved=0/16。stem=20,readout=0,pruned≈200.9。
  - 同上,只增加待剪 readout 候选数,不解决初始激活。
- `FAST_INIT=1.05, M=2 N=10 O=2 K=5 L=2` → actual `4/20/4/5`:
  - final SR≈**0.359**,conflict≈**0.641**,solved=0/16。stem=20,readout=20,stable=40。
  - FULL 4seed:典型签名 `XCXX/XXCX/XCXC`;邻近 wrong drive≈0.97-1.32,刚好过 motor 阈值,形成 `output0|output1`、`output2|output3` 邻近冲突。
  - 与 L=1 的 `2/10/2/5 + FAST_INIT=1.05 => SR=1.0` 对比,说明等比放大后 output 间距变密,nearest readout 的相邻输出泄漏变成冲突。
- `FAST_INIT=1.05, M=2 N=10 O=5 K=5 L=2` → actual `4/20/10/5`:
  - final SR=0,conflict=1.0。stem=20,readout=50,stable=70,pruned≈4。
  - 单点:correctD≈4.95,wrongD≈3.79-4.82,邻近错误输出强过阈。

**当前解释:**
1. 单纯等比放大神经元基数不能解决默认阈值锁,因为每个 input 仍只连 k=5 个 medium,单条 stem 初始/衰减后的 per-medium drive 不变甚至略低。
2. 在外生过阈条件下,L=2 反而暴露输出空间分辨率问题:输出层更密,同一个 input basin 的 5 个 medium 会跨到相邻 output,wrong drive 只要略过 1.0 就变 conflict。
3. 后续如果继续 scale,不能只做 `m/n/o × L`。要同步研究:输出侧侧抑制/winner-take-one、每 input basin 的 readout 配额、或者随 L 调整 `k/threshold/距离核`,否则大基数会把"邻近输出泄漏"放大成稳定冲突。

**epoch300 曲线 + 单点/权重复跑(16seed,checkpoint=1/20/50/100/150/200/250/300):**
- `4/20/4/5` 默认 `fastInit=0.12`:
  - SR 全程 0,noop 全程 1。epoch20 readout 已清空;epoch250 后累计 prune 再增一轮。final:stem=20,readout=0,pruned≈160.9。
  - 单点:每 input `stemN=5,fireN=0,stemEff≈0.33`;readout weight totals 全 output=0。结论:长跑只让 stem 继续衰减/候选继续被剪,不会自发解锁。
- `4/20/10/5` 默认:
  - 同样 SR=0/noop=1;final readout=0,pruned≈400.9。O 更多只放大被剪 readout 数。
- `FAST_INIT=1.05,4/20/4/5`:
  - 曲线:epoch1 SR=0/conflict=1 → epoch20 SR=0.344 → epoch50 SR=0.422 峰值 → epoch100 SR=0.359 → epoch150 0.172 → epoch200 0.109 → epoch250 0.031 → epoch300 **SR=0/conflict=1**。
  - final 单点:每 input `fireN=5,correctD≈5.44-5.52,wrongD≈1.24-1.31`,所有 input 都 conflict。
  - FULL 4seed drive map:input0 总是 `output0≈5.1-5.7 + output1≈1.2-1.5`,input1 是 `output1≈5.3-5.8 + output0≈1.25-1.40`,input2/3 同理在 `output2/output3` 邻近冲突。readout totals 每 output live=5,stable≈5.3-5.5。
  - 解释:早期有一段 partial window,但 Hebbian capture 继续把邻近 wrong output 固化到阈值以上,最终所有 basin 变双输出。
- `FAST_INIT=1.05,4/20/10/5`:
  - 曲线从 epoch1 到 epoch300 都是 conflict=1。final:readout=50,stable=70,pruned≈4。
  - 单点:correctD≈5.17-5.26,wrongD≈3.76-4.95。FULL drive map 显示 input0 可同时点亮 output0/1/2/3,input1 点亮 output0..5,input2 点亮 output4..9,input3 点亮 output6..9。
  - 解释:输出层更密时,一个 input basin 的 5 个 medium 横跨多个 motor;所有邻近输出都会被稳定固化,没有竞争机制时天然走向多输出 conflict。

### E14. 图形化 WebUI 观察工具(2026-07-01)

**目的:**
- 做一个薄 UI/IO 层,用于观察世界模型、网格拓扑、连接形成/剪枝、单点刺激路径、motor drive 和 readout 权重分布。
- 不把机制计算搬到浏览器;服务端复用 `dist/src/core` 的 topology/development/mechanism,浏览器只绘图和展示。

**实现:**
- 新增 `scripts/webui_server.cjs` / `npm run webui`。
- 新增 `webui/index.html`,`webui/app.js`,`webui/styles.css`。
- 两个视图:
  - `natural`:自然 `m/n/o/k × L` 网格,展示 checkpoint 曲线、拓扑图、连接状态(candidate/active/stable/dormant/pruned)、单点刺激 `stemN/fireN/correctD/wrongD`、输出权重 totals。
  - `valence`:1/2/1 中点世界,展示 `toxinGate/objectStimA/objectStimB/nutrientGate`、激素代理开关、刺激组合快照、`toxinD/nutrD`。
- 按手绘图补充 `gridWorld` 视图:
  - 真实世界是一维线:agent 在中点,营养/毒物/物体可出现在左/右/中点。
  - 网络是二维坐标:感受层包含 `nutrientSense/object*/toxinSense`,medium 是 rows×cols 网格,motor 是 `leftMotor/rightMotor`。
  - 物体位置只激活最近的 `object*` 感受器;营养/毒物通过 value sense 进入。这样避免把左右直接写死到 nutrient/toxin。
  - checkpoint 曲线显示 `correct/left/right/conflict/noop`;运行视图显示世界线、二维连接迁移、case 快照、left/right drive 和 motor 权重 totals。

**验证:**
- `node --check scripts/webui_server.cjs && node --check webui/app.js` 通过。
- `/api/simulate` 已用 natural/valence/gridWorld 三种模型做 HTTP spot-check,均返回 graph/path 或 graph/cases;gridWorld 1000epoch checkpoint 返回 `1,5,10,20,50,100,150,200,250,300,400,500,750,1000`。
- `npm test` 42/42 通过。

### D. 任务复杂度扩展 — 让 vacuous gate 变非空真(可与 C 并行)
- [ ] 扩 pattern 数 / 降 lr / 加长 episode,把 fresh 饱和点推后
      → 让 continued-learning sep 能取正值,而非恒 0
- [ ] 收紧后重跑 transfer matrix,确认 continued-learning sep 跨格出现正方差

## 工程经验反哺

1. 文档先约束实现:任何机制改动都要先写清楚它解决哪一个结构缺口,以及不能声称解决什么。
2. 实现必须反哺文档:advantage 更新确实修复了 complex Family A 的旧 conflict 表型,所以旧结论"SR=0.5 是上限"必须降级为历史基线,不能继续写成当前事实。
3. 诊断指标要跟着机制变:advantage 后错误形态从 conflict 转向 noop/弱激活,后续探针必须同时记录 conflictRate 和 noopRate。
4. 通过项不能自动升级为强证据:rewardOnly frozen sep 仍主要是"非零 vs fresh noop",continued-learning sep 仍是"不退化",wrong-prior -1 只是 1ep 压力失败;epoch curve 已显示 3ep 恢复。
5. 小机制改动要有 trace 字段和回归测试:本次 `rewardBaseline` / `rewardAdvantage` 进入 trace,避免未来只看最终 SR 而丢掉 credit-assignment 证据。

## 不做(本阶段 out of scope)

- 真实环境迁移(L4 之后)
- 新 sensor / 物种 / y-offset 拓扑扩展(arbitration matrix boundary 6 / complex boundary 6,独立支线)
- SNN 内生仲裁(resolver 仍是 post-hoc linear classifier)

## 底线结论

1. L0-L3 通过,但 continued-learning gate 退化为"预训练不退化",rewardOnly separation 主要测"非零 vs 零",均非强证据。
2. advantage 更新是有效的第一步:complex Family A rewardOnly 从旧 conflict 塌缩恢复到 SR=1.0;2D-challenge rewardOnly 仍 SR=0.5 但 noopRate=0.857;多物体 rewardOnly 仍 SR=0.5/conflict=0.333。
3. **A/B 探针证伪了"双侧共增强→conflict 塌缩"假说**:complex Family A 训练全程 trainConflict=0、双侧 fastWeight 对称衰减、靠 spike-count 仲裁解决;challenge 的 noop 是 fastWeight 衰减低于 motor 阈值 + conflict-gated exploration 在训练期掩盖 noop 所致;多物体 conflict 是 compositional vote-tie。三者机制不同,不能用同一个 bilateral 修复。
4. rewardOnly credit assignment 仍未完整解决,但**缺口重定向**:优先级从"loser-suppression/互斥"转为"① ε-greedy 探索(揭 noop)+ ② fastWeight 衰减/阈值 + ③ reward-driven stable depotentiation(wrong-prior 在 rewardOnly 下会卡更久)"。
5. wrong-prior = 慢恢复(supervised 下):stable depotentiation 1ep 清空 dual-lock,fast-path 2-3ep 恢复。非 stable lock-in、非无法 unlearn。**但 rewardOnly 下结论反转(见 8)**。
6. 新诊断工具:`audit:rewardonly:collapse`(complex)+ `audit:rewardonly:challenge-collapse`(challenge),per-epoch 双侧 fastWeight 对称性 + 训练 conflict/noop 曲线,作为 C 档修复的回归基准。
7. **长程验证推翻"fast 衰减跌破阈"假说,坐实真根因**:24seed×300ep 显示 rewardOnly 是"爬升到峰(~200ep SR 0.865)→ 灾难性悬崖(300ep 0/24 全死)",非单调恢复。真因不是 inter→motor fast/credit,而是 **sensory→inter 结构干线(init stable=1.1)被 `stableDecay=0.99999` 侵蚀,~200-250ep 跌破 inter axon 阈值 1.0 → inter 停发 → 整条 motor 链静默**(两 tick 架构下 inter somaPotential 是单次传导无累积,故是硬悬崖)。吸收态不可逆。反证:全局 `stableDecay=1.0` 悬崖消失。**修复**:给 `Synapse.decayProtected` 标记结构性干线,`decayWeights` 跳过其 stableDecay(learned 突触仍衰减)。修后 300ep 0/24→19/24 solved、transfer gate 不翻负。教训:① "权重跌破阈"得看**哪条**权重(sensory→inter 干线 vs inter→motor 学习突触),不能笼统;② 长程(>200ep)才暴露的悬崖不会被 40ep audit 看见,评估必须有长程基线;③ 结构性硬线与可遗忘记忆不该共用同一 passive decay。
8. **wrong-prior × rewardOnly 实测坐实 stable 去固化缺口(2026-06-30,`scripts/wrongprior_rewardonly.cjs`,6seed×300ep)**:supervised+reverseMapping 注入 wrong-prior(stable=2.0=maxWeight,dualLock=6/6),两臂都用正确映射继续学。supervised 臂 1ep 恢复、1ep 清 dualLock(`wasWronglyActive` 砍 stable 2.0→0.073);**rewardOnly 臂 0/6 在 300ep 恢复、0/6 清 dualLock,wrongMaxStable 全程钉死 2.000,SR 全程 0**。根因:rewardOnly `applyRewardLearning` 只动 fast(deltaStable:0)、不碰 stable;stable=2.0 单凭自己持续驱动错误 motor;且**自维持锁**——错误 stable 驱动错误 motor 发放→coactivity→`captureStableWeights` 每步回补 fast→stable,stable 钉在 maxWeight 不被动衰减;正确通路 correctMaxFast 全程 0(eligibility=0 被 skip,bootstrap 不起来)。**注**:此前判此题"不可测"(rewardOnly+reverseMapping 注入是 no-op,因 reward 不读 expectedAction),bypass 设计用 supervised 注入 + rewardOnly 正确映射 unlearn 绕开。结论:reward-driven stable 去固化**必要**但**不充分**(解不了 correct-path bootstrap,须与第1条同治)。
9. **STDP/BAP 大改后的长程复测给出负结论**:decayProtected 的 stem cliff 没复发,但 rewardOnly 长程行为从 master 修复后的 mean SR 0.948、19/24 solved 退化到本分支 mean SR 0.344、0/8 solved。当前问题不是"又坠崖",而是"food 能 partial 学会,toxin 回避 credit 不够,200ep 后 conflict/noop 混合"。因此 STDP/BAP 分支暂不能升为行为基线;下一步必须先处理 toxin 回避的正向强化信号。
