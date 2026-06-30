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

**E 档不做(本分支 out of scope):** ~~物理时间轴(tick→ms)、传导延迟队列~~(E2 后降级为可选)、严格 τ₊/τ₋ 生物数值对齐、reward→stable 去固化开关。

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
