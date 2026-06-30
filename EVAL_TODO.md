# 评估阶段方向与工程记忆

Type: project
Date: 2026-06-30
Scope: `/root/research/nerve_stimulate_v2`

> 本文件是当前唯一的评估指导文档。原 `nerve-stimulate-v2-eval-direction.md` 的项目记忆已合并到这里,避免 TODO 与 memory 状态分裂。

## 当前实证基线

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
- [x] 新增测试确认 rewardOnly trace 中 advantage 信号与 raw reward 可分离;`npm test` 16/16 通过
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
- [ ] 把 supervised 的 `wasWronglyActive` 去固化逻辑以 reward-driven 形式补进 rewardOnly
      (当前 rewardOnly 仍无 stableWeight depotentiation 路径,只能靠 stableDecay=0.99999 被动遗忘;A/B 证实 supervised stable 去固化很快,rewardOnly 缺这个 → wrong-prior 在 rewardOnly 下会卡更久,待测)
- [ ] 多物体 compositional vote-tie:先看修上面两项后是否自愈
- [ ] 修复后重跑目标:
      2D-challenge rewardOnly SR > 0.5 且 noopRate 明显下降(< 0.3);
      2D-complex multi-object rewardOnly SR > 0.5 且 conflictRate < 0.3;
      transfer matrix rewardOnly sep 不因修复翻负(ε-greedy 已证会翻负,新机制必须不翻负);
      **新增回归**:`audit:rewardonly:challenge-collapse` 训练期 trainNoopRate > 0(不再被掩盖)、最终 evalNoopRate 下降。

**C 档进度:ε-greedy(常数)净负已回退;长程 noop 悬崖已修(decayProtected sensory stem,24/24 stuck→0/24)。剩余:rewardOnly 40ep 短期 noop(bootstrap/commit,非悬崖)、多物体 compositional vote-tie、rewardOnly stable depotentiation。**

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
5. wrong-prior = 慢恢复:stable depotentiation 1ep 清空 dual-lock,fast-path 2-3ep 恢复。非 stable lock-in、非无法 unlearn。
6. 新诊断工具:`audit:rewardonly:collapse`(complex)+ `audit:rewardonly:challenge-collapse`(challenge),per-epoch 双侧 fastWeight 对称性 + 训练 conflict/noop 曲线,作为 C 档修复的回归基准。
7. **长程验证推翻"fast 衰减跌破阈"假说,坐实真根因**:24seed×300ep 显示 rewardOnly 是"爬升到峰(~200ep SR 0.865)→ 灾难性悬崖(300ep 0/24 全死)",非单调恢复。真因不是 inter→motor fast/credit,而是 **sensory→inter 结构干线(init stable=1.1)被 `stableDecay=0.99999` 侵蚀,~200-250ep 跌破 inter axon 阈值 1.0 → inter 停发 → 整条 motor 链静默**(两 tick 架构下 inter somaPotential 是单次传导无累积,故是硬悬崖)。吸收态不可逆。反证:全局 `stableDecay=1.0` 悬崖消失。**修复**:给 `Synapse.decayProtected` 标记结构性干线,`decayWeights` 跳过其 stableDecay(learned 突触仍衰减)。修后 300ep 0/24→19/24 solved、transfer gate 不翻负。教训:① "权重跌破阈"得看**哪条**权重(sensory→inter 干线 vs inter→motor 学习突触),不能笼统;② 长程(>200ep)才暴露的悬崖不会被 40ep audit 看见,评估必须有长程基线;③ 结构性硬线与可遗忘记忆不该共用同一 passive decay。
