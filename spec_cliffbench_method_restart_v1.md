# CliffBench 方法线重启执行计划（v1�?
Date: 2026-03-24

Status:
- 方法线重�?spec
- 不替�?[`spec.md`](./spec.md)
- 用于约束独立方法项目的执行边界、实验设置与成功门槛

Related references:
- benchmark 主协议：[`spec.md`](./spec.md)
- benchmark 完成度对照：[`spec_f.md`](./spec_f.md)
- 当前实验总总结：[`EXPERIMENT_SUMMARY_2026-03-24.md`](./EXPERIMENT_SUMMARY_2026-03-24.md)
- 当前方法线状态：[`paper_latex/notes/training-episode-protocol-status.md`](./paper_latex/notes/training-episode-protocol-status.md)
- 当前协议评估：[`paper_latex/notes/episode-protocol-eval.md`](./paper_latex/notes/episode-protocol-eval.md)

## 0. 项目定位与硬边界

### 0.1 项目目标

把当前项目从“benchmark 内部的一系列局�?repair / protocol 试探”升级为一个独立的方法项目�?
> Cliff-sensitive local decision boundary learning for few-shot molecular classification

核心目标不是提升平均 few-shot 指标，而是�?cliff-heavy 局部区域形成更稳定的决策边界，并且不显著伤�?non-cliff control side�?
这个目标与当前论文的核心诊断保持一致：

- CliffBench 的关键发现不是简单的 ranking 差异
- 更关键的�?ranking-layer �?decision-layer 的分�?/ collapse
- 未来方法设计应围绕：
  - local boundary
  - calibration
  - support-query perturbation robustness

Paper-facing alignment:

- 这一方法线服务于未来独立方法�?- 不要求当�?benchmark 主稿立即升级
- 其成功标准必须高�?benchmark 内部 exploratory 试探

### 0.2 不可动边�?
1. benchmark 主稿身份不变
当前仍然�?stronger diagnostic benchmark paper，不强行改写�?benchmark-plus-method paper�?
Evidence:
- [`EXPERIMENT_SUMMARY_2026-03-24.md`](./EXPERIMENT_SUMMARY_2026-03-24.md)

2. final substrate 不动
[`outputs/fsmol_cliff_release_v4`](./outputs/fsmol_cliff_release_v4) 继续作为 final benchmark substrate�?
Evidence:
- [`EXPERIMENT_SUMMARY_2026-03-24.md`](./EXPERIMENT_SUMMARY_2026-03-24.md)

3. 方法探索统一�?intermediate 上先�?
[`outputs/fsmol_cliff_release_v4_covext_intermediate`](./outputs/fsmol_cliff_release_v4_covext_intermediate) 是唯一方法开�?substrate，主 profile �?`relaxed_covext_10_10`�?
这个 intermediate 已经被确认是有效�?evidence-strengthening layer，而不是替�?final 主表�?
Evidence:
- [`EXPERIMENT_SUMMARY_2026-03-24.md`](./EXPERIMENT_SUMMARY_2026-03-24.md)
- [`paper_latex/notes/coverage-extension-decision.md`](./paper_latex/notes/coverage-extension-decision.md)

4. 不回头救�?`NO-GO` �?exact family
下面这些 exact family 统一视为已关闭：
- decision-aware threshold repair
- local-boundary-repair（旧 kNN repair 版）
- fixed-support hard-negative replacement
- partial hard-negative augmentation
- current episode-construction sweep

解释�?- 这里的“关闭”指不再沿同一 exact family 做小修小�?- 不排除未来另起一条真正不同的方法�?
Evidence:
- [`EXPERIMENT_SUMMARY_2026-03-24.md`](./EXPERIMENT_SUMMARY_2026-03-24.md)

### 0.3 统一实验设置

除非单独说明，三条方法线统一使用当前 benchmark 的正式设置：

- substrate: `relaxed_covext_10_10`
- release tier: `intermediate`
- task family: 2-way few-shot
- support size: `16 / class`
- query size: `16 / class`
- per task / seed:
  - `400` standard episodes
  - `400` adversarial episodes
- seeds: `0..4`
- aggregation:
  - task-level macro aggregation
  - paired bootstrap `10,000` iterations

Protocol note:

- 所有新方法线都必须首先在这个统一设置下证明自�?- 只有在这一层成立后，才允许谈迁移到更复�?backbone 或训练策�?
Reference:
- [`spec.md`](./spec.md)
- [`paper_latex/main.tex`](./paper_latex/main.tex)

### 0.4 统一成功门槛

除非单独说明，三条方法线统一相对各自 backbone baseline 评估；在方法稿门槛上，优先相�?`ProtoNet` 判定�?
#### Primary win

必须同时尽量满足�?
- adversarial `C-BAcc`:
  - paired delta `> 0`
  - `95%` CI 尽量全正
- adversarial `SCR`:
  - paired delta `< 0`
  - `95%` CI 尽量全负
- adversarial `SS-SCR`:
  - paired delta `<= 0`
  - 最�?CI 也不跨到明显正向

#### Safety constraints

必须同时满足�?
- adversarial `SQ-PSR`:
  - 不能出现 clean negative
- adversarial `NC-BAcc` / `NC-PSR`:
  - 不能出现 clean negative
- standard `C-BAcc` / `SCR`:
  - 不能出现明显退�?
#### Paper-upgrade gate

只有当方法相�?`ProtoNet` baseline �?clean win，并且其改善模式能比现有 strongest simple baseline 更完整、更稳时，才考虑方法稿升级�?
当前 strongest simple baseline 至少包括�?
- `kNN-cliff-aware`

当前 strongest balanced baseline�?
- `ProtoNet`

解释�?
- beating vanilla `kNN` 不再足够
- beating `kNN-cliff-aware` 也仍然只是最低限度的 stronger-baseline gate
- 真正的方法稿升级，必须对 `ProtoNet` 这类更强 balanced baseline 也表现出 clean win

Evidence:
- [`EXPERIMENT_SUMMARY_2026-03-24.md`](./EXPERIMENT_SUMMARY_2026-03-24.md)
- [`paper_latex/main.tex`](./paper_latex/main.tex)

## 1. 当前执行含义

这一�?v1 spec 的含义不是“当前已有成功方法”，而是�?
- benchmark 诊断已经完成
- benchmark 主稿身份已经稳定
- �?repair / protocol families 已完成一轮足够严格的筛�?- 方法线现在应从零散试探切换为独立方法项目

因此，后续所有方法工作应满足�?
- 不再以“补�?benchmark 里某个失败点”为 framing
- 而以“学�?cliff-sensitive local decision boundary”为 framing
- 并且始终接受 stronger-baseline gate 约束

## 2. 当前默认结论

在这�?v1 spec 生效时，默认结论是：

- benchmark 主稿继续维持 stronger diagnostic benchmark paper
- 当前没有任何已测方法 family 足以升级�?benchmark-plus-method paper
- 任何方法稿升级都需要在�?spec 定义的统一门槛下重新建立证�?
## 3. 备注

- 本文件是方法线执�?spec，不�?benchmark 协议 spec
- [`spec.md`](./spec.md) 继续保持冻结 benchmark 协议地位
- [`spec_f.md`](./spec_f.md) 继续只记�?benchmark 主工程完成度

## 4. Plan A：ProtoNet + Local Calibration Head

### 4.1 核心问题

当前 evidence 显示，few-shot 分子模型可能保留 hard-pair ranking，却�?cliff-side decision �?collapse�?
因此，方法重点不应是改阈值规则，而应是学习一�?episode-local calibration / boundary correction mechanism�?
这一路线与当�?benchmark 主文和实验总结保持一致：

- 关键问题不是简单平均分�?- 而是 ranking-layer �?decision-layer 的分�?/ collapse
- 如果方法线要重启，优先级最高的应是�?  - local boundary correction
  - local calibration
  - support-conditioned robustness

References:
- [`paper_latex/main.tex`](./paper_latex/main.tex)
- [`EXPERIMENT_SUMMARY_2026-03-24.md`](./EXPERIMENT_SUMMARY_2026-03-24.md)

### 4.2 方法定义

�?ProtoNet 原始输出之上增加一�?local calibration head，对 query �?active score 做局部再校准�?
`s'(q) = s_proto(q) + g_theta(z_q, z_S+, z_S-, phi_local)`

其中�?
- `s_proto(q)`：原�?ProtoNet active score
- `z_q`：query embedding
- `z_S+`：正�?prototype / support summary
- `z_S-`：负�?prototype / support summary
- `phi_local`：局�?cliff-sensitive features

允许使用的局部特征，只限当前 assay-local、episode-local 可得信息�?
- query 到正 / �?prototype �?margin
- query 到最近正 / 最近负 support 的距离差
- support 内部类内离散�?- 正负 prototype 距离
- query �?support graph 上的 neighborhood disagreement 指标
- support / query similarity density
- optional：anchor-side cliff density summary

硬限制：

- 不得使用未来信息
- 不得使用�?assay 信息
- 不得重写 benchmark �?episode 定义

禁止事项�?
- 不做 rule-based threshold trick
- 不做 post-hoc 只改 decision rule
- 不改 benchmark �?episode 定义
- 不做 aggressive hard negative replacement

### 4.3 实现拆解

#### Phase A0：基线重建与接口准备

目标�?
- �?ProtoNet baseline 跑通成可插拔版�?
推荐产物�?
- `src/fsmol_cliff/protonet_base.py`
- `src/fsmol_cliff/protonet_local_calibrated.py`
- `tests/test_protonet_local_calibration.py`

要做的事�?
- 抽出 ProtoNet 原始 score path
- 明确 discrete prediction path 仍使用固�?reporting rule，但 score 输入可替换为 `s'`
- �?evaluator 中记录：
  - raw score
  - calibrated score
  - raw margin
  - calibrated margin

完成判据�?
- baseline 数字与当�?  [`outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_protonet_relaxed_covext_10_10.aggregate.json`](./outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_protonet_relaxed_covext_10_10.aggregate.json)
  对齐到浮点误差范�?
#### Phase A1：query-only local calibration ProtoNet

目标�?
- 先在不引入额�?support-aware 模块的情况下，验证局�?calibration head 本身是否能改�?cliff-side decision signal

最小定义：

- `g_theta` 只使用：
  - `z_q`
  - `z_S+`
  - `z_S-`
  - `phi_local`
- 不引�?query-to-support attention
- 不引入额�?uncertainty estimator

核心记录�?
- ProtoNet raw score �?calibrated score �?paired delta
- cliff slice �?non-cliff control slice 分开报表

Go / No-Go�?
- �?A1 连相�?ProtoNet baseline �?adversarial `C-BAcc` / `SCR` 都不能形成方向正确的 paired signal，则不继续扩大模型复杂度

#### Phase A2：support-aware calibrated ProtoNet

目标�?
- �?calibration �?query-only correction 提升�?support-conditioned local boundary correction

新增模块�?
- support summary encoder
- class-conditional uncertainty estimator
- optional：query-to-support attention over top-k nearest support

训练目标增加�?
- `L_collapse_penalty`

解释�?
- �?cliff pair 上对 collapse 施加惩罚
- 鼓励 active / inactive 两端不要被映射成同一离散决策

最小实验矩阵：

- `top-k �?{2, 4}`
- `attention �?{off, on}`
- `collapse penalty lambda �?{0, small}`

Go / No-Go�?
- �?A2 相对 A1 没有进一步提�?`SCR / SS-SCR`
- 且计算复杂度明显升高
- 则停�?A1 级别，不继续加复杂度

#### Phase A3：strict slice 验证

目标�?
- 检查方法是否只�?relaxed 上的偶然改善

做法�?
- 不把 strict 当主 claim substrate
- 只做 stress-test verification
- �?strict slice 的方向是否与 relaxed 一�?
成功判据�?
- strict 不要�?clean significance
- 只要求不出现方向性反�?
### 4.4 风险与排�?
主要风险�?
- calibration head 只是在做 disguised threshold shifting
- 提升 `C-BAcc` 但伤�?`SQ-PSR`
- 对少�?assay 过拟�?
对应排查�?
- 保存 raw vs calibrated score distribution
- 输出 task-level delta waterfall
- cliff / control slice 分开报表
- 固定 pair direction = `active -> inactive`
- 不改 metric path

### 4.5 这条线的最终交�?
最低成功版本：

- 一�?inference-time local calibration ProtoNet
- 相对 baseline �?adversarial `C-BAcc`、`SCR` �?clean 改善
- �?control side 不坏

最佳成功版本：

- 一个可训练�?support-aware local calibration ProtoNet
- 成为第一条真正过 stronger-baseline gate 的方法线

### 4.6 当前执行状态（2026-03-25�?
当前状态：

- `A0`: `GO`
- `A1`: `NO-GO`

`A0` 结论�?
- ProtoNet baseline 已经被抽成可插拔 score path
- identity calibration bundle 不改变当前默认评测语�?- fresh baseline aggregate 与当�?intermediate ProtoNet aggregate 对齐�?`max_abs_diff = 0.0`

`A1` 结论�?
- query-only local calibration 没有通过当前 gate
- 相对 ProtoNet baseline�?  - adversarial `C-BAcc`: `+0.004735`, CI `[-0.002780, 0.014564]`
  - adversarial `SCR`: `-0.003914`, CI `[-0.016333, 0.004319]`
  - standard `C-BAcc`: `-0.013094`, CI `[-0.031532, -0.000557]`

解释�?
- 这个最小版本更�?episode-local threshold / score refit，而不是足够强�?local boundary correction
- 它抓到了一�?adversarial 方向信号，但太弱
- 同时出现�?standard-side clean negative，违�?safety 约束

执行含义�?
- 不继续做 `A2`
- 按执行顺序切换到 `Plan B / B0`

## 5. Plan B：Cliff-Aware Prototype Shaping / Margin Learning

### 5.1 核心问题

如果 `ProtoNet` 已经是当�?strongest balanced baseline，那么进一步前进的方向不应是继续做 episode hack，而应是直接改�?prototype 几何结构，让 cliff-side 的局部分界更清晰�?
当前论文主文也已经明确指出，未来方法应关�?local boundary formation，而不只是 aggregate averages�?
References:
- [`paper_latex/main.tex`](./paper_latex/main.tex)
- [`EXPERIMENT_SUMMARY_2026-03-24.md`](./EXPERIMENT_SUMMARY_2026-03-24.md)

### 5.2 方法定义

�?ProtoNet episodic training 中加�?cliff-aware prototype shaping 目标，改�?embedding / prototype geometry，而不是只在推理时�?calibration�?
### 5.3 方法组件

#### B1：Prototype margin regularization

对于 cliff-associated active / inactive query，对其相�?prototype 的距离差施加 margin�?
- `d(q+, p+) + gamma < d(q+, p-)`
- `d(q-, p-) + gamma < d(q-, p+)`

解释�?
- 目标不是拉大全局类间间距
- 而是�?cliff-associated query 在局�?decision boundary 上留出更清晰�?margin

#### B2：Cliff-pair separation loss

对同 assay 内的 cliff pair `(a+, n-)`，直接在 embedding 空间施加分离约束�?
- `L_cliff_sep = max(0, m - (d(a-, p-) - d(a-, p+)))`

实现形式可写成：

- triplet-style
- contrastive-style
- margin-ranking-style

要求�?
- 形式可以�?- 但语义必须保持为 cliff pair 的局部分离，而不是普通全局 metric loss

#### B3：Control preservation regularization

为了避免“为了修 cliff，破�?control”：

- �?non-cliff control pair 增加保守正则
- 保持�?score ordering / boundary 不被显著推坏

### 5.4 实现拆解

#### Phase B0：先做最�?loss injection

目标�?
- 不改模型结构
- 只在 ProtoNet 训练目标里加入一�?cliff margin loss

推荐产物�?
- `src/fsmol_cliff/training_losses/cliff_margin.py`
- `configs/protonet_cliff_margin.yaml`

最小实验矩阵：

- `margin gamma �?{0.05, 0.1, 0.2}`
- `lambda_cliff �?{0.1, 0.3, 1.0}`
- `control preservation �?{off, on}`

Go / No-Go�?
- 如果所有组合都出现 `SQ-PSR` �?`NC-*` 明显下降，则 B0 直接 `NO-GO`

#### Phase B1：分�?cliff �?non-cliff 的双路目�?
目标�?
- 让训练过程显式知�?cliff �?control 的不同角�?
损失�?
- `L = L_proto + lambda_1 * L_cliff_sep + lambda_2 * L_control_preserve`

训练样本构造：

- 每个 episode 内显式标记：
  - cliff-associated query
  - non-cliff control query
  - same-scaffold cliff subset

loss 分配�?
- cliff-associated query、control query、same-scaffold cliff subset 使用不同权重

关键点：

- same-scaffold 只作�?stress-aware auxiliary
- 不单独成为主训练目标

原因�?
- current same-scaffold-targeted episode variant 已经�?hard `NO-GO`

Reference:
- [`EXPERIMENT_SUMMARY_2026-03-24.md`](./EXPERIMENT_SUMMARY_2026-03-24.md)

#### Phase B2：Prototype decomposition

目标�?
- 防止单个 prototype 把局部结构平均掉

做法�?
- 每类不只一�?prototype
- 使用 `K=2` sub-prototypes / class
- �?query 采用 mixture-of-prototypes scoring
- cliff-associated query 允许自动落到更合适的 local prototype

为何值得试：

- 当前 collapse 的一个可能来源，就是类内局部异质性被单原型过度平�?
风险�?
- 方法复杂度明显升�?- coverage 小时更容易过拟合

Go / No-Go�?
- 只有 B1 已出�?clean positive，才允许进入 B2
- �?B2 提升不超�?B1，优先保留更简单的 B1

### 5.5 关键评估

主评估：

- adversarial `C-BAcc`
- adversarial `SCR`
- adversarial `SS-SCR`

次评估：

- adversarial `SQ-PSR`
- adversarial `NC-BAcc / NC-PSR`
- standard `C-BAcc / SCR`

附加诊断�?
- prototype distance histogram
- cliff vs control query margin histogram
- same-scaffold vs non-same-scaffold 子集差异

### 5.6 最终交�?
最低成功版本：

- 一个只�?training loss �?ProtoNet-cliff-margin 版本
- �?clean 压低 adversarial `SCR`
- 并维�?`SQ-PSR / NC-*` 不坏

最佳成功版本：

- 一个真正通过 local representation shaping 改善 cliff-side decision quality �?ProtoNet family
- 可作为独立方�?paper 主体

### 5.7 当前执行状态（2026-03-26�?
当前状态：

- `B0 pilot`: `NO-GO`

评估对象�?
- training run:
  - [`outputs/b0_pilot/FSMol_ProtoNetCliffMargin_2026-03-25_18-29-17`](./outputs/b0_pilot/FSMol_ProtoNetCliffMargin_2026-03-25_18-29-17)
- evaluation aggregate:
  - `/tmp/protonet_b0_pilot_eval.aggregate.json`
- paired comparison:
  - `/tmp/protonet_b0_pilot_eval.paired_comparison.json`

关键结果，相�?ProtoNet baseline�?
- standard `SCR`: `+0.051966`, CI `[0.020776, 0.079919]`
- adversarial `C-BAcc`: `-0.050101`, CI `[-0.098343, -0.012539]`
- adversarial `SCR`: `+0.043597`, CI `[-0.000963, 0.092247]`
- adversarial `SS-SCR`: `+0.056047`, CI `[0.008821, 0.109823]`
- adversarial `NC-BAcc`: `-0.025999`, CI `[-0.052581, -0.000244]`
- adversarial `NC-PSR`: `-0.153425`, CI `[-0.221040, -0.077174]`

解释�?
- 这条最�?cliff-margin loss injection 同时没过 `Primary win` �?`Safety constraints`
- 它不是“接近成功”，而是明显�?cliff-side decision �?control side 都推坏了
- 因此按执行顺序，不继续扩�?`B1 / B2`

执行含义�?
- `Plan B` 当前最小版本已失败
- 按顺序转�?`Plan C / C0`

## 6. Plan C：Support-Query Perturbation Consistency Learning

### 6.1 核心问题

当前 episode-construction family 之所以重要，不是因为那些具体 variant 成功了，而是因为它们提供了一个研究信号：

> support-query perturbation 会显著影�?cliff-side decision behavior

当前论文 discussion 也已经把 robustness under support-query perturbation 视为未来核心方法目标�?
因此，这条方法线不再手工设计 episode 规则，而是把“扰动鲁棒性”直接写成训练目标�?
References:
- [`paper_latex/main.tex`](./paper_latex/main.tex)
- [`EXPERIMENT_SUMMARY_2026-03-24.md`](./EXPERIMENT_SUMMARY_2026-03-24.md)

### 6.2 方法定义

对同一�?assay-local episode，构造多个语义保持但局部扰动不同的 support / query 视图，要求模型在 cliff-relevant 决策上保持一致�?
注意�?
这里�?perturbation 不是�?
- 改标�?- aggressive support-negative replacement
- �?benchmark 定义

而是�?
- support 内部轻微重采�?- query 排列与局部邻域视图变�?- support subset dropout
- prototype noise / embedding dropout
- minor local neighborhood recomposition

### 6.3 扰动类型设计

#### P1：Support subset dropout

- �?support 中每类随机移�?`1�?` 个样�?- 保证 episode 仍合�?- 比较�?episode �?dropout episode �?cliff-side 预测差异

#### P2：Top-k neighborhood perturbation

- �?query 使用 top-k support neighbor view
- 改变 `k` 或轻微替换边�?support
- 检�?cliff query 决策是否稳定

#### P3：Prototype noise perturbation

- �?prototype 注入小高斯噪声或 embedding dropout
- 检�?cliff-associated query �?decision consistency

#### P4：Pair-preserving local reweighting

- 不替�?support negatives
- 只对局部支持样本权重做轻微 reweighting
- 检查模型对支持权重扰动是否过敏

### 6.4 训练目标

#### Consistency loss

对原 episode 与扰�?episode �?query score / prediction 施加一致性约束：

- `L_cons = || s(q; E) - s(q; E_tilde) ||^2`

#### Cliff-focused consistency

只对 cliff-associated query 加更高权重：

- `L_cliff_cons = w_cliff * || s(q_cliff; E) - s(q_cliff; E_tilde) ||^2`

#### Collapse-aware disagreement penalty

- 如果两个视图都把 cliff pair collapse 成同类，则增加惩�?
总损失：

- `L = L_task + lambda_1 * L_cons + lambda_2 * L_cliff_cons + lambda_3 * L_collapse_aware`

### 6.5 实现拆解

#### Phase C0：只�?inference robustness audit

目标�?
- 先不训练
- 先测 baseline ProtoNet 对扰动的敏感�?
要输出的东西�?
- perturbation sensitivity report
- 每个 query �?score variance
- cliff vs control �?variance gap
- same-scaffold cliff �?variance gap

Go / No-Go�?
- 如果 ProtoNet �?cliff query �?sensitivity 明显高于 control，则说明这条线值得训练�?- 如果几乎没有 sensitivity gap，则 C 线优先级下降

#### Phase C1：单扰动一致性训�?
先只选一个最稳的扰动�?
- 推荐�?support subset dropout 开�?
训练�?
- baseline ProtoNet + consistency loss
- 不引入多路复�?augment

最小矩阵：

- `dropout strength �?{1, 2}`
- `lambda_cons �?{0.05, 0.1, 0.3}`
- `cliff weight �?{1, 2, 4}`

Go / No-Go�?
- �?consistency training 只提�?standard stability，但 adversarial `SCR / C-BAcc` 不改善，�?C1 `NO-GO`

#### Phase C2：双扰动联合一致�?
只有 C1 有正信号才做�?
- support subset dropout + prototype noise
- �?support subset dropout + top-k neighborhood perturbation

目标�?
- �?robustness 是否可以从“一个角度稳定”提升为“局部决策整体稳定�?
#### Phase C3：扰动鲁棒性与校准联动

如果 C1 / C2 有效，可以与 Plan A 合并�?
- calibration head 输出�?perturbation-consistency 联训

此时形成更完整的方法 family�?
- local calibration
- local robustness
- collapse-aware training

### 6.6 风险与排�?
风险�?
- consistency 只会让模型更保守，反而加�?collapse
- 对所�?query 一视同仁会稀�?cliff focus
- 扰动设计不当，变�?disguised episode trick

排查�?
- �?cliff / control 分层 consistency
- 统计 collapse pair 在不同视图下是否更趋�?- 检查是否出�?`SCR` 下降�?`C-BAcc` 也下降的“更稳但更错”模�?
### 6.7 最终交�?
最低成功版本：

- ProtoNet + 单扰�?consistency training
- �?adversarial `SCR` �?`SS-SCR` �?clean 改善

最佳成功版本：

- ProtoNet + cliff-focused perturbation consistency learning
- 形成“局部边界鲁棒性”的方法主张

### 6.8 当前执行状态（2026-03-26�?
当前状态：

- `C0 pilot`: 弱信号，优先级下�?
pilot 设置�?
- perturbation: `support subset dropout`
- split: `adversarial`
- seed: `0`
- �?task 取前 `2` �?episodes
- dropout strength: `{1, 2}`
- 每个 strength 使用 `2` �?support shuffle 视图

pilot 产物�?
- [`outputs/c0_pilot_protonet_support_dropout_audit.json`](./outputs/c0_pilot_protonet_support_dropout_audit.json)

pilot 结果�?
- `episodes_analyzed = 20`
- `tasks_analyzed = 10`
- `cliff_control_variance_gap_mean = 0.001115982880235436`
- `same_scaffold_cliff_control_variance_gap_mean = 0.0010416293698955967`

解释�?
- ProtoNet 在这�?support dropout audit 下，cliff query 的扰动敏感性只�?control 略高
- 这个 gap 太小，不足以支撑�?`Plan C` 升为当前第一优先
- 这更像一个弱研究信号，而不是一个足以直接训练化的强入口

执行含义�?
- `Plan C` 没有被否�?- 但当�?`C0` 结果不支持把它上升为�?`Plan A / Plan B` 更强的方法主�?
### 6.9 放大�?`C0` 结果�?026-03-26�?
当前状态：

- expanded `C0`: `NO-GO`

expanded audit 设置�?
- seeds: `0..4`
- split: `adversarial`
- �?task �?`5` �?episodes
- perturbation:
  - support subset dropout
- dropout strength:
  - `{1, 2}`
- 每个 strength `4` �?support shuffle 视图

产物�?
- [`outputs/c0_expanded_protonet_support_dropout_audit.json`](./outputs/c0_expanded_protonet_support_dropout_audit.json)

核心结果�?
- `episodes_analyzed = 250`
- `tasks_analyzed = 10`
- `cliff_control_variance_gap_mean = 0.00030871558285348037`
- `same_scaffold_cliff_control_variance_gap_mean = -0.0001433979206547812`

分布结果�?
- cliff gap:
  - positive episodes: `66`
  - negative episodes: `54`
  - zero episodes: `130`
- same-scaffold cliff gap:
  - positive episodes: `48`
  - negative episodes: `58`

解释�?
- 放大�?cliff/control sensitivity gap 没有稳定拉开
- same-scaffold cliff 的方向甚至没有保持为�?- 因此 `C0` 不再只是“弱信号”，而是放大后仍未形成可训练化入�?
执行含义�?
- 不继�?`C1 / C2 / C3`
- 当前 `Plan C` 也应视为关闭

## 7. 三条计划的执行顺�?
推荐顺序�?
1. 第一优先：Plan A（Local Calibration Head�?2. 第二优先：Plan B（Prototype Shaping�?3. 第三优先：Plan C（Perturbation Consistency�?
### 7.1 第一优先：Plan A

原因�?
- 与当前“decision-layer collapse / calibration”诊断最直接对应
- 最容易与旧 `NO-GO` family 区分开
- 最容易做出最小成功原�?
### 7.2 第二优先：Plan B

原因�?
- 如果 A 成功，B 是自然更强版�?- 更像真正�?representation-level method
- 但训练复杂度更高，解释成本更�?
### 7.3 第三优先：Plan C

原因�?
- 研究味更强，但更容易写散
- 更适合作为 A / B 的增强项
- 或在 C0 audit 先证�?sensitivity gap 后再推进

## 8. 下一阶段决策建议

### 8.1 当前阶段结论

在本轮按顺序执行之后，当前状态是�?
- `A0 = GO`
- `A1 = NO-GO`
- `B0 pilot = NO-GO`
- `C0 pilot = 弱信号，优先级下降`

这意味着�?
- 现有最�?local calibration 版本没有形成 clean method win
- 现有最�?prototype-shaping 版本明确推坏�?cliff side �?control side
- 现有最�?perturbation audit 只显示很弱的 cliff/control sensitivity gap

因此，当前不能再把任一条方法线直接包装成“即将成功的方法稿主线”�?
### 8.2 默认建议

默认建议�?
- 先停�?broad method family 扩张
- 固化当前结果
- 把当前方法线结论写清楚为�?  - benchmark diagnosis 是成立的
  - 但最小方法版本尚未跨�?stronger-baseline gate

当前不建议直接继续做�?
- `A2`
- `B1`
- `B2`
- `C1`
- `C2`
- `C3`

原因�?
- 这些阶段都以当前较弱或失败的前置结果为起�?- 在前置阶段没�?clean positive 之前继续加复杂度，大概率只会增加时间成本与解释负�?
### 8.3 如果只允许再推进一条线

如果只允许再推进一条线，当前推荐顺序应调整为：

1. 先放�?`C0` audit
2. �?gap 变强，再进入 `C1`
3. �?gap 仍弱，则停止当前方法线扩�?
理由�?
- `C0` 是目前三条线里成本最低、假设最清晰的一条验证路�?- 它不会像 `A2 / B1 / B2` 一样直接引入更高训练复杂度
- 如果连放大后�?sensitivity gap 都不成立，则说明 “perturbation consistency�?不足以成为当前方法线主轴

### 8.4 放大�?`C0` 的最小执行建�?
如果继续，推荐的最小放大版 `C0` 为：

- seeds: `0..4`
- split: `adversarial`
- �?task 至少 `5` �?episodes
- perturbation:
  - support subset dropout
- dropout strength:
  - `{1, 2}`
- 每个 strength 至少 `4` �?support shuffle 视图

成功信号�?
- `cliff_control_variance_gap_mean` 持续为正
- same-scaffold cliff �?gap 也同方向
- 结果不只由极少数 task 驱动

若这些条件不成立�?
- 直接停止 Plan C 的训练化推进

### 8.5 �?Plan A / Plan B 的处理建�?
#### Plan A

- 当前 `A1` 已经说明 query-only local calibration 不够
- 除非重新定义一个与 A1 本质不同�?stronger local calibration family，否则不建议继续 `A2`

允许重启 A 线的条件�?
- 需要新�?calibration family 明确摆脱 “episode-local score refit�?倾向
- 并且其局部特征必须比当前 A1 更接近真正的 query-specific boundary geometry

#### Plan B

- 当前 `B0 pilot` 已经�?hard `NO-GO`
- 在没有新证据前，不建议继�?`B1 / B2`

允许重启 B 线的条件�?
- 需要先证明当前 cliff margin loss 的失败不是因�?loss 定义本身过粗，而不是因�?prototype shaping 方向整体失效
- 如果不能给出新的 loss family 论证，则默认关闭 B �?
### 8.6 推荐输出�?
无论是否继续，下一阶段至少应补齐以下输出物�?
- 方法线阶段总结
- 每条方法线的 `GO / NO-GO` 证据�?- 当前 strongest-baseline gate 的失败模式总结
- 若继续做 `C0` 放大版，则补一�?perturbation sensitivity report

### 8.7 当前一句话判断

当前最稳妥的判断是�?
- CliffBench 的诊断结论已经足够强
- 但当前方法线只证明了“哪里值得尝试”，还没有证明“哪条方法已经成功�?
### 8.8 当前总收�?
截至当前，按顺序执行后的结论是：

- `Plan A`
  - `A0 = GO`
  - `A1 = NO-GO`
- `Plan B`
  - `B0 pilot = NO-GO`
- `Plan C`
  - `C0 pilot = weak`
  - expanded `C0 = NO-GO`

因此�?
- 当前三条主方法线都没有形成可继续升级的方法稿主线
- 当前最合理的处理是停止方法线扩张，保留 benchmark-first 论文身份
