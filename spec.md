# FS-Mol-Cliff Benchmark & Hypothesis Validation Protocol v4.0

> 正式规范  
> 版本：v4.0  
> 状态：冻结候选规范  
> 语言：中文  
> 适用范围：FS-Mol assay-level few-shot molecular binary classification

---

## 1. 目标

本规范定义一个面向 FS-Mol few-shot 分子二分类的正式 benchmark 与验证协议，用于回答三层问题：

1. 平均 few-shot 指标是否掩盖了模型在 high-similarity discordant / activity-cliff 区域的失败。
2. 这些失败是否与全局相似性依赖、scaffold shortcut 与 decision-layer collapse 有关。
3. 若将 cliff 显式纳入训练或 episode 构造，模型是否能提升 cliff 相关指标、降低 collapse，并在非 cliff 区域不显著恶化。

v4.0 保留 strict / relaxed 双 profile，并将 attrition audit、threshold sensitivity、support-side scoring 和结果分级纳入正式规范。

---

## 2. 核心科学命题

- `H1` 现象层：平均指标会掩盖 cliff / high-similarity discordant failure。
- `H2` 机制层：failure 与 shortcut-like collapse 一致，尤其体现在 decision layer。
- `H3` 干预层：cliff-aware 方法应同时改善 cliff 指标并降低 collapse，而不是只提升平均分。

---

## 3. v4.0 的关键更新

1. `strict` 明确定位为 mini benchmark / stress test。  
2. `relaxed` 明确定位为主 benchmark / 多模型统计比较。  
3. attrition audit 必须发布，并必须包含 threshold sensitivity。  
4. `ProtoNet` 被纳入正式 few-shot 基线套件。  
5. 所有报告 `SQ-PSR` / `SS-SQ-PSR` 的方法，必须提供 support-side scoring 机制。  
6. 对需要 validation 的方法，允许 deterministic support-valid compatibility 规则。  
7. 正式结果必须带 `result_tier ∈ {final, exploratory, intermediate}`。

---

## 4. benchmark profiles

### 4.1 Strict

- `tau = 0.85`
- `delta = 1.0`
- `min_cliff_pairs = 25`
- `min_noncliff_pairs = 10`

用途：
- 机制压力测试
- collapse / failure mode 分析
- adversarial stress test

说明：
- 可正式发布
- 但默认视作 mini benchmark，不建议单独承担论文主表

### 4.2 Relaxed

- `tau = 0.80`
- `delta = 1.0`
- `min_cliff_pairs = 25`
- `min_noncliff_pairs = 10`

用途：
- 主 benchmark
- 多模型统计比较
- H1 / H2 / H3 主结果

说明：
- 推荐承担主表和主结论

---

## 5. 数据与 assay-local 原则

- 任务类型仅限 FS-Mol assay-level few-shot 二分类。
- 连续活性仅允许使用 `LogRegressionProperty`。
- cliff、control、scaffold、hard negative、episode 构造、所有指标都必须在 assay 内完成。
- 禁止跨 assay 构造 pair 或统一连续活性尺度。

---

## 6. 合法样本与预处理

记录仅在满足以下条件时可用：

1. `LogRegressionProperty` 非空且为有限数值；
2. `Relation` 表示单点精确测量；
3. RDKit 可解析并 sanitize；
4. 以 canonical isomeric SMILES 作为身份键。

不执行：
- salt collapse
- tautomer canonicalization
- largest fragment 选择
- 额外电荷归一化

重复记录处理：
- 标签冲突整组删除
- `max(r) - min(r) > 0.5` 整组删除
- 其余保留 `median(r)` 与共同标签

scaffold：
- 使用 Bemis-Murcko canonical scaffold SMILES
- 空 scaffold 记为 `EMPTY_SCAFFOLD`

---

## 7. pair 定义

使用 Morgan fingerprint：
- `radius = 2`
- `nBits = 2048`
- `useChirality = True`
- `useBondTypes = True`

定义：

- `H_t^disc = {(i,j) | i ∈ V_t^+, j ∈ V_t^-, sim(i,j) ≥ tau}`
- `C_t = {(i,j) ∈ H_t^disc | |r_i-r_j| ≥ delta}`
- `D_t = {(i,j) ∈ H_t^disc | |r_i-r_j| < delta}`
- `C_t^scaf = {(i,j) ∈ C_t | scaf_i = scaf_j != EMPTY_SCAFFOLD}`
- `D_t^scaf = {(i,j) ∈ D_t | scaf_i = scaf_j != EMPTY_SCAFFOLD}`

方向固定为 `active -> inactive`，不要求 `r_active > r_inactive`。

---

## 8. hard negative pool

对每个 `i ∈ A_t`：

- `H_i = {j | (i,j) ∈ C_t}`
- 排序键：
  1. `sim(i,j)` 降序
  2. `|r_i-r_j|` 降序
  3. `id_j` 升序
- 仅保留前 `32` 个

---

## 9. task eligibility

一个 assay 进入某 profile 的 benchmark，必须满足：

1. `|V_t| ≥ 50`
2. `|V_t^+| ≥ 15`
3. `|V_t^-| ≥ 15`
4. `|H_t^disc| ≥ min_cliff_pairs + min_noncliff_pairs`
5. `|C_t| ≥ min_cliff_pairs`
6. `|D_t| ≥ min_noncliff_pairs`
7. `|A_t| ≥ 10`
8. `|N_t^cliff| ≥ 10`

若 `M_avail(t) ≥ 2`，则进入 adversarial-eligible 列表。

---

## 10. attrition audit 与 threshold sensitivity

attrition audit 是正式发布资产，而不是内部辅助脚本。

每个 assay 必须记录：
- 原始 assay 是否存在
- legal samples 阶段
- active/inactive minimums 阶段
- `H_t^disc` 支撑阶段
- `C_t`
- `D_t`
- `A_t`
- `N_t^cliff`
- `M_avail`
- 最终是否 benchmark eligible
- 是否 adversarial eligible

必须发布：
- `attrition_summary.json`
- `attrition_by_assay.parquet`
- `threshold_sensitivity.json` 或 `threshold_sensitivity.parquet`
- `benchmark_decision_note.md`

最少 sweep：
- `tau ∈ {0.80, 0.85}`
- `delta ∈ {0.5, 1.0}`
- `min_cliff_pairs ∈ {10, 25}`
- `min_noncliff_pairs ∈ {5, 10}`

每组输出：
- eligible assay count
- adversarial eligible assay count
- total cliff pairs
- total anchors
- same-scaffold cliff coverage

---

## 11. episode protocol

### 11.1 Standard

standard split 的 n-way、support/query 大小、class balance 必须和冻结的 FS-Mol official 配置一致。  
正式评测以发布的 `episodes_standard_<profile>.parquet` 为准，不允许临时重采样替代。

### 11.2 Adversarial

二分图：
- 左：`A_t`
- 右：`N_t^cliff`
- 边：`C_t`

`M_avail(t)` 定义为其最大匹配大小。

注入数：

`m = min(floor(0.5 * |Q^-|), |S^+|, M_avail(t))`

若 `m < 2`，该 task 不参与 adversarial 指标统计。

### 11.3 Validation compatibility

若方法需要 validation split 才能稳定运行，可从 support 集按固定、deterministic 规则切出 `support_valid`：

- 默认规则：每类从 support 中保留最后 `k` 个样本作为 validation
- `k`、顺序规则和随机种子必须固定并公开
- 该兼容规则不得改变 query 集
- 必须在结果中说明该方法采用了 support-valid compatibility

---

## 12. 模型输出与 scoring 规则

设模型输出 active score `s(x) ∈ R`。

离散化：
- two-class logits / probs：`argmax`
- 单一概率：阈值 `0.5`
- 单一实数：必须公开固定离散化规则

若方法无法生成固定离散预测，则不得提交基于 `ŷ` 的主结果。

所有 pair ranking 统一使用：

- `1` if `s_i > s_j`
- `0.5` if `s_i = s_j`
- `0` if `s_i < s_j`

### Support-side scoring requirement

凡报告：
- `SQ-PSR`
- `SS-SQ-PSR`

的方法，必须公开 support-side score 的获得方式。  
允许方式包括：
- post-adaptation classifier score
- support prototype score
- support forward pass score

不允许只对 query 打分却声称完成 `SQ-PSR` 报告。

---

## 13. 核心指标

### Ranking-layer

- `Q-PSR`
- `SQ-PSR`
- `NC-PSR`
- `SS-Q-PSR`
- `SS-SQ-PSR`

### Decision-layer

- `C-BAcc`
- `NC-BAcc`
- `SCR`
- `SS-SCR`

### 官方平均指标

推荐至少报告：
- `average_precision_score`
- `delta_auprc = AP - fraction_positive_query`

推荐最小组合：
- `delta_auprc`
- `Q-PSR`
- `SQ-PSR`
- `SCR`

---

## 14. 结果分级

每个结果必须显式标注：

- `final`
- `exploratory`
- `intermediate`

以下情况不得标为 `final`：
- 非完整 episode 数
- 部分 assay
- 中途 checkpoint
- 临时 profile
- smoke 结果

---

## 15. 聚合与 CI

task 内：
- `score = mean over valid episodes`
- 同时报告 `coverage`
- `num_valid_episodes`
- `mean_num_valid_pairs_per_episode`

主表：
- task-level macro average
- 5-seed 平均

CI：
- task-level bootstrap
- `10000` 次重采样
- `95% CI`

模型比较：
- task-level paired bootstrap

---

## 16. 正式比较模型套件

对 v4.0 正式 few-shot benchmark 比较，推荐最小模型集：

- `kNN`
- `randomForest`
- `ProtoNet`
- `MAML`

理由：
- `RF` 是强单任务基线
- `MAML` 是核心 optimization-based meta-learning baseline
- `ProtoNet` 是 FS-Mol 原文里最值得重视的 strongest baseline，并且最贴近相似性驱动 few-shot 判断与 cliff shortcut 问题

`GNN-MT` 与 `MAT` 可作为扩展模型，但不建议替代 `ProtoNet`。

对 H3，还必须额外包含至少一个：
- `cliff-aware variant`

---

## 17. H1 / H2 / H3 claim rules

### H1

正式声称支持 H1，至少需要：
- `3` 个及以上模型 / recipe
- 一个官方平均指标
- `C-BAcc`
- `Q-PSR`
- `NC-BAcc`
- `NC-PSR`
- macro + CI

### H2

正式声称支持 H2，至少需要：
- `SCR`
- `SS-SCR`
- `Q-PSR`
- `SS-Q-PSR`

推荐补充：
- `SQ-PSR`
- `SS-SQ-PSR`

### H3

正式声称支持 H3，至少需要：
- baseline
- cliff-aware variant
- `ΔOfficial`
- `ΔC-BAcc`
- `ΔQ-PSR`
- `ΔSQ-PSR`
- `ΔSCR`
- `ΔSS-SCR`
- `ΔNC-BAcc`
- `ΔNC-PSR`

只有 cliff 改善、collapse 下降、control 未显著恶化三者同时成立，才允许使用强表述。

---

## 18. 发布资产

每个正式 release 必须包含：

- `benchmark_manifest.json`
- `fsmol_cliff_strict_all.json`
- `fsmol_cliff_relaxed_all.json`
- `fsmol_cliff_strict_adv_eligible.json`
- `fsmol_cliff_relaxed_adv_eligible.json`
- `episodes_standard_strict.parquet`
- `episodes_adversarial_strict.parquet`
- `episodes_standard_relaxed.parquet`
- `episodes_adversarial_relaxed.parquet`
- pair / hard negative / molecule annotation 资产
- attrition / threshold sensitivity / decision note
- task-level result tables
- 生成与聚合脚本

---

## 19. v4.0 的总定位

v4.0 明确承认：

- `strict` 是高精度、低覆盖的 mechanism stress benchmark
- `relaxed` 是主 benchmark
- `ranking-layer` 与 `decision-layer` 必须分开看
- `ProtoNet` 不能缺席正式 few-shot 比较

一句话概括：

> strict 用来证明问题尖锐存在，relaxed 用来证明问题具统计意义；ProtoNet 是 few-shot 正式基线套件中的核心模型，而不是可有可无的补充模型。
