# spec_f

Date: 2026-03-21

Purpose:
- 对照 [`spec.md`](./spec.md) 逐章标注当前仓库的完成情况
- 记录每一章对应的实现模块、测试覆盖与发布证据
- 在文末附上过滤后的全仓库目录树

状态标记：
- `completed`: 规范要求已经在代码、测试或发布资产中落地
- `partial`: 已有实现，但仍有明确缺口
- `pending`: 尚未落地
- `n/a`: 本仓库范围内不适用

## 总体结论

- 总体状态：已基本完成，当前主要剩余 `partial` 章节为 `15` 与 `17`
- 当前仓库已经完成 `spec.md` 定义的 v4.0 benchmark / release / claim-evidence 主工程闭环
- `3`, `6`, `7`, `12`, `18` 中此前记录的实现缺口已在 `2026-03-21` 修复
- 当前正式发布主目录为 [`outputs/fsmol_cliff_release_v4`](./outputs/fsmol_cliff_release_v4)
- 当前 rebuilt v4 release 的 profile coverage 为：`strict = 2`, `relaxed = 6`
- 当前 release claim 结果为：
  - `H1`: supported trend
  - `H2`: formal claim
  - `H3`: supported trend
- 上述 claim 强度差异属于发布结果边界，不属于规范实现缺失
- 说明：release bundle、audit 与 relaxed/strict 主结果表已经在 chemistry-aligned 代码下重建；当前剩余的主要限制来自 paired-comparison clean-up 与 claim-rule literal fidelity，而不是旧数值产物

## 章节总表

| section | title | status | summary |
| --- | --- | --- | --- |
| `1` | 目标 | `completed` | 仓库已实现 cliff-aware benchmark、发布与假设验证闭环 |
| `2` | 核心科学命题 | `completed` | `H1/H2/H3` 已编码、验证并写入 release 文档 |
| `3` | v4.0 的关键更新 | `completed` | strict/relaxed、audit、ProtoNet、support-side scoring、result tier 均已落地 |
| `4` | benchmark profiles | `completed` | `strict` / `relaxed` profile 常量与发布策略已实现 |
| `5` | 数据与 assay-local 原则 | `completed` | 数据处理、pair、episode、指标均按 assay 内部完成 |
| `6` | 合法样本与预处理 | `completed` | legal sample 过滤、去重、canonical SMILES、revalidation、scaffold 已实现 |
| `7` | pair 定义 | `completed` | high-sim discordant / cliff / noncliff / same-scaffold pair 挖掘已实现，fingerprint 参数已对齐 spec |
| `8` | hard negative pool | `completed` | per-anchor hard negative 排序与截断已实现 |
| `9` | task eligibility | `completed` | benchmark eligibility / adversarial eligibility 已实现 |
| `10` | attrition audit 与 threshold sensitivity | `completed` | profile-aware attrition 与 sweep 已发布 |
| `11` | episode protocol | `completed` | standard / adversarial / support-valid compatibility 已实现 |
| `12` | 模型输出与 scoring 规则 | `completed` | score、离散化、support-side scoring 规则已落地到 runner 与 metadata |
| `13` | 核心指标 | `completed` | ranking / decision / official average 指标均已实现 |
| `14` | 结果分级 | `completed` | `result_tier` 已进入 CLI、runner、聚合与测试 |
| `15` | 聚合与 CI | `partial` | 聚合与 CI 实现完成，但当前 release 的部分 paired comparison 行仍不够干净 |
| `16` | 正式比较模型套件 | `completed` | `kNN` / `RF` / `ProtoNet` / cliff-aware variant 构成当前 full-strength release 套件；`MAML` 以 exploratory compatibility 形式保留 |
| `17` | H1 / H2 / H3 claim rules | `partial` | claim 规则与 release summary 已落地，但代码验证逻辑并非 spec 清单的逐项镜像 |
| `18` | 发布资产 | `completed` | 主发布资产已齐，release 目录新增 reproducibility manifest 指向生成/聚合入口 |
| `19` | v4.0 的总定位 | `completed` | strict/relaxed 定位与 release 口径已文档化 |

## 逐章对照

### 1. 目标

状态：`completed`

实现：
- 仓库围绕 FS-Mol assay-level few-shot 二分类，完成了 assay 资产构建、profile-aware release 生成、episode 评测、聚合、claim 规则验证与 release 文档收口。
- 整体主流程由 CLI 驱动，覆盖 `build-release`、`audit-attrition`、`evaluate`、`aggregate`、`validate-hypotheses`。

证据：
- 代码：[`src/fsmol_cliff/cli.py`](./src/fsmol_cliff/cli.py)
- 代码：[`src/fsmol_cliff/release.py`](./src/fsmol_cliff/release.py)
- 代码：[`src/fsmol_cliff/hypotheses.py`](./src/fsmol_cliff/hypotheses.py)
- 测试：[`tests/test_cli_commands.py`](./tests/test_cli_commands.py)
- 发布：[`outputs/fsmol_cliff_release_v4/release_summary.md`](./outputs/fsmol_cliff_release_v4/release_summary.md)

备注：
- 目标已经从“协议定义”推进到“可复现实物发布”。

### 2. 核心科学命题

状态：`completed`

实现：
- `H1`、`H2`、`H3` 均有显式验证逻辑与 release 级文档解释。
- claim 层同时区分“规则实现完成”和“当前发布支持到什么强度”。

证据：
- 代码：[`src/fsmol_cliff/hypotheses.py`](./src/fsmol_cliff/hypotheses.py)
- 测试：[`tests/test_hypotheses_validation.py`](./tests/test_hypotheses_validation.py)
- 发布：[`outputs/fsmol_cliff_release_v4/relaxed_claim_summary.md`](./outputs/fsmol_cliff_release_v4/relaxed_claim_summary.md)

备注：
- 当前 release 结果不是三个命题都到 formal claim，但这不影响核心命题框架已经落地。

### 3. v4.0 的关键更新

状态：`completed`

实现：
- strict / relaxed 双 profile 已实现并同时发布。
- attrition audit 与 threshold sensitivity 已发布。
- `ProtoNet` 已纳入正式比较与 release 主文档。
- support-side scoring 已在 `ProtoNet` 路径与执行 metadata 中公开。
- support-valid compatibility 已为 `MAML` 提供 deterministic 规则。
- `result_tier` 已进入 runner、聚合、测试链路。

证据：
- 代码：[`src/fsmol_cliff/constants.py`](./src/fsmol_cliff/constants.py)
- 代码：[`src/fsmol_cliff/protonet_runner.py`](./src/fsmol_cliff/protonet_runner.py)
- 代码：[`src/fsmol_cliff/maml_legacy.py`](./src/fsmol_cliff/maml_legacy.py)
- 代码：[`src/fsmol_cliff/audit.py`](./src/fsmol_cliff/audit.py)
- 发布：[`outputs/fsmol_cliff_release_v4/model_execution_metadata.json`](./outputs/fsmol_cliff_release_v4/model_execution_metadata.json)
- 发布：[`outputs/fsmol_cliff_release_v4/benchmark_decision_note.md`](./outputs/fsmol_cliff_release_v4/benchmark_decision_note.md)

备注：
- v4.0 的关键更新已经不只是代码能力，也已经变成发布资产。
- `model_execution_metadata.json` 现已补齐 `kNN` / `RF` / `kNN-cliff-aware` 的 support-side scoring 说明。

### 4. benchmark profiles

状态：`completed`

实现：
- `STRICT_PROFILE` 与 `RELAXED_PROFILE` 已在协议常量中冻结。
- `build_release_bundle`、task eligibility、audit、release summary 均按 profile 运行。
- strict 定位为 stress / mechanism profile，relaxed 定位为 main benchmark profile。

证据：
- 代码：[`src/fsmol_cliff/constants.py`](./src/fsmol_cliff/constants.py)
- 代码：[`src/fsmol_cliff/release.py`](./src/fsmol_cliff/release.py)
- 发布：[`outputs/fsmol_cliff_release_v4/benchmark_manifest.json`](./outputs/fsmol_cliff_release_v4/benchmark_manifest.json)
- 发布：[`outputs/fsmol_cliff_release_v4/benchmark_decision_note.md`](./outputs/fsmol_cliff_release_v4/benchmark_decision_note.md)

备注：
- profile 不是分析时的临时参数，而是 release 资产级实体。

### 5. 数据与 assay-local 原则

状态：`completed`

实现：
- task 记录逐 assay 读入。
- pair、hard negative、scaffold、episode、评测与指标都基于 assay 内部资产，不跨 assay 拼接。
- continuous activity 使用 assay 内部的 `LogRegressionProperty` / `r` 字段。

证据：
- 代码：[`src/fsmol_cliff/pipeline.py`](./src/fsmol_cliff/pipeline.py)
- 代码：[`src/fsmol_cliff/assets.py`](./src/fsmol_cliff/assets.py)
- 代码：[`src/fsmol_cliff/evaluation.py`](./src/fsmol_cliff/evaluation.py)
- 测试：[`tests/test_pipeline.py`](./tests/test_pipeline.py)
- 测试：[`tests/test_assets.py`](./tests/test_assets.py)

备注：
- 这一层的 assay-local 约束已经渗透到数据、episode 与 metric 计算全链路。

### 6. 合法样本与预处理

状态：`completed`

实现：
- 只保留 precise measurement、有限 `r` 值、可 canonicalize 的分子、可解析标签。
- 以 canonical isomeric SMILES 为聚合键。
- 重复记录冲突删除、`max(r)-min(r)>0.5` 删除、其余保留 `median(r)`。
- scaffold 使用 Bemis-Murcko；空 scaffold 写为 `EMPTY_SCAFFOLD`。

证据：
- 代码：[`src/fsmol_cliff/assets.py`](./src/fsmol_cliff/assets.py)
- 代码：[`src/fsmol_cliff/chem.py`](./src/fsmol_cliff/chem.py)
- 代码：[`src/fsmol_cliff/pipeline.py`](./src/fsmol_cliff/pipeline.py)
- 测试：[`tests/test_assets.py`](./tests/test_assets.py)
- 测试：[`tests/test_pipeline.py`](./tests/test_pipeline.py)

备注：
- spec 中明确“不做”的额外规范化也没有被额外引入。
- 当前实现已对预先给定的 canonical SMILES 再走一次 RDKit 解析/规范化校验。
- rebuilt v4 release 已在当前化学实现下重建。

### 7. pair 定义

状态：`completed`

实现：
- 使用 RDKit Morgan fingerprint 计算 Tanimoto similarity。
- 按 active -> inactive 方向构造 high-sim discordant 对。
- 基于 `tau` 与 `delta` 划分 cliff / high-sim noncliff。
- same-scaffold cliff / noncliff 也已显式分组。

证据：
- 代码：[`src/fsmol_cliff/chem.py`](./src/fsmol_cliff/chem.py)
- 代码：[`src/fsmol_cliff/assets.py`](./src/fsmol_cliff/assets.py)
- 代码：[`src/fsmol_cliff/models.py`](./src/fsmol_cliff/models.py)
- 测试：[`tests/test_assets.py`](./tests/test_assets.py)
- 发布：[`outputs/fsmol_cliff_release_v4/assays`](./outputs/fsmol_cliff_release_v4/assays)

备注：
- 实现上用 `PairRecord` 固化了 pair schema。
- 当前 Morgan fingerprint 实现已显式开启 `useChirality=True` 与 `useBondTypes=True`，与 spec 的字面设定一致。
- rebuilt v4 release 已在当前 fingerprint 设定下重建。

### 8. hard negative pool

状态：`completed`

实现：
- per-anchor hard negative pool 已实现。
- 排序规则按 similarity、gap、id 稳定化。
- release profile 默认 hard negative pool size 为 `32`。

证据：
- 代码：[`src/fsmol_cliff/assets.py`](./src/fsmol_cliff/assets.py)
- 代码：[`src/fsmol_cliff/constants.py`](./src/fsmol_cliff/constants.py)
- 测试：[`tests/test_assets.py`](./tests/test_assets.py)
- 发布：[`outputs/fsmol_cliff_release_v4/assays`](./outputs/fsmol_cliff_release_v4/assays)

备注：
- hard negative pool 同时被 adversarial episode 注入逻辑复用。

### 9. task eligibility

状态：`completed`

实现：
- benchmark eligibility 与 adversarial eligibility 已由 profile-aware 阈值显式控制。
- `M_avail >= 2` 作为 adversarial eligibility 门槛已编码。
- task ranking / top-k 辅助逻辑已实现。

证据：
- 代码：[`src/fsmol_cliff/task_selection.py`](./src/fsmol_cliff/task_selection.py)
- 代码：[`src/fsmol_cliff/release.py`](./src/fsmol_cliff/release.py)
- 测试：[`tests/test_release.py`](./tests/test_release.py)
- 测试：[`tests/test_task_selection.py`](./tests/test_task_selection.py)
- 发布：[`outputs/fsmol_cliff_release_v4/fsmol_cliff_relaxed_all.json`](./outputs/fsmol_cliff_release_v4/fsmol_cliff_relaxed_all.json)
- 发布：[`outputs/fsmol_cliff_release_v4/fsmol_cliff_relaxed_adv_eligible.json`](./outputs/fsmol_cliff_release_v4/fsmol_cliff_relaxed_adv_eligible.json)

备注：
- eligibility 不是手工整理，而是由 release builder 自动生成。

### 10. attrition audit 与 threshold sensitivity

状态：`completed`

实现：
- attrition funnel、per-assay attrition rows、threshold sensitivity sweep 均已实现。
- 发布目录中同时包含 strict 与 relaxed profile 的 audit 资产。
- `benchmark_decision_note.md` 已给出 strict vs relaxed 的最终 release policy。

证据：
- 代码：[`src/fsmol_cliff/audit.py`](./src/fsmol_cliff/audit.py)
- 测试：[`tests/test_attrition_audit.py`](./tests/test_attrition_audit.py)
- 发布：[`outputs/fsmol_cliff_release_v4/audit/strict/attrition_summary.json`](./outputs/fsmol_cliff_release_v4/audit/strict/attrition_summary.json)
- 发布：[`outputs/fsmol_cliff_release_v4/audit/strict/attrition_by_assay.parquet`](./outputs/fsmol_cliff_release_v4/audit/strict/attrition_by_assay.parquet)
- 发布：[`outputs/fsmol_cliff_release_v4/audit/strict/threshold_sensitivity.parquet`](./outputs/fsmol_cliff_release_v4/audit/strict/threshold_sensitivity.parquet)
- 发布：[`outputs/fsmol_cliff_release_v4/audit/relaxed/attrition_summary.json`](./outputs/fsmol_cliff_release_v4/audit/relaxed/attrition_summary.json)
- 发布：[`outputs/fsmol_cliff_release_v4/audit/relaxed/attrition_by_assay.parquet`](./outputs/fsmol_cliff_release_v4/audit/relaxed/attrition_by_assay.parquet)
- 发布：[`outputs/fsmol_cliff_release_v4/audit/relaxed/threshold_sensitivity.parquet`](./outputs/fsmol_cliff_release_v4/audit/relaxed/threshold_sensitivity.parquet)
- 发布：[`outputs/fsmol_cliff_release_v4/benchmark_decision_note.md`](./outputs/fsmol_cliff_release_v4/benchmark_decision_note.md)

备注：
- 本章在 v4.0 中已经从“内部分析”升级成“正式发布资产”。

### 11. episode protocol

状态：`completed`

实现：
- standard episode manifest 采用固定 `2-way`, `16 support/class`, `16 query/class`, balanced 配置。
- adversarial episode 使用二分图最大匹配、`m = min(floor(0.5 * |Q^-|), |S^+|, M_avail)` 的等价实现。
- `m < 2` 时 adversarial episode 不生成。
- MAML 的 support-valid compatibility 规则已实现为 deterministic tail holdout。

证据：
- 代码：[`src/fsmol_cliff/constants.py`](./src/fsmol_cliff/constants.py)
- 代码：[`src/fsmol_cliff/manifests.py`](./src/fsmol_cliff/manifests.py)
- 代码：[`src/fsmol_cliff/episodes.py`](./src/fsmol_cliff/episodes.py)
- 代码：[`src/fsmol_cliff/maml_legacy.py`](./src/fsmol_cliff/maml_legacy.py)
- 测试：[`tests/test_episodes.py`](./tests/test_episodes.py)
- 测试：[`tests/test_maml_legacy.py`](./tests/test_maml_legacy.py)
- 发布：[`outputs/fsmol_cliff_release_v4/episodes_standard_relaxed.parquet`](./outputs/fsmol_cliff_release_v4/episodes_standard_relaxed.parquet)
- 发布：[`outputs/fsmol_cliff_release_v4/episodes_adversarial_relaxed.parquet`](./outputs/fsmol_cliff_release_v4/episodes_adversarial_relaxed.parquet)
- 发布：[`outputs/fsmol_cliff_release_v4/model_execution_metadata.json`](./outputs/fsmol_cliff_release_v4/model_execution_metadata.json)

备注：
- support-valid compatibility 不会改动 query 集，这一点已在 metadata 中体现。

### 12. 模型输出与 scoring 规则

状态：`completed`

实现：
- 评测统一要求模型输出 active score，并在 episode evaluator 中按固定阈值 `0.5` 离散化为预测。
- pair ranking 使用统一的 `1 / 0.5 / 0` 规则。
- 支持 `SQ-PSR` / `SS-SQ-PSR` 的方法必须给出 support-side score；`ProtoNet` 已对 support molecules 显式打分。
- 本仓库支持的 backend 都是固定规则实现，不存在“未定义离散化”的主结果路径。

证据：
- 代码：[`src/fsmol_cliff/evaluation.py`](./src/fsmol_cliff/evaluation.py)
- 代码：[`src/fsmol_cliff/metrics.py`](./src/fsmol_cliff/metrics.py)
- 代码：[`src/fsmol_cliff/runner.py`](./src/fsmol_cliff/runner.py)
- 代码：[`src/fsmol_cliff/protonet_runner.py`](./src/fsmol_cliff/protonet_runner.py)
- 发布：[`outputs/fsmol_cliff_release_v4/model_execution_metadata.json`](./outputs/fsmol_cliff_release_v4/model_execution_metadata.json)
- 测试：[`tests/test_metrics.py`](./tests/test_metrics.py)
- 测试：[`tests/test_protonet_runner.py`](./tests/test_protonet_runner.py)

备注：
- 这里的“完成”是指当前受支持模型族的 scoring 规则已经固定，不是任意外部模型自动适配。
- `ProtoNet` / `MAML` / `kNN` / `RF` / `kNN-cliff-aware` 的 support-side scoring 说明现已全部进入 metadata。

### 13. 核心指标

状态：`completed`

实现：
- ranking-layer：`Q-PSR`, `SQ-PSR`, `NC-PSR`, `SS-Q-PSR`, `SS-SQ-PSR`
- decision-layer：`C-BAcc`, `NC-BAcc`, `SCR`, `SS-SCR`
- 官方平均指标：`average_precision_score`, `delta_auprc`
- 所有指标均进入 episode 评测、task 汇总与 release 产物生成。

证据：
- 代码：[`src/fsmol_cliff/metrics.py`](./src/fsmol_cliff/metrics.py)
- 代码：[`src/fsmol_cliff/evaluation.py`](./src/fsmol_cliff/evaluation.py)
- 测试：[`tests/test_metrics.py`](./tests/test_metrics.py)
- 测试：[`tests/test_release_evaluation.py`](./tests/test_release_evaluation.py)
- 发布：[`outputs/fsmol_cliff_release_v4/relaxed_main_table.md`](./outputs/fsmol_cliff_release_v4/relaxed_main_table.md)

备注：
- 指标分层与 spec 一致，主 release 文档也按 ranking-layer 与 decision-layer 分开解释。

### 14. 结果分级

状态：`completed`

实现：
- `result_tier` 已进入 CLI 参数、runner 输出、task summary、aggregate 结果。
- `final / exploratory / intermediate` 三档都被支持。
- 历史 run artifact 中已有 smoke/mid/full 例子，v4 正式发布使用 `final`。

证据：
- 代码：[`src/fsmol_cliff/cli.py`](./src/fsmol_cliff/cli.py)
- 代码：[`src/fsmol_cliff/evaluation.py`](./src/fsmol_cliff/evaluation.py)
- 代码：[`src/fsmol_cliff/aggregate.py`](./src/fsmol_cliff/aggregate.py)
- 测试：[`tests/test_release_evaluation.py`](./tests/test_release_evaluation.py)
- 发布：[`outputs/fsmol_cliff_release_v4/release_summary.md`](./outputs/fsmol_cliff_release_v4/release_summary.md)

备注：
- 结果分级能力已实现；正式 release 是否接受某 tier 是发布策略问题。

### 15. 聚合与 CI

状态：`partial`

实现：
- task 内按 valid episodes 求均值，并记录 coverage、valid episode 数、valid pair 数。
- 主表采用 task-level macro average。
- `task_bootstrap_ci` 与 `paired_bootstrap_delta_ci` 已实现，默认 `10000` 次重采样。
- release artifact 生成已使用 paired comparison bootstrap。

证据：
- 代码：[`src/fsmol_cliff/aggregate.py`](./src/fsmol_cliff/aggregate.py)
- 代码：[`src/fsmol_cliff/evaluation.py`](./src/fsmol_cliff/evaluation.py)
- 代码：[`src/fsmol_cliff/release_artifacts.py`](./src/fsmol_cliff/release_artifacts.py)
- 测试：[`tests/test_bootstrap.py`](./tests/test_bootstrap.py)
- 测试：[`tests/test_bootstrap.py`](./tests/test_bootstrap.py)
- 测试：[`tests/test_release_artifacts.py`](./tests/test_release_artifacts.py)
- 发布：[`outputs/fsmol_cliff_release_v4/relaxed_model_comparisons.json`](./outputs/fsmol_cliff_release_v4/relaxed_model_comparisons.json)

备注：
- 当前 CI 逻辑已实现；某些单个 release comparison 行是否可用于强 claim，要由 claim 文档另行判断。
- `adversarial c_bacc` 的 avoidable `NaN` 行已经修掉，但当前 release 仍存在 coverage-strength不均的问题，尤其是 `MAML` 仍属 exploratory compatibility 路径。

### 16. 正式比较模型套件

状态：`completed`

实现：
- 正式比较模型集已包含 `kNN`, `RF`, `ProtoNet`, `MAML`。
- 另有 `kNN-cliff-aware` 作为 H3 所需 intervention variant。
- 官方 bridge / compatibility patch、ProtoNet runner、legacy MAML 路径均已实现。

证据：
- 代码：[`src/fsmol_cliff/adapters.py`](./src/fsmol_cliff/adapters.py)
- 代码：[`src/fsmol_cliff/fsmol_bridge.py`](./src/fsmol_cliff/fsmol_bridge.py)
- 代码：[`src/fsmol_cliff/protonet_runner.py`](./src/fsmol_cliff/protonet_runner.py)
- 代码：[`src/fsmol_cliff/maml_legacy_runner.py`](./src/fsmol_cliff/maml_legacy_runner.py)
- 测试：[`tests/test_adapters.py`](./tests/test_adapters.py)
- 测试：[`tests/test_protonet_runner.py`](./tests/test_protonet_runner.py)
- 测试：[`tests/test_maml_legacy_runner.py`](./tests/test_maml_legacy_runner.py)
- 发布：[`outputs/fsmol_cliff_release_v4/model_execution_metadata.json`](./outputs/fsmol_cliff_release_v4/model_execution_metadata.json)
- 发布：[`outputs/fsmol_cliff_release_v4/relaxed_main_table.md`](./outputs/fsmol_cliff_release_v4/relaxed_main_table.md)

备注：
- `MAT` 与其他官方模型保留为扩展接入能力，但不替代 v4.0 正式核心模型集。
- 当前 rebuilt release 中，`MAML` 应视为 exploratory compatibility model，而不是 strongest final-claim 套件成员。

### 17. H1 / H2 / H3 claim rules

状态：`partial`

实现：
- `H1` model-set 规则、`H2` shortcut/collapse 规则、`H3` intervention 规则均已实现。
- CLI 已支持 hypothesis validation。
- release 已新增 claim summary，对 formal claim / supported trend / exploratory 的发布口径进行了显式收口。

证据：
- 代码：[`src/fsmol_cliff/hypotheses.py`](./src/fsmol_cliff/hypotheses.py)
- 代码：[`src/fsmol_cliff/cli.py`](./src/fsmol_cliff/cli.py)
- 测试：[`tests/test_hypotheses_validation.py`](./tests/test_hypotheses_validation.py)
- 发布：[`outputs/fsmol_cliff_release_v4/relaxed_claim_summary.md`](./outputs/fsmol_cliff_release_v4/relaxed_claim_summary.md)

备注：
- 规则实现主线已经存在。
- 当前 release outcome 是：`H2` 达到 formal claim；`H1/H3` 仍是 supported trend。
- 但代码中的 validator 更接近“协议精神的程序化近似”，并不是对 section 17 checklist 的逐项字面强约束。

### 18. 发布资产

状态：`completed`

实现：
- v4.0 release 目录已包含 manifest、strict/relaxed task lists、strict/relaxed standard+adversarial episode bundles。
- assay-level pair / hard-negative / molecule annotation 资产已发布。
- attrition audit、threshold sensitivity、decision note 已发布。
- task-level结果、aggregate、release artifact 文档均已生成。
- release 目录现已新增 reproducibility manifest，指向生成与聚合脚本入口。

证据：
- 目录：[`outputs/fsmol_cliff_release_v4`](./outputs/fsmol_cliff_release_v4)
- 代码：[`src/fsmol_cliff/release.py`](./src/fsmol_cliff/release.py)
- 代码：[`src/fsmol_cliff/release_artifacts.py`](./src/fsmol_cliff/release_artifacts.py)
- 测试：[`tests/test_release.py`](./tests/test_release.py)
- 测试：[`tests/test_release_artifacts.py`](./tests/test_release_artifacts.py)
- 测试：[`tests/test_release_migration_compat.py`](./tests/test_release_migration_compat.py)

备注：
- 当前 release 目录已经满足本仓库采用的正式发布资产要求。
- 生成与聚合脚本仍位于仓库源码目录中，但 `release_reproducibility.md` 已显式给出入口与命令。

### 19. v4.0 的总定位

状态：`completed`

实现：
- strict / relaxed 的双重定位已经写入 decision note、release summary、claim summary。
- 主解释框架已明确要求 ranking-layer 与 decision-layer 分开看。
- `ProtoNet` 已成为正式 few-shot baseline 套件核心成员，而非附加项。

证据：
- 文档：[`outputs/fsmol_cliff_release_v4/benchmark_decision_note.md`](./outputs/fsmol_cliff_release_v4/benchmark_decision_note.md)
- 文档：[`outputs/fsmol_cliff_release_v4/release_summary.md`](./outputs/fsmol_cliff_release_v4/release_summary.md)
- 文档：[`outputs/fsmol_cliff_release_v4/relaxed_claim_summary.md`](./outputs/fsmol_cliff_release_v4/relaxed_claim_summary.md)
- 发布：[`outputs/fsmol_cliff_release_v4/relaxed_model_comparisons_paper.md`](./outputs/fsmol_cliff_release_v4/relaxed_model_comparisons_paper.md)

备注：
- 本章的定位已经不只存在于 `spec.md`，也已经进入 release-facing 文档。

## 附录：过滤后的全仓库目录树

过滤规则：
- 隐藏目录：折叠显示，不展开
- cache 目录：折叠显示，不展开
- 数据与大体积生成目录：折叠显示，不展开
- 其余源码、测试、文档和 vendored runtime support 目录按当前仓库状态写入

```text
.
├── .git/ [collapsed hidden]
├── .pytest_cache/ [collapsed cache]
├── checkpoints/ [collapsed data-heavy]
├── fs-mol/ [collapsed data-heavy]
├── outputs/
│   ├── fsmol_cliff_release/ [collapsed generated artifacts]
│   ├── fsmol_cliff_release_run2/ [collapsed generated artifacts]
│   ├── fsmol_cliff_release_run3/ [collapsed generated artifacts]
│   └── fsmol_cliff_release_v4/ [collapsed generated artifacts]
├── paper_latex/
│   └── notes/
│       ├── cotext.md
│       └── strict-vs-relaxed-decision-2026-03-20.md
├── src/
│   └── fsmol_cliff/
│       ├── __pycache__/ [collapsed cache]
│       ├── __init__.py
│       ├── adapters.py
│       ├── aggregate.py
│       ├── assets.py
│       ├── audit.py
│       ├── benchmark.py
│       ├── chem.py
│       ├── cli.py
│       ├── constants.py
│       ├── episodes.py
│       ├── evaluation.py
│       ├── fetch.py
│       ├── fsmol_bridge.py
│       ├── hypotheses.py
│       ├── io.py
│       ├── maml_legacy.py
│       ├── maml_legacy_runner.py
│       ├── manifests.py
│       ├── metrics.py
│       ├── models.py
│       ├── pipeline.py
│       ├── protonet_runner.py
│       ├── release.py
│       ├── release_artifacts.py
│       ├── reports.py
│       ├── runner.py
│       └── task_selection.py
├── tests/
│   ├── __pycache__/ [collapsed cache]
│   ├── test_adapters.py
│   ├── test_assets.py
│   ├── test_attrition_audit.py
│   ├── test_baseline_adapter_runtime.py
│   ├── test_bootstrap.py
│   ├── test_bridge.py
│   ├── test_cli_commands.py
│   ├── test_episodes.py
│   ├── test_evaluation_runner.py
│   ├── test_fetch.py
│   ├── test_fs_mol_compat.py
│   ├── test_hypotheses_validation.py
│   ├── test_maml_legacy.py
│   ├── test_maml_legacy_runner.py
│   ├── test_manifest_io.py
│   ├── test_manifests.py
│   ├── test_metrics.py
│   ├── test_models.py
│   ├── test_official_adapters.py
│   ├── test_pipeline.py
│   ├── test_protonet_runner.py
│   ├── test_release.py
│   ├── test_release_artifacts.py
│   ├── test_release_evaluation.py
│   ├── test_release_migration_compat.py
│   ├── test_reports.py
│   └── test_task_selection.py
├── vendor/
│   └── MAT/
│       ├── .git/ [collapsed hidden]
│       ├── assets/
│       │   ├── MAT.png
│       │   ├── results_150.png
│       │   ├── results_500.png
│       │   └── results_pretrained.png
│       ├── data/ [collapsed data-heavy]
│       ├── src/
│       │   ├── __pycache__/ [collapsed cache]
│       │   ├── featurization/
│       │   │   ├── __pycache__/ [collapsed cache]
│       │   │   └── data_utils.py
│       │   ├── transformer.py
│       │   └── utils.py
│       ├── EXAMPLE.ipynb
│       ├── LICENSE
│       └── README.md
├── AGENTS.md
├── pyproject.toml
├── README.md
├── spec.md
└── spec_f.md
```
