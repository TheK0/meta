# Experiment Summary

Date: 2026-03-24

This note summarizes the main lines of work attempted after the initial `v4.0` benchmark/release paper state, what was changed in code, what was only evaluated locally under `outputs/`, what was reverted, and what conclusions currently hold.

## 1. Current Repository State

### Stable committed state

The repository currently preserves the following committed benchmark-side upgrades:

- `21ea3cc feat: add coverage-extension profile support`
- `6cbfd43 fix: make release artifacts tier-safe`
- `b2ba168 docs: freeze final release policy`
- `bc10d10 docs: record coverage-extension evaluation status`
- `f07cf30 Revert "feat: add decision-aware kNN backend"`

Interpretation:

- coverage-extension infrastructure is committed and usable
- release artifacts are tier-safe
- the final benchmark identity remains frozen
- the earlier `decision-aware` method experiment was explicitly reverted

### Current uncommitted state

At the moment of writing, the worktree is not fully clean. The only active code-level experiment left in the worktree is the start of the episode-protocol route:

- modified: `src/fsmol_cliff/manifests.py`
- modified: `tests/test_manifests.py`
- untracked: `docs/superpowers/plans/2026-03-24-training-episode-protocol-go-no-go-plan.md`
- untracked: `paper_latex/notes/training-episode-protocol-status.md`

Interpretation:

- no decision-layer repair code remains in the working tree
- the current active direction is episode-construction work

## 2. Baseline Paper Identity Before Method Experiments

Rollback-safe paper identity:

- stronger diagnostic benchmark paper
- final benchmark substrate: `outputs/fsmol_cliff_release_v4`
- claim anchor:
  - `H1`: supported trend
  - `H2`: formal claim
  - `H3`: supported trend

Primary evidence:

- [`outputs/fsmol_cliff_release_v4/release_summary.md`](./outputs/fsmol_cliff_release_v4/release_summary.md)
- [`outputs/fsmol_cliff_release_v4/relaxed_claim_summary.md`](./outputs/fsmol_cliff_release_v4/relaxed_claim_summary.md)
- [`outputs/fsmol_cliff_release_v4/relaxed_numeric_summary.md`](./outputs/fsmol_cliff_release_v4/relaxed_numeric_summary.md)

Interpretation:

- all later routes should be judged relative to this stable anchor
- no later experiment should invalidate this landing point

## 3. Coverage-Extension Program

### What changed

Committed code changes:

- auxiliary profiles were added for coverage-extension
- CLI profile handling was generalized
- audit threshold inference was made generic for arbitrary registered profiles
- release artifact builders were made `result_tier`-safe

Relevant files:

- [`src/fsmol_cliff/constants.py`](./src/fsmol_cliff/constants.py)
- [`src/fsmol_cliff/cli.py`](./src/fsmol_cliff/cli.py)
- [`src/fsmol_cliff/audit.py`](./src/fsmol_cliff/audit.py)
- [`src/fsmol_cliff/release_artifacts.py`](./src/fsmol_cliff/release_artifacts.py)

### Intermediate release built

Main intermediate release:

- [`outputs/fsmol_cliff_release_v4_covext_intermediate`](./outputs/fsmol_cliff_release_v4_covext_intermediate)

Key coverage change from final relaxed to intermediate `relaxed_covext_10_10`:

- eligible assays: `6 -> 10`
- adversarial-eligible assays: `6 -> 10`
- total cliff pairs: `325 -> 407`
- total anchors: `200 -> 268`
- same-scaffold cliff pairs: `171 -> 229`

Primary evidence:

- [`outputs/fsmol_cliff_release_v4/audit/relaxed/attrition_summary.json`](./outputs/fsmol_cliff_release_v4/audit/relaxed/attrition_summary.json)
- [`outputs/fsmol_cliff_release_v4/audit/relaxed/threshold_sensitivity.parquet`](./outputs/fsmol_cliff_release_v4/audit/relaxed/threshold_sensitivity.parquet)
- [`outputs/fsmol_cliff_release_v4_covext_intermediate/audit/relaxed_covext_10_10/attrition_summary.json`](./outputs/fsmol_cliff_release_v4_covext_intermediate/audit/relaxed_covext_10_10/attrition_summary.json)
- [`outputs/fsmol_cliff_release_v4_covext_intermediate/audit/relaxed_covext_10_10/threshold_sensitivity.parquet`](./outputs/fsmol_cliff_release_v4_covext_intermediate/audit/relaxed_covext_10_10/threshold_sensitivity.parquet)

### Main conclusion

Result: `GO` as an evidence-strengthening layer.

Interpretation:

- coverage expansion was real, not cosmetic
- adversarial-eligible support increased in lockstep
- cliff signal was not diluted
- the intermediate release should be retained as an appendix / extended-table robustness layer
- the final relaxed release should still remain the only `final` substrate

## 4. Full-Strength Intermediate Benchmark Result

Intermediate full-strength model set:

- `kNN`
- `RF`
- `ProtoNet`
- `kNN-cliff-aware`

Aggregate snapshot:

- `kNN`
  - standard: `c_bacc=0.524372`, `q_psr=0.553892`, `scr=0.912433`
  - adversarial: `c_bacc=0.510799`, `sq_psr=0.561747`, `scr=0.905575`
- `RF`
  - standard: `c_bacc=0.519904`, `q_psr=0.636408`, `scr=0.883934`
  - adversarial: `c_bacc=0.523557`, `sq_psr=0.916869`, `scr=0.911728`
- `ProtoNet`
  - standard: `c_bacc=0.527575`, `q_psr=0.583579`, `scr=0.829293`
  - adversarial: `c_bacc=0.561089`, `sq_psr=0.786000`, `scr=0.842863`
- `kNN-cliff-aware`
  - standard: `c_bacc=0.530921`, `q_psr=0.565363`, `scr=0.889642`
  - adversarial: `c_bacc=0.533033`, `sq_psr=0.572072`, `scr=0.844232`

Primary evidence:

- [`outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_relaxed_covext_10_10.aggregate.json`](./outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_relaxed_covext_10_10.aggregate.json)
- [`outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_rf_relaxed_covext_10_10.aggregate.json`](./outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_rf_relaxed_covext_10_10.aggregate.json)
- [`outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_protonet_relaxed_covext_10_10.aggregate.json`](./outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_protonet_relaxed_covext_10_10.aggregate.json)
- [`outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_cliff_aware_relaxed_covext_10_10.aggregate.json`](./outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_cliff_aware_relaxed_covext_10_10.aggregate.json)

### Main benchmark-level conclusions from the intermediate release

- `H2` remained supported on the expanded substrate
- `H3` became materially stronger
- `H1` still should not be upgraded

Evidence:

- `kNN -> RF`
  - adversarial `SQ-PSR`: `+0.355122`, CI `[0.313384, 0.385869]`
  - adversarial `C-BAcc`: `+0.012759`, CI `[-0.014416, 0.056074]`
  - adversarial `SCR`: `+0.006153`, CI `[-0.026304, 0.042388]`
- `kNN -> kNN-cliff-aware`
  - adversarial `C-BAcc`: `+0.022234`, CI `[0.002132, 0.045002]`
  - adversarial `SCR`: `-0.061343`, CI `[-0.108198, -0.021282]`
  - adversarial `NC-PSR`: `+0.037420`, CI `[0.013171, 0.060661]`

Interpretation:

- ranking--decision decoupling remains real under larger coverage
- the intervention signal becomes stronger
- control-side evidence does not reveal obvious hidden damage

## 5. Decision-Layer Repair Attempts

All of these were exploratory local method attempts on top of the intermediate substrate.

### 5.1 `decision-aware` threshold repair

Code status:

- implemented and committed as `8bea0e0`
- then explicitly reverted by [`f07cf30`](./.git/COMMIT_EDITMSG) via `git revert`

Result: `NO-GO`

Main evidence versus `kNN-cliff-aware`:

- adversarial `C-BAcc`: `+0.005857`, CI `[-0.012311, 0.028371]`
- adversarial `SCR`: `+0.017833`, CI `[-0.002699, 0.041030]`
- adversarial `NC-BAcc`: `+0.000700`, CI `[-0.011557, 0.012649]`

Interpretation:

- threshold-only repair did not improve the key adversarial decision/collapse targets cleanly
- it therefore failed the Phase 2A gate

Evidence source:

- [`outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_decision_aware_relaxed_covext_10_10.aggregate.json`](./outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_decision_aware_relaxed_covext_10_10.aggregate.json)

### 5.2 `local-boundary-repair`

Code status:

- run locally as an experiment
- not retained in the current working tree

Result: `NO-GO`

Main evidence versus `kNN-cliff-aware`:

- adversarial `C-BAcc`: `+0.025943`, CI `[-0.037967, 0.099965]`
- adversarial `SCR`: `-0.003136`, CI `[-0.065342, 0.066683]`
- adversarial `NC-BAcc`: `-0.053019`, CI `[-0.124137, 0.000497]`

Interpretation:

- standard collapse improved
- but the adversarial repair signal did not cleanly pass
- control-side moved in the wrong direction

Evidence source:

- [`outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_local_boundary_repair_relaxed_covext_10_10.aggregate.json`](./outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_local_boundary_repair_relaxed_covext_10_10.aggregate.json)

### Overall repair conclusion

The direct decision-layer repair line did not produce a protocol-worthy repair family.

Interpretation:

- benchmark diagnosis remained useful
- repair validation remained useful
- but the attempted repair methods did not justify upgrading the paper into a benchmark-plus-repair-method paper

## 6. Support-Protocol Attempts

These were protocol-level changes on the intermediate substrate, evaluated against the stronger `kNN-cliff-aware` baseline.

### 6.1 `fixed-support-hard-negative-replacement`

Result: `NO-GO`

Evidence versus `kNN-cliff-aware`:

- adversarial `C-BAcc`: `-0.001771`, CI `[-0.013826, 0.012403]`
- adversarial `SQ-PSR`: `-0.003784`, CI `[-0.005722, -0.001503]`
- adversarial `SCR`: `+0.014204`, CI `[0.000884, 0.028715]`
- adversarial `SS-SCR`: `+0.022139`, CI `[0.007247, 0.037068]`

Interpretation:

- support-negative replacement was too aggressive
- it degraded the ranking/collapse balance of the stronger baseline

Evidence source:

- [`outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_support_replacement_relaxed_covext_10_10.aggregate.json`](./outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_support_replacement_relaxed_covext_10_10.aggregate.json)

### 6.2 `partial-hard-negative-augmentation`

Result: `NO-GO`

Evidence versus `kNN-cliff-aware`:

- standard `C-BAcc`: `-0.004637`, CI `[-0.008669, -0.001252]`
- standard `SCR`: `+0.006306`, CI `[0.001520, 0.010964]`
- adversarial `C-BAcc`: `-0.011732`, CI `[-0.028682, 0.004171]`
- adversarial `SQ-PSR`: `-0.010528`, CI `[-0.021050, -0.003241]`
- adversarial `SCR`: `+0.014638`, CI `[-0.001143, 0.033016]`
- adversarial `SS-SCR`: `+0.016551`, CI `[0.002319, 0.033547]`

Interpretation:

- even the more conservative support protocol still degraded the stronger support-side baseline
- this suggests that directly manipulating support negatives is not the most promising protocol family

Evidence source:

- [`outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_partial_hard_negative_augmentation_relaxed_covext_10_10.aggregate.json`](./outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_partial_hard_negative_augmentation_relaxed_covext_10_10.aggregate.json)

### Support-protocol conclusion

- support protocol clearly matters relative to vanilla `kNN`
- but neither tested support protocol beat `kNN-cliff-aware`
- therefore the support-protocol line, as currently instantiated, should be treated as `NO-GO`

## 7. Episode-Construction Attempt

### 7.1 `query-targeted support negatives`

Code status:

- builder/manifests implementation is currently uncommitted in the working tree
- the originally evaluated directory
  [`outputs/fsmol_cliff_release_v4_covext_intermediate_query_targeted_support_neg`](./outputs/fsmol_cliff_release_v4_covext_intermediate_query_targeted_support_neg)
  should now be treated as a pre-correction local artifact
- root-cause note:
  - the earlier local implementation changed the adversarial RNG namespace and therefore resampled the full adversarial episode skeleton
  - that behavior did not match the intended definition of this variant
- corrected release variant:
  [`outputs/fsmol_cliff_release_v4_covext_intermediate_query_targeted_support_neg_corrected`](./outputs/fsmol_cliff_release_v4_covext_intermediate_query_targeted_support_neg_corrected)

Result:

- relative to vanilla intermediate `kNN`: `GO`
- relative to `kNN-cliff-aware`: `NO-GO`

Evidence versus vanilla intermediate `kNN` on the corrected release:

- adversarial `C-BAcc`: `+0.028222`, CI `[0.000352, 0.063484]`
- adversarial `SCR`: `-0.050479`, CI `[-0.087093, -0.019716]`
- adversarial `SS-SCR`: `-0.055465`, CI `[-0.099378, -0.020178]`
- adversarial `NC-BAcc`: `+0.016291`, CI `[0.000626, 0.033399]`
- adversarial `NC-PSR`: `+0.036526`, CI `[0.016899, 0.059819]`
- adversarial `SQ-PSR`: `+0.006837`, CI `[-0.009844, 0.026247]`
- standard `C-BAcc`: `+0.000000`, CI `[0.000000, 0.000000]`
- standard `SCR`: `+0.000000`, CI `[0.000000, 0.000000]`

Evidence versus `kNN-cliff-aware` on the corrected release:

- adversarial `C-BAcc`: `+0.005988`, CI `[-0.007839, 0.025257]`
- adversarial `SCR`: `+0.010864`, CI `[-0.008886, 0.030524]`
- adversarial `SQ-PSR`: `-0.003488`, CI `[-0.008147, 0.001516]`
- adversarial `SS-SCR`: `+0.020048`, CI `[-0.008250, 0.053903]`
- adversarial `NC-BAcc`: `-0.003515`, CI `[-0.023699, 0.015919]`
- adversarial `NC-PSR`: `-0.000894`, CI `[-0.023195, 0.021324]`

Interpretation:

- episode construction matters more than the failed support-protocol variants
- after correcting the implementation, the variant still remains positive versus vanilla `kNN`
- the corrected implementation is cleaner because it leaves the baseline adversarial skeleton unchanged and only rewrites `support_neg_ids`
- however, the current variant still does not surpass the stronger `kNN-cliff-aware` baseline

Evidence sources:

- [`outputs/fsmol_cliff_release_v4_covext_intermediate_query_targeted_support_neg_corrected/task_results_knn_relaxed_covext_10_10.aggregate.json`](./outputs/fsmol_cliff_release_v4_covext_intermediate_query_targeted_support_neg_corrected/task_results_knn_relaxed_covext_10_10.aggregate.json)
- [`outputs/fsmol_cliff_release_v4_covext_intermediate_query_targeted_support_neg_corrected/paired_comparison_knn_and_cliff_aware_vs_query_targeted_support_neg_corrected.json`](./outputs/fsmol_cliff_release_v4_covext_intermediate_query_targeted_support_neg_corrected/paired_comparison_knn_and_cliff_aware_vs_query_targeted_support_neg_corrected.json)
- [`outputs/fsmol_cliff_release_v4_covext_intermediate_query_targeted_support_neg_corrected/episode_protocol_note.md`](./outputs/fsmol_cliff_release_v4_covext_intermediate_query_targeted_support_neg_corrected/episode_protocol_note.md)
- [`src/fsmol_cliff/manifests.py`](./src/fsmol_cliff/manifests.py)
- [`tests/test_manifests.py`](./tests/test_manifests.py)

### 7.2 `same_scaffold_query_targeted`

Code status:

- implemented as an explicit adversarial episode variant in the working tree
- evaluated on:
  [`outputs/fsmol_cliff_release_v4_covext_intermediate_same_scaffold_query_targeted`](./outputs/fsmol_cliff_release_v4_covext_intermediate_same_scaffold_query_targeted)

Result:

- relative to vanilla intermediate `kNN`: `NO-GO`
- relative to `kNN-cliff-aware`: `NO-GO`

Evidence versus vanilla intermediate `kNN`:

- adversarial `C-BAcc`: `+0.002586`, CI `[-0.050017, 0.058396]`
- adversarial `SQ-PSR`: `+0.013816`, CI `[-0.017866, 0.055753]`
- adversarial `SCR`: `+0.008893`, CI `[-0.001459, 0.021869]`
- adversarial `SS-SCR`: `+0.007664`, CI `[-0.004500, 0.022456]`

Evidence versus `kNN-cliff-aware`:

- adversarial `C-BAcc`: `-0.019648`, CI `[-0.073260, 0.027052]`
- adversarial `SCR`: `+0.070236`, CI `[0.033092, 0.110647]`
- adversarial `SS-SCR`: `+0.083176`, CI `[0.040950, 0.128621]`
- adversarial `NC-BAcc`: `-0.019472`, CI `[-0.037618, -0.000795]`

Interpretation:

- preferring same-scaffold injected cliff pairs did not produce a cleaner boundary signal
- collapse got materially worse
- this is a hard `NO-GO`

Evidence sources:

- [`outputs/fsmol_cliff_release_v4_covext_intermediate_same_scaffold_query_targeted/task_results_knn_relaxed_covext_10_10.aggregate.json`](./outputs/fsmol_cliff_release_v4_covext_intermediate_same_scaffold_query_targeted/task_results_knn_relaxed_covext_10_10.aggregate.json)
- [`outputs/fsmol_cliff_release_v4_covext_intermediate_same_scaffold_query_targeted/paired_comparison.json`](./outputs/fsmol_cliff_release_v4_covext_intermediate_same_scaffold_query_targeted/paired_comparison.json)

### 7.3 `anchor_coverage_first`

Code status:

- implemented as an explicit adversarial episode variant in the working tree
- evaluated on:
  [`outputs/fsmol_cliff_release_v4_covext_intermediate_anchor_coverage_first`](./outputs/fsmol_cliff_release_v4_covext_intermediate_anchor_coverage_first)

Result:

- relative to vanilla intermediate `kNN`: `NO-GO`
- relative to `kNN-cliff-aware`: `NO-GO`

Evidence versus vanilla intermediate `kNN`:

- adversarial `C-BAcc`: `+0.008886`, CI `[-0.009426, 0.034788]`
- adversarial `SQ-PSR`: `-0.013828`, CI `[-0.032922, 0.004156]`
- adversarial `SCR`: `+0.010092`, CI `[-0.014236, 0.036220]`
- adversarial `SS-SCR`: `+0.008884`, CI `[-0.035377, 0.048836]`

Evidence versus `kNN-cliff-aware`:

- adversarial `C-BAcc`: `-0.013348`, CI `[-0.035446, 0.007235]`
- adversarial `SQ-PSR`: `-0.024153`, CI `[-0.037809, -0.010700]`
- adversarial `SCR`: `+0.071435`, CI `[0.028842, 0.114614]`
- adversarial `SS-SCR`: `+0.084397`, CI `[0.018096, 0.148807]`
- adversarial `NC-PSR`: `-0.036284`, CI `[-0.062545, -0.011332]`

Interpretation:

- prioritizing anchors with larger cliff-negative coverage did not produce a cleaner adversarial episode
- the variant sacrifices ranking and collapse behavior relative to `kNN-cliff-aware`
- this is also a hard `NO-GO`

Evidence sources:

- [`outputs/fsmol_cliff_release_v4_covext_intermediate_anchor_coverage_first/task_results_knn_relaxed_covext_10_10.aggregate.json`](./outputs/fsmol_cliff_release_v4_covext_intermediate_anchor_coverage_first/task_results_knn_relaxed_covext_10_10.aggregate.json)
- [`outputs/fsmol_cliff_release_v4_covext_intermediate_anchor_coverage_first/paired_comparison.json`](./outputs/fsmol_cliff_release_v4_covext_intermediate_anchor_coverage_first/paired_comparison.json)

### 7.4 `paired_hardness_balanced`

Code status:

- implemented as an explicit adversarial episode variant in the working tree
- evaluated on:
  [`outputs/fsmol_cliff_release_v4_covext_intermediate_paired_hardness_balanced`](./outputs/fsmol_cliff_release_v4_covext_intermediate_paired_hardness_balanced)

Result:

- relative to vanilla intermediate `kNN`: `NO-GO`
- relative to `kNN-cliff-aware`: `NO-GO`

Evidence versus vanilla intermediate `kNN`:

- adversarial `C-BAcc`: `+0.018946`, CI `[-0.005416, 0.057247]`
- adversarial `SQ-PSR`: `+0.008525`, CI `[0.001800, 0.016022]`
- adversarial `SCR`: `-0.017176`, CI `[-0.031944, -0.004527]`
- adversarial `SS-SCR`: `-0.018294`, CI `[-0.042195, 0.003123]`

Evidence versus `kNN-cliff-aware`:

- adversarial `C-BAcc`: `-0.003288`, CI `[-0.031345, 0.026435]`
- adversarial `SCR`: `+0.044167`, CI `[0.000824, 0.090941]`
- adversarial `SS-SCR`: `+0.057218`, CI `[-0.000318, 0.115106]`
- adversarial `NC-PSR`: `-0.029329`, CI `[-0.067971, 0.004881]`

Interpretation:

- balancing injected cliff-pair hardness gives a cleaner signal than the same-scaffold and anchor-coverage rules
- but it still fails the stronger-baseline gate because collapse worsens relative to `kNN-cliff-aware`
- this is not a paper-upgrade result

Evidence sources:

- [`outputs/fsmol_cliff_release_v4_covext_intermediate_paired_hardness_balanced/task_results_knn_relaxed_covext_10_10.aggregate.json`](./outputs/fsmol_cliff_release_v4_covext_intermediate_paired_hardness_balanced/task_results_knn_relaxed_covext_10_10.aggregate.json)
- [`outputs/fsmol_cliff_release_v4_covext_intermediate_paired_hardness_balanced/paired_comparison.json`](./outputs/fsmol_cliff_release_v4_covext_intermediate_paired_hardness_balanced/paired_comparison.json)

### 7.5 `query_cluster_separation_by_neg_diversity`

Code status:

- implemented as an explicit adversarial episode variant in the working tree
- evaluated on:
  [`outputs/fsmol_cliff_release_v4_covext_intermediate_query_cluster_separation_by_neg_diversity`](./outputs/fsmol_cliff_release_v4_covext_intermediate_query_cluster_separation_by_neg_diversity)

Result:

- relative to vanilla intermediate `kNN`: `NO-GO`
- relative to `kNN-cliff-aware`: `NO-GO`

Evidence versus vanilla intermediate `kNN`:

- adversarial `C-BAcc`: `+0.016485`, CI `[-0.004774, 0.051254]`
- adversarial `SQ-PSR`: `+0.009369`, CI `[0.001159, 0.018597]`
- adversarial `SCR`: `-0.009923`, CI `[-0.023472, 0.001877]`
- adversarial `SS-SCR`: `-0.009226`, CI `[-0.036427, 0.016747]`

Evidence versus `kNN-cliff-aware`:

- adversarial `C-BAcc`: `-0.005749`, CI `[-0.034720, 0.023516]`
- adversarial `SCR`: `+0.051420`, CI `[0.002789, 0.103138]`
- adversarial `SS-SCR`: `+0.066286`, CI `[0.001244, 0.133241]`

Interpretation:

- avoiding hub negatives produces a slightly cleaner signal than the earlier same-scaffold or anchor-coverage rules
- but the stronger-baseline gate still fails because collapse worsens relative to `kNN-cliff-aware`
- this remains a `NO-GO`

Evidence sources:

- [`outputs/fsmol_cliff_release_v4_covext_intermediate_query_cluster_separation_by_neg_diversity/task_results_knn_relaxed_covext_10_10.aggregate.json`](./outputs/fsmol_cliff_release_v4_covext_intermediate_query_cluster_separation_by_neg_diversity/task_results_knn_relaxed_covext_10_10.aggregate.json)
- [`outputs/fsmol_cliff_release_v4_covext_intermediate_query_cluster_separation_by_neg_diversity/paired_comparison.json`](./outputs/fsmol_cliff_release_v4_covext_intermediate_query_cluster_separation_by_neg_diversity/paired_comparison.json)

### 7.6 `query_cluster_separation_by_anchor_neg_mix`

Code status:

- implemented as an explicit adversarial episode variant in the working tree
- evaluated on:
  [`outputs/fsmol_cliff_release_v4_covext_intermediate_query_cluster_separation_by_anchor_neg_mix`](./outputs/fsmol_cliff_release_v4_covext_intermediate_query_cluster_separation_by_anchor_neg_mix)

Result:

- relative to vanilla intermediate `kNN`: `NO-GO`
- relative to `kNN-cliff-aware`: `NO-GO`

Evidence versus vanilla intermediate `kNN`:

- adversarial `C-BAcc`: `-0.009232`, CI `[-0.027852, 0.006624]`
- adversarial `SQ-PSR`: `-0.001912`, CI `[-0.024263, 0.015919]`
- adversarial `SCR`: `+0.003177`, CI `[-0.007274, 0.014780]`
- adversarial `SS-SCR`: `+0.007910`, CI `[-0.010200, 0.030710]`

Evidence versus `kNN-cliff-aware`:

- adversarial `C-BAcc`: `-0.031466`, CI `[-0.051745, -0.013005]`
- adversarial `SCR`: `+0.064520`, CI `[0.024419, 0.108454]`
- adversarial `SS-SCR`: `+0.083422`, CI `[0.029609, 0.143822]`
- adversarial `NC-PSR`: `-0.026561`, CI `[-0.051750, -0.005222]`

Interpretation:

- mixing high-coverage and low-coverage anchors does not stabilize the query-side perturbation story
- this variant is worse than `query_cluster_separation_by_neg_diversity`
- this is also a hard `NO-GO`

Evidence sources:

- [`outputs/fsmol_cliff_release_v4_covext_intermediate_query_cluster_separation_by_anchor_neg_mix/task_results_knn_relaxed_covext_10_10.aggregate.json`](./outputs/fsmol_cliff_release_v4_covext_intermediate_query_cluster_separation_by_anchor_neg_mix/task_results_knn_relaxed_covext_10_10.aggregate.json)
- [`outputs/fsmol_cliff_release_v4_covext_intermediate_query_cluster_separation_by_anchor_neg_mix/paired_comparison.json`](./outputs/fsmol_cliff_release_v4_covext_intermediate_query_cluster_separation_by_anchor_neg_mix/paired_comparison.json)

## 8. Overall Interpretation

What clearly worked:

- the benchmark and release protocol itself
- coverage-extension as an evidence-strengthening layer
- the diagnostic reading of H2 on larger coverage
- the stronger H3 signal on the intermediate release
- the ability of the protocol/checklist to reject fake or weak repair/protocol variants

What clearly did not work:

- threshold-only decision repair
- nearest-support local boundary patching
- support-negative replacement
- conservative partial hard-negative augmentation

What partially worked:

- query-targeted episode construction

Interpretation:

- the strongest emerging research signal is no longer “repair the decision rule directly”
- it became “episode construction matters, but the current sequential variant sweep still does not beat the strongest simple cliff-aware baseline”
- the only cleanly positive protocol-side signal remains the corrected `query-targeted support negatives` variant versus vanilla `kNN`
- all later variants in the same family failed the stronger-baseline gate

## 9. Best Current Paper Identity

Best current paper identity:

- stronger diagnostic benchmark paper
- with an evidence-strengthening intermediate appendix

Not yet justified:

- benchmark + successful repair method paper
- benchmark + successful training/episode protocol paper

Reason:

- no tested repair/protocol family has yet passed the strongest comparison against the current best local cliff-aware baseline
- the current episode-construction sweep should therefore be treated as exhausted rather than as an open near-miss

## 10. Recommended Next Step

If experimentation continues, the most defensible handling now is:

- stop direct decision-rule repair
- stop further support-negative protocol tweaking
- stop the current episode-construction variant family sweep
- keep the corrected `query-targeted support negatives` result only as appendix / future-work evidence that episode construction is a meaningful axis
- do not port any current episode variant to `ProtoNet` as if it had already cleared the stronger baseline gate
