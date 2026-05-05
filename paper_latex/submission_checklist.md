# CliffBench v5 — Submission Checklist

Date: 2026-05-05 | Status: Evidence consolidation complete

---

## Final Claim Hierarchy

| ID | Claim | Status | Key Evidence |
|----|-------|--------|-------------|
| **FC1** | CliffBench v5 is a valid activity-cliff diagnostic benchmark derived from FS-Mol | **formal claim** | 157 raw assays -> 6-10 eligible tasks under controlled thresholds; deterministic reproducibility; threshold sensitivity analysis |
| **FC2** | FS-Mol baselines exhibit ranking-decision mismatch / decision-layer collapse | **formal claim** | RF SQ-PSR=0.917 but C-BAcc=0.524, SCR=0.912; ProtoNet most balanced (C-BAcc=0.561, SCR=0.843); SS-SCR exceeds SCR across all models |
| **FC3** | 22 shallow interventions fail the stronger-baseline gate without clean repair | **formal claim** | 0/22 pass stronger-baseline gate; most common failures: control-side harm, vanilla-only wins, structural no-effect, insufficient supervision |
| **SC** | MoleculeACE provides external pair-level supporting evidence, not direct replication | **external supporting evidence** | RF Q-PSR > kNN Q-PSR in 22/25 targets (88%); C-BAcc nearly identical; SCR directionally consistent; pair-level protocol differs from FS-Mol episodes |

## Allowed Claims

- CliffBench exposes ranking-decision mismatch in few-shot molecular classification
- Systematic intervention audit shows no shallow repair passes stronger-baseline gate
- MoleculeACE provides external supporting evidence consistent with ranking-decision decoupling
- The current evidence package supports a diagnostic benchmark manuscript

## Forbidden Claims

- "MoleculeACE directly replicates FS-Mol" — NO (pair-level, not episode-based)
- "CASE-Net improves activity-cliff prediction" — NO (v1/v2 both NO-GO)
- "We solve activity-cliff few-shot prediction" — NO (no successful method)
- "Activity cliffs are proven to require 3D protein-ligand modeling" — NO (not tested)
- "2D molecular features alone cannot predict cliffs" — NO (only tested specific descriptors)
- "JCIM-ready" — NO (use "suitable for JCIM-style submission subject to review")

---

## Main Paper Tables (4)

| Table | Title | Key numbers |
|-------|-------|-------------|
| Table 1 | Attrition funnel and v5 profile coverage | 157->6-10 tasks; 4 profiles; bottleneck: high-sim discordant support |
| Table 2 | FS-Mol v5 baseline results (ext_10_10, adversarial) | 4 models × 6 metrics; task-level macro mean; 10k bootstrap CI |
| Table 3 | Hypothesis and diagnostic evidence summary | FC1-FC3 + SC; per-hypothesis primary evidence and limitations |
| Table 4 | MoleculeACE external pair-level diagnostic | 30 targets; kNN vs RF; 25 with test cliff pairs; 2k bootstrap CI |

## Supplement Tables (5)

| Table | Title |
|-------|-------|
| S1 | 22-family intervention registry with per-family details |
| S2 | CASE-Net v1/v2 pair-level diagnostic details |
| S3 | MoleculeACE per-target results (30 rows) |
| S4 | Reproducibility checklist |
| S5 | Threshold sensitivity and attrition audit details |

---

## Result File Paths

| Artifact | Path |
|----------|------|
| V5 release (all profiles) | `outputs/fsmol_cliff_release_v5/` |
| V5 kNN results (relaxed) | `outputs/fsmol_cliff_release_v5/task_results_kNN_relaxed.parquet` |
| V5 kNN results (ext_10_10) | `outputs/fsmol_cliff_release_v5/task_results_kNN_relaxed_covext_10_10.parquet` |
| V5 kNN-cliff-aware results (relaxed) | `outputs/fsmol_cliff_release_v5/task_results_kNN-cliff-aware_relaxed.parquet` |
| V5 kNN-cliff-aware results (ext_10_10) | `outputs/fsmol_cliff_release_v5/task_results_kNN-cliff-aware_relaxed_covext_10_10.parquet` |
| V5 randomForest results (relaxed) | `outputs/fsmol_cliff_release_v5/task_results_randomForest_relaxed.parquet` |
| V5 randomForest results (ext_10_10) | `outputs/fsmol_cliff_release_v5/task_results_randomForest_relaxed_covext_10_10.parquet` |
| V5 ProtoNet results (relaxed) | `outputs/fsmol_cliff_release_v5/task_results_protonet_relaxed.parquet` |
| V5 ProtoNet results (ext_10_10) | `outputs/fsmol_cliff_release_v5/task_results_protonet_relaxed_covext_10_10.parquet` |
| MoleculeACE v2 results | `outputs/moleculeace_validation/moleculeace_results_v2.json` |
| MoleculeACE per-target CSV | `outputs/moleculeace_validation/moleculeace_per_target_v2.csv` |
| MoleculeACE summary CSV | `outputs/moleculeace_validation/moleculeace_summary_v2.csv` |
| MoleculeACE bootstrap JSON | `outputs/moleculeace_validation/moleculeace_bootstrap_v2.json` |
| CASE-Net v2 pair-level report | `outputs/case_net_v2/pair_level_report.json` |

---

## Metric Definitions

### FS-Mol episode-level metrics

| Metric | Definition |
|--------|-----------|
| C-BAcc | Balanced accuracy over cliff query molecules: 0.5 * (TPR_cliff + TNR_cliff) |
| NC-BAcc | Balanced accuracy over noncliff query molecules |
| SCR | Score Collapse Rate: fraction of (anchor, cliff-negative) pairs where both molecules get the same binary prediction |
| SS-SCR | SCR restricted to same-scaffold pairs |
| Q-PSR | Query Pair Success Rate: fraction of query-positive/query-negative pairs correctly ranked |
| SQ-PSR | Support-Query PSR: fraction of support-positive/query-negative pairs correctly ranked |
| NC-PSR | Noncliff PSR: Q-PSR restricted to noncliff pairs |
| SS-Q-PSR | Same-scaffold Q-PSR |
| SS-SQ-PSR | Same-scaffold SQ-PSR |

### MoleculeACE pair-level metrics (v2 corrected)

| Metric | Definition |
|--------|-----------|
| C-BAcc | mean over cliff pairs: 0.5 * (1(active_pred==1) + 1(inactive_pred==0)) |
| NC-BAcc | mean over highsim_noncliff pairs: 0.5 * (1(active_pred==1) + 1(inactive_pred==0)) |
| SCR | fraction of high-sim test pairs with same binary prediction |
| Q-PSR | fraction of high-sim test pairs with active_score > inactive_score |
| NC-PSR | Q-PSR restricted to noncliff pairs |
| C-ActiveAcc | fraction of cliff-pair active molecules predicted active (legacy one-sided) |
| NC-InactiveAcc | fraction of noncliff-pair inactive molecules predicted inactive (legacy one-sided) |

---

## MoleculeACE Protocol Caveats

1. **Protocol**: Pair-level train/test evaluation, not few-shot episode sampling.
2. **Labels**: Median-split binarization per target, not assay-native binary labels.
3. **Models**: kNN (k=5) and RF (500 trees) on Morgan fingerprints only. ProtoNet not evaluated (graph features unavailable).
4. **Pair availability**: Only 25/30 targets have >=1 test cliff pair at tau=0.80; 8/30 have >=5. Macro means are driven by 8-13 targets.
5. **tau matching**: Uses FS-Mol v5 tau=0.80. MoleculeACE's own cliff definition uses tau=0.90.
6. **Interpretation**: External supporting evidence, not direct replication of FS-Mol few-shot protocol.

---

## Reproducibility Commands

```bash
# Full test suite
python -m pytest tests/ \
  --deselect tests/test_bootstrap.py::test_cli_exposes_expected_top_level_subcommands -q
# Expected: 179 passed, 1 deselected

# Build v5 release (any profile)
PYTHONPATH=src python -m fsmol_cliff.cli build-release \
  --data-dir fs-mol --output-dir outputs/fsmol_cliff_release_v5 \
  --profile relaxed --seeds "[0, 1, 2, 3, 4]" --episodes-per-split 400

# Run baseline evaluation (example: kNN on relaxed)
PYTHONPATH=src python -m fsmol_cliff.cli evaluate \
  --release-dir outputs/fsmol_cliff_release_v5 --profile relaxed \
  --backend local --model-name kNN \
  --split-types '["standard","adversarial"]' \
  --output outputs/fsmol_cliff_release_v5/task_results_kNN_relaxed.parquet

# Aggregate
PYTHONPATH=src python -m fsmol_cliff.cli aggregate \
  --input <parquet> --output <json>

# MoleculeACE external validation
PYTHONPATH=src python -m fsmol_cliff.moleculeace_eval \
  <MoleculeACE data dir> outputs/moleculeace_validation

# CASE-Net v2 pair-level (for supplement only)
PYTHONPATH=src python -m fsmol_cliff.case_train_v2 \
  --data-dir fs-mol --max-train 500 --max-valid 100 \
  --output-dir outputs/case_net_v2
```

---

## Commit Hashes (evidence chain)

```
fda22f2 docs: freeze claim hierarchy, add manuscript outline and Results draft
084e7ee chore: cleanup MoleculeACE bootstrap seed, CASE-Net naming, historical notes
7d3ad6a docs: revise paper tables with corrected metrics and cautious wording
6a90093 fix: correct MoleculeACE pair-level metrics and clean CASE-Net naming
0f83416 fix: P1 reproducibility hardening
c6a7643 fix: skip torch_scatter compat patch when native CUDA scatter is available
594662f perf: cache assay molecular neighbor index across episodes
9a3a78f fix: reorder upstream patch registry so leaf modules load first
db1542e refactor: harden FS-Mol bridge and eliminate runner/protonet_runner duplication
```

---

## Remaining Limitations Before Submission

| Limitation | Severity | Mitigation |
|-----------|----------|------------|
| core_strict and ext_10_5 not evaluated | Low | Declared as sensitivity profiles; episodes released |
| MAML not evaluated on v5 | Low | Declared exploratory in paper |
| 6-10 FS-Mol tasks is narrow | Medium | Acknowledged in Discussion; MoleculeACE provides external evidence |
| MoleculeACE is pair-level, not episode-based | Medium | Documented as protocol difference; framed as external supporting evidence |
| No successful repair method | Medium | 22-family audit documented; framed as systematic negative result |
| torch.load(weights_only=False) | Low | Mitigated by checkpoint hash-lock; noted in reproducibility supplement |
| MoleculeACE v2 results not independently reproduced | Medium | All code and data sources documented; deterministic bootstrap seed (42) |
| ProtoNet checkpoint dependency on FS-Mol external repo | Low | Hash-locked; environment variable override documented |
| Hardcoded FS-Mol path in MAML runner (legacy) | Low | Fixed for main pipeline; MAML path still needs attention for full repro |
