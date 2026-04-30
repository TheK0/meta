# Method Restart v1 Closing Memo

Date: 2026-04-30

## Scope Lock

This memo formalizes the closeout of the v1 method exploration phase. It does not alter the benchmark protocol or the benchmark paper identity.

## Fixed Benchmark Identity

- **Current paper identity**: stronger diagnostic benchmark paper (unchanged by this closeout)
- **Benchmark main paper is not upgraded** by this closeout — no repair or protocol method has passed the stronger-baseline gate
- **Final substrate**: `outputs/fsmol_cliff_release_v4`
- **Method-development substrate**: `outputs/fsmol_cliff_release_v4_covext_intermediate`
- **Main profile**: `relaxed_covext_10_10`
- **Formal claim anchor**: H2 (ranking--decision decoupling)
- **Supported trends**: H1 (cliff gap), H3 (decision-layer collapse, strengthened on intermediate)
- **All future method work** remains under the stronger-baseline gate

## A1 Closeout

**Family**: Query-only local score refit (logistic regression calibration)

**Result**: NO-GO

**Evidence**: Query-only local calibration showed slight adversarial signal but remained too weak and caused standard-side harm. Specifically, the calibrated scores did not materially improve cliff-side decisions (C-BAcc) over the ProtoNet baseline while degrading non-cliff control metrics.

**Decision**: A1 is closed. Query-only episode-local logistic refit is not a valid expansion entry.

## B0 Closeout

**Family**: Coarse cliff-margin loss injection (ProtoNet training with cliff-aware auxiliary loss)

**Result**: NO-GO

**Evidence**: Minimal cliff-margin loss injection moved the main decision and safety metrics in the wrong direction. The margin penalty did not produce a coherent improvement pattern — collapse metrics either stayed flat or worsened.

**Decision**: B0 is closed. The current margin-loss family (cliff_margin lambda + control_preservation regularizer) is not restartable.

## Expanded C0 Closeout

**Family**: Support-subset-dropout variance-gap entry (perturbation audit as mechanism discovery)

**Result**: NO-GO

**Evidence**: Support-dropout sensitivity gap did not hold after scaling up to full intermediate evaluation. The cliff-vs-control query score variance gap was not stable across tasks. No consistency-training entry exists.

**Decision**: C0 is closed. Support-dropout variance gap does not authorize robustness training.

## Closed Exact Families

The following exact families are **closed** and must not be extended:

| Family | Status |
|--------|--------|
| decision-aware threshold repair | closed |
| local-boundary-repair (kNN nearest-support patch) | closed |
| fixed-support hard-negative replacement | closed |
| partial-hard-negative augmentation | closed |
| query-targeted support negatives (episode construction) | historical evidence only |
| same_scaffold_query_targeted | closed |
| anchor_coverage_first | closed |
| paired_hardness_balanced | closed |
| query_cluster_separation_by_neg_diversity | closed |
| query_cluster_separation_by_anchor_neg_mix | closed |
| A1 (query-only local score refit) | closed |
| B0 (coarse cliff-margin loss injection) | closed |
| C0 (support-subset-dropout variance-gap) | closed |

## Single Remaining Restart Direction

Only one method direction is authorized for restart:

> **Boundary-aware / uncertainty-aware calibration for ProtoNet**

This direction is explicitly different from A1:
- No logistic regression or learned score refit
- No threshold shifting disguised as calibration
- Uses support-conditioned boundary uncertainty features
- Applies deterministic margin shrinkage, not learned re-weighting

## Operational Rules For Next Work

- Do not continue `A2`
- Do not continue `B1` / `B2`
- Do not continue `C1` / `C2` / `C3`
- Old kNN decision / support / episode exact families stay closed
- Only a genuinely new boundary-aware / uncertainty-aware calibration family may restart
- Robustness stays audit-only in the current phase
- Representation work stays frozen by default
- Any new method pilot must apply the stronger-baseline gate (beat `kNN-cliff-aware` minimum, `ProtoNet` for paper upgrade)

## Prerequisite Check Before Any New Method Pilot

- `docs/method_restart_v1_closing_memo.md` exists
- `docs/method_go_no_go_table.md` exists
- `docs/stronger_baseline_gate_failure_patterns.md` exists
- `docs/closed_families_registry.md` exists
- No old family is being extended under a new name
- Any future pilot explicitly maps itself to an allowed reopen rule in `docs/closed_families_registry.md`
