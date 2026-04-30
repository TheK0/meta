# Closed Families Registry

Date: 2026-04-30

## Status Legend

- `closed`: do not extend inside the same family
- `historical evidence`: keep only as reference
- `reopen only with new justification`: requires a separate memo before any experiment

## Registered Closed Families

### A1 family — query-only local score refit
- **Status**: `closed`
- **Description**: Episode-local logistic regression calibration over 6 local features (raw_score, raw_margin, prototype_gap, support_dispersion, cross_class_density, cross_class_cliff_fraction)
- **Failure pattern**: weak directional signal with standard-side harm
- **Evidence**: `outputs/fsmol_cliff_release_v4_covext_intermediate/`

### B0 family — coarse cliff-margin loss injection
- **Status**: `closed`
- **Description**: ProtoNet training with cliff-aware auxiliary loss (λ_cliff * cliff_margin_loss + control_preservation penalty)
- **Failure pattern**: broad wrong-way degradation
- **Evidence**: `configs/protonet_cliff_margin.yaml`

### C0 family — support-subset-dropout variance-gap entry
- **Status**: `closed`
- **Description**: Support molecule dropout perturbation to measure cliff-vs-control query score variance gap
- **Failure pattern**: unstable or non-scaling mechanism signal
- **Evidence**: `src/fsmol_cliff/protonet_perturbation_audit.py`

### decision-aware threshold repair
- **Status**: `closed`
- **Description**: Per-episode decision threshold adjustment based on local cliff density
- **Failure pattern**: weak directional signal with minimal primary movement
- **Evidence**: `outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_decision_aware_relaxed_covext_10_10.aggregate.json`

### local-boundary-repair
- **Status**: `closed`
- **Description**: kNN boundary patching using nearest-support neighbors
- **Failure pattern**: broad wrong-way degradation on control side
- **Evidence**: `outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_local_boundary_repair_relaxed_covext_10_10.aggregate.json`

### fixed-support hard-negative replacement
- **Status**: `closed`
- **Description**: Replace support negatives with hard-negative cliff candidates
- **Failure pattern**: broad wrong-way degradation; too aggressive for ranking/collapse balance
- **Evidence**: `outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_support_replacement_relaxed_covext_10_10.aggregate.json`

### partial-hard-negative augmentation
- **Status**: `closed`
- **Description**: Conservative partial augmentation of support negatives with hard negatives
- **Failure pattern**: beats vanilla only — degraded stronger baseline even with conservative settings
- **Evidence**: `outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_partial_hard_negative_augmentation_relaxed_covext_10_10.aggregate.json`

### query-targeted support negatives (episode construction)
- **Status**: `historical evidence`
- **Description**: Rewrite support_neg_ids using query-cliff-targeted selection in adversarial episodes
- **Failure pattern**: beats vanilla but not the stronger baseline
- **Evidence**: `outputs/fsmol_cliff_release_v4_covext_intermediate_query_targeted_support_neg_corrected/`
- **Note**: Retained as appendix / future-work evidence that episode construction is a meaningful axis, but not restartable as a method direction

### same_scaffold_query_targeted
- **Status**: `closed`
- **Description**: Prefer same-scaffold cliff pairs for adversarial injection
- **Failure pattern**: broad wrong-way degradation; collapse materially worse
- **Evidence**: `outputs/fsmol_cliff_release_v4_covext_intermediate_same_scaffold_query_targeted/`

### anchor_coverage_first
- **Status**: `closed`
- **Description**: Prioritize anchors with largest cliff-negative coverage for injection
- **Failure pattern**: broad wrong-way degradation; sacrificed ranking and collapse
- **Evidence**: `outputs/fsmol_cliff_release_v4_covext_intermediate_anchor_coverage_first/`

### paired_hardness_balanced
- **Status**: `closed`
- **Description**: Balance injected cliff pairs by hardness (similarity + activity gap)
- **Failure pattern**: cleaner than same-scaffold/anchor-coverage but still failed stronger-baseline gate on collapse
- **Evidence**: `outputs/fsmol_cliff_release_v4_covext_intermediate_paired_hardness_balanced/`

### query_cluster_separation_by_neg_diversity
- **Status**: `closed`
- **Description**: Avoid hub negatives in adversarial injection to improve query cluster separation
- **Failure pattern**: collapse still worsened relative to cliff-aware despite cleaner signal
- **Evidence**: `outputs/fsmol_cliff_release_v4_covext_intermediate_query_cluster_separation_by_neg_diversity/`

### query_cluster_separation_by_anchor_neg_mix
- **Status**: `closed`
- **Description**: Mix high-coverage and low-coverage anchors for query-side perturbation
- **Failure pattern**: broad wrong-way degradation; worst variant in episode-construction sweep
- **Evidence**: `outputs/fsmol_cliff_release_v4_covext_intermediate_query_cluster_separation_by_anchor_neg_mix/`

## Prohibited Actions

- Do not continue `A2`
- Do not continue `B1` / `B2`
- Do not continue `C1` / `C2` / `C3`
- Do not port any old episode variant directly to ProtoNet
- Do not continue support-negative tweaking
- Do not return to threshold tricks or decision-rule repair
- Do not reopen any episode-construction exact family under informal reframing
- Do not convert audit findings into training without a separate justification artifact

## Allowed Reopen Rules

- New calibration work is allowed only as a new family distinct from A1
  - Must be boundary-aware / uncertainty-aware (not logistic refit)
  - Must use support-conditioned features, not query-only features
  - Documented in `docs/new_calibration_family_proposal.md`
- Robustness work is allowed only as audit-only mechanism study
  - No training hook authorized
  - Documented in `docs/collapse_specific_perturbation_audit_plan.md`
- Representation work is blocked unless `docs/new_representation_family_justification.md` exists and proves the idea is not a continuation of B0

## Representation Family Freeze

- Default status: frozen
- Do not restart prototype-shaping / margin-loss work from B0
- The only allowed reopen artifact is `docs/new_representation_family_justification.md`
- That memo must prove the proposal is not a margin / lambda / control regularizer continuation of B0
