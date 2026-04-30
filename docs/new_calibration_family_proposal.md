# Boundary-Aware Calibration Family Proposal

Date: 2026-04-30

## Objective

Add a boundary-aware uncertainty calibration mode to ProtoNet that deterministically shrinks raw margins in cliff-sensitive regions without using machine-learned calibration or threshold shifting.

## Why `A1` Was Not Enough

A1 (query-only local score refit) used LogisticRegression trained on 6 local features per episode. It showed slight adversarial signal but:
- Required per-episode model fitting (200 iterations of liblinear)
- Used cross-class cliff density features that depend on assay-level cliff/noncliff pairing
- Showed standard-side harm (clean negative deltas on control metrics)
- The learned refit did not materially improve cliff decisions over the ProtoNet baseline

## Chosen Restart Family

**Uncertainty-aware boundary calibration** — deterministic margin shrinkage based on support-conditioned boundary uncertainty.

Correction form:
```
composite_uncertainty = (local_ambiguity + dispersion_ratio + neighborhood_disagreement) / 3
calibrated_margin = raw_margin * (1 - uncertainty_scale * composite_uncertainty)
```

Key properties:
- No machine learning — fully deterministic
- Only shrinks margin magnitude toward zero; preserves sign
- Identity at zero uncertainty (uncertainty_scale=0 → no change)
- Uncertainty is bounded in [0, 1]

## Local Feature Contract

Only assay-local and episode-local signals are used:

| Feature | Computation | Meaning |
|---------|------------|---------|
| `prototype_margin` | |mean(pos_support_margins) - mean(neg_support_margins)| | Separation between class support centroids |
| `support_dispersion` | pstdev(pos) + pstdev(neg) | Within-class spread of support margins |
| `local_ambiguity` | 1 - min(|margin| / margin_scale, 1) | How close molecule is to decision boundary |
| `neighborhood_disagreement` | fraction of top-K support neighbors with opposite label | Local chemical label inconsistency |

Explicitly rejected:
- Query-only episode-local logistic refit (A1 pattern)
- Decision-rule replacement or threshold shifting
- Cross-assay information or global statistics
- Cross-class density or cliff fraction features (to avoid dependency on assay pair structure)

## Why This Is Not A Threshold Trick

The calibration does not shift a global threshold. It applies per-molecule margin shrinkage proportional to local uncertainty. High-confidence molecules (large |margin|, low local ambiguity) are unchanged. Low-confidence molecules near the boundary in chemically ambiguous neighborhoods are pulled toward 0.5. This is a continuous per-molecule transformation, not a binary decision rule.

## Minimal Pilot Definition

- **Substrate**: `outputs/fsmol_cliff_release_v4_covext_intermediate`
- **Profile**: `relaxed_covext_10_10`
- **Config**: `configs/protonet_boundary_aware_calibration_uncertainty.yaml`
- **Grid**: top_k=[2,4], uncertainty_scale=[0.1,0.2], margin_floor=[0.1]
- **Selected row**: top_k=2, uncertainty_scale=0.1, margin_floor=0.1

## `GO` / `NO-GO` Rule

**Primary gate** (vs ProtoNet baseline):
- adversarial C-BAcc: delta > 0, 95% CI preferably all positive
- adversarial SCR: delta < 0, 95% CI preferably all negative
- adversarial SS-SCR: delta <= 0

**Safety gate**:
- No clean negative on adversarial SQ-PSR, NC-BAcc, NC-PSR
- No clean negative on standard C-BAcc, SCR

**Decision rule**:
- GO if primary metrics show coherent improvement pattern without safety leakage
- NO-GO if any primary metric moves significantly in the wrong direction, or if any safety metric shows a clean negative
- If only a slight adversarial gain with standard-side harm → close immediately (A1 pattern)
