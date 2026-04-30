# Boundary Calibration Pilot Readout

Date: 2026-04-30

Status: **PRELIMINARY** — limited-episode smoke test confirms pipeline integrity.
Full evaluation (400 standard + 400 adversarial episodes per task/seed) needed for gate decision.

## Chosen Config

```yaml
calibration_mode: boundary_uncertainty
top_k: 2
uncertainty_scale: 0.1
margin_floor: 0.1
```

## Smoke Test Results (max_episodes=5, 10 tasks × 5 seeds)

### Boundary_uncertainty aggregate (adversarial split)

| Metric | Score | 95% CI |
|--------|-------|--------|
| C-BAcc | 0.585 | [0.517, 0.670] |
| SCR | 0.863 | [0.763, 0.956] |
| SS-SCR | 0.906 | [0.776, 1.000] |
| SQ-PSR | 0.875 | [0.819, 0.919] |
| NC-BAcc | 0.508 | [0.491, 0.528] |
| NC-PSR | 0.649 | [0.545, 0.761] |

### ProtoNet baseline reference (adversarial split, full 400 episodes)

| Metric | Score | 95% CI |
|--------|-------|--------|
| C-BAcc | 0.561 | [0.518, 0.616] |
| SCR | 0.843 | [0.776, 0.903] |
| SS-SCR | — | — |
| SQ-PSR | 0.786 | — |

### Interpretation

The smoke test (5 episodes) cannot produce meaningful paired deltas because the baseline uses 400 episodes. The boundary_uncertainty scores are in the expected range and the pipeline runs without errors. Full evaluation is required.

## Gate Decision

**PENDING** — awaiting full evaluation.

## Full Evaluation Procedure

```bash
# Step 1: Full evaluation (no --max-episodes limit)
PYTHONPATH=src python -m fsmol_cliff.cli evaluate \
  --release-dir outputs/fsmol_cliff_release_v4_covext_intermediate \
  --data-dir fs-mol \
  --checkpoint checkpoints/PN-Support64_best_validation.pt \
  --output outputs/method_boundary_calibration_pilot/task_results_protonet_boundary_uncertainty_full.parquet \
  --profile relaxed_covext_10_10 \
  --result-tier intermediate \
  --backend protonet \
  --protonet-calibration-mode boundary_uncertainty \
  --protonet-calibration-top-k 2 \
  --protonet-calibration-uncertainty-scale 0.1 \
  --protonet-calibration-margin-floor 0.1

# Step 2: Aggregate
PYTHONPATH=src python -m fsmol_cliff.cli aggregate \
  --input outputs/method_boundary_calibration_pilot/task_results_protonet_boundary_uncertainty_full.parquet \
  --output outputs/method_boundary_calibration_pilot/task_results_protonet_boundary_uncertainty_full.aggregate.json

# Step 3: Paired comparison vs ProtoNet baseline
PYTHONPATH=src python -m fsmol_cliff.cli protocol-compare \
  --inputs \
    protonet_base=outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_protonet_relaxed_covext_10_10.parquet \
    boundary_uncertainty=outputs/method_boundary_calibration_pilot/task_results_protonet_boundary_uncertainty_full.parquet \
  --comparisons protonet_base:boundary_uncertainty \
  --profile relaxed_covext_10_10 \
  --result-tier intermediate \
  --output outputs/method_boundary_calibration_pilot/protonet_vs_boundary_uncertainty_full.paired_comparison.json

# Step 4: Apply the gate
# Check:
#   Primary: adversarial C-BAcc delta > 0, SCR delta < 0, SS-SCR delta <= 0
#   Safety: no clean negative on SQ-PSR, NC-BAcc, NC-PSR, standard C-BAcc/SCR
#   Decision: GO if coherent improvement without safety leakage; NO-GO otherwise
```

## Gate Metrics Checklist

- [ ] adversarial C-BAcc: delta > 0, CI preferably all positive
- [ ] adversarial SCR: delta < 0, CI preferably all negative
- [ ] adversarial SS-SCR: delta <= 0
- [ ] adversarial SQ-PSR: no clean negative
- [ ] adversarial NC-BAcc: no clean negative
- [ ] adversarial NC-PSR: no clean negative
- [ ] standard C-BAcc: no clean negative
- [ ] standard SCR: no clean negative
- [ ] Decision: GO / NO-GO
