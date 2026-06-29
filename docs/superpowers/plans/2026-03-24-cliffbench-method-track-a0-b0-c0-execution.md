# CliffBench Method Track A0/B0/C0 Execution Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the method-track restart in the required order by first making ProtoNet calibration pluggable (`A0`), then only if needed proceeding to the minimal training-loss injection for prototype shaping (`B0`) and the inference-only perturbation audit (`C0`).

**Architecture:** Keep the benchmark protocol frozen and treat this plan as method-track execution on top of the fixed intermediate substrate. `A0` is the mandatory entry point because it turns ProtoNet into a controllable baseline with raw and calibrated score paths; `B0` and `C0` are fallback entry points only if Plan A fails to produce a viable signal. Every phase is gated by targeted tests, a baseline-alignment check, and the stronger-baseline success criteria in [`spec_cliffbench_method_restart_v1.md`](../../../spec_cliffbench_method_restart_v1.md).

**Tech Stack:** Python 3.12, pytest, pandas/parquet, PyTorch, existing `fsmol_cliff` ProtoNet runner, release evaluation CLI, markdown specs in the repo root.

---

## Chunk 1: Plan A `A0` ProtoNet Baseline Extraction

### Task 1: Add a pluggable ProtoNet score bundle without changing current metrics semantics

**Files:**
- Create: `src/fsmol_cliff/protonet_base.py`
- Create: `src/fsmol_cliff/protonet_local_calibrated.py`
- Modify: `src/fsmol_cliff/protonet_runner.py`
- Modify: `src/fsmol_cliff/evaluation.py`
- Create: `tests/test_protonet_local_calibration.py`
- Modify: `tests/test_protonet_runner.py`

- [ ] **Step 1: Write the failing test for raw/calibrated score plumbing**

Add a test in `tests/test_protonet_local_calibration.py` that expects a bundle like:

```python
{
    "raw_scores": {"qa": 0.8},
    "calibrated_scores": {"qa": 0.8},
    "raw_margins": {"qa": 0.3},
    "calibrated_margins": {"qa": 0.3},
}
```

The initial calibrated path should be identity.

- [ ] **Step 2: Run the new test to verify it fails**

Run:
```bash
python -m pytest tests/test_protonet_local_calibration.py::test_identity_calibration_preserves_raw_scores -q
```

Expected:
- FAIL because the module/bundle path does not exist yet

- [ ] **Step 3: Write the failing runner regression test**

Add a test in `tests/test_protonet_runner.py` asserting:
- release-mode ProtoNet evaluation still emits the current metric rows
- raw/calibrated margins are recorded in episode-level intermediate structures or a returned score bundle
- discrete predictions still come from the fixed reporting rule applied to the active score path

- [ ] **Step 4: Run the runner regression test to verify it fails**

Run:
```bash
python -m pytest tests/test_protonet_runner.py::test_evaluate_release_with_protonet_supports_identity_calibration_bundle -q
```

Expected:
- FAIL because the score bundle interface is missing

- [ ] **Step 5: Implement the minimal score bundle path**

Implementation requirements:
- `src/fsmol_cliff/protonet_base.py`
  - extract raw ProtoNet scoring into a reusable function
  - return raw scores and raw margins
- `src/fsmol_cliff/protonet_local_calibrated.py`
  - define an identity local calibrator first
  - output calibrated scores and calibrated margins
- `src/fsmol_cliff/protonet_runner.py`
  - keep current default metrics behavior unchanged
  - allow the score source to be replaced by calibrated scores later
- `src/fsmol_cliff/evaluation.py`
  - preserve the current fixed decision path (`score >= 0.5`)
  - do not alter metric definitions

- [ ] **Step 6: Run the focused tests to verify they pass**

Run:
```bash
python -m pytest \
  tests/test_protonet_local_calibration.py \
  tests/test_protonet_runner.py \
  -q
```

Expected:
- PASS

- [ ] **Step 7: Verify baseline aggregate alignment**

Run a fresh ProtoNet release evaluation on the intermediate substrate:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli evaluate \
  --release-dir outputs/fsmol_cliff_release_v4_covext_intermediate \
  --data-dir fs-mol \
  --checkpoint checkpoints/PN-Support64_best_validation.pt \
  --output /tmp/protonet_a0_recheck.parquet \
  --profile relaxed_covext_10_10 \
  --result-tier intermediate \
  --backend protonet
```

Aggregate it:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli aggregate \
  --input /tmp/protonet_a0_recheck.parquet \
  --output /tmp/protonet_a0_recheck.aggregate.json
```

Compare against:
- `outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_protonet_relaxed_covext_10_10.aggregate.json`

Expected:
- key rows match to floating-point tolerance

- [ ] **Step 8: Commit**

```bash
git add \
  src/fsmol_cliff/protonet_base.py \
  src/fsmol_cliff/protonet_local_calibrated.py \
  src/fsmol_cliff/protonet_runner.py \
  src/fsmol_cliff/evaluation.py \
  tests/test_protonet_local_calibration.py \
  tests/test_protonet_runner.py
git commit -m "feat: extract pluggable protonet score path"
```

### Task 2: `A0` gate

**Files:**
- Modify: `paper_latex/notes/training-episode-protocol-status.md`
- Modify: `spec_cliffbench_method_restart_v1.md`

- [ ] **Step 1: Record `A0` result**

Write one of:
- `A0 GO`: baseline rebuilt and score path is pluggable without metric drift
- `A0 NO-GO`: baseline drift or reporting semantics changed

- [ ] **Step 2: If `A0 GO`, continue immediately to `A1`**

Constraint:
- do not start `B0` or `C0` if `A0` has not passed

## Chunk 2: Plan A `A1` Query-Only Local Calibration

### Task 3: Add the smallest possible query-only calibration head

**Files:**
- Modify: `src/fsmol_cliff/protonet_local_calibrated.py`
- Modify: `src/fsmol_cliff/protonet_runner.py`
- Modify: `tests/test_protonet_local_calibration.py`
- Modify: `tests/test_protonet_runner.py`

- [ ] **Step 1: Write the failing calibration-head test**

Test:
- local features limited to:
  - raw query margin
  - prototype distance gap
  - nearest positive/negative support distance difference
- calibrated score differs from raw score when the head is enabled
- raw score path remains available

- [ ] **Step 2: Run the calibration-head test to verify it fails**

Run:
```bash
python -m pytest tests/test_protonet_local_calibration.py::test_query_only_calibration_uses_local_features -q
```

- [ ] **Step 3: Implement the minimal query-only calibrator**

Implementation requirements:
- no attention
- no uncertainty estimator
- no change to episode definition
- no rule-based threshold patching

- [ ] **Step 4: Run the focused tests**

Run:
```bash
python -m pytest \
  tests/test_protonet_local_calibration.py \
  tests/test_protonet_runner.py \
  -q
```

- [ ] **Step 5: Run fresh `A1` evaluation**

Run release-mode ProtoNet on the intermediate substrate with calibration enabled, aggregate, and compare against the fresh `A0` baseline.

- [ ] **Step 6: Apply the `A1` gate**

Continue only if:
- adversarial `C-BAcc` improves directionally
- adversarial `SCR` improves directionally
- safety metrics do not cleanly degrade

- [ ] **Step 7: Commit**

```bash
git add \
  src/fsmol_cliff/protonet_local_calibrated.py \
  src/fsmol_cliff/protonet_runner.py \
  tests/test_protonet_local_calibration.py \
  tests/test_protonet_runner.py
git commit -m "feat: add query-only protonet calibration"
```

## Chunk 3: Plan B `B0` Fallback Entry

### Task 4: Only if Plan A fails, add the minimal ProtoNet cliff margin loss

**Files:**
- Create: `src/fsmol_cliff/training_losses/cliff_margin.py`
- Create: `configs/protonet_cliff_margin.yaml`
- Create: `tests/test_cliff_margin_loss.py`

- [ ] **Step 1: Do not start unless `A1` is `NO-GO`**

- [ ] **Step 2: Write the failing loss test**

Test:
- cliff-associated query receives a positive penalty when prototype margin is violated
- loss is zero when the margin is already satisfied

- [ ] **Step 3: Run the loss test to verify it fails**

Run:
```bash
python -m pytest tests/test_cliff_margin_loss.py -q
```

- [ ] **Step 4: Implement the minimal loss injection**

Matrix to run after code passes:
- `margin gamma ∈ {0.05, 0.1, 0.2}`
- `lambda_cliff ∈ {0.1, 0.3, 1.0}`
- control preservation `on/off`

- [ ] **Step 5: Run the loss test to verify it passes**

Run:
```bash
python -m pytest tests/test_cliff_margin_loss.py -q
```

- [ ] **Step 6: Run the `B0` experiment matrix**

Expected:
- if all rows show clean `SQ-PSR` or `NC-*` degradation, mark `B0 NO-GO`

## Chunk 4: Plan C `C0` Fallback Entry

### Task 5: Only if Plan B fails, run the inference-only perturbation audit

**Files:**
- Create: `src/fsmol_cliff/protonet_perturbation_audit.py`
- Create: `tests/test_protonet_perturbation_audit.py`
- Modify: `paper_latex/notes/training-episode-protocol-status.md`

- [ ] **Step 1: Do not start unless `B0` is `NO-GO`**

- [ ] **Step 2: Write the failing audit test**

Test:
- report contains:
  - per-query score variance
  - cliff/control variance gap
  - same-scaffold cliff variance gap

- [ ] **Step 3: Run the audit test to verify it fails**

Run:
```bash
python -m pytest tests/test_protonet_perturbation_audit.py -q
```

- [ ] **Step 4: Implement the minimal audit**

Preferred first perturbation:
- support subset dropout

- [ ] **Step 5: Run the audit test to verify it passes**

Run:
```bash
python -m pytest tests/test_protonet_perturbation_audit.py -q
```

- [ ] **Step 6: Run the audit on the ProtoNet baseline**

Expected:
- if cliff-query sensitivity is not meaningfully higher than control, mark `C0` lower priority

## Chunk 5: Baseline Code Verification

### Task 6: Run the current code tests before implementation starts

**Files:**
- Test only: `tests/test_protonet_runner.py`
- Test only: `tests/test_cli_commands.py`
- Test only: `tests/test_manifests.py`
- Test only: `tests/test_release.py`
- Test only: `tests/test_protocol_compare.py`

- [ ] **Step 1: Run the baseline test suite**

Run:
```bash
python -m pytest -q \
  tests/test_protonet_runner.py \
  tests/test_cli_commands.py \
  tests/test_manifests.py \
  tests/test_release.py \
  tests/test_protocol_compare.py
```

Expected:
- all currently committed/working-tree code paths pass before any new A0 work begins

- [ ] **Step 2: Record the baseline result**

Write the exact pass/fail count into the session notes or task log before starting `A0`.
