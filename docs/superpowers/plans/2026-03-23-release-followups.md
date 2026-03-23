# Release Follow-Ups Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tighten the rebuilt v4.0 release interpretation by clarifying MAML status, investigating the `adversarial c_bacc` `NaN` issue in paired comparisons, and defining a path to a full-episode `MAML` final run.

**Architecture:** Keep the current rebuilt `outputs/fsmol_cliff_release_v4` as the working release directory. First fix interpretation/documentation debt, then debug the paired-comparison pathology, and only after that decide whether to extend the runtime stack for a true full-episode `MAML` final path.

**Tech Stack:** Python 3.12, pytest, pandas, parquet/json release artifacts, current `fsmol-maml-legacy` environment, existing `release_artifacts.py` pipeline

---

## Chunk 1: MAML Positioning Tightening

### Task 1: Audit every place where MAML is currently described too strongly

**Files:**
- Modify: `outputs/fsmol_cliff_release_v4/release_summary.md`
- Modify: `outputs/fsmol_cliff_release_v4/relaxed_claim_summary.md`
- Modify: `outputs/fsmol_cliff_release_v4/relaxed_model_comparisons_paper.md`
- Modify: `README.md`
- Modify: `spec_f.md`

- [ ] **Step 1: Search for all MAML references in release-facing docs**

Run:
```bash
rg -n "MAML|exploratory|legacy|3 episodes|final" \
  README.md \
  spec_f.md \
  outputs/fsmol_cliff_release_v4/*.md
```

Expected:
- A complete list of places where `MAML` is described in benchmark summaries, claim summaries, and paper-facing comparison notes

- [ ] **Step 2: Mark the current MAML release rows as exploratory compatibility results**

Update wording to make these points explicit:
- Current rebuilt `MAML` rows come from the legacy compatibility path
- Current `MAML` rows use `3` episodes per task/seed/split, not the full `400`
- `MAML` should not be treated as equal-strength evidence in strongest-final-claim wording

- [ ] **Step 3: Verify the documentation change is internally consistent**

Run:
```bash
python - <<'PY'
from pathlib import Path
for p in [
    Path("README.md"),
    Path("spec_f.md"),
    Path("outputs/fsmol_cliff_release_v4/release_summary.md"),
    Path("outputs/fsmol_cliff_release_v4/relaxed_claim_summary.md"),
    Path("outputs/fsmol_cliff_release_v4/relaxed_model_comparisons_paper.md"),
]:
    text = p.read_text()
    assert "MAML" in text
print("ok")
PY
```

Expected:
- Documentation still references `MAML`
- No contradiction remains between “model included” and “model treated as final-strength evidence”

## Chunk 2: Paired-Comparison `adversarial c_bacc` Debugging

### Task 2: Find the root cause of `NaN` in paired `adversarial c_bacc`

**Files:**
- Inspect: `src/fsmol_cliff/release_artifacts.py`
- Inspect: `src/fsmol_cliff/aggregate.py`
- Inspect: `outputs/fsmol_cliff_release_v4/task_results_*_relaxed.parquet`
- Test or modify if needed: `tests/test_release_artifacts.py`
- Modify if root cause is confirmed: `src/fsmol_cliff/release_artifacts.py`

- [ ] **Step 1: Reproduce the `NaN` rows in isolation**

Run:
```bash
python - <<'PY'
import json
from pathlib import Path
rows = json.loads(Path("outputs/fsmol_cliff_release_v4/relaxed_model_comparisons.json").read_text())
for row in rows:
    if row["split_type"] == "adversarial" and row["metric"] == "c_bacc":
        print(row)
PY
```

Expected:
- Show the exact `adversarial c_bacc` rows containing `NaN`

- [ ] **Step 2: Inspect task-level `c_bacc` coverage for every relaxed model**

Run:
```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
base = Path("outputs/fsmol_cliff_release_v4")
for stem in [
    "knn",
    "rf",
    "protonet",
    "maml",
    "knn_cliff_aware",
]:
    df = pd.read_parquet(base / f"task_results_{stem}_relaxed.parquet")
    sub = df[(df["split_type"] == "adversarial") & (df["metric"] == "c_bacc")]
    print(stem, "tasks", sub["task_id"].nunique(), "rows", len(sub), "coverage_nonnull", sub["score"].notna().sum())
PY
```

Expected:
- Clear evidence whether the `NaN` issue comes from missing task coverage, bootstrap edge cases, or malformed pairing logic

- [ ] **Step 3: Trace the failure point**

Investigate in order:
- `_task_metric_lookup(...)`
- `_paired_task_values(...)`
- `paired_bootstrap_delta_ci(...)`
- markdown/json serialization path

Expected:
- One concrete root cause statement, for example:
  - mismatched task sets
  - all-`NaN` or partially missing task values
  - invalid bootstrap input arrays

- [ ] **Step 4: Add a regression test before changing behavior**

Create one focused test in `tests/test_release_artifacts.py` covering the exact `adversarial c_bacc` failure mode.

Run:
```bash
python -m pytest tests/test_release_artifacts.py -q
```

Expected:
- New test fails before the implementation fix

- [ ] **Step 5: Implement the minimal fix and regenerate relaxed comparisons**

Run after fix:
```bash
python -m pytest tests/test_release_artifacts.py -q
```

Expected:
- New regression test passes
- `relaxed_model_comparisons.json` no longer emits avoidable `NaN` rows for valid `adversarial c_bacc` pairings

## Chunk 3: Full-Episode MAML Final Path

### Task 3: Decide whether a true final-strength MAML path is feasible

**Files:**
- Inspect: `src/fsmol_cliff/runner.py`
- Inspect: `src/fsmol_cliff/maml_legacy.py`
- Inspect: `src/fsmol_cliff/maml_legacy_runner.py`
- Inspect: `tests/test_maml_legacy_runner.py`
- Optional create: `docs/superpowers/plans/2026-03-23-maml-final-path.md`

- [ ] **Step 1: Define what “full-episode MAML final” must mean**

Acceptance criteria:
- `400` episodes per task/seed/split, matching the main release protocol
- no smoke-only truncation
- same release bundle inputs as other models
- clearly documented support-valid compatibility rule
- output written to `task_results_maml_{profile}.parquet` with `result_tier="final"`

- [ ] **Step 2: Audit the current blocker**

Inspect whether the current limitation is:
- model runtime speed
- environment fragility
- legacy external FS-Mol dependency behavior
- missing batch/restart orchestration
- incompatibility with current support/query protocol

Expected:
- One explicit blocker statement, not just “legacy path”

- [ ] **Step 3: Choose one of two outcomes**

Outcome A:
- feasible now
- write a dedicated implementation plan for a full-episode `MAML` final path

Outcome B:
- not feasible within acceptable runtime/maintenance cost
- codify `MAML` as exploratory-only in the v4 release policy

- [ ] **Step 4: If feasible, write the follow-up execution plan**

Create:
- `docs/superpowers/plans/2026-03-23-maml-final-path.md`

That plan must include:
- runtime entrypoint
- environment assumptions
- checkpoint handling
- full verification commands

- [ ] **Step 5: If not feasible, close the loop in release docs**

Update:
- `release_summary.md`
- `relaxed_claim_summary.md`
- `README.md`

Expected:
- No ambiguity remains about whether the current `MAML` rows are “temporary final” or permanently exploratory

## Suggested Execution Order

1. Tighten `MAML` wording now
2. Debug and, if possible, fix `adversarial c_bacc` paired comparisons
3. Only then decide whether the cost of a full-episode `MAML final` path is justified

## Verification Gate

Before claiming this follow-up work is complete, run:

```bash
python -m pytest -q
```

And if the paired-comparison logic changed, also re-check:

```bash
python - <<'PY'
import json
from pathlib import Path
rows = json.loads(Path("outputs/fsmol_cliff_release_v4/relaxed_model_comparisons.json").read_text())
print("rows", len(rows))
print("adversarial_c_bacc_rows", sum(r["split_type"] == "adversarial" and r["metric"] == "c_bacc" for r in rows))
PY
```
