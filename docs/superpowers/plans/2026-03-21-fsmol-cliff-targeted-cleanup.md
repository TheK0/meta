# FS-Mol-Cliff Targeted Cleanup Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean the remaining warning/reproducibility debt and restore release/code consistency after the chemistry-alignment fixes without taking on a risky package-layout or OOP rewrite.

**Architecture:** Keep the existing flat `src/fsmol_cliff/` layout and the current runner model. Make only surgical, evidence-driven changes: warning ownership cleanup, release rebuild under current chemistry semantics, and narrowly scoped helper extraction if profiling or duplication proves it is needed. Explicitly defer namespace migration and `BaseRunner` abstraction unless new evidence justifies them.

**Tech Stack:** Python 3.12, pytest, pandas, scikit-learn, RDKit, parquet/json release artifacts

---

## Chunk 1: Warning Ownership Cleanup

### Task 1: Triage and eliminate owned warnings

**Files:**
- Modify: `tests/test_baseline_adapter_runtime.py`
- Modify: `tests/test_cli_commands.py`
- Create if needed: `tests/conftest.py`
- Reference: `src/fsmol_cliff/adapters.py`
- Reference: `src/fsmol_cliff/cli.py`

- [ ] **Step 1: Reproduce warnings with full visibility**

Run:
```bash
python -m pytest tests/ -rw
```

Expected:
- Current output shows the `joblib/loky` physical-core warning
- Current output shows the `azureml` deprecation warning

- [ ] **Step 2: Isolate the sklearn/joblib warning**

Run:
```bash
python -m pytest tests/test_baseline_adapter_runtime.py::test_score_sklearn_episode_returns_query_scores_in_manifest_order -rw -q
```

Expected:
- Reproduce or eliminate the `loky` warning in the narrowest test scope possible

- [ ] **Step 3: Isolate the adapter-status / azure warning**

Run:
```bash
python -m pytest tests/test_cli_commands.py::test_adapter_status_command_writes_availability_report -rw -q
```

Expected:
- Reproduce or eliminate the `azureml` deprecation warning in the narrowest test scope possible

- [ ] **Step 4: Apply the smallest correct fix**

Implementation rules:
- If the warning is caused by our code path and can be prevented by a deterministic parameter/env change, prefer that fix
- If the warning comes from unavoidable third-party import side effects, add a narrow test-only filter in `tests/conftest.py`
- Do not add broad global warning suppression

- [ ] **Step 5: Verify warnings are gone under strict mode**

Run:
```bash
python -m pytest tests/ -W error -q
```

Expected:
- All tests pass
- No warnings promoted to errors

## Chunk 2: Release/Code Consistency Rebuild

### Task 2: Rebuild the frozen release bundle under current chemistry semantics

**Files:**
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/benchmark_manifest.json`
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/fsmol_cliff_strict_all.json`
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/fsmol_cliff_relaxed_all.json`
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/task_summaries_strict.parquet`
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/task_summaries_relaxed.parquet`
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/episodes_standard_strict.parquet`
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/episodes_adversarial_strict.parquet`
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/episodes_standard_relaxed.parquet`
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/episodes_adversarial_relaxed.parquet`
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/assays/*`
- Reference: `src/fsmol_cliff/release.py`

- [ ] **Step 1: Back up the current release directory**

Run:
```bash
cp -R outputs/fsmol_cliff_release_v4 outputs/fsmol_cliff_release_v4.pre_rebuild
```

Expected:
- Backup directory exists before any rebuild work

- [ ] **Step 2: Rebuild the strict bundle with current code**

Run:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli build-release \
  --data-dir <fsmol_dir> \
  --output-dir outputs/fsmol_cliff_release_v4 \
  --profile strict \
  --fsmol-data-version <fsmol_data_version>
```

Expected:
- Strict manifests, episode bundles, assay assets, metadata, and reproducibility note are regenerated

- [ ] **Step 3: Rebuild the relaxed bundle with current code**

Run:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli build-release \
  --data-dir <fsmol_dir> \
  --output-dir outputs/fsmol_cliff_release_v4 \
  --profile relaxed \
  --fsmol-data-version <fsmol_data_version>
```

Expected:
- Relaxed manifests, episode bundles, assay assets, metadata, and reproducibility note are regenerated

### Task 3: Regenerate audit assets against the rebuilt release

**Files:**
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/audit/strict/attrition_summary.json`
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/audit/strict/attrition_by_assay.parquet`
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/audit/strict/threshold_sensitivity.parquet`
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/audit/relaxed/attrition_summary.json`
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/audit/relaxed/attrition_by_assay.parquet`
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/audit/relaxed/threshold_sensitivity.parquet`
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/benchmark_decision_note.md`
- Reference: `src/fsmol_cliff/audit.py`

- [ ] **Step 1: Re-run strict attrition audit**

Run:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli audit-attrition \
  --release-dir outputs/fsmol_cliff_release_v4 \
  --data-dir <fsmol_dir> \
  --output-dir outputs/fsmol_cliff_release_v4/audit/strict \
  --profile strict
```

Expected:
- Strict audit parquet/json outputs are refreshed

- [ ] **Step 2: Re-run relaxed attrition audit**

Run:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli audit-attrition \
  --release-dir outputs/fsmol_cliff_release_v4 \
  --data-dir <fsmol_dir> \
  --output-dir outputs/fsmol_cliff_release_v4/audit/relaxed \
  --profile relaxed
```

Expected:
- Relaxed audit parquet/json outputs are refreshed

## Chunk 3: Results and Documentation Refresh

### Task 4: Re-run model evaluations and regenerate release artifact tables

**Files:**
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/task_results_*`
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/relaxed_main_table.*`
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/relaxed_failure_taxonomy.*`
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/relaxed_model_comparisons.*`
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/relaxed_model_comparisons_paper.md`
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/relaxed_claim_summary.md`
- Modify/overwrite: `outputs/fsmol_cliff_release_v4/release_summary.md`
- Reference: `src/fsmol_cliff/runner.py`
- Reference: `src/fsmol_cliff/protonet_runner.py`
- Reference: `src/fsmol_cliff/release_artifacts.py`

- [ ] **Step 1: Re-run the supported model suite on the rebuilt release**

Run the existing project model-evaluation workflow for:
- local `kNN`
- local `RF`
- `kNN-cliff-aware`
- `ProtoNet`
- `MAML` compatibility path if the runtime is available

Expected:
- `task_results_*` parquet and aggregate files are regenerated under current chemistry semantics

- [ ] **Step 2: Regenerate release artifact tables from rebuilt results**

Implementation:
- Use the existing `src/fsmol_cliff/release_artifacts.py` workflow or its current invocation path
- Recompute:
  - `relaxed_main_table`
  - `relaxed_failure_taxonomy`
  - `relaxed_model_comparisons`

Expected:
- Main-table and paired-comparison artifacts now match the current code

- [ ] **Step 3: Refresh narrative docs to remove the stale-numerics caveat**

Update:
- `outputs/fsmol_cliff_release_v4/release_summary.md`
- `outputs/fsmol_cliff_release_v4/relaxed_claim_summary.md`
- `spec_f.md`

Expected:
- Documentation no longer says the checked-in release predates the chemistry fix

- [ ] **Step 4: Verify end-to-end**

Run:
```bash
python -m pytest -q
```

Expected:
- Full suite passes after warning cleanup and release refresh

## Explicit Non-Goals

- Do **not** split `src/fsmol_cliff/` into `models/`, `data/`, `eval/`, and `core/` in this plan
- Do **not** rename `models.py` into a package in this plan
- Do **not** introduce a `BaseRunner` abstract class in this plan
- Do **not** merge `maml_legacy.py` and `maml_legacy_runner.py` unless duplication is first demonstrated with concrete evidence

These changes are intentionally deferred because they create high import-surface churn, high compatibility risk, and unclear immediate payoff for the current repository state.
