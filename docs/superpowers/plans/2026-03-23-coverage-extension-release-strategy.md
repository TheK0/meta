# Coverage-Extension Release Strategy Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate whether coverage-expanded auxiliary profiles can strengthen H1/H3 evidence without changing the current v4.0 final benchmark substrate.

**Architecture:** Keep the current relaxed release as the only `final` benchmark substrate. Add one `intermediate` coverage-extension profile and one optional `exploratory` looser profile as physically separate release directories, then compare coverage, cliff density, and hypothesis strength with explicit stop/go gates. Also make release-facing artifact generation tier-safe so future mixed-tier use cannot silently read the wrong rows.

**Tech Stack:** Python 3.12, pytest, pandas/parquet, existing `fsmol_cliff` CLI/release pipeline, markdown release artifacts.

---

## File Structure

**Modify**
- `src/fsmol_cliff/constants.py`
  Purpose: register auxiliary coverage-extension profiles without changing the current strict/relaxed final definitions.
- `src/fsmol_cliff/cli.py`
  Purpose: allow auxiliary profile names in `build-release`, `audit-attrition`, and `evaluate`.
- `src/fsmol_cliff/audit.py`
  Purpose: make profile-to-threshold inference generic for all registered profiles, not just `strict`/`relaxed`.
- `src/fsmol_cliff/release_artifacts.py`
  Purpose: explicitly filter aggregate/task rows by `result_tier` in addition to `profile`.
- `tests/test_release.py`
  Purpose: cover auxiliary profile release outputs and manifest behavior.
- `tests/test_cli_commands.py`
  Purpose: cover CLI acceptance of auxiliary profile names and tiered evaluation invocations.
- `tests/test_attrition_audit.py`
  Purpose: cover audit behavior for auxiliary profiles.
- `tests/test_release_artifacts.py`
  Purpose: prove artifact builders respect `result_tier`.
- `README.md`
  Purpose: document the final vs auxiliary release policy at a short operational level.

**Create**
- `docs/superpowers/plans/2026-03-23-coverage-extension-release-strategy.md`
  Purpose: this implementation plan.
- `outputs/<intermediate-release-dir>/...`
  Purpose: intermediate coverage-extension release bundle and derived results.
- `outputs/<exploratory-release-dir>/...`
  Purpose: optional exploratory looser release bundle and derived results.
- `paper_latex/notes/coverage-extension-decision.md`
  Purpose: short paper-facing decision record summarizing whether intermediate/exploratory results stay appendix-only, become extended tables, or are dropped.

**Do not modify**
- `spec.md`
  Reason: the current final benchmark definition remains unchanged.
- Current final release under `outputs/fsmol_cliff_release_v4`
  Reason: preserve the benchmark substrate used for existing final claims.

---

## Naming Decisions

Use explicit profile names in code:

- `relaxed_covext_10_10`
  Semantics: `tau=0.80`, `delta=1.0`, `min_cliff_pairs=10`, `min_noncliff_pairs=10`
  Intended `result_tier`: `intermediate`

- `relaxed_covext_10_5`
  Semantics: `tau=0.80`, `delta=1.0`, `min_cliff_pairs=10`, `min_noncliff_pairs=5`
  Intended `result_tier`: `exploratory`

Use explicit output directories:

- Final reference release: keep current `outputs/fsmol_cliff_release_v4`
- Intermediate auxiliary release: `outputs/fsmol_cliff_release_v4_covext_intermediate`
- Exploratory auxiliary release: `outputs/fsmol_cliff_release_v4_covext_exploratory`

Reason:
- avoid changing the existing final release path
- avoid mixing `final`, `intermediate`, and `exploratory` rows in a single artifact tree
- keep paper-facing interpretation physically isolated

---

## Chunk 1: Add Auxiliary Profiles and Tier-Safe Artifact Filtering

### Task 1: Register auxiliary coverage-extension profiles

**Files:**
- Modify: `src/fsmol_cliff/constants.py`
- Test: `tests/test_release.py`
- Test: `tests/test_cli_commands.py`

- [ ] **Step 1: Write failing tests for auxiliary profile registration**

Add tests that assert:
- `PROFILE_SPECS` contains `relaxed_covext_10_10`
- `PROFILE_SPECS` contains `relaxed_covext_10_5`
- each profile keeps `similarity_threshold=0.80`
- each profile keeps `activity_gap_threshold=1.0`
- only `min_cliff_pairs` / `min_noncliff_pairs` change

- [ ] **Step 2: Run tests to verify failure**

Run:
```bash
python -m pytest tests/test_release.py tests/test_cli_commands.py -q
```

Expected:
- FAIL because the new profile keys do not exist yet

- [ ] **Step 3: Implement auxiliary profiles**

In `src/fsmol_cliff/constants.py`:
- add `RELAXED_COVEXT_10_10_PROFILE`
- add `RELAXED_COVEXT_10_5_PROFILE`
- register both in `PROFILE_SPECS`

Do not modify:
- `STRICT_PROFILE`
- `RELAXED_PROFILE`

- [ ] **Step 4: Extend CLI profile choices**

In `src/fsmol_cliff/cli.py`:
- replace hard-coded `choices=["strict", "relaxed"]`
- derive `choices=sorted(PROFILE_SPECS)` for:
  - `audit-attrition --profile`
  - `build-release --profile`
  - `evaluate --profile`

- [ ] **Step 5: Run tests to verify pass**

Run:
```bash
python -m pytest tests/test_release.py tests/test_cli_commands.py -q
```

Expected:
- PASS

- [ ] **Step 6: Commit**

```bash
git add src/fsmol_cliff/constants.py src/fsmol_cliff/cli.py tests/test_release.py tests/test_cli_commands.py
git commit -m "feat: add coverage-extension benchmark profiles"
```

### Task 2: Make audit inference generic for auxiliary profiles

**Files:**
- Modify: `src/fsmol_cliff/audit.py`
- Test: `tests/test_attrition_audit.py`

- [ ] **Step 1: Write failing audit test**

Add a test asserting that `write_attrition_audit(...)` correctly infers profile-specific `tau` and `delta` for an auxiliary profile instead of falling back to the current strict/relaxed-only branch.

- [ ] **Step 2: Run test to verify failure**

Run:
```bash
python -m pytest tests/test_attrition_audit.py -q
```

Expected:
- FAIL because `audit.py` still hard-codes only `strict` vs `relaxed`

- [ ] **Step 3: Implement generic profile inference**

In `src/fsmol_cliff/audit.py`:
- import `PROFILE_SPECS`
- when synthesizing summary rows from `task_summaries_<profile>.parquet`, resolve `tau` and `delta` from `PROFILE_SPECS[profile]`
- remove the current fallback that assumes only `0.85 if strict else 0.8`

- [ ] **Step 4: Run test to verify pass**

Run:
```bash
python -m pytest tests/test_attrition_audit.py -q
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add src/fsmol_cliff/audit.py tests/test_attrition_audit.py
git commit -m "fix: make attrition audit profile-aware for auxiliary releases"
```

### Task 3: Make release artifacts explicitly tier-safe

**Files:**
- Modify: `src/fsmol_cliff/release_artifacts.py`
- Test: `tests/test_release_artifacts.py`

- [ ] **Step 1: Write failing tests for result-tier filtering**

Add tests that construct aggregate/task rows containing:
- same `profile`
- same `split_type`
- same `metric`
- different `result_tier`

Assert that artifact builders can be called with `result_tier="final"` or `result_tier="intermediate"` and only consume rows from the requested tier.

- [ ] **Step 2: Run test to verify failure**

Run:
```bash
python -m pytest tests/test_release_artifacts.py -q
```

Expected:
- FAIL because current lookup logic filters by `profile` only

- [ ] **Step 3: Implement tier filtering**

In `src/fsmol_cliff/release_artifacts.py`:
- add `result_tier` arguments to:
  - `build_main_table_rows(...)`
  - `build_failure_taxonomy_rows(...)`
  - `build_paired_model_comparison_rows(...)` if needed for future mixed-tier inputs
- update `_aggregate_lookup(...)` and `_task_metric_lookup(...)` to filter by both `profile` and `result_tier`
- default `result_tier="final"` to preserve current final behavior

- [ ] **Step 4: Run tests to verify pass**

Run:
```bash
python -m pytest tests/test_release_artifacts.py -q
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add src/fsmol_cliff/release_artifacts.py tests/test_release_artifacts.py
git commit -m "fix: filter release artifacts by result tier"
```

---

## Chunk 2: Build and Evaluate the Intermediate Coverage-Extension Release

### Task 4: Reconfirm the final release baseline and freeze it

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add short release policy note**

In `README.md`, add a brief section stating:
- `outputs/fsmol_cliff_release_v4` remains the final benchmark substrate
- coverage-extension releases are auxiliary and do not replace the final relaxed benchmark

- [ ] **Step 2: Re-run tests**

Run:
```bash
python -m pytest -q
```

Expected:
- PASS

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: freeze final release and auxiliary coverage policy"
```

### Task 5: Build the intermediate release

**Files:**
- Output only: `outputs/fsmol_cliff_release_v4_covext_intermediate/`

- [ ] **Step 1: Build the release bundle**

Run:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli build-release \
  --data-dir fs-mol \
  --output-dir outputs/fsmol_cliff_release_v4_covext_intermediate \
  --profile relaxed_covext_10_10
```

Expected:
- writes `task_summaries_relaxed_covext_10_10.parquet`
- writes `episodes_standard_relaxed_covext_10_10.parquet`
- writes `episodes_adversarial_relaxed_covext_10_10.parquet`

- [ ] **Step 2: Run attrition audit**

Run:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli audit-attrition \
  --release-dir outputs/fsmol_cliff_release_v4_covext_intermediate \
  --data-dir fs-mol \
  --output-dir outputs/fsmol_cliff_release_v4_covext_intermediate/audit/relaxed_covext_10_10 \
  --profile relaxed_covext_10_10
```

Expected:
- writes `attrition_summary.json`
- writes `attrition_by_assay.parquet`
- writes `threshold_sensitivity.parquet`

- [ ] **Step 3: Record the coverage snapshot**

Extract and save the following values from the audit outputs:
- `eligible_assay_count`
- `adversarial_eligible_assay_count`
- `total_cliff_pairs`
- `total_anchors`
- `same_scaffold_cliff_pair_count`

Run a one-off extraction:
```bash
python - <<'PY'
import json
import pandas as pd
from pathlib import Path
root = Path("outputs/fsmol_cliff_release_v4_covext_intermediate/audit/relaxed_covext_10_10")
print(json.loads((root / "attrition_summary.json").read_text()))
df = pd.read_parquet(root / "threshold_sensitivity.parquet")
print(df.sort_values(["eligible_assay_count","adversarial_eligible_assay_count"], ascending=False).head(8).to_string(index=False))
PY
```

- [ ] **Step 4: Evaluate the full-strength model set**

Run:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli evaluate \
  --release-dir outputs/fsmol_cliff_release_v4_covext_intermediate \
  --output outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_relaxed_covext_10_10.parquet \
  --profile relaxed_covext_10_10 \
  --result-tier intermediate \
  --model-name kNN
```

```bash
PYTHONPATH=src python -m fsmol_cliff.cli evaluate \
  --release-dir outputs/fsmol_cliff_release_v4_covext_intermediate \
  --output outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_rf_relaxed_covext_10_10.parquet \
  --profile relaxed_covext_10_10 \
  --result-tier intermediate \
  --model-name RF
```

```bash
PYTHONPATH=src python -m fsmol_cliff.cli evaluate \
  --release-dir outputs/fsmol_cliff_release_v4_covext_intermediate \
  --output outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_cliff_aware_relaxed_covext_10_10.parquet \
  --profile relaxed_covext_10_10 \
  --result-tier intermediate \
  --backend cliff-aware \
  --model-name kNN
```

```bash
PYTHONPATH=src python -m fsmol_cliff.cli evaluate \
  --release-dir outputs/fsmol_cliff_release_v4_covext_intermediate \
  --data-dir fs-mol \
  --checkpoint checkpoints/PN-Support64_best_validation.pt \
  --output outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_protonet_relaxed_covext_10_10.parquet \
  --profile relaxed_covext_10_10 \
  --result-tier intermediate \
  --backend protonet
```

- [ ] **Step 5: Aggregate all model outputs**

Run for each generated parquet:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli aggregate \
  --input outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_relaxed_covext_10_10.parquet \
  --output outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_relaxed_covext_10_10.aggregate.json
```

Repeat for:
- `rf`
- `protonet`
- `knn_cliff_aware`

- [ ] **Step 6: Commit generated command notes only if policy allows**

Do not commit generated `outputs/` content unless explicitly requested. If the repository policy remains “outputs ignored”, only commit code/tests/docs changes and keep intermediate results local.

---

## Chunk 3: Decision Gates, Optional Exploratory Run, and Paper-Facing Outcome

### Task 6: Evaluate the intermediate release against stop/go gates

**Files:**
- Create: `paper_latex/notes/coverage-extension-decision.md`

- [ ] **Step 1: Check coverage gates**

Required checks:
- intermediate `eligible_assay_count` > final relaxed `eligible_assay_count`
- intermediate `adversarial_eligible_assay_count` > final relaxed `adversarial_eligible_assay_count`

Interpretation:
- if eligible rises but adversarial-eligible does not, treat the profile as weak support for H2/H3 strengthening

- [ ] **Step 2: Check cliff-density gates**

Required checks:
- `total_cliff_pairs`
- `total_anchors`
- `same_scaffold_cliff_pair_count`

Interpretation:
- if assay count increases but cliff density drops sharply, do not promote the profile beyond appendix sensitivity analysis

- [ ] **Step 3: Check H3 first**

Primary deltas to inspect:
- `\Delta C-BAcc`
- `\Delta SCR`
- `\Delta SQ-PSR`
- `\Delta SS-SCR`
- `\Delta NC-BAcc`
- `\Delta NC-PSR`

Decision:
- if directional wins remain stable and paired intervals are cleaner than the current final release, mark H3 as the main beneficiary of the coverage extension

- [ ] **Step 4: Check H1 second**

Inspect across the full-strength model set:
- official average metric
- `C-BAcc`
- `Q-PSR`
- `NC-BAcc`
- `NC-PSR`

Decision:
- if ordering and cliff/control gaps remain unstable, keep H1 as `supported trend`
- if the expanded release materially stabilizes the gap story, note H1 as strengthened but still auxiliary until explicitly re-reviewed

- [ ] **Step 5: Write the decision record**

In `paper_latex/notes/coverage-extension-decision.md`, write one of:

- **Plan outcome A**
  - keep current final relaxed as main table
  - include intermediate release in appendix / extended table
  - use it to strengthen H3 wording and possibly H1 discussion

- **Plan outcome B**
  - keep intermediate release as appendix-only robustness evidence
  - do not alter main claims

- **Plan outcome C**
  - stop the coverage-extension line
  - retain only as internal sensitivity evidence

- [ ] **Step 6: Commit the decision record**

```bash
git add paper_latex/notes/coverage-extension-decision.md
git commit -m "docs: record coverage-extension evaluation decision"
```

### Task 7: Only if the intermediate release is positive, run the exploratory profile

**Files:**
- Output only: `outputs/fsmol_cliff_release_v4_covext_exploratory/`

- [ ] **Step 1: Confirm stop/go condition**

Only continue if all are true:
- intermediate coverage increased materially
- adversarial-eligible coverage also increased
- cliff density did not collapse
- H3 evidence became meaningfully cleaner

- [ ] **Step 2: Build the exploratory release**

Run:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli build-release \
  --data-dir fs-mol \
  --output-dir outputs/fsmol_cliff_release_v4_covext_exploratory \
  --profile relaxed_covext_10_5
```

- [ ] **Step 3: Audit the exploratory release**

Run:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli audit-attrition \
  --release-dir outputs/fsmol_cliff_release_v4_covext_exploratory \
  --data-dir fs-mol \
  --output-dir outputs/fsmol_cliff_release_v4_covext_exploratory/audit/relaxed_covext_10_5 \
  --profile relaxed_covext_10_5
```

- [ ] **Step 4: Evaluate only if intermediate was clearly promising**

Repeat the full-strength model evaluation and aggregation flow from Task 5, but with:
- `--profile relaxed_covext_10_5`
- `--result-tier exploratory`

- [ ] **Step 5: Stop if exploratory weakens the story**

Do not promote exploratory results into the main comparison substrate. They are appendix-only unless a later explicit paper-writing decision says otherwise.

### Task 8: Final verification and handoff

**Files:**
- Modify if needed: `README.md`
- Modify if needed: `paper_latex/main.tex`

- [ ] **Step 1: Run the full test suite after code changes**

Run:
```bash
python -m pytest -q
```

Expected:
- PASS

- [ ] **Step 2: Verify tier-safe artifact behavior**

Run:
```bash
python -m pytest tests/test_release_artifacts.py tests/test_attrition_audit.py tests/test_release.py tests/test_cli_commands.py -q
```

Expected:
- PASS

- [ ] **Step 3: Update paper positioning if needed**

Only if Task 6 concluded `Plan outcome A`:
- add one paragraph to the paper notes explaining that coverage-extension results strengthen auxiliary evidence while leaving final claims anchored to the original relaxed final substrate

- [ ] **Step 4: Final commit**

```bash
git add README.md src/fsmol_cliff/*.py tests/*.py docs/superpowers/plans/2026-03-23-coverage-extension-release-strategy.md paper_latex/notes/coverage-extension-decision.md
git commit -m "feat: add coverage-extension release evaluation workflow"
```

---

## Decision Summary

- Keep current relaxed release as the only `final` substrate.
- Add `relaxed_covext_10_10` as the primary auxiliary `intermediate` profile.
- Add `relaxed_covext_10_5` only if the intermediate release shows a clear positive signal.
- Do not expand the raw assay pool in this phase.
- Do not mix tiers inside one release tree.
- Prioritize H3 strengthening before H1 upgrading.

## Success Criteria

This plan succeeds if:
- auxiliary profiles can be built and audited without altering the current final release
- release artifact generation becomes `result_tier`-safe
- intermediate coverage rises beyond the current `6` relaxed tasks and increases adversarial-eligible support
- a written decision record clearly states whether the intermediate/exploratory releases help, remain appendix-only, or should be dropped

## Failure Criteria

Stop the line of work if:
- intermediate coverage barely grows
- adversarial-eligible coverage does not improve
- cliff density is materially diluted
- H3 intervals do not become cleaner
- H1 remains equally unstable

In that case, preserve the current final release and treat coverage-extension as internal sensitivity analysis only.
