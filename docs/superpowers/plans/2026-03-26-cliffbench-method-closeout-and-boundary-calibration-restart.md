# CliffBench Method Closeout and Boundary-Aware Calibration Restart Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Formally close restart v1, freeze exhausted method families, and leave exactly one restartable boundary-aware / uncertainty-aware calibration line plus one audit-only collapse mechanism line.

**Architecture:** Treat this work as a method-track governance and restart package, not as an open-ended exploration loop. First produce the closeout and freeze documents that make `A1`, `B0`, `C0`, and the older kNN exact families non-restartable by default. Then implement one new ProtoNet calibration family that is explicitly different from `A1` because it uses support-conditioned boundary uncertainty rather than episode-local score refit, and keep robustness work constrained to a collapse-specific audit with no training hook.

**Tech Stack:** Markdown docs, Python 3.12, pytest, pandas/parquet, PyTorch, existing `fsmol_cliff` ProtoNet runner and CLI, YAML configs in `configs/`, output artifacts in `outputs/`.

---

## File Structure

**Create**
- `docs/method_restart_v1_closing_memo.md`
  Purpose: single narrative closeout for `A1`, `B0`, `C0`, the benchmark identity lock, and the one remaining restartable method direction.
- `docs/method_go_no_go_table.md`
  Purpose: compact evidence table covering the method-track `GO` / `NO-GO` state and artifact references.
- `docs/stronger_baseline_gate_failure_patterns.md`
  Purpose: reusable failure taxonomy for stronger-baseline gate misses.
- `docs/closed_families_registry.md`
  Purpose: authoritative "closed / do-not-extend" registry plus reopen conditions.
- `docs/new_calibration_family_proposal.md`
  Purpose: one-page restart proposal for the only allowed new method family.
- `docs/boundary_calibration_pilot_readout.md`
  Purpose: explicit pilot result and `GO` / `NO-GO` readout versus ProtoNet baseline.
- `docs/collapse_specific_perturbation_audit_plan.md`
  Purpose: constrain robustness work to audit-only mechanism discovery.
- `docs/perturbation_audit_v2_readout.md`
  Purpose: summarize the v2 audit and state whether robustness training remains blocked.
- `configs/protonet_boundary_aware_calibration_uncertainty.yaml`
  Purpose: one-family, small-grid pilot config for the new calibration line.
- `src/fsmol_cliff/protonet_boundary_calibration.py`
  Purpose: isolated implementation of the new boundary-aware / uncertainty-aware calibration family.
- `tests/test_protonet_boundary_calibration.py`
  Purpose: unit tests for local boundary uncertainty features and calibrated score bundle semantics.
- `src/fsmol_cliff/collapse_specific_perturbation_audit.py`
  Purpose: new audit-only implementation for collapse-specific perturbations that is explicitly separate from `C0`.
- `tests/test_collapse_specific_perturbation_audit.py`
  Purpose: unit tests for the new audit-only perturbation selectors and summaries.

**Modify**
- `src/fsmol_cliff/protonet_runner.py`
  Purpose: wire the new calibration mode into existing ProtoNet release evaluation without changing episode semantics.
- `src/fsmol_cliff/cli.py`
  Purpose: expose the new ProtoNet calibration mode for controlled pilot evaluation.
- `tests/test_protonet_runner.py`
  Purpose: verify the new calibration mode uses the fixed reporting rule and preserves release evaluation shape.
- `tests/test_cli_commands.py`
  Purpose: verify CLI parsing and pass-through for the new calibration mode and its explicit parameters.

**Reference only**
- `spec_cliffbench_method_restart_v1.md`
  Purpose: canonical method-track gate, failure interpretation, and restart boundary.
- `EXPERIMENT_SUMMARY_2026-03-24.md`
  Purpose: canonical summary of closed exact families and benchmark-side conclusions.
- `paper_latex/main.tex`
  Purpose: canonical paper framing source when `main.pdf` is absent from the workspace.

**Do not modify in this plan**
- `spec.md`
  Reason: benchmark protocol stays frozen.
- `spec_f.md`
  Reason: this plan does not change benchmark completion state.
- `outputs/fsmol_cliff_release_v4`
  Reason: final benchmark substrate remains fixed.
- `outputs/fsmol_cliff_release_v4_covext_intermediate`
  Reason: method-development substrate is reused, not regenerated.
- `src/fsmol_cliff/protonet_local_calibrated.py`
  Reason: `A1` stays closed; the new family must live in a new module, not as a quiet extension of query-only score refit.
- `src/fsmol_cliff/protonet_cliff_margin_train.py`
  Reason: `B0` family is closed by default.
- `src/fsmol_cliff/protonet_perturbation_audit.py`
  Reason: `C0` stays as historical evidence; v2 audit work must use a new module.

---

## Fixed Execution Contract

- Release tier: `intermediate`
- Profile: `relaxed_covext_10_10`
- Task family: `2-way` few-shot
- Episode size: `16` support / class, `16` query / class
- Per task / seed: `400` standard episodes + `400` adversarial episodes
- Seeds: `0..4`
- Aggregation: task-level macro aggregation + paired bootstrap `10,000` iterations
- Primary decision metrics:
  - adversarial `C-BAcc`
  - adversarial `SCR`
  - adversarial `SS-SCR`
- Safety metrics:
  - adversarial `SQ-PSR`
  - adversarial `NC-BAcc`
  - adversarial `NC-PSR`
  - standard `C-BAcc`
  - standard `SCR`
- Promotion rule:
  - beating vanilla `kNN` is insufficient
  - stronger-baseline gate must clear `kNN-cliff-aware`
  - paper-upgrade gate is judged primarily against `ProtoNet`
- Default stop rules:
  - if a method shows only weak adversarial signal but any clean standard-side or safety-side negative, close it immediately
  - if robustness evidence depends on support-dropout variance gap, do not convert it into training
  - if a representation idea is a continuous extension of `B0`, do not start it without a separate justification memo

## Required Inputs Before Work Starts

- `spec_cliffbench_method_restart_v1.md`
- `EXPERIMENT_SUMMARY_2026-03-24.md`
- `paper_latex/main.tex` and compiled `main.pdf` if it exists locally

## Chunk 1: Phase 1 Closeout and Evidence Freeze

### Task 1: Write the restart v1 closing memo

**Files:**
- Create: `docs/method_restart_v1_closing_memo.md`

- [ ] **Step 1: Write the memo skeleton with the required sections**

The document must contain these headings:

```markdown
# Method Restart v1 Closing Memo
## Scope Lock
## Fixed Benchmark Identity
## A1 Closeout
## B0 Closeout
## Expanded C0 Closeout
## Closed Exact Families
## Single Remaining Restart Direction
## Operational Rules For Next Work
```

- [ ] **Step 2: Fill the fixed benchmark identity section from repo evidence**

Fill both `Scope Lock` and `Fixed Benchmark Identity` with concrete content:
- current paper identity stays stronger diagnostic benchmark paper
- benchmark main paper is not upgraded by this closeout
- final substrate stays `outputs/fsmol_cliff_release_v4`
- method-development substrate stays `outputs/fsmol_cliff_release_v4_covext_intermediate`
- main profile stays `relaxed_covext_10_10`
- `H2` remains the formal claim anchor
- all future method work remains under the stronger-baseline gate

- [ ] **Step 3: Fill the freeze boundary sections**

Write explicit content for:
- `Closed Exact Families`
- `Single Remaining Restart Direction`
- `Operational Rules For Next Work`

These sections must state:
- do not continue `A2`
- do not continue `B1` / `B2`
- do not continue `C1` / `C2` / `C3`
- old kNN decision / support / episode exact families stay closed
- only a genuinely new boundary-aware / uncertainty-aware calibration family may restart
- robustness stays audit-only in the current phase
- representation work stays frozen by default

- [ ] **Step 4: Fill the three required failure summaries**

Write these conclusions verbatim in substance:
- `A1`: query-only local calibration showed slight adversarial signal but remained too weak and caused standard-side harm, so it is not a valid expansion entry
- `B0`: minimal cliff-margin loss injection moved the main decision and safety metrics in the wrong direction, so the current margin-loss family is closed
- expanded `C0`: support-dropout sensitivity gap did not hold after scaling up, so consistency learning has no current training entry

- [ ] **Step 5: Verify the memo contains the mandatory closeout and freeze statements**

Run:
```bash
rg -n 'A1 Closeout|B0 Closeout|Expanded C0 Closeout|Closed Exact Families|Single Remaining Restart Direction|Operational Rules For Next Work|stronger diagnostic benchmark paper|relaxed_covext_10_10|do not continue `A2`|do not continue `B1` / `B2`|do not continue `C1` / `C2` / `C3`|audit-only|boundary-aware / uncertainty-aware calibration' docs/method_restart_v1_closing_memo.md
```

Expected:
- all required headings and anchor phrases are present exactly once or more

- [ ] **Step 6: Commit the memo draft**

```bash
git add docs/method_restart_v1_closing_memo.md
git commit -m "docs: add method restart v1 closing memo"
```

### Task 2: Write the `GO` / `NO-GO` evidence table

**Files:**
- Create: `docs/method_go_no_go_table.md`

- [ ] **Step 1: Write the table header and required columns**

Use this table shape:

```markdown
| Family / Phase | Artifact | Comparator | Primary signal | Safety signal | Decision | Why |
|---|---|---|---|---|---|---|
```

- [ ] **Step 2: Add the mandatory rows and fill every evidence column**

The table must include at least:
- `A0`
- `A1`
- `B0 pilot`
- `C0 pilot`
- expanded `C0`
- `decision-aware threshold repair`
- `local-boundary-repair`
- `fixed-support hard-negative replacement`
- `partial-hard-negative augmentation`
- current episode-construction sweep as a grouped row

For every required row, fill:
- `Comparator`
- `Primary signal`
- `Safety signal`
- `Decision`
- `Why`

The `Decision` and `Why` cells must reflect the stronger-baseline gate outcome rather than generic prose.

- [ ] **Step 3: Make the evidence durable**

Each row must point to the actual evidence source. If any decisive evidence currently exists only in ephemeral notes or `/tmp`, copy the decisive metrics and provenance into the table row text so the closeout remains durable.

Prefer durable artifacts such as:
- `outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_protonet_relaxed_covext_10_10.aggregate.json`
- `outputs/c0_expanded_protonet_support_dropout_audit.json`

- [ ] **Step 4: Verify the mandatory rows and decisions are present**

Run:
```bash
rg -n "A0|A1|B0 pilot|C0 pilot|expanded C0|decision-aware threshold repair|local-boundary-repair|fixed-support hard-negative replacement|partial-hard-negative augmentation|episode-construction sweep|GO|NO-GO|weak|stronger-baseline gate|Comparator|Primary signal|Safety signal|Decision|Why" docs/method_go_no_go_table.md
```

Expected:
- every required family or phase is present and every row is fully classified

- [ ] **Step 5: Commit the evidence table**

```bash
git add docs/method_go_no_go_table.md
git commit -m "docs: add method go no-go evidence table"
```

### Task 3: Write the stronger-baseline gate failure patterns document

**Files:**
- Create: `docs/stronger_baseline_gate_failure_patterns.md`

- [ ] **Step 1: Create the reusable failure taxonomy**

The document must include these sections:

```markdown
# Stronger-Baseline Gate Failure Patterns
## Weak Directional Signal With Standard-Side Harm
## Broad Wrong-Way Degradation
## Unstable Or Non-Scaling Mechanism Signal
## Beats Vanilla But Not The Stronger Baseline
## Policy For Closing A Family
```

- [ ] **Step 2: Map concrete families to each pattern**

Required mappings:
- `A1` -> weak directional signal with standard-side harm
- `B0` -> broad wrong-way degradation
- expanded `C0` -> unstable or non-scaling mechanism signal
- `query-targeted support negatives` -> beats vanilla but not the stronger baseline

- [ ] **Step 3: Write the actual closure policy**

Write explicitly:
- the gate is not "is there any positive number"
- the gate is "is there a coherent improvement pattern without safety leakage"
- near-miss results do not keep a family alive by default
- a family that only beats vanilla but not the stronger baseline stays closed or historical-only
- exhausted episode-construction exact families do not reopen under informal reframing

- [ ] **Step 4: Verify all required patterns, family names, and closure policy rules exist**

Run:
```bash
rg -n "Weak Directional Signal With Standard-Side Harm|Broad Wrong-Way Degradation|Unstable Or Non-Scaling Mechanism Signal|Beats Vanilla But Not The Stronger Baseline|Policy For Closing A Family|A1|B0|expanded C0|query-targeted support negatives|near-miss|beats vanilla but not the stronger baseline|episode-construction exact families" docs/stronger_baseline_gate_failure_patterns.md
```

Expected:
- all required headings and family mappings are present

- [ ] **Step 5: Commit the failure-pattern document**

```bash
git add docs/stronger_baseline_gate_failure_patterns.md
git commit -m "docs: add stronger baseline gate failure taxonomy"
```

## Chunk 2: Phase 2 Closed-Family Freeze

### Task 4: Create the closed-families registry

**Files:**
- Create: `docs/closed_families_registry.md`

- [ ] **Step 1: Write the registry header and status legend**

Start with:

```markdown
# Closed Families Registry
## Status Legend
- `closed`: do not extend inside the same family
- `historical evidence`: keep only as reference
- `reopen only with new justification`: requires a separate memo before any experiment
```

- [ ] **Step 2: Register every family that must be frozen**

The registry must include:
- `A1 family` = query-only local score refit
- `B0 family` = coarse cliff-margin loss injection
- `C0 family` = support-subset-dropout variance-gap entry
- `decision-aware threshold repair`
- `local-boundary-repair`
- `fixed-support hard-negative replacement`
- `partial-hard-negative augmentation`
- `query-targeted support negatives`
- `same_scaffold_query_targeted`
- `anchor_coverage_first`
- `paired_hardness_balanced`
- `query_cluster_separation_by_neg_diversity`
- `query_cluster_separation_by_anchor_neg_mix`

Mark `query-targeted support negatives` as historical evidence only, not restartable.

- [ ] **Step 3: Add the prohibited actions block**

Write these items explicitly:
- do not continue `A2`
- do not continue `B1` / `B2`
- do not continue `C1` / `C2` / `C3`
- do not port any old episode variant directly to ProtoNet
- do not continue support-negative tweaking
- do not return to threshold tricks or decision-rule repair

- [ ] **Step 4: Add the only allowed reopen rules**

Record:
- new calibration work is allowed only as a new family distinct from `A1`
- robustness work is allowed only as audit-only mechanism study
- representation work is blocked unless `docs/new_representation_family_justification.md` exists and proves the idea is not a continuation of `B0`

- [ ] **Step 5: Verify the registry contains every closed family and prohibited action**

Run:
```bash
rg -n 'A1 family|B0 family|C0 family|decision-aware threshold repair|local-boundary-repair|fixed-support hard-negative replacement|partial-hard-negative augmentation|query-targeted support negatives|same_scaffold_query_targeted|anchor_coverage_first|paired_hardness_balanced|query_cluster_separation_by_neg_diversity|query_cluster_separation_by_anchor_neg_mix|do not continue `A2`|do not continue `B1` / `B2`|do not continue `C1` / `C2` / `C3`|do not continue support-negative tweaking|do not return to threshold tricks or decision-rule repair|do not port any old episode variant directly to ProtoNet|new_representation_family_justification.md' docs/closed_families_registry.md
```

Expected:
- all required closed families and blocked actions are present

- [ ] **Step 6: Commit the registry**

```bash
git add docs/closed_families_registry.md
git commit -m "docs: register closed method families"
```

### Task 5: Cross-check that closeout is complete before new work starts

**Files:**
- Modify: `docs/method_restart_v1_closing_memo.md`

- [ ] **Step 1: Add a short prerequisite block to the closing memo**

Append:

```markdown
## Prerequisite Check Before Any New Method Pilot
- `docs/method_restart_v1_closing_memo.md` exists
- `docs/method_go_no_go_table.md` exists
- `docs/stronger_baseline_gate_failure_patterns.md` exists
- `docs/closed_families_registry.md` exists
- no old family is being extended under a new name
- any future pilot explicitly maps itself to an allowed reopen rule in `docs/closed_families_registry.md`
```

- [ ] **Step 2: Verify the prerequisite block exists**

Run:
```bash
rg -n "Prerequisite Check Before Any New Method Pilot|closed_families_registry.md|no old family is being extended under a new name|maps itself to an allowed reopen rule" docs/method_restart_v1_closing_memo.md
```

Expected:
- the memo explicitly blocks premature pilot work

- [ ] **Step 3: Commit the prerequisite update**

```bash
git add docs/method_restart_v1_closing_memo.md
git commit -m "docs: block new pilots until family freeze is complete"
```

## Chunk 3: Phase 3 Single New Method Line - Boundary-Aware Calibration

### Task 6: Write the one-page new calibration family proposal

**Files:**
- Create: `docs/new_calibration_family_proposal.md`

- [ ] **Step 1: Write the proposal header and problem statement**

Start with:

```markdown
# Boundary-Aware Calibration Family Proposal
## Objective
## Why `A1` Was Not Enough
## Chosen Restart Family
## Local Feature Contract
## Why This Is Not A Threshold Trick
## Minimal Pilot Definition
## `GO` / `NO-GO` Rule
```

- [ ] **Step 2: Choose exactly one restart family for the pilot**

Select:
- `uncertainty-aware boundary calibration`

Do not leave multiple candidate families active in the proposal.

- [ ] **Step 3: Define the local feature contract**

The proposal must explicitly use only assay-local and episode-local signals, such as:
- raw ProtoNet margin
- prototype margin magnitude
- support dispersion
- local ambiguity
- neighborhood disagreement

The proposal must explicitly reject:
- query-only episode-local logistic refit
- decision-rule replacement
- threshold shifting disguised as calibration
- cross-assay information

- [ ] **Step 4: Verify the proposal names the chosen family and the rejection criteria**

Run:
```bash
rg -n 'uncertainty-aware boundary calibration|Why `A1` Was Not Enough|not a threshold trick|support dispersion|local ambiguity|neighborhood disagreement|cross-assay information' docs/new_calibration_family_proposal.md
```

Expected:
- the proposal clearly defines one family and rejects the old `A1` pattern

- [ ] **Step 5: Commit the proposal**

```bash
git add docs/new_calibration_family_proposal.md
git commit -m "docs: propose boundary-aware calibration restart family"
```

### Task 7: Add the minimal boundary-aware calibration implementation

**Files:**
- Create: `src/fsmol_cliff/protonet_boundary_calibration.py`
- Create: `tests/test_protonet_boundary_calibration.py`
- Modify: `src/fsmol_cliff/protonet_runner.py`
- Modify: `src/fsmol_cliff/cli.py`
- Modify: `tests/test_protonet_runner.py`
- Modify: `tests/test_cli_commands.py`

- [ ] **Step 1: Write the failing unit tests for the new score bundle and anti-threshold behavior**

Add tests in `tests/test_protonet_boundary_calibration.py` like:

```python
def test_boundary_uncertainty_calibration_returns_scores_margins_and_uncertainty() -> None:
    bundle = apply_boundary_uncertainty_calibration(
        episode=episode,
        assay_context=assay_context,
        raw_scores={"qa": 0.54, "qn": 0.46, "a1": 0.80, "n1": 0.20},
        raw_margins={"qa": 0.04, "qn": -0.04, "a1": 0.30, "n1": -0.30},
        top_k=2,
        uncertainty_scale=0.2,
        margin_floor=0.1,
    )
    assert set(bundle) == {
        "raw_scores",
        "calibrated_scores",
        "raw_margins",
        "calibrated_margins",
        "uncertainty_summary",
    }
    assert bundle["uncertainty_summary"]["qa"]["local_ambiguity"] >= 0.0

def test_boundary_uncertainty_calibration_is_identity_at_zero_uncertainty_and_only_shrinks_margin() -> None:
    bundle = apply_boundary_uncertainty_calibration(...)
    assert bundle["calibrated_margins"]["qa"] == bundle["raw_margins"]["qa"]
    assert abs(high_uncertainty_bundle["calibrated_margins"]["qa"]) <= abs(high_uncertainty_bundle["raw_margins"]["qa"])
```

- [ ] **Step 2: Run the unit test to verify it fails**

Run:
```bash
python -m pytest tests/test_protonet_boundary_calibration.py::test_boundary_uncertainty_calibration_returns_scores_margins_and_uncertainty -q
```

Expected:
- FAIL because the new module does not exist yet

- [ ] **Step 3: Write the failing runner and CLI regression tests for the new mode**

Add a test in `tests/test_protonet_runner.py` asserting:
- `boundary_uncertainty` is an accepted ProtoNet calibration mode
- release evaluation still writes the usual metric rows
- discrete predictions still come from the fixed reporting rule over calibrated scores
- the returned score bundle carries `uncertainty_summary`

Add a test in `tests/test_cli_commands.py` asserting:
- `boundary_uncertainty` is accepted by the parser
- `--protonet-calibration-top-k`
- `--protonet-calibration-uncertainty-scale`
- `--protonet-calibration-margin-floor`
are passed through to ProtoNet evaluation

- [ ] **Step 4: Run the runner and CLI regression tests to verify they fail**

Run:
```bash
python -m pytest \
  tests/test_protonet_runner.py::test_evaluate_release_with_protonet_supports_boundary_uncertainty_mode \
  tests/test_cli_commands.py::test_evaluate_command_supports_boundary_uncertainty_parameters \
  -q
```

Expected:
- FAIL because the mode and parameters are not wired yet

- [ ] **Step 5: Implement the new calibration module with a deterministic boundary-uncertainty correction**

Implementation requirements:
- `src/fsmol_cliff/protonet_boundary_calibration.py`
  - compute assay-local, episode-local uncertainty features only
  - include at least `prototype_margin`, `support_dispersion`, `local_ambiguity`, and `neighborhood_disagreement`
  - produce `uncertainty_summary`
  - apply a correction of the form `m'(q) = m(q) * (1 - alpha * uncertainty(q))`
  - do not add any free bias or threshold offset term
  - return identity when uncertainty is zero
  - only shrink margin magnitude toward zero while preserving sign
  - convert margins back to calibrated scores with clipping into `[0.0, 1.0]`
- `src/fsmol_cliff/protonet_runner.py`
  - import the new module without changing `identity` or `query_only`
  - add a new calibration mode name: `boundary_uncertainty`
  - accept `top_k`, `uncertainty_scale`, and `margin_floor` as explicit calibration parameters
- `src/fsmol_cliff/cli.py`
  - add `boundary_uncertainty` to `--protonet-calibration-mode`
  - add explicit CLI flags for `--protonet-calibration-top-k`, `--protonet-calibration-uncertainty-scale`, and `--protonet-calibration-margin-floor`
- `tests/test_protonet_runner.py`
  - preserve the current release result shape
- `tests/test_cli_commands.py`
  - preserve CLI coverage for ProtoNet evaluation parameter plumbing

- [ ] **Step 6: Run the focused tests to verify they pass**

Run:
```bash
python -m pytest \
  tests/test_protonet_boundary_calibration.py \
  tests/test_protonet_runner.py \
  tests/test_cli_commands.py \
  -q
```

Expected:
- PASS

- [ ] **Step 7: Commit the implementation**

```bash
git add \
  src/fsmol_cliff/protonet_boundary_calibration.py \
  src/fsmol_cliff/protonet_runner.py \
  src/fsmol_cliff/cli.py \
  tests/test_protonet_boundary_calibration.py \
  tests/test_protonet_runner.py \
  tests/test_cli_commands.py
git commit -m "feat: add boundary-aware protonet calibration mode"
```

### Task 8: Add the minimal pilot config and run the first gated evaluation

**Files:**
- Create: `configs/protonet_boundary_aware_calibration_uncertainty.yaml`
- Create: `docs/boundary_calibration_pilot_readout.md`

- [ ] **Step 1: Write the one-family, small-grid pilot config**

Use a config shaped like:

```yaml
method: protonet_boundary_aware_calibration
base_checkpoint: checkpoints/PN-Support64_best_validation.pt
release_dir: outputs/fsmol_cliff_release_v4_covext_intermediate
profile: relaxed_covext_10_10
result_tier: intermediate
calibration_mode: boundary_uncertainty
grid:
  top_k: [2, 4]
  uncertainty_scale: [0.1, 0.2]
  margin_floor: [0.1]
selected_row:
  top_k: 2
  uncertainty_scale: 0.1
  margin_floor: 0.1
```

- [ ] **Step 2: Run the first pilot row**

Run:
```bash
mkdir -p outputs/method_boundary_calibration_pilot
PYTHONPATH=src python -m fsmol_cliff.cli evaluate \
  --release-dir outputs/fsmol_cliff_release_v4_covext_intermediate \
  --data-dir fs-mol \
  --checkpoint checkpoints/PN-Support64_best_validation.pt \
  --output outputs/method_boundary_calibration_pilot/task_results_protonet_boundary_uncertainty_relaxed_covext_10_10_k2_u0p1_m0p1.parquet \
  --profile relaxed_covext_10_10 \
  --result-tier intermediate \
  --backend protonet \
  --protonet-calibration-mode boundary_uncertainty \
  --protonet-calibration-top-k 2 \
  --protonet-calibration-uncertainty-scale 0.1 \
  --protonet-calibration-margin-floor 0.1
```

Expected:
- a new pilot parquet is written under `outputs/method_boundary_calibration_pilot/`

- [ ] **Step 3: Aggregate the pilot result**

Run:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli aggregate \
  --input outputs/method_boundary_calibration_pilot/task_results_protonet_boundary_uncertainty_relaxed_covext_10_10_k2_u0p1_m0p1.parquet \
  --output outputs/method_boundary_calibration_pilot/task_results_protonet_boundary_uncertainty_relaxed_covext_10_10_k2_u0p1_m0p1.aggregate.json
```

Expected:
- aggregate JSON exists for the pilot run

- [ ] **Step 4: Compare the pilot against ProtoNet baseline with paired bootstrap**

Run:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli protocol-compare \
  --inputs \
    protonet_base=outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_protonet_relaxed_covext_10_10.parquet \
    boundary_uncertainty=outputs/method_boundary_calibration_pilot/task_results_protonet_boundary_uncertainty_relaxed_covext_10_10_k2_u0p1_m0p1.parquet \
  --comparisons protonet_base:boundary_uncertainty \
  --profile relaxed_covext_10_10 \
  --result-tier intermediate \
  --output outputs/method_boundary_calibration_pilot/protonet_vs_boundary_uncertainty_k2_u0p1_m0p1.paired_comparison.json
```

Expected:
- paired comparison JSON exists with all primary and safety metrics

- [ ] **Step 5: Write the pilot readout and apply the gate**

`docs/boundary_calibration_pilot_readout.md` must report:
- the chosen config row from `selected_row`
- paired delta and `95%` CI for adversarial `C-BAcc`, `SCR`, `SS-SCR`
- paired delta and `95%` CI for adversarial `SQ-PSR`, `NC-BAcc`, `NC-PSR`
- paired delta and `95%` CI for standard `C-BAcc`, `SCR`
- explicit `GO` or `NO-GO`

The readout must also state:
- if the result is only a slight adversarial gain with any clean standard-side or safety-side negative, close this family immediately
- `GO` requires the spec-defined directionality versus ProtoNet, not just a numerically positive row

- [ ] **Step 6: Verify the readout contains the gate metrics**

Run:
```bash
rg -n "paired delta|95% CI|C-BAcc|SCR|SS-SCR|SQ-PSR|NC-BAcc|NC-PSR|standard|GO|NO-GO|close this family immediately|versus ProtoNet" docs/boundary_calibration_pilot_readout.md
```

Expected:
- the readout explicitly applies the gate rather than narrating around it

- [ ] **Step 7: Commit the config and readout**

```bash
git add \
  configs/protonet_boundary_aware_calibration_uncertainty.yaml \
  docs/boundary_calibration_pilot_readout.md
git commit -m "docs: record boundary calibration pilot gate"
```

## Chunk 4: Phase 4 Audit-Only Robustness and Phase 5 Representation Freeze

### Task 9: Write the collapse-specific perturbation audit plan

**Files:**
- Create: `docs/collapse_specific_perturbation_audit_plan.md`

- [ ] **Step 1: Write the audit-only scope lock**

The document must open with:

```markdown
# Collapse-Specific Perturbation Audit Plan
## Scope Lock
- robustness work is audit-only in the current phase
- no consistency training starts from this plan
- `support subset dropout` is historical evidence, not the active audit entry
- no audit outcome in this chunk authorizes robustness training
```

- [ ] **Step 2: Define the allowed audit families**

Include exactly these allowed families:
- `R1`: boundary-neighbor perturbation
- `R2`: pair-order perturbation
- `R3`: local ambiguity perturbation

State explicitly:
- this chunk implements `R1` first
- `R2` and `R3` are documented allowed audit families, not required immediate implementations

- [ ] **Step 3: Define the prohibited follow-on actions**

Write explicitly:
- do not start consistency training
- do not continue `C0 -> C1`
- do not reuse support-dropout variance gap as the training gate
- no audit result in this chunk can directly reopen robustness training
- any future robustness training reopen requires a separate future justification artifact and plan

- [ ] **Step 4: Verify the allowed and blocked audit rules are present**

Run:
```bash
rg -n 'audit-only|boundary-neighbor perturbation|pair-order perturbation|local ambiguity perturbation|this chunk implements `R1` first|do not start consistency training|do not continue `C0 -> C1`|support subset dropout|no audit outcome in this chunk authorizes robustness training|separate future justification artifact and plan' docs/collapse_specific_perturbation_audit_plan.md
```

Expected:
- the plan clearly separates audit from training

- [ ] **Step 5: Commit the audit plan**

```bash
git add docs/collapse_specific_perturbation_audit_plan.md
git commit -m "docs: add collapse specific perturbation audit plan"
```

### Task 10: Implement the audit v2 surface and write the readout

**Files:**
- Create: `src/fsmol_cliff/collapse_specific_perturbation_audit.py`
- Create: `tests/test_collapse_specific_perturbation_audit.py`
- Create: `docs/perturbation_audit_v2_readout.md`

- [ ] **Step 1: Write the failing tests for the new audit selectors and summaries**

Add tests covering at least:

```python
def test_select_boundary_neighbors_prefers_cross_class_support_near_small_margin() -> None:
    ...

def test_measure_pair_order_flip_rate_tracks_rank_and_decision_breaks() -> None:
    ...

def test_summarize_cliff_vs_control_break_rates_reports_matched_control_gap() -> None:
    ...

def test_summarize_collapse_specific_audit_reports_task_distribution() -> None:
    ...
```

- [ ] **Step 2: Run the new audit test file to verify it fails**

Run:
```bash
python -m pytest tests/test_collapse_specific_perturbation_audit.py -q
```

Expected:
- FAIL because the new module does not exist yet

- [ ] **Step 3: Implement the audit v2 module**

Implementation requirements:
- keep it separate from `src/fsmol_cliff/protonet_perturbation_audit.py`
- implement boundary-neighbor perturbation first
- report both pair-order breaks and decision-collapse breaks
- report cliff-vs-control or cliff-vs-matched-noncliff break rates explicitly
- include per-task counts so a few tasks cannot masquerade as a global signal
- write outputs under `outputs/perturbation_audit_v2/`

- [ ] **Step 4: Run the audit tests to verify they pass**

Run:
```bash
python -m pytest tests/test_collapse_specific_perturbation_audit.py -q
```

Expected:
- PASS

- [ ] **Step 5: Run the first v2 audit**

Run:
```bash
mkdir -p outputs/perturbation_audit_v2
PYTHONPATH=src python -m fsmol_cliff.collapse_specific_perturbation_audit \
  --release-dir outputs/fsmol_cliff_release_v4_covext_intermediate \
  --data-dir fs-mol \
  --checkpoint checkpoints/PN-Support64_best_validation.pt \
  --profile relaxed_covext_10_10 \
  --split adversarial \
  --seeds "[0, 1, 2, 3, 4]" \
  --episodes-per-task 5 \
  --perturbation boundary_neighbor \
  --output outputs/perturbation_audit_v2/boundary_neighbor_audit.json
```

Expected:
- a v2 audit JSON exists and reports task-level distribution

- [ ] **Step 6: Write the readout and enforce the decision rule**

`docs/perturbation_audit_v2_readout.md` must state:
- which perturbation was run
- whether the collapse-sensitive signal is stable across tasks
- whether the signal is cliff-specific rather than random-noise sensitivity
- the cliff-vs-control or matched-control break-rate summary
- whether robustness training remains blocked

The default conclusion must be:
- robustness training remains blocked in this plan regardless of audit outcome
- if the signal is stable and cliff-specific, record it only as mechanism evidence for a separate future justification path
- if the signal is not stable and distributed across tasks, do not open any robustness training family

- [ ] **Step 7: Verify the readout contains the block condition**

Run:
```bash
rg -n "stable across tasks|cliff-specific|cliff-vs-control|matched-control|robustness training remains blocked|separate future justification path|do not open any robustness training family" docs/perturbation_audit_v2_readout.md
```

Expected:
- the readout makes the training block explicit

- [ ] **Step 8: Commit the audit implementation and readout**

```bash
git add \
  src/fsmol_cliff/collapse_specific_perturbation_audit.py \
  tests/test_collapse_specific_perturbation_audit.py \
  docs/perturbation_audit_v2_readout.md
git commit -m "feat: add collapse specific perturbation audit v2"
```

### Task 11: Keep the representation family frozen by default

**Files:**
- Modify: `docs/closed_families_registry.md`

- [ ] **Step 1: Add the explicit representation freeze note if it is not already present**

Append:

```markdown
## Representation Family Freeze
- default status: frozen
- do not restart prototype-shaping / margin-loss work from `B0`
- the only allowed reopen artifact is `docs/new_representation_family_justification.md`
- that memo must prove the proposal is not a `margin / lambda / control regularizer` continuation of `B0`
```

- [ ] **Step 2: Verify the freeze note is present**

Run:
```bash
rg -n 'Representation Family Freeze|default status: frozen|new_representation_family_justification.md|not a .* continuation of `B0`' docs/closed_families_registry.md
```

Expected:
- the registry blocks representation restarts by default

- [ ] **Step 3: Commit the freeze note**

```bash
git add docs/closed_families_registry.md
git commit -m "docs: freeze representation family by default"
```

---

## Completion Criteria

This plan is complete only when all of the following are true:

- the four closeout / freeze docs exist:
  - `docs/method_restart_v1_closing_memo.md`
  - `docs/method_go_no_go_table.md`
  - `docs/stronger_baseline_gate_failure_patterns.md`
  - `docs/closed_families_registry.md`
- no old family remains implicitly extendable
- exactly one new method family is active:
  - boundary-aware / uncertainty-aware calibration
- robustness work is recorded only as audit-only mechanism discovery
- representation work remains frozen unless a separate justification memo is approved
- every new pilot result is judged under the stronger-baseline gate rather than narrative optimism

## Execution Order

1. Finish Chunk 1.
2. Finish Chunk 2.
3. Do not start Chunk 3 until Chunk 1 and Chunk 2 are complete.
4. Do not start Task 10 unless Task 9 is complete.
5. Do not start any representation experiment from this plan.
