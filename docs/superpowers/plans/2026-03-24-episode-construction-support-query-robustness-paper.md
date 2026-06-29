# Episode-Construction / Support-Query Robustness Paper Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether episode construction can support a benchmark-plus-protocol paper by showing that support-query robustness is a meaningful intervention axis under the existing FS-Mol-Cliff validation gates.

**Architecture:** Keep [`outputs/fsmol_cliff_release_v4`](../../../outputs/fsmol_cliff_release_v4) as the fixed claim anchor and treat [`outputs/fsmol_cliff_release_v4_covext_intermediate`](../../../outputs/fsmol_cliff_release_v4_covext_intermediate) as the only method-development substrate. Use explicit adversarial episode-variant releases, paired comparisons against both `kNN` and `kNN-cliff-aware`, and a hard promotion gate before any ProtoNet port or paper framing upgrade.

**Tech Stack:** Python 3.12, pytest, pandas/parquet, existing `fsmol_cliff` CLI/release pipeline, `release_artifacts.build_paired_model_comparison_rows`, markdown notes in `paper_latex/notes`, LaTeX in `paper_latex/main.tex`.

---

## File Structure

**Modify**
- `src/fsmol_cliff/manifests.py`
  Purpose: host deterministic episode-construction variants beyond the corrected `query_targeted_support_neg` baseline.
- `src/fsmol_cliff/release.py`
  Purpose: build episode-variant releases from a fixed base release and record protocol metadata.
- `src/fsmol_cliff/cli.py`
  Purpose: expose episode-variant release building and comparison helpers as explicit CLI commands.
- `tests/test_manifests.py`
  Purpose: lock the behavioral contract of each episode variant.
- `tests/test_release.py`
  Purpose: verify episode-variant releases preserve the fixed substrate and only change intended manifests.
- `tests/test_cli_commands.py`
  Purpose: verify CLI entry points for episode-variant release building.
- `paper_latex/notes/training-episode-protocol-status.md`
  Purpose: keep the route-level status and promotion gate current.
- `paper_latex/main.tex`
  Purpose: only upgrade framing if a variant clears the stronger baseline gate.

**Create**
- `docs/superpowers/plans/2026-03-24-episode-construction-support-query-robustness-paper.md`
  Purpose: this plan.
- `src/fsmol_cliff/protocol_compare.py`
  Purpose: reusable paired-comparison helper for protocol variants against fixed baselines.
- `tests/test_protocol_compare.py`
  Purpose: verify paired-comparison filtering and output shape.
- `paper_latex/notes/episode-protocol-eval.md`
  Purpose: collect per-variant results and gate decisions.
- `paper_latex/notes/protocol-checklist-summary.md`
  Purpose: centralize paired-comparison checklist rows for the paper.
- `paper_latex/notes/episode-robustness-paper-outline.md`
  Purpose: article outline tied to gate outcomes.

**Do not modify**
- `spec.md`
  Reason: the formal benchmark protocol remains fixed.
- `spec_f.md`
  Reason: this route is an experiment/article branch, not a change to benchmark completion status.
- `outputs/fsmol_cliff_release_v4`
  Reason: the final release stays the claim anchor.

## Chunk 1: Freeze the Article Scope

### Task 1: Lock the current evidence boundary and paper thesis

**Files:**
- Modify: `paper_latex/notes/training-episode-protocol-status.md`
- Modify: `paper_latex/notes/episode-protocol-eval.md`
- Create: `paper_latex/notes/episode-robustness-paper-outline.md`

- [ ] **Step 1: Write the article thesis and failure boundary**

Record explicitly:
- working article thesis = episode construction is a meaningful intervention axis for cliff-sensitive few-shot classification
- current evidence state = positive versus vanilla `kNN`, not yet positive versus `kNN-cliff-aware`
- rollback identity = stronger diagnostic benchmark paper

- [ ] **Step 2: Record the corrected reference artifact**

Reference:
- `outputs/fsmol_cliff_release_v4_covext_intermediate_query_targeted_support_neg_corrected`
- `outputs/fsmol_cliff_release_v4_covext_intermediate_query_targeted_support_neg_corrected/paired_comparison_knn_and_cliff_aware_vs_query_targeted_support_neg_corrected.json`

Expected note:
- this corrected artifact is the clean episode-construction reference point for the paper route

- [ ] **Step 3: Draft the article outline**

Sections:
- problem framing: ranking--decision split is not the only design axis
- method framing: episode construction as support-query robustness intervention
- evaluation contract: paired protocol gates
- results tree: vanilla `kNN` gain, stronger-baseline gate, ProtoNet portability gate
- failure fallback: if no stronger-baseline win, paper remains benchmark-first

- [ ] **Step 4: Verify the notes are internally consistent**

Run:
```bash
rg -n "query-targeted|kNN-cliff-aware|rollback|support-query robustness" \
  paper_latex/notes/training-episode-protocol-status.md \
  paper_latex/notes/episode-protocol-eval.md \
  paper_latex/notes/episode-robustness-paper-outline.md
```

Expected:
- all three files contain the same thesis and gate language

- [ ] **Step 5: Commit**

```bash
git add \
  paper_latex/notes/training-episode-protocol-status.md \
  paper_latex/notes/episode-protocol-eval.md \
  paper_latex/notes/episode-robustness-paper-outline.md
git commit -m "docs: freeze episode-construction paper scope"
```

## Chunk 2: Make Episode-Variant Evaluation Reusable

### Task 2: Add a dedicated episode-variant release builder

**Files:**
- Modify: `src/fsmol_cliff/release.py`
- Modify: `src/fsmol_cliff/cli.py`
- Modify: `tests/test_release.py`
- Modify: `tests/test_cli_commands.py`

- [ ] **Step 1: Write the failing release test**

Add a test covering:
- input = existing base release dir
- output = new variant release dir
- invariant = assay assets, standard manifests, benchmark manifest profile block remain unchanged
- variant change = adversarial manifest plus `adversarial_episode_variant` metadata

- [ ] **Step 2: Run the release test to verify it fails**

Run:
```bash
python -m pytest tests/test_release.py::test_build_episode_variant_release_rewrites_only_adversarial_manifests -q
```

Expected:
- FAIL because helper/CLI path does not exist yet

- [ ] **Step 3: Write the failing CLI test**

Add a parser/command test for:
- `fsmol-cliff build-episode-variant-release`
- required args:
  - `--base-release-dir`
  - `--output-dir`
  - `--profile`
  - `--adversarial-episode-variant`

- [ ] **Step 4: Run the CLI test to verify it fails**

Run:
```bash
python -m pytest tests/test_cli_commands.py::test_build_episode_variant_release_command_rewrites_release -q
```

Expected:
- FAIL because the subcommand is missing

- [ ] **Step 5: Implement the minimal helper**

Implementation requirements:
- copy the base release tree to a new output directory
- load `task_summaries_<profile>.parquet`
- rebuild only `episodes_adversarial_<profile>.parquet` using the chosen variant
- write `benchmark_manifest.json` with `adversarial_episode_variant`
- write `episode_protocol_note.md`
- rewrite `release_reproducibility.md` with the new command form

- [ ] **Step 6: Run the targeted tests to verify they pass**

Run:
```bash
python -m pytest \
  tests/test_release.py::test_build_episode_variant_release_rewrites_only_adversarial_manifests \
  tests/test_cli_commands.py::test_build_episode_variant_release_command_rewrites_release \
  -q
```

Expected:
- PASS

- [ ] **Step 7: Commit**

```bash
git add src/fsmol_cliff/release.py src/fsmol_cliff/cli.py tests/test_release.py tests/test_cli_commands.py
git commit -m "feat: add episode-variant release builder"
```

### Task 3: Add a reusable paired-comparison helper

**Files:**
- Create: `src/fsmol_cliff/protocol_compare.py`
- Modify: `src/fsmol_cliff/cli.py`
- Create: `tests/test_protocol_compare.py`

- [ ] **Step 1: Write the failing comparison test**

Cover:
- reads two or more task-result parquet files
- filters by `profile` and `result_tier`
- emits rows with `baseline_model`, `treatment_model`, `split_type`, `metric`, `delta_mean`, `ci_low`, `ci_high`

- [ ] **Step 2: Run the comparison test to verify it fails**

Run:
```bash
python -m pytest tests/test_protocol_compare.py::test_protocol_compare_writes_paired_rows -q
```

Expected:
- FAIL because the helper/module does not exist

- [ ] **Step 3: Implement the minimal helper**

Implementation requirements:
- thin wrapper over `build_paired_model_comparison_rows`
- parquet inputs only
- JSON output only
- no new statistics logic

- [ ] **Step 4: Run the comparison test to verify it passes**

Run:
```bash
python -m pytest tests/test_protocol_compare.py -q
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add src/fsmol_cliff/protocol_compare.py src/fsmol_cliff/cli.py tests/test_protocol_compare.py
git commit -m "feat: add protocol comparison helper"
```

## Chunk 3: Implement the Next Episode Variants

### Task 4: Add `anchor_diversified_adversarial`

**Files:**
- Modify: `src/fsmol_cliff/manifests.py`
- Modify: `tests/test_manifests.py`

- [ ] **Step 1: Write the failing manifest test**

Variant definition:
- keep support/query sizes fixed
- keep standard manifests unchanged
- for adversarial manifests, choose injected cliff pairs by maximizing distinct anchors before similarity tie-breaks
- do not rewrite support negatives after the fact

Test behavior:
- more unique anchors than baseline when enough anchors are available
- no duplicate molecules across episode slots

- [ ] **Step 2: Run the manifest test to verify it fails**

Run:
```bash
python -m pytest tests/test_manifests.py::test_build_anchor_diversified_adversarial_episode_manifests_prefers_distinct_anchors -q
```

Expected:
- FAIL because the builder is missing

- [ ] **Step 3: Implement the minimal builder**

Implementation requirements:
- build on the same deterministic infrastructure as existing adversarial manifests
- only change pair selection policy
- register the variant in the adversarial episode registry

- [ ] **Step 4: Run the targeted manifest tests**

Run:
```bash
python -m pytest tests/test_manifests.py -k 'anchor_diversified or adversarial' -q
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add src/fsmol_cliff/manifests.py tests/test_manifests.py
git commit -m "feat: add anchor-diversified adversarial episodes"
```

### Task 5: Add `same_scaffold_query_targeted`

**Files:**
- Modify: `src/fsmol_cliff/manifests.py`
- Modify: `tests/test_manifests.py`

- [ ] **Step 1: Write the failing manifest test**

Variant definition:
- keep support/query sizes fixed
- prefer same-scaffold cliff pairs for injected query negatives when such pairs exist
- fall back to the baseline adversarial rule if same-scaffold pairs are unavailable

Test behavior:
- same-scaffold cliff pairs are chosen before cross-scaffold pairs
- fallback path matches baseline behavior when no same-scaffold pair exists

- [ ] **Step 2: Run the manifest test to verify it fails**

Run:
```bash
python -m pytest tests/test_manifests.py::test_build_same_scaffold_query_targeted_episode_manifests_prefers_same_scaffold_pairs -q
```

Expected:
- FAIL because the builder is missing

- [ ] **Step 3: Implement the minimal builder**

Implementation requirements:
- reuse current cliff-pair schema
- keep deterministic ordering explicit
- register the variant in the adversarial episode registry

- [ ] **Step 4: Run the targeted manifest tests**

Run:
```bash
python -m pytest tests/test_manifests.py -k 'same_scaffold_query_targeted or adversarial' -q
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add src/fsmol_cliff/manifests.py tests/test_manifests.py
git commit -m "feat: add same-scaffold query-targeted episodes"
```

## Chunk 4: Evaluate and Gate Each Variant

### Task 6: Evaluate `anchor_diversified_adversarial` on the intermediate substrate

**Files:**
- Modify: `paper_latex/notes/episode-protocol-eval.md`
- Modify: `paper_latex/notes/protocol-checklist-summary.md`

- [ ] **Step 1: Build the variant release**

Run:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli build-episode-variant-release \
  --base-release-dir outputs/fsmol_cliff_release_v4_covext_intermediate \
  --output-dir outputs/fsmol_cliff_release_v4_covext_intermediate_anchor_diversified \
  --profile relaxed_covext_10_10 \
  --adversarial-episode-variant anchor_diversified_adversarial
```

Expected:
- new release directory exists with rewritten adversarial manifest and protocol note

- [ ] **Step 2: Run `kNN` on the variant release**

Run:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli evaluate \
  --release-dir outputs/fsmol_cliff_release_v4_covext_intermediate_anchor_diversified \
  --output outputs/fsmol_cliff_release_v4_covext_intermediate_anchor_diversified/task_results_knn_relaxed_covext_10_10.parquet \
  --profile relaxed_covext_10_10 \
  --result-tier intermediate \
  --model-name kNN \
  --backend local
```

Expected:
- task-result parquet written

- [ ] **Step 3: Aggregate and compare**

Run:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli aggregate \
  --input outputs/fsmol_cliff_release_v4_covext_intermediate_anchor_diversified/task_results_knn_relaxed_covext_10_10.parquet \
  --output outputs/fsmol_cliff_release_v4_covext_intermediate_anchor_diversified/task_results_knn_relaxed_covext_10_10.aggregate.json
```

Run:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli protocol-compare \
  --inputs \
    kNN=outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_relaxed_covext_10_10.parquet \
    kNN-cliff-aware=outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_cliff_aware_relaxed_covext_10_10.parquet \
    anchor_diversified=outputs/fsmol_cliff_release_v4_covext_intermediate_anchor_diversified/task_results_knn_relaxed_covext_10_10.parquet \
  --comparisons kNN:anchor_diversified kNN-cliff-aware:anchor_diversified \
  --profile relaxed_covext_10_10 \
  --result-tier intermediate \
  --output outputs/fsmol_cliff_release_v4_covext_intermediate_anchor_diversified/paired_comparison.json
```

Expected:
- paired comparison JSON written

- [ ] **Step 4: Apply the gate**

GO only if:
- adversarial `C-BAcc` improves versus `kNN-cliff-aware`
- adversarial `SCR` and `SS-SCR` both fall directionally
- adversarial `SQ-PSR` does not clearly degrade
- controls do not show obvious damage

- [ ] **Step 5: Record the result**

Document:
- aggregate snapshot
- paired deltas
- task-level interpretation
- explicit `GO` or `NO-GO`

- [ ] **Step 6: Commit**

```bash
git add paper_latex/notes/episode-protocol-eval.md paper_latex/notes/protocol-checklist-summary.md
git commit -m "docs: record anchor-diversified episode results"
```

### Task 7: Evaluate `same_scaffold_query_targeted` on the intermediate substrate

**Files:**
- Modify: `paper_latex/notes/episode-protocol-eval.md`
- Modify: `paper_latex/notes/protocol-checklist-summary.md`

- [ ] **Step 1: Build the variant release**

Run:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli build-episode-variant-release \
  --base-release-dir outputs/fsmol_cliff_release_v4_covext_intermediate \
  --output-dir outputs/fsmol_cliff_release_v4_covext_intermediate_same_scaffold_query_targeted \
  --profile relaxed_covext_10_10 \
  --adversarial-episode-variant same_scaffold_query_targeted
```

- [ ] **Step 2: Run `kNN` and aggregate**

Run:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli evaluate \
  --release-dir outputs/fsmol_cliff_release_v4_covext_intermediate_same_scaffold_query_targeted \
  --output outputs/fsmol_cliff_release_v4_covext_intermediate_same_scaffold_query_targeted/task_results_knn_relaxed_covext_10_10.parquet \
  --profile relaxed_covext_10_10 \
  --result-tier intermediate \
  --model-name kNN \
  --backend local
```

Run:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli aggregate \
  --input outputs/fsmol_cliff_release_v4_covext_intermediate_same_scaffold_query_targeted/task_results_knn_relaxed_covext_10_10.parquet \
  --output outputs/fsmol_cliff_release_v4_covext_intermediate_same_scaffold_query_targeted/task_results_knn_relaxed_covext_10_10.aggregate.json
```

- [ ] **Step 3: Compare against fixed baselines**

Run:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli protocol-compare \
  --inputs \
    kNN=outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_relaxed_covext_10_10.parquet \
    kNN-cliff-aware=outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_cliff_aware_relaxed_covext_10_10.parquet \
    same_scaffold_query_targeted=outputs/fsmol_cliff_release_v4_covext_intermediate_same_scaffold_query_targeted/task_results_knn_relaxed_covext_10_10.parquet \
  --comparisons kNN:same_scaffold_query_targeted kNN-cliff-aware:same_scaffold_query_targeted \
  --profile relaxed_covext_10_10 \
  --result-tier intermediate \
  --output outputs/fsmol_cliff_release_v4_covext_intermediate_same_scaffold_query_targeted/paired_comparison.json
```

- [ ] **Step 4: Apply the same gate and record the result**

Use the same `GO` / `NO-GO` criteria as Task 6.

- [ ] **Step 5: Commit**

```bash
git add paper_latex/notes/episode-protocol-eval.md paper_latex/notes/protocol-checklist-summary.md
git commit -m "docs: record same-scaffold episode results"
```

## Chunk 5: Port Only a Winning Variant

### Task 8: Port exactly one `GO` variant to ProtoNet

**Files:**
- Modify: `src/fsmol_cliff/protonet_runner.py`
- Modify: `tests/test_protonet_runner.py`
- Modify: `paper_latex/notes/episode-protocol-eval.md`

- [ ] **Step 1: Stop if no variant beat `kNN-cliff-aware`**

Expected:
- if all variants are `NO-GO`, do not do this task

- [ ] **Step 2: Write the failing ProtoNet test**

Cover:
- the winning release variant loads through the standard release-mode ProtoNet path
- no manifest-path special casing breaks

- [ ] **Step 3: Run the ProtoNet test to verify it fails**

Run:
```bash
python -m pytest tests/test_protonet_runner.py::test_evaluate_release_with_protonet_accepts_episode_variant_release -q
```

Expected:
- FAIL because the winning path is not yet wired

- [ ] **Step 4: Implement the minimal ProtoNet compatibility change**

Constraint:
- do not change ProtoNet scoring semantics
- only make sure the release-mode path accepts the winning episode variant cleanly

- [ ] **Step 5: Run the ProtoNet test to verify it passes**

Run:
```bash
python -m pytest tests/test_protonet_runner.py::test_evaluate_release_with_protonet_accepts_episode_variant_release -q
```

- [ ] **Step 6: Run the winning ProtoNet evaluation**

Run:
```bash
PYTHONPATH=src python -m fsmol_cliff.cli evaluate \
  --release-dir <winning_release_dir> \
  --data-dir <fsmol_dir> \
  --checkpoint <protonet_ckpt> \
  --output <winning_release_dir>/task_results_protonet_relaxed_covext_10_10.parquet \
  --profile relaxed_covext_10_10 \
  --result-tier intermediate \
  --backend protonet
```

Expected:
- ProtoNet parquet written

- [ ] **Step 7: Aggregate, compare, and record**

Use the same aggregate + `protocol-compare` flow as the `kNN` evaluations, but swap in `ProtoNet` as the baseline.

- [ ] **Step 8: Commit**

```bash
git add src/fsmol_cliff/protonet_runner.py tests/test_protonet_runner.py paper_latex/notes/episode-protocol-eval.md
git commit -m "feat: port winning episode protocol to protonet"
```

## Chunk 6: Paper Go/No-Go

### Task 9: Upgrade or stop the article route explicitly

**Files:**
- Modify: `paper_latex/notes/episode-robustness-paper-outline.md`
- Modify: `paper_latex/notes/training-episode-protocol-status.md`
- Modify maybe: `paper_latex/main.tex`

- [ ] **Step 1: Evaluate the article gate**

Article `GO` requires:
- at least one episode variant beats `kNN-cliff-aware` cleanly on the intermediate substrate
- the gain is not just a query-easing artifact
- at least one portability check is positive or at minimum non-contradictory

Article `NO-GO` if:
- all variants remain below `kNN-cliff-aware`
- gains are unstable across same-scaffold / collapse metrics
- portability collapses

- [ ] **Step 2: If `NO-GO`, write the failure-stable closeout**

Write:
- episode construction matters
- current variants are insufficient for a protocol paper
- keep this as future work or appendix discussion, not a main-paper upgrade

- [ ] **Step 3: If `GO`, upgrade the outline into paper edits**

Promote:
- the intervention-axis framing
- the fixed paired checklist
- the winning variant and portability result

Do not promote:
- any rejected variant
- any result that only beats vanilla `kNN`

- [ ] **Step 4: Verify the final state**

Run:
```bash
python -m pytest -q \
  tests/test_manifests.py \
  tests/test_release.py \
  tests/test_cli_commands.py \
  tests/test_protocol_compare.py
```

Expected:
- PASS

If ProtoNet wiring changed, also run:
```bash
python -m pytest tests/test_protonet_runner.py -q
```

- [ ] **Step 5: Commit**

```bash
git add \
  paper_latex/notes/episode-robustness-paper-outline.md \
  paper_latex/notes/training-episode-protocol-status.md \
  paper_latex/main.tex
git commit -m "docs: resolve episode-construction paper gate"
```
