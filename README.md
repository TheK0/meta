# FS-Mol-Cliff

Independent implementation of the FS-Mol-Cliff benchmark and hypothesis-validation protocol for assay-level few-shot molecular classification.

## Overview

This repository builds and evaluates a cliff-aware benchmark on top of FS-Mol. It covers:

- assay-local asset construction for cliff pairs, hard negatives, and molecule annotations
- frozen release bundle generation for `strict` and `relaxed` benchmark profiles
- evaluation runners for sklearn baselines, official-style adapters, ProtoNet, and legacy MAML compatibility
- task-level aggregation, paired comparisons, and hypothesis-oriented reporting for `H1` / `H2` / `H3`

The current project state is centered on the `v4.0` release workflow under [`outputs/fsmol_cliff_release_v4`](./outputs/fsmol_cliff_release_v4).

## Current Release Status

FS-Mol-Cliff `v4.0` uses a dual-profile release policy:

- `relaxed`: main benchmark / formal comparison profile
- `strict`: mechanism stress test / mini benchmark

Current release facts:

- raw test assays: `157`
- relaxed-eligible assays: `6`
- strict-eligible assays: `2`
- full-strength release families: `kNN`, `RF`, `ProtoNet`, `kNN-cliff-aware`
- exploratory compatibility family: `MAML`

Current claim status on the released relaxed artifacts:

- `H1`: supported trend
- `H2`: formal claim
- `H3`: supported trend

Release policy note:

- [`outputs/fsmol_cliff_release_v4`](./outputs/fsmol_cliff_release_v4) remains the current `final` benchmark substrate.
- Any future coverage-extension releases are auxiliary `intermediate` / `exploratory` layers for robustness analysis and do not replace the current final relaxed benchmark.

Release entry points:

- [Release Summary](./outputs/fsmol_cliff_release_v4/release_summary.md)
- [Relaxed Claim Summary](./outputs/fsmol_cliff_release_v4/relaxed_claim_summary.md)
- [Paper-Grade Paired Comparisons](./outputs/fsmol_cliff_release_v4/relaxed_model_comparisons_paper.md)
- [Benchmark Decision Note](./outputs/fsmol_cliff_release_v4/benchmark_decision_note.md)
- [Release Reproducibility](./outputs/fsmol_cliff_release_v4/release_reproducibility.md)

## Method Track

The benchmark protocol and the post-benchmark method exploration are tracked separately:

- [Benchmark Protocol Spec](./spec.md)
- [Benchmark Completion Status](./spec_f.md)
- [Method Restart Spec](./spec_cliffbench_method_restart_v1.md)
- [Experiment Summary](./EXPERIMENT_SUMMARY_2026-03-24.md)

Current default reading:

- `spec.md` remains the frozen benchmark protocol
- `spec_cliffbench_method_restart_v1.md` defines the restarted independent method track
- `EXPERIMENT_SUMMARY_2026-03-24.md` records which exploratory families were tried and why they passed or failed

## Installation

Python `3.12+` is required.

Minimal install:

```bash
pip install -e .
```

Development install with tests:

```bash
pip install -e ".[dev]"
```

Install chemistry / parquet extras when building assets or reading parquet outputs locally:

```bash
pip install -e ".[chem,dev]"
```

If you invoke modules directly without installation, use `PYTHONPATH=src`.

## Common Commands

Run the full test suite:

```bash
python -m pytest -q
```

Check official adapter availability:

```bash
PYTHONPATH=src python -m fsmol_cliff.cli adapter-status --output /tmp/status.json
```

Build a frozen release bundle:

```bash
PYTHONPATH=src python -m fsmol_cliff.cli build-release \
  --data-dir <fsmol_dir> \
  --output-dir <out> \
  --profile relaxed
```

Audit assay attrition and threshold sensitivity for a release:

```bash
PYTHONPATH=src python -m fsmol_cliff.cli audit-attrition \
  --release-dir <release_dir> \
  --data-dir <fsmol_dir> \
  --output-dir <audit_out> \
  --profile relaxed
```

Evaluate a release bundle with a sklearn baseline:

```bash
PYTHONPATH=src python -m fsmol_cliff.cli evaluate \
  --release-dir <release_dir> \
  --output <task_results.parquet> \
  --profile relaxed \
  --model-name RF
```

Aggregate task-level outputs:

```bash
PYTHONPATH=src python -m fsmol_cliff.cli aggregate \
  --input <task_results.parquet> \
  --output <aggregate.json>
```

## Project Layout

Core package: [`src/fsmol_cliff`](./src/fsmol_cliff)

Key modules:

- [`assets.py`](./src/fsmol_cliff/assets.py), [`pipeline.py`](./src/fsmol_cliff/pipeline.py), [`release.py`](./src/fsmol_cliff/release.py): build assay assets and frozen release bundles
- [`audit.py`](./src/fsmol_cliff/audit.py): attrition and threshold-sensitivity reporting
- [`episodes.py`](./src/fsmol_cliff/episodes.py), [`manifests.py`](./src/fsmol_cliff/manifests.py): standard and adversarial episode generation
- [`evaluation.py`](./src/fsmol_cliff/evaluation.py), [`runner.py`](./src/fsmol_cliff/runner.py), [`aggregate.py`](./src/fsmol_cliff/aggregate.py): scoring, task summaries, and macro aggregation
- [`release_artifacts.py`](./src/fsmol_cliff/release_artifacts.py): main-table, taxonomy, and paired-comparison artifact generation
- [`hypotheses.py`](./src/fsmol_cliff/hypotheses.py), [`reports.py`](./src/fsmol_cliff/reports.py): claim validation and report rendering
- [`adapters.py`](./src/fsmol_cliff/adapters.py), [`fsmol_bridge.py`](./src/fsmol_cliff/fsmol_bridge.py), [`protonet_runner.py`](./src/fsmol_cliff/protonet_runner.py), [`maml_legacy_runner.py`](./src/fsmol_cliff/maml_legacy_runner.py): model/runtime integration

Tests: [`tests`](./tests)

Vendored runtime support:

- [`vendor/MAT`](./vendor/MAT)

Release artifacts:

- [`outputs/fsmol_cliff_release_v4`](./outputs/fsmol_cliff_release_v4)

## Key Outputs

The `v4.0` release directory contains the main published assets:

- benchmark manifests and frozen episode bundles
- assay-level cliff / hard-negative / annotation assets
- model task results and aggregated summaries
- relaxed main table, failure taxonomy, and full paired comparisons
- claim and release summary documents for paper-facing interpretation

Recommended reading order:

1. [Release Summary](./outputs/fsmol_cliff_release_v4/release_summary.md)
2. [Relaxed Claim Summary](./outputs/fsmol_cliff_release_v4/relaxed_claim_summary.md)
3. [Relaxed Main Table](./outputs/fsmol_cliff_release_v4/relaxed_main_table.md)
4. [Relaxed Failure Taxonomy](./outputs/fsmol_cliff_release_v4/relaxed_failure_taxonomy.md)
5. [Paper-Grade Paired Comparisons](./outputs/fsmol_cliff_release_v4/relaxed_model_comparisons_paper.md)

## Notes

- `strict` is intentionally kept as a published mechanism profile even though it is too small for benchmark-wide formal claims by itself.
- `relaxed` is the main comparison substrate because it preserves the cliff definition while expanding structural-similarity coverage enough to support a larger task set.
- `MAML` is currently available only through the legacy compatibility path; rebuilt release rows are exploratory and are not part of the strongest final-claim substrate.
- A full-episode `MAML final` path is not treated as an immediate release requirement because the current legacy runner is still smoke-oriented (`max_episodes=3`, `max_num_epochs=1`, `patience=1`) and would need a separate runtime-quality upgrade before it could be considered final-strength.
- Official adapter availability depends on local environment and external FS-Mol runtime compatibility. Use `adapter-status` before assuming a model family is runnable.
