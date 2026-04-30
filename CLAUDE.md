# CLAUDE.md — CliffBench Project Context

## What This Project Is

**CliffBench** — an activity-cliff diagnostic benchmark for assay-local few-shot molecular classification. It uses activity cliffs as probes to reveal decision-layer collapse: models may preserve hard-pair ranking while failing to form usable classification boundaries in cliff-heavy regions.

The project spans the full pipeline: raw FS-Mol data ingestion → assay asset construction → episode generation → model scoring → statistical aggregation → hypothesis validation → paper rendering.

**Paper:** "Activity Cliffs as Probes of Decision-Layer Collapse in Few-Shot Molecular Classification"
- Target: NeurIPS 2026 (using NeurIPS 2025 template as fallback)
- Status: Anonymous Authors, pre-submission
- Source: `paper_latex/main.tex`
- Note: compiled PDF currently absent from the workspace; compile `paper_latex/main.tex` to get `main.pdf`

---

## Frozen Paper Identity (DO NOT CHANGE WITHOUT EXPLICIT DECISION)

- **Paper type:** stronger diagnostic benchmark paper — NOT a benchmark+method paper
- **Claim ladder:**
  - H1 (Cliff Gap / average-metric misalignment): **supported trend**
  - H2 (Ranking-Decision Decoupling / decision-layer collapse): **formal claim**
  - H3 (Intervention / cliff-aware repair): **supported trend**
- **Final benchmark substrate:** `outputs/fsmol_cliff_release_v4` (2 strict + 6 relaxed assays)
- **Method-development substrate:** `outputs/fsmol_cliff_release_v4_covext_intermediate` (10 assays, profile `relaxed_covext_10_10`)
- **H2 formal claim evidence:** `outputs/fsmol_cliff_release_v4/relaxed_claim_summary.md`
- **Paper limitations (from Discussion section):**
  1. Narrow task coverage (6 relaxed + 2 strict)
  2. Behavioral evidence, not causal proof of mechanism
  3. MAML retained only as exploratory compatibility
  4. Missing threshold-fragility analysis and richer control reporting

---

## Key Documents — Read Order

| # | File | Purpose |
|---|------|---------|
| 1 | `paper_latex/main.tex` | The paper — narrative, motivation, results |
| 2 | `spec.md` | Frozen benchmark protocol specification |
| 3 | `spec_f.md` | Completion status by spec section (2 partial remain) |
| 4 | `EXPERIMENT_SUMMARY_2026-03-24.md` | Canonical summary of ALL method exploration: what was tried, what worked/failed, final conclusions |
| 5 | `outputs/fsmol_cliff_release_v4/relaxed_claim_summary.md` | H1/H2/H3 evidence with exact numbers |
| 6 | `spec_cliffbench_method_restart_v1.md` | Method-line governance: hard constraints, success gates, allowed directions |
| 7 | `docs/closed_families_registry.md` | All 13 closed families with prohibited actions |
| 8 | `docs/method_go_no_go_table.md` | Evidence table for every evaluated family |
| 9 | `docs/new_calibration_family_proposal.md` | The ONE allowed restart direction |
| 10 | `docs/boundary_calibration_pilot_readout.md` | Pilot status and gate checklist |
| 11 | `docs/method_restart_v1_closing_memo.md` | Formal closeout of v1 method exploration |
| 12 | `docs/stronger_baseline_gate_failure_patterns.md` | Reusable failure taxonomy |

---

## Architecture and Data Flow

```
FS-Mol raw data (external checkout at $FSMOL_EXTERNAL_ROOT or /Volumes/macplus/project/meta/external/FS-Mol)
  │
  ├─ chem.py           RDKit: Morgan fingerprints, Tanimoto similarity, Murcko scaffolds
  ├─ assets.py         Filter assay records, mine cliff/noncliff pairs
  ├─ pipeline.py       Build assay asset bundles (pairs.jsonl, annotations.parquet, diagnostics.json)
  ├─ task_selection.py Filter/rank tasks for benchmark eligibility
  │
  ├─ models.py         PairRecord, BenchmarkManifest dataclasses
  ├─ constants.py      EpisodeConfig, ProtocolConstants (τ=0.85, δ=1.0), BenchmarkProfile
  ├─ episodes.py       Adversarial episode construction (bipartite matching + DFS injection selection)
  ├─ manifests.py      Standard + 7 adversarial episode manifest variants
  │
  ├─ release.py        Build frozen release bundles
  ├─ fetch.py          Data fetching + source manifest
  ├─ benchmark.py      Benchmark manifest with git commit resolution
  │
  ├─ evaluation.py     Core evaluation loop: score episodes → compute 9 metrics
  ├─ runner.py         Release evaluation orchestrator (sklearn/ProtoNet/MAML backends)
  ├─ metrics.py        9 core metrics in 3 families (classification, pair-ranking, collapse)
  │
  ├─ aggregate.py      Bootstrap CI (10k iterations), macro aggregation
  ├─ hypotheses.py     H1/H2/H3 validation logic
  ├─ audit.py          Attrition audit + threshold sensitivity
  ├─ protocol_compare.py  Cross-model paired comparison
  │
  ├─ adapters.py       Sklearn/official/cliff-aware model adapters
  ├─ fsmol_bridge.py   External FS-Mol codebase bridge (hash-locked source patches)
  ├─ torch_scatter_compat.py  Pure-PyTorch scatter ops (MPS compatibility)
  │
  ├─ protonet_runner.py       ProtoNet model loading + low-level scoring
  ├─ protonet_local_calibrated.py  Identity + query-only logistic calibration
  ├─ protonet_boundary_calibration.py  NEW: boundary-aware uncertainty calibration
  ├─ protonet_cliff_margin_train.py    B0: cliff-margin ProtoNet training (CLOSED)
  ├─ protonet_perturbation_audit.py    C0: perturbation stability audit (CLOSED)
  │
  ├─ training_losses/cliff_margin.py  B0 loss functions
  ├─ maml_legacy.py / maml_legacy_runner.py  Legacy TensorFlow MAML (exploratory)
  │
  └─ cli.py            11 subcommands: fetch → build → evaluate → aggregate → validate
```

**Module map** (what imports what, no circular deps):
- `chem` ← `assets` ← `pipeline` ← `release` ← `cli`
- `constants`, `models` ← `episodes` ← `manifests` ← `release`
- `evaluation` ← `runner` ← `cli`
- `adapters` ← `runner`
- `fsmol_bridge` ← `adapters`, `protonet_runner`
- `protonet_runner` ← `runner` (via `_pn` module lookup)

---

## Hardware and Environment Requirements

- **External FS-Mol checkout:** required at `/Volumes/macplus/project/meta/external/FS-Mol` or set `FSMOL_EXTERNAL_ROOT` env var
- **ProtoNet checkpoint:** `checkpoints/PN-Support64_best_validation.pt` (~73 MB, PyTorch)
- **Python:** 3.12+
- **Key deps:** pandas, scipy, scikit-learn, PyTorch, RDKit (optional but recommended)
- **MAML legacy:** requires separate `fsmol-maml-legacy` conda environment with TensorFlow 2.x
- **FS-Mol data:** `fs-mol/` directory with assay JSONL files

---

## Protocol Constants (FROZEN)

```python
EpisodeConfig:    N-way=2, 16 support/class, 16 query/class
ProtocolConstants: τ=0.85 (similarity), δ=1.0 (activity gap), hard_neg_pool=32
BenchmarkProfile: strict (τ=0.90, δ=1.0), relaxed (τ=0.85, δ=1.0),
                  relaxed_covext_10_10, relaxed_covext_10_5
```

---

## Core Metrics (9 in 3 families)

| Family | Metrics | What it measures |
|--------|---------|-----------------|
| Classification | `c_bacc`, `nc_bacc` | Balanced accuracy on cliff/non-cliff query subsets |
| Pair-ranking | `q_psr`, `nc_psr`, `sq_psr`, `ss_q_psr`, `ss_nc_psr`, `ss_sq_psr` | Pair success rate (does model rank active > inactive?) |
| Collapse | `scr`, `ss_scr` | Score collapse rate (fraction of pairs getting same binary prediction) |

---

## Current Results Snapshot (relaxed_covext_10_10, intermediate tier)

### Aggregate (adversarial split, full 400 episodes)

| Model | C-BAcc | SCR | SQ-PSR | NC-BAcc |
|-------|--------|-----|--------|---------|
| ProtoNet | 0.561 [0.518,0.616] | 0.843 [0.776,0.903] | 0.786 | — |
| kNN | 0.511 [0.452,0.568] | 0.906 [0.872,0.940] | 0.562 | — |
| kNN-cliff-aware | 0.533 [0.481,0.593] | 0.844 [0.773,0.906] | 0.572 | — |
| RF | 0.524 [0.486,0.588] | 0.912 [0.872,0.947] | 0.917 | — |

### Key deltas (paired bootstrap, 10k iterations)

| Comparison | Metric | Delta | 95% CI |
|-----------|--------|-------|--------|
| kNN → RF (adversarial) | SQ-PSR | +0.355 | [0.313, 0.386] |
| kNN → RF (adversarial) | C-BAcc | +0.013 | [-0.014, 0.056] |
| kNN → kNN-cliff-aware (adversarial) | C-BAcc | +0.022 | [0.002, 0.045] |
| kNN → kNN-cliff-aware (adversarial) | SCR | -0.061 | [-0.108, -0.021] |

**Interpretation:** RF is the clearest ranking-competent-but-decision-collapsed case (SQ-PSR very high, C-BAcc near random). ProtoNet is most balanced (lowest SCR). kNN-cliff-aware shows modest but consistent gains.

Evidence sources:
- `outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_protonet_relaxed_covext_10_10.aggregate.json`
- `outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_relaxed_covext_10_10.aggregate.json`
- `outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_knn_cliff_aware_relaxed_covext_10_10.aggregate.json`
- `outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_rf_relaxed_covext_10_10.aggregate.json`

---

## Method Exploration: ALL Closed Families

**Hard rule:** All 19 evaluated families are closed. Do not extend, iterate, or reframe ANY of them. The closed families registry at `docs/closed_families_registry.md` is authoritative.

| Family | Key failure mode |
|--------|-----------------|
| decision-aware threshold repair | Minimal primary signal; no clean improvement |
| local-boundary-repair | Standard collapse improved but adversarial signal not clean; control-side harm |
| fixed-support hard-negative replacement | Too aggressive; degraded ranking/collapse balance |
| partial-hard-negative augmentation | Even conservative version degraded stronger baseline |
| query-targeted support negatives (episode) | Beats vanilla kNN only; fails vs kNN-cliff-aware → historical evidence |
| same_scaffold_query_targeted | Collapse materially worse; hard NO-GO |
| anchor_coverage_first | Sacrificed ranking and collapse; hard NO-GO |
| paired_hardness_balanced | Cleaner than above but still failed stronger-baseline gate |
| query_cluster_separation_by_neg_diversity | Collapse worsened relative to cliff-aware |
| query_cluster_separation_by_anchor_neg_mix | Worst variant; broad wrong-way degradation |
| A1 (query-only logistic calibration) | Weak signal + standard-side harm |
| B0 (cliff-margin loss injection) | Broad wrong-way degradation on primary metrics |
| C0 (support-dropout perturbation audit) | Mechanism signal did not scale |

---

## Active Method Line: Boundary-Aware Calibration

**Status:** Pilot implementation complete; full evaluation pending.

**What it does:** Deterministic margin shrinkage based on support-conditioned boundary uncertainty:
```
composite_uncertainty = (local_ambiguity + dispersion_ratio + neighborhood_disagreement) / 3
calibrated_margin = raw_margin × (1 - uncertainty_scale × composite_uncertainty)
```

**Key difference from A1:** No logistic regression, no learned refit, no threshold shifting. Pure per-molecule margin shrinkage proportional to local uncertainty.

**Files:**
- `src/fsmol_cliff/protonet_boundary_calibration.py` — implementation
- `configs/protonet_boundary_aware_calibration_uncertainty.yaml` — pilot config
- `docs/new_calibration_family_proposal.md` — design + GO/NO-GO rules
- `docs/boundary_calibration_pilot_readout.md` — pilot status + gate checklist

**CLI:** `--protonet-calibration-mode boundary_uncertainty --protonet-calibration-top-k 2 --protonet-calibration-uncertainty-scale 0.1 --protonet-calibration-margin-floor 0.1`

**Pilot smoke test result:** Pipeline confirmed end-to-end. Full evaluation (8000 episodes) needed for gate decision.

**Success gate (vs ProtoNet baseline):**
- Primary: adv C-BAcc delta > 0, adv SCR delta < 0, adv SS-SCR delta <= 0
- Safety: no clean negative on SQ-PSR, NC-BAcc, NC-PSR, standard C-BAcc, standard SCR

**To run the full evaluation:**
```bash
# Step 1: Full evaluation
PYTHONPATH=src python -m fsmol_cliff.cli evaluate \
  --release-dir outputs/fsmol_cliff_release_v4_covext_intermediate \
  --data-dir fs-mol \
  --checkpoint checkpoints/PN-Support64_best_validation.pt \
  --output outputs/method_boundary_calibration_pilot/task_results_protonet_boundary_uncertainty_full.parquet \
  --profile relaxed_covext_10_10 --result-tier intermediate --backend protonet \
  --protonet-calibration-mode boundary_uncertainty \
  --protonet-calibration-top-k 2 --protonet-calibration-uncertainty-scale 0.1 \
  --protonet-calibration-margin-floor 0.1

# Step 2: Aggregate
PYTHONPATH=src python -m fsmol_cliff.cli aggregate \
  --input outputs/method_boundary_calibration_pilot/task_results_protonet_boundary_uncertainty_full.parquet \
  --output outputs/method_boundary_calibration_pilot/...aggregate.json

# Step 3: Compare vs ProtoNet baseline
PYTHONPATH=src python -m fsmol_cliff.cli protocol-compare \
  --inputs protonet_base=outputs/fsmol_cliff_release_v4_covext_intermediate/task_results_protonet_relaxed_covext_10_10.parquet \
           boundary_uncertainty=outputs/method_boundary_calibration_pilot/task_results_protonet_boundary_uncertainty_full.parquet \
  --comparisons protonet_base:boundary_uncertainty \
  --profile relaxed_covext_10_10 --result-tier intermediate \
  --output outputs/method_boundary_calibration_pilot/protonet_vs_boundary_uncertainty_full.paired_comparison.json
```

---

## Engineering Health

### Recently Fixed (session 2026-04-30)
- FS-Mol bridge: hash-locked upstream sources, sys.path context manager, early error on missing checkout, post-load symbol validation
- Code duplication: eliminated ~90 lines of dead code between runner.py and protonet_runner.py
- Duplicate test function: removed silent test loss in test_cli_commands.py
- Shared functions: extracted resolve_manifest_path, resolve_assay_path, load_assay_context to io.py

### Known Issues (not yet addressed)

| Priority | Issue | Location |
|----------|-------|----------|
| P0 | Hardcoded path `/Volumes/macplus/project/meta/external/FS-Mol` for MAML subprocess | `runner.py:110` |
| P1 | `pipeline.py` fallback defaults (τ=0.85, δ=1.0) duplicate `constants.py` — drift risk | `pipeline.py:37-39` |
| P1 | `@lru_cache(maxsize=None)` on recursive DFS search — unbounded memory | `episodes.py:180` |
| P2 | RDKit unavailable → silent degradation to empty benchmark (zero cliff pairs) | `chem.py` |
| P2 | `torch.load(weights_only=False)` — pickle deserialization risk | `protonet_runner.py:83` |
| P2 | Pre-existing test failure: CLI subcommand count mismatch | `tests/test_bootstrap.py:110` |
| P3 | `/tmp` file collision in concurrent MAML subprocess runs | `runner.py:149` |

### Working Tree State (uncommitted)

Seven files have uncommitted modifications (mostly from pre-existing episode-construction work):
- `src/fsmol_cliff/episodes.py`, `manifests.py`, `release.py`, `evaluation.py`
- `tests/test_manifests.py`, `tests/test_release.py`
- `README.md`

Thirty untracked files/directories that should be committed as-is (historical evidence):
- Method infrastructure: `protonet_base.py`, `protonet_local_calibrated.py`, `protonet_cliff_margin.py`, `protonet_cliff_margin_train.py`, `protonet_perturbation_audit.py`, `torch_scatter_compat.py`, `training_losses/`, `protocol_compare.py`
- Config: `configs/protonet_cliff_margin.yaml`
- Tests: `test_protonet_local_calibration.py`, `test_protonet_perturbation_audit.py`, `test_protonet_cliff_margin_integration.py`, `test_protonet_cliff_margin_train.py`, `test_torch_scatter_compat.py`, `test_cliff_margin_loss.py`, `test_protocol_compare.py`
- Docs: `spec_cliffbench_method_restart_v1.md`, `EXPERIMENT_SUMMARY_2026-03-24.md`, `paper_latex/notes/*.md`, `docs/superpowers/plans/*.md`

---

## Development Commands

```bash
# Full test suite (expect 172+ passed, 1 pre-existing skip)
python -m pytest tests/ --deselect tests/test_bootstrap.py::test_cli_exposes_expected_top_level_subcommands -q

# Focused test
python -m pytest tests/test_protonet_boundary_calibration.py -v

# CLI (always with PYTHONPATH=src)
PYTHONPATH=src python -m fsmol_cliff.cli adapter-status --output /tmp/status.json
PYTHONPATH=src python -m fsmol_cliff.cli evaluate --release-dir ... --backend protonet ...

# Aggregate results
PYTHONPATH=src python -m fsmol_cliff.cli aggregate --input <parquet> --output <json>

# Paired comparison
PYTHONPATH=src python -m fsmol_cliff.cli protocol-compare --inputs ... --comparisons ... --profile ... --output ...

# Validate hypotheses
PYTHONPATH=src python -m fsmol_cliff.cli validate-hypotheses --input <json> --output <json>
```

---

## Hard Constraints for Any Future Work

1. **Do not modify the frozen benchmark protocol** (`spec.md`)
2. **Do not change the paper identity** from "stronger diagnostic benchmark paper"
3. **Do not reopen any closed method family** without a separate justification memo
4. **Any new method must pass the stronger-baseline gate:** beat `kNN-cliff-aware` (minimum), `ProtoNet` (paper upgrade)
5. **Robustness work is audit-only** — no training hook authorized from perturbation findings
6. **Representation work is frozen** — requires `docs/new_representation_family_justification.md` proving it's not a B0 continuation
7. **All method development uses the intermediate substrate** (`outputs/fsmol_cliff_release_v4_covext_intermediate`, profile `relaxed_covext_10_10`)
