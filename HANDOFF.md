# CliffBench v5 — Project Handoff

Date: 2026-05-05 | Status: **Evidence consolidation complete. Manuscript assembly phase. No new experiments needed.**

---

## What This Project Is

CliffBench v5 is an activity-cliff diagnostic benchmark for assay-local few-shot molecular classification. It uses activity cliffs (pairs of structurally similar molecules with large potency differences) as probes to expose **decision-layer collapse**: models that can rank hard pairs correctly but cannot form reliable binary classification boundaries around cliffs.

**Paper target**: JCIM / Journal of Cheminformatics.
**Core contribution**: A systematic diagnostic benchmark + external validation showing ranking-decision mismatch, not a method paper.

---

## Where Things Stand

### Frozen (do not change)

- **Claim hierarchy**: 3 formal claims + 1 supporting claim
- **V5 benchmark substrate**: `outputs/fsmol_cliff_release_v5/` (4 profiles)
- **Baseline results**: 4 models evaluated on core_relaxed and ext_10_10
- **22 intervention families**: all NO-GO, documented in `docs/closed_families_registry.md`
- **CASE-Net v1/v2**: both NO-GO, retained as negative intervention appendix
- **Benchmark protocol**: `spec.md` (frozen)

### Ready for manuscript

- **Results draft**: `paper_latex/main_draft.md` — R1-R4 complete, Abstract/Methods/Discussion as outlines
- **Tables**: 4 main paper + 5 supplement, drafted in `paper_latex/notes/paper_tables_draft.md`
- **Submission checklist**: `paper_latex/submission_checklist.md`
- **Outline**: `paper_latex/outline.md`

### Needs work before submission

| Item | Effort | Priority |
|------|--------|----------|
| Update `paper_latex/main.tex` LaTeX from `main_draft.md` | 2-3 hours | P0 |
| Expand Abstract from placeholder | 30 min | P0 |
| Expand Introduction from outline | 1-2 hours | P1 |
| Expand Related Work from outline | 1-2 hours | P1 |
| Expand Methods (3.1-3.4) from outline | 2-3 hours | P1 |
| Expand Discussion (5.1-5.4) from outline | 1-2 hours | P2 |
| Run core_strict and ext_10_5 evaluations | 1 hour compute | P3 |
| MoleculeACE v2 independent repro check | 30 min | P3 |
| Update `README.md` to v5 | 30 min | P3 |
| Clean `paper_latex/notes/` — archive historical files | 15 min | P3 |

---

## Claim Hierarchy (Frozen)

| ID | Claim | Status |
|----|-------|--------|
| **FC1** | CliffBench v5 is a valid activity-cliff diagnostic benchmark derived from FS-Mol | formal claim |
| **FC2** | FS-Mol baselines exhibit ranking-decision mismatch / decision-layer collapse | formal claim |
| **FC3** | 22 shallow interventions fail the stronger-baseline gate without clean repair | formal claim |
| **SC** | MoleculeACE provides external pair-level supporting evidence, not direct replication | external supporting evidence |

### Forbidden Claims

- "MoleculeACE directly replicates FS-Mol" → use "external pair-level supporting evidence"
- "We solve activity-cliff few-shot prediction" → no successful method
- "Activity cliffs require 3D protein-ligand modeling" → not tested
- "JCIM-ready" → use "suitable for JCIM-style submission subject to review"
- "CASE-Net improves..." → v1/v2 both NO-GO

---

## Key Results (for paper)

### FS-Mol v5 baseline (extended_relaxed_10_10, 10 tasks, adversarial)

| Metric | kNN | kNN-cliff-aware | randomForest | ProtoNet |
|--------|-----|-----------------|-------------|----------|
| C-BAcc | 0.511 | 0.533 | 0.524 | **0.561** |
| SCR | 0.906 | **0.844** | 0.912 | **0.843** |
| SQ-PSR | 0.562 | 0.572 | **0.917** | 0.786 |

- **ProtoNet**: strongest balanced model (highest C-BAcc, lowest SCR)
- **RF**: clearest ranking-competent/decision-collapsed (SQ-PSR=0.917 but C-BAcc=0.524, SCR=0.912)
- **kNN-cliff-aware**: modest SCR improvement (-0.061) with clean NC-BAcc gain, but C-BAcc CI crosses zero

### MoleculeACE v2 (25 targets with test cliff pairs)

| Metric | kNN | RF |
|--------|-----|----|
| C-BAcc | 0.522 [0.493,0.552] | 0.526 [0.509,0.547] |
| SCR | 0.875 [0.828,0.918] | 0.902 [0.859,0.941] |
| Q-PSR | 0.279 [0.193,0.370] | 0.647 [0.553,0.735] |

- RF Q-PSR > kNN Q-PSR in 22/25 targets (88%)
- C-BAcc nearly identical → classic mismatch pattern weaker than FS-Mol but ranking advantage is cross-dataset robust

---

## Project Layout

```
cliff/
  CLAUDE.md                          # AI context file (loads automatically)
  HANDOFF.md                         # This file
  AGENTS.md                          # Engineering guidelines
  spec.md                            # Frozen benchmark protocol
  spec_f.md                          # Protocol completion status
  EXPERIMENT_SUMMARY_2026-03-24.md   # Method exploration record
  
  src/fsmol_cliff/                   # Core package (35+ modules)
    cli.py                           # 11 subcommands
    constants.py                     # ProtocolConstants, BenchmarkProfile
    assets.py / pipeline.py          # Assay asset construction
    episodes.py / manifests.py       # Episode generation
    release.py                       # Frozen release builder
    evaluation.py / runner.py        # Evaluation engine
    metrics.py                       # 9 core metrics
    aggregate.py / hypotheses.py     # Bootstrap CI, H1-H3 validation
    protonet_runner.py               # ProtoNet model loading + scoring
    fsmol_bridge.py                  # FS-Mol external repo bridge (hash-locked)
    moleculeace_eval.py              # MoleculeACE pair-level evaluation
    signed_relations.py              # CASE-Net v1 relation dataset builder (NO-GO)
    case_adapter.py                  # CASE-Net v1 pair featurizer (NO-GO)
    case_runner.py                   # CASE-Net v1 evidence fusion (NO-GO)
    case_relation_trainer.py         # CASE-Net v2 pretrained relation head (NO-GO)
    case_train_v2.py                 # CASE-Net v2 training CLI (NO-GO)
    protonet_boundary_calibration.py # boundary calibration (NO-GO)
    protonet_local_calibrated.py     # A1 query-only calibration (NO-GO)
    protonet_cliff_margin_train.py   # B0 cliff-margin training (NO-GO)
    torch_scatter_compat.py          # MPS scatter compatibility
  
  tests/                             # 35 test files, 179 passed, 1 deselected
  
  outputs/
    fsmol_cliff_release_v5/          # V5 release (4 profiles)
    fsmol_cliff_release_v4/          # V4 release (frozen, 2+6 tasks)
    fsmol_cliff_release_v4_covext_intermediate/ # Method dev substrate
    moleculeace_validation/          # MoleculeACE v2 results
    method_boundary_calibration_pilot/ # boundary calib pilot results
    case_net_ablation/               # CASE-Net v1 ablation results
    case_net_v2/                     # CASE-Net v2 pair-level report
  
  paper_latex/
    main.tex                         # LaTeX source (v4 — needs update)
    main_draft.md                    # v5 manuscript skeleton (CURRENT)
    outline.md                       # 8-section structure
    submission_checklist.md          # Claim hierarchy, metrics, repro commands
    notes/
      results_draft.md               # R1-R4 Results full draft
      paper_tables_draft.md          # Tables 1-4 + S1-S5
      supplement_comprehensive.md    # Heavy supplement (partly superseded by v2)
  
  docs/                              # Method governance (all frozen)
    method_restart_v1_closing_memo.md
    method_go_no_go_table.md
    closed_families_registry.md
    stronger_baseline_gate_failure_patterns.md
  
  configs/                           # YAML configs for methods
  checkpoints/                       # ProtoNet model (73MB)
  fs-mol/                            # FS-Mol data (test split used)
```

---

## Key Commands

```bash
# Full test suite
python -m pytest tests/ \
  --deselect tests/test_bootstrap.py::test_cli_exposes_expected_top_level_subcommands -q
# → 179 passed, 1 deselected

# Build a release profile
PYTHONPATH=src python -m fsmol_cliff.cli build-release \
  --data-dir fs-mol --output-dir outputs/fsmol_cliff_release_v5 \
  --profile relaxed --seeds "[0,1,2,3,4]" --episodes-per-split 400

# Run baseline evaluation
PYTHONPATH=src python -m fsmol_cliff.cli evaluate \
  --release-dir outputs/fsmol_cliff_release_v5 --profile relaxed \
  --backend local --model-name kNN \
  --output outputs/.../task_results_kNN_relaxed.parquet

# ProtoNet evaluation (needs checkpoint + FS-Mol external repo)
PYTHONPATH=src FSMOL_EXTERNAL_ROOT=/path/to/FS-Mol \
python -m fsmol_cliff.cli evaluate \
  --release-dir outputs/fsmol_cliff_release_v5 --profile relaxed_covext_10_10 \
  --backend protonet --device cuda \
  --checkpoint checkpoints/PN-Support64_best_validation.pt \
  --output .../task_results_protonet_ext10.parquet

# Aggregate results
PYTHONPATH=src python -m fsmol_cliff.cli aggregate \
  --input <parquet> --output <json>

# MoleculeACE external validation
PYTHONPATH=src python -m fsmol_cliff.moleculeace_eval \
  <MoleculeACE_data_dir> outputs/moleculeace_validation

# CASE-Net v2 pair-level (supplement only)
PYTHONPATH=src python -m fsmol_cliff.case_train_v2 \
  --data-dir fs-mol --max-train 500 --max-valid 100 \
  --output-dir outputs/case_net_v2
```

---

## Key Technical Details

### Protocol parameters (frozen)
- Fingerprint: Morgan ECFP4, 2048-bit, radius=2
- tau (similarity threshold): 0.80 (relaxed), 0.85 (strict)
- delta (activity gap): 1.0 log unit
- Episode: 2-way, 16 support/class, 16 query/class
- 400 standard + 400 adversarial per task/seed/split
- 5 seeds (0-4)
- Bootstrap: 10,000 iterations, task-level resampling
- MoleculeACE bootstrap: 2,000 iterations, seed=42

### Metric families
- Classification: C-BAcc, NC-BAcc (balanced accuracy on cliff/noncliff query subsets)
- Pair-ranking: Q-PSR, SQ-PSR, NC-PSR, SS-Q-PSR, SS-SQ-PSR
- Collapse: SCR, SS-SCR (fraction of pairs with same binary prediction)
- Note: MoleculeACE C-BAcc uses pair-balanced formula: mean over pairs of 0.5*(1(active_pred==1)+1(inactive_pred==0))

### Reproducibility
- Episode hashes verified identical between v4 and v5 (deterministic seed protocol)
- ProtoNet checkpoint hash-locked (SHA256 in `protonet_runner.py`)
- FS-Mol external bridge hash-locked (4 upstream source files)
- RDKit fail-fast guard prevents silent empty benchmark
- `torch.load(weights_only=False)` risk mitigated by checkpoint hash-lock
- MAML hardcoded path fixed

### Engineering fixes (session 2026-04-30 to 2026-05-05)
- FS-Mol bridge: hash-lock, sys.path context manager, patch order fix, CUDA detection
- Pipeline: default values from constants, RDKit fail-fast
- Episodes: lru_cache bounded (was unlimited)
- runner/protonet_runner: ~90 lines dead code removed, shared functions extracted to io.py
- MAML: hardcoded path replaced with env-var-aware lookup

---

## Remaining Limitations

| Limitation | Severity | Mitigation |
|-----------|----------|------------|
| core_strict + ext_10_5 not evaluated | Low | Releases built; declared sensitivity profiles |
| MAML not evaluated on v5 | Low | Declared exploratory |
| 6-10 FS-Mol tasks is narrow | Medium | Discussed; MoleculeACE external evidence |
| MoleculeACE pair-level, not episode-level | Medium | Documented; framed as external supporting evidence |
| No successful repair method | Medium | 22-family audit; framed as systematic negative result |
| torch.load(weights_only=False) | Low | Hash-locked; noted in supplement |
| MoleculeACE v2 results not independently reproduced | Medium | Code+data documented; deterministic seed |
| `paper_latex/main.tex` not yet updated to v5 | Medium | `main_draft.md` has current content |

---

## If You're Picking This Up

### To write the paper
1. Read `paper_latex/main_draft.md` — it has the complete Results and skeleton
2. Read `paper_latex/notes/results_draft.md` — detailed Results with numbers
3. Read `paper_latex/notes/paper_tables_draft.md` — formatted tables
4. Read `paper_latex/submission_checklist.md` — claim hierarchy, caveats
5. Expand Abstract, Introduction, Methods, Discussion from outlines
6. Sync to `paper_latex/main.tex`

### To verify results
1. `python -m pytest tests/ --deselect ... -q` → should pass 179
2. Check `outputs/fsmol_cliff_release_v5/` for release files
3. Check `outputs/moleculeace_validation/moleculeace_results_v2.json` for MoleculeACE v2
4. All result file paths are in `submission_checklist.md`

### To add a new experiment
1. New methods go on `outputs/fsmol_cliff_release_v4_covext_intermediate` (dev substrate)
2. V5 release (`outputs/fsmol_cliff_release_v5`) is for final baseline only
3. Must pass stronger-baseline gate (beat kNN-cliff-aware minimum, ProtoNet for upgrade)
4. Read `docs/closed_families_registry.md` before proposing anything

### What NOT to do
- Do not reopen closed method families without a justification memo
- Do not change frozen benchmark protocol
- Do not rerun CASE-Net v1/v2 full evaluation
- Do not claim MoleculeACE directly replicates FS-Mol
- Do not develop new methods on v5 final substrate
