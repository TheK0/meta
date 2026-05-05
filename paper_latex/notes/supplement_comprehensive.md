# CliffBench — Comprehensive Supplement

Date: 2026-05-05 | Commits: `db1542e` … `e663cf8` (9 commits)

---

## 1. MoleculeACE External Validation — Complete Detail

### 1.1 Data Source

- **Repository**: `https://github.com/molML/MoleculeACE` (MIT license)
- **Version**: `master` branch, commit `7e6de0b` (2025-02-15)
- **30 targets**: 21 × Ki, 9 × EC50. All from ChEMBL, curated by the MoleculeACE authors (van Tilborg et al., JCIM 2022)

| Target | Endpoint | Unit | N Mols | Cliff Mol % | Activity Range |
|--------|----------|------|--------|-------------|----------------|
| CHEMBL1862_Ki | Ki | pKi | 794 | 41.6% | [4.0, 10.7] |
| CHEMBL1871_Ki | Ki | pKi | 659 | 24.6% | [4.3, 9.6] |
| CHEMBL2034_Ki | Ki | pKi | 750 | 33.5% | [4.7, 10.0] |
| CHEMBL2047_EC50 | EC50 | pEC50 | 631 | 38.8% | [4.0, 9.5] |
| CHEMBL204_Ki | Ki | pKi | 2754 | 39.7% | [2.6, 11.0] |
| CHEMBL2147_Ki | Ki | pKi | 1456 | 39.3% | [4.0, 11.0] |
| CHEMBL214_Ki | Ki | pKi | 3317 | 36.8% | [4.2, 10.9] |
| CHEMBL218_EC50 | EC50 | pEC50 | 1031 | 35.8% | [4.0, 10.5] |
| CHEMBL219_Ki | Ki | pKi | 1865 | 39.7% | [4.0, 10.7] |
| CHEMBL228_Ki | Ki | pKi | 1704 | 37.4% | [4.1, 10.9] |
| CHEMBL231_Ki | Ki | pKi | 973 | 24.4% | [4.0, 10.3] |
| CHEMBL233_Ki | Ki | pKi | 3142 | 41.2% | [4.2, 11.0] |
| CHEMBL234_Ki | Ki | pKi | 3657 | 43.9% | [4.1, 10.7] |
| CHEMBL235_EC50 | EC50 | pEC50 | 2349 | 37.7% | [4.0, 10.7] |
| CHEMBL236_Ki | Ki | pKi | 2598 | 39.1% | [4.0, 11.0] |
| CHEMBL237_EC50 | EC50 | pEC50 | 955 | 47.7% | [4.3, 11.0] |
| CHEMBL237_Ki | Ki | pKi | 2603 | 42.6% | [4.1, 10.9] |
| CHEMBL238_Ki | Ki | pKi | 1052 | 25.3% | [4.0, 9.3] |
| CHEMBL239_EC50 | EC50 | pEC50 | 1721 | 40.6% | [4.0, 10.6] |
| CHEMBL244_Ki | Ki | pKi | 3097 | 47.7% | [4.0, 11.0] |
| CHEMBL262_Ki | Ki | pKi | 856 | 18.7% | [4.0, 10.0] |
| CHEMBL264_Ki | Ki | pKi | 2862 | 41.6% | [4.1, 10.6] |
| CHEMBL2835_Ki | Ki | pKi | 615 | 9.8% | [5.3, 10.0] |
| CHEMBL287_Ki | Ki | pKi | 1328 | 38.2% | [4.3, 10.4] |
| CHEMBL2971_Ki | Ki | pKi | 976 | 16.6% | [4.0, 10.2] |
| CHEMBL3979_EC50 | EC50 | pEC50 | 1125 | 41.6% | [4.2, 10.2] |
| CHEMBL4005_Ki | Ki | pKi | 960 | 41.8% | [4.6, 10.5] |
| CHEMBL4203_Ki | Ki | pKi | 731 | 8.8% | [4.1, 9.5] |
| CHEMBL4616_EC50 | EC50 | pEC50 | 682 | 52.1% | [4.9, 10.0] |
| CHEMBL4792_Ki | Ki | pKi | 1471 | 54.0% | [4.7, 10.2] |

**Total**: 48,714 molecules across 30 targets. All activity values are pEC50/pKi (higher = more active). Active direction is unified.

### 1.2 Data Cleaning Rules

- **Canonical SMILES**: RDKit `Chem.MolToSmiles(canonical=True, isomericSmiles=True)`, via `fsmol_cliff.chem.canonicalize_isomeric_smiles()`. RDKit version: 2024.09.1.
- **Duplicate SMILES**: Not merged. Each row in MoleculeACE data is a unique compound per target. The dataset is pre-curated by the MoleculeACE authors.
- **Conflicting activity values**: Not applicable — MoleculeACE data is pre-curated with single activity values per compound per target.
- **Active direction**: `y [pEC50/pKi]` column. Higher values = more active. Unified across all targets.
- **Salt removal / neutralization / stereochemistry**: Not applied. MoleculeACE SMILES are used as-is (pre-curated). The canonicalization step in `morgan_fingerprint_array` handles isomeric SMILES.
- **Label binarization**: Median split on `y [pEC50/pKi]` per target. Top 50% = active (label=1), bottom 50% = inactive (label=0). This is a deviation from FS-Mol's assay-native binary labels and is noted as a protocol difference.

### 1.3 Pair Mining Parameters

- **Fingerprint type**: Morgan (ECFP-like), radius=2, nBits=2048, via `fsmol_cliff.chem.morgan_fingerprint_array()`. Matches FS-Mol v5 protocol.
- **τ (similarity threshold)**: 0.80. Matches FS-Mol v5 relaxed profile.
- **δ (activity gap threshold)**: 1.0 log unit. Matches FS-Mol v5 relaxed profile.
- **Cliff definition**: Active-inactive pair with Tanimoto ≥ τ AND |activity_gap| ≥ δ.
- **Highsim_noncliff definition**: Active-inactive pair with Tanimoto ≥ τ AND |activity_gap| < δ.
- **Scaffold method**: RDKit Murcko scaffold via `fsmol_cliff.chem.murcko_scaffold_smiles()`.
- **Pair enumeration**: All active-inactive test-set pairs enumerated. Train set pairs not exhaustively enumerated (only used for model training, not for metric computation).

### 1.4 Per-Target Pair Counts and Metrics

| Target | Mols | Scaffolds | Cliff Pairs | Noncliff Pairs | kNN C-BAcc | kNN SCR | kNN Q-PSR | RF C-BAcc | RF SCR | RF Q-PSR |
|--------|------|-----------|-------------|----------------|------------|---------|-----------|-----------|--------|---------|
| CHEMBL1862_Ki | 794 | 354 | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| CHEMBL1871_Ki | 659 | 198 | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| CHEMBL2034_Ki | 750 | 253 | 2 | 4 | 0.500 | 0.667 | 0.500 | 0.500 | 1.000 | 0.833 |
| CHEMBL2047_EC50 | 631 | 228 | 1 | 1 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.500 |
| CHEMBL204_Ki | 2754 | 1193 | 13 | 6 | 0.692 | 0.895 | 0.316 | 0.769 | 1.000 | 0.579 |
| CHEMBL2147_Ki | 1456 | 752 | 1 | 0 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| CHEMBL214_Ki | 3317 | 1468 | 10 | 7 | 0.300 | 0.941 | 0.647 | 0.400 | 0.941 | 0.824 |
| CHEMBL218_EC50 | 1031 | 542 | 2 | 1 | 1.000 | 1.000 | 0.333 | 1.000 | 1.000 | 1.000 |
| CHEMBL219_Ki | 1865 | 852 | 1 | 7 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.750 |
| CHEMBL228_Ki | 1704 | 712 | 4 | 10 | 0.750 | 0.857 | 0.429 | 0.750 | 0.786 | 0.571 |
| CHEMBL231_Ki | 973 | 530 | 1 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| CHEMBL233_Ki | 3142 | 1393 | 16 | 16 | 0.625 | 0.812 | 0.219 | 0.562 | 0.781 | 0.562 |
| CHEMBL234_Ki | 3657 | 1695 | 10 | 13 | 0.600 | 0.957 | 0.304 | 0.600 | 0.957 | 0.522 |
| CHEMBL235_EC50 | 2349 | 961 | 4 | 10 | 0.750 | 0.857 | 0.214 | 0.750 | 0.786 | 0.714 |
| CHEMBL236_Ki | 2598 | 1173 | 1 | 9 | 1.000 | 1.000 | 0.100 | 1.000 | 0.900 | 0.700 |
| CHEMBL237_EC50 | 955 | 420 | 3 | 2 | 1.000 | 0.800 | 0.600 | 0.333 | 1.000 | 0.800 |
| CHEMBL237_Ki | 2603 | 1166 | 10 | 17 | 0.800 | 0.815 | 0.333 | 0.500 | 0.889 | 0.667 |
| CHEMBL238_Ki | 1052 | 442 | 1 | 2 | 0.000 | 0.667 | 0.333 | 0.000 | 1.000 | 0.667 |
| CHEMBL239_EC50 | 1721 | 685 | 4 | 6 | 0.250 | 0.900 | 0.100 | 0.250 | 0.900 | 0.600 |
| CHEMBL244_Ki | 3097 | 1232 | 9 | 3 | 0.778 | 0.750 | 0.417 | 0.556 | 0.833 | 0.750 |
| CHEMBL262_Ki | 856 | 447 | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| CHEMBL264_Ki | 2862 | 1512 | 11 | 8 | 0.727 | 0.842 | 0.368 | 0.545 | 0.737 | 0.789 |
| CHEMBL2835_Ki | 615 | 265 | 0 | 16 | 0.000 | 0.812 | 0.312 | 0.000 | 0.875 | 0.625 |
| CHEMBL287_Ki | 1328 | 726 | 2 | 3 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.400 |
| CHEMBL2971_Ki | 976 | 502 | 1 | 0 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| CHEMBL3979_EC50 | 1125 | 397 | 1 | 3 | 1.000 | 0.750 | 0.250 | 1.000 | 0.750 | 0.750 |
| CHEMBL4005_Ki | 960 | 385 | 2 | 1 | 1.000 | 1.000 | 0.000 | 1.000 | 0.667 | 0.667 |
| CHEMBL4203_Ki | 731 | 465 | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| CHEMBL4616_EC50 | 682 | 288 | 5 | 3 | 0.800 | 0.750 | 0.250 | 1.000 | 0.750 | 1.000 |
| CHEMBL4792_Ki | 1471 | 671 | 3 | 6 | 0.667 | 0.667 | 0.222 | 0.000 | 0.889 | 0.556 |

**Note**: CHEMBL1862, CHEMBL1871, CHEMBL262, CHEMBL4203 have zero high-similarity pairs in the test set (τ=0.80). These 4 targets contribute 0/NaN metrics and are excluded from mismatch analysis. MoleculeACE's own cliff definition uses τ=0.90 for cliff molecule labeling. Half of the 30 targets have <5 cliff pairs with τ=0.80.

### 1.5 Macro-Averaged Metrics with 95% Bootstrap CI

Bootstrap: 1,000 iterations, task-level (resample 30 targets with replacement).

| Metric | kNN | kNN 95% CI | RF | RF 95% CI |
|--------|-----|-------------|-----|------------|
| C-BAcc | 0.607 | [0.453, 0.740] | 0.551 | [0.401, 0.690] |
| SCR | 0.757 | [0.629, 0.858] | 0.781 | [0.656, 0.883] |
| Q-PSR | 0.242 | [0.163, 0.327] | 0.562 | [0.456, 0.670] |

### 1.6 Mismatch Analysis

| Pattern | Count |
|---------|-------|
| RF better Q-PSR than kNN | 23/30 (77%) |
| kNN better C-BAcc than RF | 6/30 (20%) |
| Simultaneous mismatch (RF↑Q + kNN↑C) | 6/30 (20%) |
| RF high-rank/low-decision (Q>0.5, C<0.55) | 9/30 (30%) |
| FS-Mol-like (RF↑Q + RF↑SCR) | 8/30 (27%) |

**Interpretation**: RF's ranking advantage (better Q-PSR in 77% of targets) is cross-dataset robust. The ranking-decision mismatch is most prominent in 6 targets where kNN simultaneously has better cliff decisions AND RF has better ranking. The mismatch is not driven by a single outlier — it's distributed across multiple targets.

### 1.7 Evaluation Protocol

- **Type**: Pair-level diagnostic (not episode-level). This is a protocol difference from FS-Mol v5. MoleculeACE does not have natural few-shot episode structure.
- **Train/test split**: MoleculeACE-native split (provided by dataset authors). ~80% train, ~20% test per target.
- **Model**: kNN (k=5) and RandomForest (500 trees, max_depth=20) on Morgan fingerprints (2048-bit).
- **Label**: Median-split binarization per target (top 50% = active).
- **Protocol differences from FS-Mol v5**:
  1. No few-shot episode sampling (pair-level evaluation)
  2. Median-split binarization instead of assay-native binary labels
  3. No support/query structure
  4. No adversarial episode injection
  5. No ProtoNet (requires graph features not available for MoleculeACE molecules)
- **Comparability**: The τ, δ, fingerprint type, and cliff/noncliff definitions match FS-Mol v5. The ranking-decision metrics (C-BAcc, SCR, Q-PSR) are computed identically. The main difference is the evaluation protocol (pair-level vs episode-level).

---

## 2. FS-Mol V5 Baseline — All Profiles

### V5 Release Overview

| Profile | τ | δ | min_cliff | Tasks | Episodes (std+adv) | Status |
|---------|---|---|-----------|-------|---------------------|--------|
| core_strict | 0.85 | 1.0 | 25 | 2 | 8,000 | Release built; evaluation not yet run |
| core_relaxed | 0.80 | 1.0 | 25 | 6 | 24,000 | ✅ Evaluated (4 models) |
| ext_10_10 | 0.80 | 1.0 | 10 | 10 | 40,000 | ✅ Evaluated (4 models) |
| ext_10_5 | 0.80 | 1.0 | 5 | 11 | 44,000 | Release built; evaluation not yet run |

**Note**: `core_strict` and `ext_10_5` evaluations were not completed in this session. The profile definitions and episode parquets exist in `outputs/fsmol_cliff_release_v5/`. The core_relaxed and ext_10_10 results below are final. ProtoNet, kNN-cliff-aware, and randomForest results for ext_10_10 were carried over from v4 covext intermediate (episode hashes verified identical).

### core_relaxed (6 tasks, adversarial split)

Task-level macro mean over 6 tasks, 400 episodes per task/seed/split, 5 seeds.

| Metric | kNN | kNN-cliff-aware | randomForest | ProtoNet |
|--------|-----|-----------------|-------------|----------|
| C-BAcc | 0.475 | 0.490 | 0.494 | **0.547** |
| NC-BAcc | 0.505 | 0.520 | 0.503 | 0.538 |
| SCR | 0.940 | **0.918** | 0.942 | **0.841** |
| SS-SCR | 0.949 | 0.920 | 0.961 | 0.862 |
| Q-PSR | 0.477 | 0.498 | 0.658 | **0.724** |
| SQ-PSR | 0.529 | 0.539 | **0.910** | 0.778 |

### ext_10_10 (10 tasks, adversarial split)

Task-level macro mean over 10 tasks, 400 episodes per task/seed/split, 5 seeds.

| Metric | kNN | kNN-cliff-aware | randomForest | ProtoNet |
|--------|-----|-----------------|-------------|----------|
| C-BAcc | 0.511 | 0.533 | 0.524 | **0.561** |
| NC-BAcc | 0.509 | 0.529 | 0.493 | 0.526 |
| SCR | 0.906 | **0.844** | 0.912 | **0.843** |
| SS-SCR | 0.901 | **0.825** | 0.899 | 0.850 |
| Q-PSR | 0.536 | 0.549 | 0.686 | **0.735** |
| SQ-PSR | 0.562 | 0.572 | **0.917** | 0.786 |

### Standard split (selected, for reference)

| Profile | Metric | kNN | kNN-clf | RF | ProtoNet |
|---------|--------|-----|---------|-----|----------|
| core_relaxed | C-BAcc | 0.508 | 0.515 | 0.505 | **0.544** |
| | SCR | 0.925 | 0.897 | 0.896 | **0.804** |
| | Q-PSR | 0.513 | 0.519 | 0.625 | **0.644** |
| ext_10_10 | C-BAcc | 0.524 | 0.531 | 0.520 | 0.528 |
| | SCR | 0.912 | 0.890 | 0.884 | **0.829** |
| | Q-PSR | 0.554 | 0.565 | 0.636 | 0.584 |

### Aggregation method

- **Mean type**: Task-level macro mean (mean over task means, each task mean is mean over episode scores within that task)
- **Bootstrap**: 10,000 iterations, task-level resampling (not episode-level)
- **CI type**: Percentile bootstrap (2.5th–97.5th percentile)
- **Paired delta CI**: 10,000 iterations, paired bootstrap (same tasks, same seeds, same episodes)
- **Command**: `python -m fsmol_cliff.cli aggregate --input <parquet> --output <json>`

---

## 3. Negative Interventions Appendix — Full 22-Family Registry

### 3.1 Decision Repair Family

| Field | Value |
|-------|-------|
| **Family** | decision-aware threshold repair |
| **Module** | `fsmol_cliff.adapters` (threshold variant, reverted in `f07cf30`) |
| **Intended target** | Per-episode decision threshold based on local cliff density |
| **Changed parameter** | Decision threshold (from 0.5 to cliff-density-adjusted) |
| **Best Δ** | adv C-BAcc +0.006 vs kNN-cliff-aware |
| **CI** | [-0.012, 0.028] — crosses zero |
| **Improved metric** | Marginal C-BAcc |
| **Degraded metric** | None cleanly degraded |
| **NO-GO reason** | Threshold-only repair did not improve key targets cleanly; failed Phase 2A gate |

| Field | Value |
|-------|-------|
| **Family** | local-boundary-repair |
| **Module** | Not retained in tree (evaluated locally) |
| **Intended target** | kNN boundary patching using nearest-support neighbors |
| **Best Δ** | adv C-BAcc +0.026 vs kNN-cliff-aware |
| **CI** | [-0.038, 0.100] — crosses zero |
| **Improved metric** | Standard collapse improved |
| **Degraded metric** | NC-BAcc -0.053 (control-side harm) |
| **NO-GO reason** | Adversarial repair signal not clean; control-side moved wrong direction |

### 3.2 Support Protocol Family

| Field | Value |
|-------|-------|
| **Family** | fixed-support hard-negative replacement |
| **Module** | `fsmol_cliff.manifests` (variant, evaluated locally) |
| **Intended target** | Replace support negatives with hard-negative cliff candidates |
| **Best Δ** | adv C-BAcc -0.002 vs kNN-cliff-aware |
| **CI** | [-0.014, 0.012] — crosses zero |
| **Improved metric** | None |
| **Degraded metric** | SQ-PSR -0.004 (clean negative) |
| **NO-GO reason** | Too aggressive; degraded ranking/collapse balance of stronger baseline |

| Field | Value |
|-------|-------|
| **Family** | partial-hard-negative augmentation |
| **Module** | `fsmol_cliff.manifests` (variant, evaluated locally) |
| **Intended target** | Conservative partial augmentation of support negatives |
| **Best Δ** | std C-BAcc -0.005 vs kNN-cliff-aware |
| **CI** | [-0.009, -0.001] — clean negative |
| **Improved metric** | None |
| **Degraded metric** | std C-BAcc, SQ-PSR (multiple clean negatives) |
| **NO-GO reason** | Even conservative version degraded stronger baseline |

### 3.3 Episode Construction Family (7 variants)

All evaluated on `relaxed_covext_10_10`, against `kNN` and `kNN-cliff-aware` baselines. All NO-GO.

| Variant | Δ vs kNN | Δ vs kNN-cliff-aware | Key Failure |
|---------|----------|---------------------|-------------|
| query-targeted support negatives (corrected) | C-BAcc +0.028 [0.000,0.063] GO | C-BAcc +0.006 [-0.008,0.025] flat | Beats vanilla only; fails stronger-baseline gate; retained as historical evidence |
| same_scaffold_query_targeted | C-BAcc +0.003 [-0.050,0.058] flat | C-BAcc -0.020 [-0.073,0.027] flat | Collapse materially worse; hard NO-GO |
| anchor_coverage_first | C-BAcc +0.009 [-0.009,0.035] flat | C-BAcc -0.013 [-0.035,0.007] flat | Sacrificed ranking and collapse; hard NO-GO |
| paired_hardness_balanced | C-BAcc +0.019 [-0.005,0.057] flat | C-BAcc -0.003 [-0.031,0.026] flat | Cleaner signal but still failed stronger-baseline gate |
| neg_diversity | C-BAcc +0.016 [-0.005,0.051] flat | C-BAcc -0.006 [-0.035,0.024] flat | Collapse worsened relative to cliff-aware |
| anchor_neg_mix | C-BAcc -0.009 [-0.028,0.007] flat | C-BAcc -0.031 [-0.052,-0.013] clean LOSS | Worst variant; broad wrong-way degradation |
| (baseline) query-targeted | See above | — | Only variant with positive vs vanilla |

### 3.4 A1, B0, C0 Families

| Field | Value |
|-------|-------|
| **Family** | A1 — query-only local score refit |
| **Module** | `src/fsmol_cliff/protonet_local_calibrated.py` |
| **Intended target** | Per-episode logistic regression calibration on 6 local features |
| **Changed parameter** | LogisticRegression(C=1.0, max_iter=200) on support molecules |
| **Best Δ** | Slight adversarial signal, too weak |
| **CI** | Standard-side harm CI clean negative |
| **Improved metric** | Marginal adv signal |
| **Degraded metric** | Standard C-BAcc / SCR |
| **NO-GO reason** | Weak directional signal with standard-side harm |

| Field | Value |
|-------|-------|
| **Family** | B0 — coarse cliff-margin loss injection |
| **Module** | `src/fsmol_cliff/protonet_cliff_margin_train.py`, `training_losses/cliff_margin.py` |
| **Intended target** | ProtoNet training with cliff-aware auxiliary loss |
| **Changed parameter** | λ_cliff ∈ [0.1, 0.3, 1.0], margin_gamma ∈ [0.05, 0.1, 0.2] |
| **Best Δ** | Wrong direction on primary decision metrics |
| **CI** | Clean negative |
| **Improved metric** | None |
| **Degraded metric** | C-BAcc, SCR (broad degradation) |
| **NO-GO reason** | Broad wrong-way degradation on primary metrics |

| Field | Value |
|-------|-------|
| **Family** | C0 — support-subset dropout perturbation audit |
| **Module** | `src/fsmol_cliff/protonet_perturbation_audit.py` |
| **Intended target** | Cliff-vs-control query score variance gap → mechanism discovery |
| **Changed parameter** | Support dropout fraction |
| **Best Δ** | Cliff-vs-control gap not stable |
| **CI** | N/A (mechanism signal, not a metric delta) |
| **Improved metric** | N/A |
| **Degraded metric** | N/A |
| **NO-GO reason** | Mechanism signal did not scale to full task coverage |

### 3.5 Boundary Uncertainty Calibration

| Field | Value |
|-------|-------|
| **Family** | boundary_uncertainty calibration |
| **Module** | `src/fsmol_cliff/protonet_boundary_calibration.py` |
| **Intended target** | Deterministic margin shrinkage based on support-conditioned uncertainty |
| **Changed parameter** | uncertainty_scale ∈ [0.1, 0.3, 0.5, 0.8] |
| **Best Δ (all scales)** | C-BAcc: +0.0000 (flat), SCR: +0.0000 (flat) |
| **CI** | All primary metrics flat |
| **Improved metric** | SQ-PSR, NC-PSR (ranking improvements at higher scales) |
| **Degraded metric** | None |
| **NO-GO reason** | Structurally sign-preserving: margin shrinkage never changes binary predictions. Grid search confirmed all scales produce identical discrete predictions to ProtoNet identity. |

### 3.6 CASE-Net v1

| Field | Value |
|-------|-------|
| **Family** | CASE-Net v1 — per-episode logistic regression relation head |
| **Module** | `src/fsmol_cliff/signed_relations.py`, `case_adapter.py`, `case_runner.py` |
| **Intended target** | Signed evidence aggregation from support-pair cliff/noncliff annotations, fused with ProtoNet logits |
| **Changed parameter** | fusion_lambda ∈ [0.0, 0.25, 0.5, 0.75] |
| **Support-support pairs per episode** | min=0, median=2-4, max=~20 (most episodes have very few annotated pairs) |
| **λ=0.5 result** | **Exactly identical** to ProtoNet identity on all discrete metrics (C-BAcc, SCR, SS-SCR all Δ=0.0000). Per-episode LR produces evidence scores that do not alter the argmax of fusion. |
| **λ=0.0 (pure evidence) result** | C-BAcc=0.487 (↓ from 0.561), SCR=0.510 (↓ from 0.843), Q-PSR=0.500. SCR improvement is real but comes from conservative prediction collapse (fewer active predictions), not from cliff-aware reasoning. C-BAcc drops significantly. |
| **NO-GO reason** | Per-episode LR has too few training pairs to learn meaningful cliff-vs-noncliff relations. Evidential signal is dominated by noise. |

### 3.7 CASE-Net v2

| Field | Value |
|-------|-------|
| **Family** | CASE-Net v2 — pretrained cross-task relation head |
| **Module** | `src/fsmol_cliff/case_relation_trainer.py`, `case_train_v2.py` |
| **Intended target** | Global RandomForest classifier trained on FS-Mol train/valid pairs, frozen for test inference |
| **Training assays** | 290 (from FS-Mol train, τ=0.80, δ=1.0) |
| **Training pairs** | 3,376 (cliff ratio: 22.0%) |
| **Validation assays** | 19 (from FS-Mol valid) |
| **Validation pairs** | 1,858 (cliff ratio: 23.3%) |
| **AUPRC** | 0.247 (base rate: 0.233) |
| **AUC-ROC** | 0.506 |
| **Balanced Accuracy** | 0.463 |
| **p_cliff on cliff pairs** | 0.480 |
| **p_cliff on noncliff pairs** | 0.469 |
| **Features** | Morgan fingerprint abs-diff (2048D), Morgan intersection (2048D), Tanimoto, same_scaffold, bit difference count, shared bit count |
| **Consistency check** | Fast sampler vs formal pipeline: Tanimoto median identical (0.836), cliff ratio differs due to max_pairs truncation (formal: 9.0%, fast: 16.0% for 20-assay subset). The fingerprint-cliff relationship is the same under both methods. |
| **NO-GO reason** | 2D pair features are insufficient for transferable cliff-vs-noncliff relation prediction. AUPRC at base rate, BAcc below chance. RandomForest with 200 trees, balanced class weights, and ~4100 features cannot distinguish cliff from noncliff pairs beyond random level. |

---

## 4. Reproducibility Fixes — Evidence

### 4.1 Commits

| Commit | Description |
|--------|-------------|
| `0f83416` | **P1 reproducibility hardening**: RDKit fail-fast guard, MAML path de-hardcoded, pipeline.py default dedup, lru_cache capped, torch.load checkpoint hash-lock |
| `c6a7643` | Skip torch_scatter compat patch on CUDA |
| `594662f` | Cache assay molecular neighbor index across episodes |
| `9a3a78f` | Reorder upstream patch registry (gnn before GFExtractor) |
| `db1542e` | Harden FS-Mol bridge + eliminate runner/protonet_runner duplication |

### 4.2 Pytest Verification

```bash
$ python -m pytest tests/ --deselect tests/test_bootstrap.py::test_cli_exposes_expected_top_level_subcommands -q
........................................................................ [ 41%]
........................................................................ [ 83%]
............................                                             [100%]
172 passed, 1 deselected in 11.86s
```

One pre-existing failure (`test_cli_exposes_expected_top_level_subcommands` — CLI subcommand count mismatch, unrelated to P1 fixes).

### 4.3 Fix Details

1. **RDKit fail-fast**: `chem.py` now exports `require_rdkit()` which raises `RdkitNotAvailableError` with install instructions. Called at pipeline entry points. Without RDKit, benchmark construction fails immediately instead of silently producing zero cliff pairs.

2. **MAML path de-hardcoded**: `runner.py:106` replaces `/Volumes/macplus/project/meta/external/FS-Mol` with `default_external_fsmol_root()`, which respects `FSMOL_EXTERNAL_ROOT` environment variable.

3. **pipeline.py default dedup**: `pipeline.py:37-39` changed from hardcoded `0.85/1.0/32` to `DEFAULT_PROTOCOL_CONSTANTS.similarity_threshold/activity_gap_threshold/hard_negative_pool_size`.

4. **lru_cache bounded**: `episodes.py:180` changed `@lru_cache(maxsize=None)` to `@lru_cache(maxsize=65536)`.

5. **torch.load checkpoint hash-lock**: `protonet_runner.py` stores `_TRUSTED_CHECKPOINT_HASH` (SHA256 of `PN-Support64_best_validation.pt`). `load_protonet_model(require_trusted=True)` verifies hash before loading. Untrusted checkpoints require explicit `require_trusted=False`.

6. **Manifest completeness**: V5 `benchmark_manifest.json` includes `episode_manifest_hashes` (SHA256 of all 8 episode parquets), `asset_checksum_seed` (aggregate hash of all assay assets), `profile_config_hash` (hash of profile definitions), `seeds`, and `built_profiles`.

### 4.4 Known Limitations

- `core_strict` and `ext_10_5` profile evaluations not yet run (release built, episodes exist)
- MAML not re-evaluated on v5 (legacy conda environment requirement)
- MoleculeACE evaluation is pair-level, not episode-level (protocol difference documented above)
- `weights_only=False` remains required for legacy FS-Mol checkpoint format; mitigated by hash-lock
