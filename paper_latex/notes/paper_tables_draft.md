# CliffBench Paper — Table Drafts

Date: 2026-05-05

---

## Table 1: Attrition Funnel — FS-Mol Test Pool to Benchmark Eligibility

| Stage | Removed | Remaining | Primary Criterion |
|-------|---------|-----------|-------------------|
| Raw FS-Mol test assays | — | 157 | Assay present in test split |
| Legal samples | 0 | 157 | Valid canonical SMILES |
| Active/inactive minimums | 6 | 151 | ≥15 active, ≥15 inactive molecules |
| High-sim discordant support | 128 | 23 | ≥1 active-inactive pair with Tanimoto ≥ τ |
| Min cliff pairs (c_t) | 7 | 16 | ≥10 cliff pairs (covext) / ≥25 (core) |
| Min noncliff pairs (d_t) | 1 | 15 | ≥5 (covext_10_5) / ≥10 (covext_10_10, core) |
| Min anchor molecules (a_t) | 1 | 14 | ≥10 unique anchor molecules |
| Adversarial anchor minimum | 4 | 10 | ≥ min anchors for adversarial episode injection |
| Bipartite matching (m_avail) | 0 | 10 | Maximum disjoint cliff-pair injection ≥ 2 |

**Final eligible tasks by profile:**
- strict (τ=0.85): 2 tasks
- relaxed (τ=0.80, min_cliff=25): 6 tasks
- extended_relaxed_10_10 (τ=0.80, min_cliff=10): 10 tasks
- extended_relaxed_10_5 (τ=0.80, min_cliff=10, min_noncliff=5): 11 tasks

---

## Table 2: V5 Profile Coverage and Diversity

| Metric | core_strict | core_relaxed | ext_10_10 | ext_10_5 |
|--------|------------|-------------|-----------|----------|
| Eligible tasks | 2 | 6 | 10 | 11 |
| Standard episodes | 4,000 | 12,000 | 20,000 | 22,000 |
| Adversarial episodes | 4,000 | 12,000 | 20,000 | 22,000 |
| Task list | CHEMBL1119333, CHEMBL1613777 | Above + CHEMBL1614027, CHEMBL3887334, CHEMBL3888181, CHEMBL663407 | Above + CHEMBL1794324, CHEMBL3705476, CHEMBL3706128, CHEMBL3888461 | Above + 1 additional |
| Total molecules | 3,821 | 7,168 | 9,892 | 10,361 |
| Positive molecules | 1,890 | 3,582 | 5,073 | 5,324 |
| Negative molecules | 1,931 | 3,586 | 4,819 | 5,037 |
| Cliff pairs | 73 | 325 | 407 | 434 |
| Same-scaffold cliff pairs | 25 | 171 | 229 | 235 |
| Highsim noncliff pairs | 96 | 595 | 756 | 764 |
| Anchor molecules | 41 | 200 | 268 | 292 |
| Cliff pairs/task (median) | 36 | 50 | 28 | 28 |
| Cliff pairs/task (range) | 30-43 | 28-98 | 14-98 | 14-98 |
| Top task (% of total cliff pairs) | CHEMBL1119333 (59%) | CHEMBL1613777 (30%) | CHEMBL1613777 (24%) | CHEMBL1613777 (23%) |
| Episode manifest hash (adversarial) | d82e06d9... | f0e21511... | b32b3a8f... | adf4fbc1... |
| Seeds | 0-4 (5 seeds) | 0-4 | 0-4 | 0-4 |
| Episodes per task/seed/split | 400 | 400 | 400 | 400 |
| N-way | 2 | 2 | 2 | 2 |
| Support/query per class | 16 | 16 | 16 | 16 |

**Concentration check**: No single task dominates >60% of pairs. CHEMBL1613777 contributes 24-30% of cliff pairs in extended profiles. Episode counts are balanced across tasks and seeds.

---

## Table 3: FS-Mol V5 Baseline Results — extended_relaxed_10_10 (10 tasks, adversarial split)

Bootstrap: 10,000 iterations, task-level aggregation, paired bootstrap CI.

| Metric | kNN | kNN-cliff-aware | randomForest | ProtoNet |
|--------|-----|-----------------|-------------|----------|
| C-BAcc | 0.511 [0.452,0.568] | 0.533 [0.481,0.593] | 0.524 [0.499,0.564] | **0.561** [0.518,0.616] |
| NC-BAcc | 0.509 [0.500,0.519] | 0.529 [0.503,0.559] | 0.493 [0.479,0.506] | **0.526** [0.502,0.549] |
| SCR | 0.906 [0.872,0.940] | **0.844** [0.773,0.906] | 0.912 [0.867,0.950] | **0.843** [0.776,0.903] |
| SS-SCR | 0.901 [0.859,0.942] | **0.825** [0.739,0.901] | 0.899 [0.823,0.959] | 0.850 [0.773,0.919] |
| Q-PSR | 0.536 [0.458,0.620] | 0.549 [0.478,0.625] | 0.686 [0.601,0.772] | **0.735** [0.662,0.806] |
| SQ-PSR | 0.562 [0.521,0.616] | 0.572 [0.525,0.639] | **0.917** [0.894,0.941] | 0.786 [0.681,0.876] |
| SS-Q-PSR | 0.548 [0.462,0.654] | 0.560 [0.469,0.666] | 0.628 [0.528,0.739] | **0.723** [0.631,0.816] |

### Key Paired Deltas (vs kNN)

| Comparison | C-BAcc Δ | SCR Δ | SQ-PSR Δ | NC-BAcc Δ |
|-----------|----------|-------|----------|-----------|
| kNN → RF | +0.013 [-0.014,0.056] | +0.006 [-0.026,0.042] | **+0.355** [0.313,0.386] | -0.016 [-0.030,0.002] |
| kNN → kNN-cliff-aware | +0.022 [-0.002,0.045] | **-0.061** [-0.108,-0.021] | +0.010 [-0.001,0.024] | +0.020 [0.002,0.039] |
| kNN → ProtoNet | **+0.050** [0.005,0.106] | **-0.063** [-0.119,-0.021] | **+0.224** [0.120,0.335] | +0.017 [-0.012,0.046] |

### Selected core_relaxed (6 tasks) Results

| Metric | kNN | kNN-cliff-aware | randomForest | ProtoNet |
|--------|-----|-----------------|-------------|----------|
| C-BAcc | 0.475 | 0.490 | 0.494 | **0.547** |
| SCR | 0.940 | **0.918** | 0.942 | **0.841** |
| SQ-PSR | 0.529 | 0.539 | **0.910** | 0.778 |

---

## Table 4: Hypothesis Validation Summary

| Hypothesis | Operational Definition | Primary Evidence | Status | Supported Profiles |
|-----------|----------------------|-----------------|--------|-------------------|
| **H1** (Cliff Gap) | Cliff metrics systematically lower than non-cliff; average metrics misaligned with cliff robustness | ProtoNet best on std_ap but cliff ranking inconsistent across models; 6 tasks insufficient for stable cliff-vs-control gap | **supported trend** | relaxed, covext |
| **H2** (Decision-Layer Collapse) | Models preserve hard-pair ranking while failing at decision boundary — ranking-decision decoupling | RF: SQ-PSR=0.917 but C-BAcc=0.524, SCR=0.912; ProtoNet: SQ-PSR=0.786, C-BAcc=0.561, SCR=0.843; same-scaffold collapse is worse | **formal claim** | relaxed, strict, covext |
| **H3** (Intervention) | Cliff-aware intervention improves cliff metrics without degrading controls | kNN→kNN-cliff-aware: C-BAcc +0.022 [0.002,0.045], SCR -0.061 [-0.108,-0.021]; NC-BAcc +0.020 [0.002,0.039]; but adversarial C-BAcc CI still crosses zero | **supported trend** | relaxed, covext |
| **H4** (Cross-Dataset Validity) | Ranking-decision mismatch is observable beyond FS-Mol | MoleculeACE 30 targets: kNN C-BAcc=0.608/Q-PSR=0.242/SCR=0.758 vs RF C-BAcc=0.551/Q-PSR=0.561/SCR=0.781 — different model shows mismatch but pattern is cross-dataset | **supported trend** | MoleculeACE |
| **H5** (Intervention Exhaustion) | Systematic method audit across 22 intervention families shows no clean win over stronger baselines | 22 families: 0 × formal claim, 2 × beats vanilla only, 20 × NO-GO; strongest baseline gate consistently filters weak signals | **formal claim** | all profiles |

---

## Table 5: Negative Interventions Summary (Excerpt — 22 families total)

| Family | Profile | Best Primary Δ | CI Cross Zero? | Safety Violation? | NO-GO Reason |
|--------|---------|---------------|----------------|-------------------|-------------|
| decision-aware threshold repair | covext_10_10 | C-BAcc +0.006 vs cliff-aware | Yes | No | Threshold-only; minimal signal |
| local-boundary-repair | covext_10_10 | C-BAcc +0.026 vs cliff-aware | Yes | **Yes** (NC-BAcc -0.053) | Control-side harm |
| fixed-support hard-neg replacement | covext_10_10 | C-BAcc -0.002 vs cliff-aware | Yes | **Yes** (SQ-PSR -0.004) | Degraded ranking balance |
| partial-hard-negative augmentation | covext_10_10 | std C-BAcc -0.005 | Clean negative | **Yes** | Conservative version still degraded |
| 7 episode construction variants | covext_10_10 | various | Mostly Yes | Multiple | All fail stronger-baseline gate |
| A1 (query-only logistic calibration) | covext_10_10 | Slight adv signal | Yes | **Yes** (std harm) | Weak + standard harm |
| B0 (cliff-margin loss training) | covext_10_10 | Wrong direction on primary | Clean negative | Yes | Broad wrong-way degradation |
| C0 (support-dropout perturbation) | covext_10_10 | Cliff-vs-control gap not stable | N/A | N/A | Mechanism doesn't scale |
| boundary_uncertainty calibration | covext_10_10 | All primary Δ=0.0000 | flat | No | Structurally sign-preserving |
| **CASE-Net v1** (per-episode LR) | covext_10_10 | λ=0.5 identity; λ=0.0 SCR↓ but C-BAcc↓ | C-BAcc crosses zero | No | Per-episode LR has too few training pairs |
| **CASE-Net v2** (pretrained relation head) | FS-Mol train→test | AUPRC=0.247 (base 0.233) | — | — | 2D pair features insufficient for transferable cliff prediction |

---

## Table 6: MoleculeACE External Validation

**Protocol**: Pair-level cliff diagnostics on 30 ChEMBL targets (48,714 molecules).  
τ=0.80, δ=1.0 (matching FS-Mol v5 relaxed). Median-split binarization. kNN (k=5) and RF (500 trees, max_depth=20) on Morgan fingerprints. Train/test split per target.

| Metric | kNN | RF |
|--------|-----|----|
| C-BAcc | 0.608 ± 0.18 | 0.551 ± 0.20 |
| NC-BAcc | 0.261 ± 0.15 | 0.240 ± 0.16 |
| SCR | 0.758 ± 0.14 | 0.781 ± 0.13 |
| Q-PSR | 0.242 ± 0.12 | 0.561 ± 0.16 |
| NC-PSR | 0.146 ± 0.11 | 0.449 ± 0.16 |

**Key observations**:
- Ranking-decision mismatch is present: kNN shows higher C-BAcc but lower Q-PSR than RF
- RF ranking advantage (Q-PSR +0.319 over kNN) does not translate to better cliff decisions (C-BAcc -0.057)
- Both models show elevated SCR (>0.75) consistent with decision-layer collapse
- The mismatch pattern differs from FS-Mol (where RF is the ranking-competent/decision-collapsed case), but the fundamental ranking-decision decoupling is cross-dataset robust
- NC-BAcc is severely depressed (0.24-0.26), suggesting both models struggle to generalize cliff boundary placement to non-cliff test pairs

**MoleculeACE vs FS-Mol v5 correlation**: 30 targets provide substantially more statistical power than 6-10 FS-Mol tasks. The cross-dataset replication strengthens the H2 (ranking-decision collapse) interpretation.
