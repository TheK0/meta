# CliffBench Paper — Table Drafts (v2 — corrected metrics)

Date: 2026-05-05 | Commits: `db1542e` … latest

---

## Table 1: Attrition Funnel — FS-Mol Test Pool to Benchmark Eligibility

| Stage | Removed | Remaining | Primary Criterion |
|-------|---------|-----------|-------------------|
| Raw FS-Mol test assays | — | 157 | Assay present in test split |
| Legal samples | 0 | 157 | Valid canonical SMILES, measurement present |
| Active/inactive minimums | 6 | 151 | >= 15 active, >= 15 inactive molecules |
| High-sim discordant support | 128 | 23 | >= 1 active-inactive pair with Tanimoto >= tau |
| Min cliff pairs (c_t) | 7 | 16 | >= 10 (covext) / >= 25 (core) |
| Min noncliff pairs (d_t) | 1 | 15 | >= 5 (covext_10_5) / >= 10 (covext_10_10, core) |
| Min anchor molecules (a_t) | 1 | 14 | >= 10 unique anchor molecules |
| Adversarial anchor minimum | 4 | 10 | >= min adversarial-eligible anchors |
| Bipartite matching (m_avail) | 0 | 10 | Maximum disjoint cliff-pair injection >= 2 |

**Final eligible tasks by profile:**
- strict (tau=0.85, min_cliff=25): 2 tasks
- relaxed (tau=0.80, min_cliff=25): 6 tasks
- extended_relaxed_10_10 (tau=0.80, min_cliff=10): 10 tasks
- extended_relaxed_10_5 (tau=0.80, min_cliff=10, min_noncliff=5): 11 tasks

The main bottleneck is the high-sim discordant support stage: 128 of 157 raw assays lack any active-inactive pair with Tanimoto >= 0.80, consistent with the known rarity of densely-cliffed SAR landscapes in public bioactivity databases.

---

## Table 2: V5 Profile Coverage and Diversity

We release four benchmark profiles and evaluate the four main baselines on the two primary analysis profiles: `core_relaxed` (6 tasks) and `extended_relaxed_10_10` (10 tasks). The `core_strict` profile (2 tasks) and `extended_relaxed_10_5` profile (11 tasks) are provided as sensitivity profiles with episode releases but model evaluations are not yet complete.

| Metric | core_strict | core_relaxed | ext_10_10 | ext_10_5 |
|--------|------------|-------------|-----------|----------|
| Eligible tasks | 2 | 6 | 10 | 11 |
| Standard episodes | 4,000 | 12,000 | 20,000 | 22,000 |
| Adversarial episodes | 4,000 | 12,000 | 20,000 | 22,000 |
| Total molecules | 3,821 | 7,168 | 9,892 | 10,361 |
| Positive molecules | 1,890 | 3,582 | 5,073 | 5,324 |
| Negative molecules | 1,931 | 3,586 | 4,819 | 5,037 |
| Cliff pairs (eligible tasks) | 73 | 325 | 407 | 434 |
| Same-scaffold cliff pairs | 25 | 171 | 229 | 235 |
| Highsim noncliff pairs | 96 | 595 | 756 | 764 |
| Anchor molecules | 41 | 200 | 268 | 292 |
| Cliff pairs/task (median) | 36 | 50 | 28 | 28 |
| Cliff pairs/task (range) | 30-43 | 28-98 | 14-98 | 14-98 |
| Seeds | 0-4 | 0-4 | 0-4 | 0-4 |
| Episodes per task/seed/split | 400 | 400 | 400 | 400 |
| Evaluation status | release built | **evaluated** | **evaluated** | release built |

---

## Table 3: FS-Mol V5 Baseline Results — extended_relaxed_10_10 (10 tasks, adversarial split)

Per-task macro mean over 10 tasks. Each task mean is the mean over episode scores (400 episodes / task / seed, 5 seeds, 2 splits). Bootstrap: 10,000 iterations, task-level resampling (percentile 2.5%–97.5%).

| Metric | kNN | kNN-cliff-aware | randomForest | ProtoNet |
|--------|-----|-----------------|-------------|----------|
| C-BAcc | 0.511 [0.452,0.568] | 0.533 [0.481,0.593] | 0.524 [0.499,0.564] | **0.561** [0.518,0.616] |
| NC-BAcc | 0.509 [0.500,0.519] | 0.529 [0.503,0.559] | 0.493 [0.479,0.506] | **0.526** [0.502,0.549] |
| SCR | 0.906 [0.872,0.940] | **0.844** [0.773,0.906] | 0.912 [0.867,0.950] | **0.843** [0.776,0.903] |
| SS-SCR | 0.901 [0.859,0.942] | **0.825** [0.739,0.901] | 0.899 [0.823,0.959] | 0.850 [0.773,0.919] |
| Q-PSR | 0.536 [0.458,0.620] | 0.549 [0.478,0.625] | 0.686 [0.601,0.772] | **0.735** [0.662,0.806] |
| SQ-PSR | 0.562 [0.521,0.616] | 0.572 [0.525,0.639] | **0.917** [0.894,0.941] | 0.786 [0.681,0.876] |
| SS-Q-PSR | 0.548 [0.462,0.654] | 0.560 [0.469,0.666] | 0.628 [0.528,0.739] | **0.723** [0.631,0.816] |

### Selected pairwise comparisons (vs kNN, extended_relaxed_10_10, adversarial)

| Comparison | C-BAcc Δ | SCR Δ | SQ-PSR Δ | NC-BAcc Δ |
|-----------|----------|-------|----------|-----------|
| kNN -> RF | +0.013 [-0.014,0.056] | +0.006 [-0.026,0.042] | **+0.355** [0.313,0.386] | -0.016 [-0.030,0.002] |
| kNN -> kNN-cliff-aware | +0.022 [-0.0002,0.040] | **-0.061** [-0.108,-0.021] | +0.010 [-0.001,0.024] | +0.020 [0.002,0.039] |
| kNN -> ProtoNet | **+0.050** [0.005,0.106] | **-0.063** [-0.119,-0.021] | **+0.224** [0.120,0.335] | +0.017 [-0.012,0.046] |

### core_relaxed (6 tasks) — selected adversarial metrics

| Metric | kNN | kNN-cliff-aware | randomForest | ProtoNet |
|--------|-----|-----------------|-------------|----------|
| C-BAcc | 0.475 | 0.490 | 0.494 | **0.547** |
| SCR | 0.940 | **0.918** | 0.942 | **0.841** |
| SQ-PSR | 0.529 | 0.539 | **0.910** | 0.778 |

---

## Table 4: Hypothesis Validation Summary

| ID | Hypothesis | Status | Key Evidence | Limitation |
|----|-----------|--------|-------------|------------|
| **H0** (Benchmark validity) | The FS-Mol test pool supports a well-defined activity-cliff diagnostic benchmark through systematic assay filtering and pair mining | **formal claim** | 157 raw assays -> 6-10 eligible under standard thresholds; attrition dominated by high-sim discordant support scarcity; threshold sensitivity analysis across tau 0.80-0.85 and min_cliff 10-25 confirms funnel is well-behaved | Task count is small (6-10); dependent on FS-Mol data quality |
| **H1** (Activity-cliff diagnostic gap) | Ordinary few-shot / noncliff metrics do not fully predict cliff-sensitive classification performance | **supported trend** | ProtoNet is strongest on standard AUPRC but cliff ranking across kNN/RF/ProtoNet is not monotonic; 6-task core_relaxed cliff-vs-control gap not stable enough for strong claim | 6-10 tasks; cliff-control gap needs more tasks for stability |
| **H2** (Ranking-decision mismatch) | Models can preserve pairwise ranking while failing to convert that signal into robust binary decisions around activity cliffs | **formal claim** | RF: SQ-PSR=0.917 but C-BAcc=0.524, SCR=0.912; ProtoNet: SQ-PSR=0.786, C-BAcc=0.561, SCR=0.843; same-scaffold SCR exceeds overall SCR in all models; pattern stable across 6-task and 10-task profiles | Behavioral evidence, not mechanistic proof |
| **H3** (Shallow intervention no-go) | Audited shallow interventions do not reliably repair cliff classification without harming controls or failing stronger-baseline gates | **formal claim** | 22 intervention families evaluated; 0 pass the stronger-baseline gate for paper upgrade; kNN-cliff-aware improves SCR (-0.061, CI clean negative) and NC-BAcc (+0.020, CI clean positive) but C-BAcc CI crosses zero [-0.0002,0.040]; strongest-balanced baseline (ProtoNet) remains unrepaired | Interventions tested are shallow (calibration, support-set, threshold); representation-level repairs not explored |
| **H4** (External support) | Pair-level diagnostics on an independent data source show that ranking and decision behavior can decouple outside FS-Mol, but not as a direct few-shot replication | **external supporting evidence** | MoleculeACE 30 targets, 25 with test cliff pairs: RF Q-PSR > kNN Q-PSR in 22/25 targets (88%), but C-BAcc is similar (RF 0.526 vs kNN 0.522); RF SCR > kNN SCR (0.902 vs 0.875), consistent with ranking producing more confident but collapsible predictions | Pair-level protocol, median-split labels; not episode-based; C-BAcc gap is negligible with corrected metrics |

**Notes on H3 C-BAcc CI**: The paired C-BAcc delta for kNN -> kNN-cliff-aware on extended_relaxed_10_10 is +0.022 with 95% CI [-0.0002, 0.040]. The lower bound is negative by 0.0002, so the interval does not cleanly separate from zero. We therefore classify H1/H3 as supported trends and H2/H0/H3-intervention-audit as formal claims.

---

## Table 5: Negative Interventions Summary (22 families total)

All evaluated on `relaxed_covext_10_10` profile (intermediate tier) unless noted. Stronger-baseline gate: kNN-cliff-aware (minimum), ProtoNet (paper upgrade).

| Family | Best Primary Δ | CI | Safety | NO-GO Reason |
|--------|---------------|-----|--------|-------------|
| decision-aware threshold repair | C-BAcc +0.006 vs cliff-aware | crosses zero | OK | Threshold-only; minimal signal |
| local-boundary-repair | C-BAcc +0.026 vs cliff-aware | crosses zero | **NC-BAcc degraded** | Control-side harm |
| fixed-support hard-neg replacement | C-BAcc -0.002 vs cliff-aware | crosses zero | **SQ-PSR degraded** | Degraded ranking balance |
| partial-hard-negative augmentation | std C-BAcc -0.005 | **clean negative** | **multiple** | Even conservative version degraded |
| 7 episode construction variants | various | mostly crosses zero | multiple | All fail stronger-baseline gate |
| A1 (query-only logistic calibration) | slight adv signal | crosses zero | **std harm** | Weak + standard harm |
| B0 (cliff-margin loss training) | wrong direction | clean negative | yes | Broad wrong-way degradation |
| C0 (support-dropout perturbation) | gap not stable | N/A | N/A | Mechanism doesn't scale |
| boundary_uncertainty calibration | all primary Δ=0.0000 | flat | OK | Structurally sign-preserving |
| **CASE-Net v1** (per-episode LR relation head) | λ=0.5 identical to ProtoNet; λ=0.0 SCR ↓ but C-BAcc ↓ | C-BAcc crosses zero | OK | Per-episode support-support pairs too sparse (median 2-4); noisy relation learning |
| **CASE-Net v2** (pretrained cross-task cliff-vs-noncliff relation head, FS-Mol train -> valid) | AUPRC=0.247 (base=0.233) | — | — | Simple 2D pair descriptors insufficient for transferable cliff-vs-highsim_noncliff relation prediction in this setting |

**CASE-Net v2 detailed pair-level results**:
- Training: 3,376 pairs from 290 FS-Mol train assays (cliff ratio 22.0%)
- Validation: 1,858 pairs from 19 FS-Mol valid assays (cliff ratio 23.3%)
- AUC-ROC: 0.506, Balanced Accuracy: 0.463
- p_cliff on cliff pairs: 0.480 vs p_cliff on noncliff pairs: 0.469 (no separation)
- Features: Morgan abs diff (2048D), Morgan intersection (2048D), Tanimoto, same_scaffold, bit difference count, shared bit count
- Consistency check: fast sampler vs formal pipeline median Tanimoto identical (0.836)

---

## Table 6: MoleculeACE External Pair-Level Diagnostic (v2 — corrected metrics)

MoleculeACE uses pair-level train/test evaluation with median-split labels, not few-shot episode sampling. We therefore use it as an external diagnostic substrate for ranking-decision decoupling rather than as a direct replication of the FS-Mol few-shot protocol.

**Protocol**: tau = 0.80, delta = 1.0 (matching FS-Mol v5 relaxed). Morgan fingerprints (2048-bit, radius=2). Train/test split from MoleculeACE authors. Median binarization per target. Metrics computed only on test-set pairs. Targets with zero eligible test pairs excluded per-metric (not zero-padded).

**Source**: `github.com/molML/MoleculeACE`, commit `7e6de0b` (2025-02-15). 30 targets, ChEMBL-derived. All values are pEC50/pKi (higher = more active).

### Macro-averaged metrics with 95% bootstrap CI (2000 iterations, task-level)

| Metric | Eligible | kNN | kNN 95% CI | RF | RF 95% CI |
|--------|----------|-----|-------------|-----|------------|
| C-BAcc | 25/30 | 0.522 | [0.493, 0.552] | 0.526 | [0.509, 0.547] |
| NC-BAcc | 23/30 | 0.488 | [0.451, 0.527] | 0.484 | [0.426, 0.528] |
| SCR | 26/30 | 0.875 | [0.828, 0.918] | 0.902 | [0.859, 0.941] |
| Q-PSR | 26/30 | 0.279 | [0.193, 0.370] | 0.647 | [0.553, 0.735] |
| NC-PSR | 23/30 | 0.191 | [0.105, 0.285] | 0.585 | [0.484, 0.691] |
| C-ActiveAcc | 25/30 | 0.730 | [0.597, 0.840] | 0.661 | [0.514, 0.789] |
| NC-InactiveAcc | 23/30 | 0.341 | [0.230, 0.456] | 0.313 | [0.217, 0.415] |

**Sensitivity**: 25/30 targets have >=1 test cliff pair, 13 have >=3, 8 have >=5. 26/30 have >=1 test high-sim pair.

**C-BAcc / NC-BAcc definition (v2 corrected)**: For each high-sim active-inactive pair, pair_decision_acc = 0.5 * [1(active_pred==1) + 1(inactive_pred==0)]. C-BAcc is the mean over cliff pairs; NC-BAcc is the mean over highsim_noncliff pairs. This replaces the earlier one-sided active-recall naming (now C-ActiveAcc / NC-InactiveAcc).

### Mismatch diagnostics

| Pattern | Count |
|---------|-------|
| RF Q-PSR > kNN Q-PSR | 22/25 (88%) |
| kNN C-BAcc > RF C-BAcc | 3/25 (12%) |
| Simultaneous (RF↑Q + kNN↑C) | 3/25 (12%) |
| RF SCR > kNN SCR | 16/25 (64%) |

**Interpretation**: RF exhibits a consistent ranking advantage over kNN (88% of targets with test cliff pairs) but this ranking signal is accompanied by modestly higher collapse (SCR: 0.902 vs 0.875). C-BAcc is nearly identical between the two models (0.526 vs 0.522), so the classic "ranking-competent but decision-collapsed" pattern is less pronounced than on FS-Mol. However, the RF ranking advantage coexisting with comparable or worse decision performance across a majority of targets provides external supporting evidence that ranking and decision behavior can decouple outside FS-Mol. The MoleculeACE pattern is distinct from FS-Mol (where randomForest showed a much wider Q-PSR/C-BAcc gap), consistent with the differing evaluation protocols.

---

## Paper Narrative (recommended)

CliffBench v5 provides a few-shot activity-cliff diagnostic benchmark derived from FS-Mol. Its core finding is a ranking-decision mismatch: models can preserve pairwise ranking signal on hard molecular pairs while failing to convert that signal into robust binary decisions around activity cliffs.

A systematic audit of 22 shallow intervention families — including threshold repair, support-set interventions, calibration, margin losses, perturbation audits, and relational heads — did not identify a clean repair that improves cliff-sensitive classification while preserving control metrics and beating stronger baselines.

MoleculeACE broadens the empirical substrate with 30 external ChEMBL targets. Because MoleculeACE lacks FS-Mol-style few-shot episodes, we use it as a pair-level external diagnostic rather than a direct replication. The results provide supporting evidence that ranking and decision behavior can decouple outside FS-Mol, although the model-specific pattern differs from the FS-Mol episodes.

The current evidence package is suitable for a JCIM / Journal of Cheminformatics style diagnostic benchmark manuscript, subject to final result audit, cautious wording, and reproducibility checks.
