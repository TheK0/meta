# CliffBench v5 — Manuscript Outline

**Target**: JCIM / Journal of Cheminformatics
**Status**: Evidence consolidation complete; Results draft in progress

---

## 1. Introduction

- Few-shot molecular classification is attractive for data-poor assays
- Average metrics can mask chemically critical local failures
- Activity cliffs as diagnostic probes: small structural changes, large potency shifts
- CliffBench: assay-local few-shot benchmark derived from FS-Mol, exposing decision-layer collapse
- Contributions:
  1. V5 tiered benchmark release with 4 profiles across 2-11 tasks
  2. Systematic baseline evaluation showing ranking-decision mismatch
  3. Audit of 22 shallow intervention families, all failing stronger-baseline gate
  4. External pair-level diagnostic on 30 MoleculeACE targets

## 2. Related Work

- Few-shot molecular property prediction (FS-Mol, ProtoNet, MAML, MAT)
- Activity cliff benchmarks (MoleculeACE, ASC-ADMET)
- Decision-layer diagnostics in molecular ML
- Intervention methods (calibration, contrastive training)

## 3. CliffBench v5 Benchmark Construction

### 3.1 Data source and assay-local protocol
- FS-Mol test pool: 157 assays
- Assay-local principle: all pair mining, episode generation, and scoring within single assays
- Molecule filtering: canonical SMILES, precise measurements, deduplication
- Activity cliff definition: Tanimoto >= tau, |activity_gap| >= delta

### 3.2 Attrition funnel
- 157 -> 6-10 eligible tasks depending on profile
- Main bottleneck: high-sim discordant support (128 removed)
- Threshold sensitivity analysis across tau 0.80-0.85, min_cliff 10-25

### 3.3 Tiered profiles
- core_strict (tau=0.85, min_cliff=25): 2 tasks
- core_relaxed (tau=0.80, min_cliff=25): 6 tasks
- extended_relaxed_10_10 (tau=0.80, min_cliff=10): 10 tasks
- extended_relaxed_10_5 (tau=0.80, min_cliff=10, min_noncliff=5): 11 tasks

### 3.4 Episode protocol
- 2-way few-shot, 16 support/class, 16 query/class
- 400 standard + 400 adversarial episodes per task/seed/split
- Adversarial episodes: cliff-pair injection via bipartite matching
- Deterministic seed-based episode generation for reproducibility

### 3.5 Evaluation protocol
- Four baselines: kNN, randomForest, ProtoNet, kNN-cliff-aware
- Nine core metrics in three families: classification, pair-ranking, collapse
- Task-level macro aggregation with 10,000-iteration bootstrap CI

## 4. Baseline Results

### 4.1 Main Table: extended_relaxed_10_10 (10 tasks)
- ProtoNet: strongest balanced model (C-BAcc=0.561, SCR=0.843)
- randomForest: clearest ranking-competent/decision-collapsed example (SQ-PSR=0.917, C-BAcc=0.524, SCR=0.912)
- kNN-cliff-aware: modest SCR improvement over kNN (-0.061) with preserved controls

### 4.2 Ranking-decision mismatch (H2 evidence)
- RF preserves ranking (SQ-PSR=0.917) but decisions are fragile (SCR=0.912)
- Same-scaffold SCR systematically exceeds overall SCR across all models
- ProtoNet shows more balanced profile but same-scaffold vulnerability persists

### 4.3 core_relaxed (6 tasks) — consistent pattern
- Same ranking-decision mismatch, smaller task count

## 5. Intervention Audit

### 5.1 Audit design
- Stronger-baseline gate: must beat kNN-cliff-aware (minimum), ProtoNet (upgrade)
- Primary metrics: C-BAcc, SCR, SS-SCR
- Safety metrics: NC-BAcc, NC-PSR, SQ-PSR (no clean negative)
- 22 families across 6 categories

### 5.2 Intervention categories and outcomes
- Decision repair (2): threshold repair, local-boundary repair — both NO-GO
- Support-set interventions (2): hard-negative replacement, partial augmentation — both NO-GO
- Episode construction (7 variants): all fail stronger-baseline gate
- Calibration (3): A1 query-only LR (NO-GO), boundary_uncertainty (NO-GO), CASE-Net v1 per-episode LR (NO-GO)
- Training (1): B0 cliff-margin loss (NO-GO)
- Relational (1): CASE-Net v2 pretrained relation head (NO-GO at pair-level gate)

### 5.3 Patterns in failure
- Control-side harm: most common failure mode
- Vanilla-only wins: some methods beat kNN but not kNN-cliff-aware
- Structural limitations: sign-preserving calibration cannot change discrete predictions
- Insufficient supervision: per-episode support pairs too sparse for relation learning

## 6. MoleculeACE External Pair-Level Diagnostic

### 6.1 Motivation and protocol differences
- FS-Mol task count is small (6-10); external evidence needed
- MoleculeACE: 30 ChEMBL targets, 48,714 molecules
- Pair-level evaluation (not episode-based)
- Matching tau=0.80, delta=1.0, Morgan fingerprint protocol

### 6.2 Results
- RF ranking advantage in 22/25 targets (88%)
- C-BAcc nearly identical between kNN (0.522) and RF (0.526)
- RF SCR (0.902) > kNN SCR (0.875)
- Consistent with ranking-decision decoupling, but pattern differs from FS-Mol

### 6.3 Interpretation
- MoleculeACE as external supporting evidence, not direct replication
- Protocol differences documented: pair-level, median-split labels, no episodes
- Strengthens the generalizability claim while acknowledging methodological boundaries

## 7. Discussion

### 7.1 Benchmark validity
- CliffBench captures a real and diagnosable failure mode
- Task count limitation mitigated by MoleculeACE external evidence
- Deterministic reproducibility enables independent verification

### 7.2 Why interventions fail
- Shallow interventions cannot fix deep ranking-decision decoupling
- 2D pair descriptors insufficient for transferable cliff relation prediction
- Representation-level approaches remain unexplored

### 7.3 Limitations
- 6-10 FS-Mol tasks is narrow for a benchmark
- MoleculeACE is pair-level, not episode-based
- MAML not fully evaluated
- No successful repair method identified
- Behavioral evidence, not mechanistic proof

### 7.4 Future work
- Larger-scale cliff-specific benchmarks
- External data sources beyond ChEMBL
- Representation-level interventions (pre-training, contrastive learning)
- Controlled prospective validation

## 8. Conclusion

CliffBench v5 exposes a consistent failure mode — ranking-decision mismatch — in few-shot molecular classification. Activity cliffs serve as high-value diagnostic probes that reveal when models can rank hard pairs correctly but cannot convert that ranking into reliable binary decisions. 22 shallow intervention families fail to repair this gap while preserving control metrics. MoleculeACE provides external supporting evidence that ranking and decision behavior can decouple beyond FS-Mol, though with a distinct model-specific pattern due to protocol differences. The current evidence package supports CliffBench as a diagnostic benchmark for activity-cliff-sensitive model evaluation in few-shot molecular learning.

---

## Tables (main paper — 4 tables)

| Table | Title | Location |
|-------|-------|----------|
| Table 1 | Attrition funnel and profile coverage | Section 3 |
| Table 2 | FS-Mol v5 main baseline results (extended_relaxed_10_10) | Section 4 |
| Table 3 | Hypothesis and diagnostic evidence summary | Section 4 |
| Table 4 | MoleculeACE external pair-level diagnostic | Section 6 |

## Supplement (SI)

- S1: 22-family intervention registry with per-family details
- S2: CASE-Net v1/v2 pair-level diagnostics (per-episode pair counts, AUPRC, BAcc)
- S3: MoleculeACE per-target results (30 rows)
- S4: Reproducibility checklist (commit hashes, pytest output, hash-lock details)
- S5: Threshold sensitivity and attrition audit details
