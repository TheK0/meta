# Activity Cliffs as Probes of Decision-Layer Collapse in Few-Shot Molecular Classification

**Target**: JCIM / Journal of Cheminformatics
**Status**: Evidence complete; Results drafted; Abstract/Methods/Discussion as outlines below

---

## Abstract

[PLACEHOLDER — to be finalized after full draft]

Few-shot molecular classification models may preserve pairwise ranking on hard
molecular pairs while failing to form robust binary decisions around activity
cliffs — regions where structurally similar molecules exhibit large potency
shifts. We introduce CliffBench v5, an assay-local few-shot diagnostic benchmark
derived from FS-Mol that systematically evaluates this ranking-decision
mismatch across four baseline models and 22 intervention families. On a 10-task
extended profile, ProtoNet achieves the best balance (C-BAcc=0.561, SCR=0.843)
while randomForest shows the clearest decoupling (SQ-PSR=0.917 but C-BAcc=0.524,
SCR=0.912). All 22 shallow intervention families fail a stronger-baseline gate,
most commonly through control-side harm or vanilla-only improvements. An
external pair-level diagnostic on 30 MoleculeACE targets provides supporting
evidence that ranking-decision decoupling is not FS-Mol-specific, although the
model-specific pattern differs due to protocol differences. CliffBench v5
provides a reproducible diagnostic substrate for evaluating whether few-shot
molecular classifiers form reliable decisions in the chemically critical regions
where activity cliffs concentrate.

---

## 1. Introduction

[OUTLINE]

- Few-shot molecular classification is attractive for data-poor assays
- Average few-shot metrics can obscure chemically critical local failures
- Activity cliffs: small structural changes, large potency shifts
- Decision-layer collapse: ranking remains partly intact, but binary decisions
  become fragile in cliff-heavy regions
- Existing benchmarks: FS-Mol (broad few-shot, no cliff diagnostics),
  MoleculeACE (cliff-focused, supervised, not few-shot)
- CliffBench v5: assay-local few-shot benchmark derived from FS-Mol that
  exposes decision-layer collapse through complementary metric families and
  adversarial episode evaluation
- Contributions:
  1. V5 tiered benchmark release (4 profiles, 2-11 tasks)
  2. Systematic baseline evaluation (4 models, 9 metrics)
  3. Audit of 22 shallow intervention families
  4. External pair-level diagnostic on 30 MoleculeACE targets

## 2. Related Work

[OUTLINE]

- Few-shot molecular property prediction: FS-Mol, ProtoNet, MAML, MAT,
  GNN-based meta-learning
- Activity cliff benchmarks: MoleculeACE (supervised, pair-level, 30 targets),
  ASC-ADMET
- Model diagnostics: calibration, uncertainty quantification, failure mode
  analysis in molecular ML
- Intervention methods: contrastive training, margin losses, support-set
  augmentation, relational learning
- Relationship to this work: CliffBench fills the gap between broad few-shot
  benchmarks and supervised cliff benchmarks by providing assay-local,
  episode-level cliff diagnostics

## 3. CliffBench v5: Benchmark Construction

[OUTLINE — detailed in outline.md Section 3]

### 3.1 Assay-local protocol
- FS-Mol test pool: 157 assays
- Assay-local principle: per-assay pair mining, episode generation, scoring
- Molecule filtering: canonical SMILES, precise measurements, deduplication
- Activity cliff definition: Tanimoto >= tau, |gap| >= delta
- Fingerprint: Morgan ECFP4, 2048-bit, radius=2

### 3.2 Attrition funnel
- 157 -> 6-10 eligible tasks (extended_relaxed_10_10)
- Main bottleneck: 128/157 assays lack any high-sim discordant pair
- Threshold sensitivity: tau 0.80-0.85, min_cliff 10-25
- Extended COVEXT profiles increase task count from 6 to 10 by relaxing
  min_cliff from 25 to 10

### 3.3 Tiered profiles
- Four profiles: core_strict, core_relaxed, ext_10_10, ext_10_5
- Two analysis profiles: core_relaxed (6 tasks), ext_10_10 (10 tasks)
- Episode protocol: 2-way, 16/class support, 16/class query
- 400 standard + 400 adversarial episodes per task/seed/split
- Adversarial episodes: cliff-pair injection via maximum bipartite matching
- Deterministic seed-based generation for reproducibility

### 3.4 Evaluation protocol
- Four baselines: kNN, randomForest, ProtoNet (pretrained), kNN-cliff-aware
- Nine core metrics: classification (C-BAcc, NC-BAcc), pair-ranking
  (Q-PSR, SQ-PSR, NC-PSR, SS-Q-PSR, SS-SQ-PSR), collapse (SCR, SS-SCR)
- Task-level macro aggregation, 10,000-iteration bootstrap CI
- Paired bootstrap for model comparisons

**Table 1**: Attrition funnel + profile coverage.

## 4. Results

### R1: Cliff-rich FS-Mol tasks are scarce but sufficient for a diagnostic benchmark

We applied the CliffBench construction pipeline to the full FS-Mol test pool of
157 assays. Under the core_relaxed profile (tau=0.80, delta=1.0, min_cliff=25),
6 of 157 assays (3.8%) met all eligibility criteria, rising to 10 assays (6.4%)
under the extended_relaxed_10_10 profile (min_cliff=10). The main attrition
bottleneck was the high-similarity discordant support stage: 128 of 157 assays
(81.5%) lacked any active-inactive molecular pair with Tanimoto similarity >=
0.80, regardless of activity gap. This is consistent with the known scarcity of
densely-cliffed structure-activity landscapes in public bioactivity databases.

Threshold sensitivity analysis varying tau from 0.80 to 0.85 and min_cliff from
10 to 25 showed that the attrition funnel is well-behaved: small parameter
changes produce predictable changes in eligible assay count without qualitative
shifts in assay composition. At tau=0.85 and min_cliff=25 (core_strict),
eligible assays drop to 2, while tau=0.80 and min_cliff=10 (ext_10_10) yields
10 assays.

The 10 eligible extended_relaxed_10_10 tasks contain 9,892 molecules, 407 cliff
pairs, and 756 high-similarity noncliff pairs, with a median of 28 cliff pairs
per task (range 14-98). No single task dominates the pair distribution:
CHEMBL1613777 contributes 24% of total cliff pairs. Each profile generates 400
standard and 400 adversarial episodes per task per seed (5 seeds), totaling
40,000 episodes for extended_relaxed_10_10.

**Table 1**: Attrition funnel and profile coverage statistics.

### R2: Baselines reveal a ranking-decision mismatch

We evaluated four baseline models — kNN (k=5), randomForest (500 trees,
max_depth=20), ProtoNet (pretrained FS-Mol GNN), and kNN-cliff-aware
(cliff-augmented support negatives) — on the extended_relaxed_10_10 profile
(Table 2). All models used Morgan fingerprints (2048-bit, radius=2). ProtoNet
additionally used the FS-Mol GNN graph feature extractor.

**ProtoNet is the strongest balanced model**. On the adversarial split, ProtoNet
achieves C-BAcc=0.561 (95% CI [0.518, 0.616]) and SCR=0.843 [0.776, 0.903],
representing the best cliff-side classification and the lowest collapse rate.
Versus kNN, ProtoNet improves C-BAcc by +0.050 [0.005, 0.106] and reduces SCR
by -0.063 [-0.119, -0.021].

**randomForest reveals ranking-decision decoupling**. RF achieves the highest
adversarial pair-ranking score (SQ-PSR=0.917 [0.894, 0.941], +0.355 over kNN)
but this ranking advantage does not translate into better cliff decisions:
C-BAcc=0.524 [0.499, 0.564], only marginally higher than kNN (+0.013
[-0.014, 0.056]). RF's SCR (0.912 [0.867, 0.950]) is the highest among all
models, indicating the most severe decision-layer collapse. RF learns to rank
hard pairs correctly but fails to convert that ranking into a usable
active/inactive boundary.

**Same-scaffold pairs are systematically harder**. Across all four models,
same-scaffold SCR meets or exceeds overall SCR. For RF, SS-SCR=0.899; for kNN,
SS-SCR=0.901. ProtoNet shows the smallest scaffold penalty (SS-SCR=0.850 vs
SCR=0.843), while kNN-cliff-aware achieves the lowest absolute SS-SCR (0.825).

**kNN-cliff-aware provides modest but incomplete improvement**. The cliff-aware
augmentation improves SCR over vanilla kNN by -0.061 [-0.108, -0.021] and
NC-BAcc by +0.020 [0.002, 0.039] with clean statistical separation. However,
the C-BAcc gain (+0.022) has a 95% CI of [-0.0002, 0.040] which narrowly
crosses zero.

The ranking-decision mismatch is visible under both standard and adversarial
evaluation. On the standard split, ProtoNet again achieves the best balance
(C-BAcc=0.528, SCR=0.829), while RF shows the largest Q-PSR/C-BAcc gap
(Q-PSR=0.636, C-BAcc=0.520).

**Table 2**: FS-Mol v5 baseline results (4 models, 6 metrics, adversarial split).  
**Table 3**: Hypothesis and diagnostic evidence summary.

### R3: Shallow interventions fail the stronger-baseline gate

We systematically evaluated 22 intervention families spanning six categories:
decision repair, support-set modification, episode construction, calibration,
training loss modification, and relational learning. Every intervention was
evaluated on the relaxed_covext_10_10 intermediate substrate. The
stronger-baseline gate requires a method to surpass kNN-cliff-aware on primary
metrics (C-BAcc, SCR) without causing clean negatives on safety metrics
(NC-BAcc, NC-PSR, SQ-PSR).

**No intervention family passed the stronger-baseline gate** (Table S1). The
most common failure modes were:

1. **Control-side harm** (8 families). Methods such as local-boundary-repair
   and fixed-support hard-negative replacement improved one primary metric but
   degraded noncliff control metrics by statistically significant margins.

2. **Vanilla-only wins** (3 families). Episode-construction approaches showed
   clean improvements over vanilla kNN but failed to beat the stronger
   kNN-cliff-aware baseline.

3. **Structural no-effect** (2 families). The boundary_uncertainty calibration
   and CASE-Net v1 at lambda=0.5 produced predictions identical to ProtoNet on
   all discrete metrics. Sign-preserving calibration is structurally incapable
   of changing binary predictions.

4. **Insufficient supervision** (1 family). CASE-Net v1's per-episode logistic
   regression had a median of 2-4 annotated support-support pairs per episode,
   far too few to learn meaningful cliff-vs-noncliff distinctions.

5. **No transferable signal** (1 family). CASE-Net v2's cross-task RandomForest
   achieved AUPRC=0.247 (base rate 0.233) and BAcc=0.463 on 1,858 held-out
   validation pairs from 19 FS-Mol valid assays. The predicted probabilities on
   cliff (0.480) and noncliff (0.469) pairs were indistinguishable.

**Implications**. The systematic failure of shallow interventions suggests that
the ranking-decision mismatch is not trivially repairable through threshold
adjustment, support-set modification, or calibration. The consistent pattern of
control-side harm when interventions do have an effect indicates that
cliff-sensitive repair requires mechanisms that explicitly model the local
structure-activity relationship rather than globally re-weighting scores or
augmenting negatives.

**Table S1**: 22-family intervention registry.

### R4: MoleculeACE provides external pair-level supporting evidence

To assess whether the ranking-decision decoupling observed on FS-Mol generalizes
beyond the FS-Mol data distribution, we conducted a pair-level diagnostic on 30
MoleculeACE targets (48,714 molecules). Because MoleculeACE does not provide
few-shot episode splits, we used the dataset authors' train/test partitions,
median-split binarization per target, and pair-level metric computation with
matching tau=0.80 and delta=1.0. This protocol differs from FS-Mol's
episode-based few-shot evaluation; we therefore interpret MoleculeACE results
as external supporting evidence rather than a direct replication.

**Results**. Of 30 MoleculeACE targets, 25 have >=1 test cliff pair at
tau=0.80, 13 have >=3, and 8 have >=5 (Table 4). RF shows higher Q-PSR than
kNN in 22 of 25 targets (88%). The macro-averaged RF Q-PSR (0.647 [0.553,
0.735]) substantially exceeds kNN Q-PSR (0.279 [0.193, 0.370]), consistent with
the FS-Mol finding of an RF ranking advantage. C-BAcc is nearly identical
between models (kNN=0.522 [0.493, 0.552]; RF=0.526 [0.509, 0.547]). RF SCR
(0.902 [0.859, 0.941]) modestly exceeds kNN SCR (0.875 [0.828, 0.918]) in 16
of 25 targets (64%), directionally consistent with the FS-Mol pattern but
narrower in magnitude.

**Interpretation**. MoleculeACE provides external supporting evidence that the
RF ranking advantage is cross-dataset robust and that ranking improvement can
coexist with modestly elevated collapse outside FS-Mol. However, the
model-specific mismatch pattern differs: FS-Mol shows a wide Q-PSR/C-BAcc gap
for RF, while MoleculeACE shows similar C-BAcc but elevated SCR. These
differences reflect the distinct evaluation protocols (pair-level vs
episode-level) and assay compositions. MoleculeACE is an external diagnostic
substrate, not a direct replication of the FS-Mol few-shot protocol.

**Table 4**: MoleculeACE external pair-level diagnostic.
**Table S3**: MoleculeACE per-target results.

## 5. Discussion

[OUTLINE — detailed in outline.md Section 7]

### 5.1 Benchmark validity
- CliffBench captures a real and diagnosable failure mode
- Task count limitation partially mitigated by MoleculeACE external evidence
- Deterministic reproducibility enables independent verification

### 5.2 Why interventions fail
- Shallow interventions cannot fix deep ranking-decision decoupling
- Simple 2D pair descriptors insufficient for transferable cliff relation
  prediction in the tested setup
- Representation-level approaches remain unexplored

### 5.3 Limitations
- 6-10 FS-Mol tasks is narrow for a benchmark
- MoleculeACE is pair-level, not episode-based
- MAML not fully evaluated on v5
- No successful repair method identified
- Behavioral evidence, not mechanistic proof

### 5.4 Future work
- Larger-scale cliff-specific benchmarks
- External data sources beyond ChEMBL
- Representation-level interventions
- Controlled prospective validation

## 6. Conclusion

CliffBench v5 exposes a consistent failure mode — ranking-decision mismatch — in
few-shot molecular classification. Activity cliffs serve as high-value
diagnostic probes that reveal when models can rank hard pairs correctly but
cannot convert that ranking into reliable binary decisions. ProtoNet achieves
the most balanced profile, while randomForest is the clearest example of
ranking-competent but decision-collapsed behavior. A systematic audit of 22
shallow intervention families finds no method that repairs this gap while
preserving control metrics and beating stronger baselines. MoleculeACE provides
external pair-level supporting evidence that ranking and decision behavior can
decouple beyond FS-Mol, though with a distinct model-specific pattern reflecting
protocol differences. The current evidence package supports CliffBench as a
diagnostic benchmark for activity-cliff-sensitive model evaluation in few-shot
molecular learning.

---

## Tables

| Table | Title | Section |
|-------|-------|---------|
| Table 1 | Attrition funnel and v5 profile coverage | R1 |
| Table 2 | FS-Mol v5 baseline results (ext_10_10, adversarial) | R2 |
| Table 3 | Hypothesis and diagnostic evidence summary | R2 |
| Table 4 | MoleculeACE external pair-level diagnostic | R4 |

## Supporting Information

| Table | Title |
|-------|-------|
| S1 | 22-family intervention registry |
| S2 | CASE-Net v1/v2 pair-level diagnostic details |
| S3 | MoleculeACE per-target results |
| S4 | Reproducibility checklist |
| S5 | Threshold sensitivity and attrition audit |
