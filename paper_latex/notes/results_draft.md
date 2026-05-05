# Results — Draft (2026-05-05)

---

## R1: Cliff-rich FS-Mol tasks are scarce but sufficient for a diagnostic benchmark

We applied the CliffBench construction pipeline to the full FS-Mol test pool of 157 assays. Under the core_relaxed profile (tau=0.80, delta=1.0, min_cliff=25), 6 of 157 assays (3.8%) met all eligibility criteria, rising to 10 assays (6.4%) under the extended_relaxed_10_10 profile (min_cliff=10). The main attrition bottleneck was the high-similarity discordant support stage: 128 of 157 assays (81.5%) lacked any active-inactive molecular pair with Tanimoto similarity >= 0.80, regardless of activity gap. This is consistent with the known scarcity of densely-cliffed structure-activity landscapes in public bioactivity databases.

**Threshold sensitivity**. We varied tau from 0.80 to 0.85 and the minimum cliff-pair threshold from 10 to 25. At tau=0.85 and min_cliff=25 (core_strict), eligible assays drop to 2, while tau=0.80 and min_cliff=10 (extended_relaxed_10_10) yields 10 assays. Further reducing min_noncliff to 5 (extended_relaxed_10_5) adds 1 additional assay. The sensitivity analysis confirms that the attrition funnel is well-behaved: small changes in thresholds produce predictable changes in assay count without qualitative shifts in the eligible assay composition.

**Profile characteristics**. The 10 eligible extended_relaxed_10_10 tasks contain 9,892 molecules (5,073 positive, 4,819 negative), 407 cliff pairs, and 756 high-similarity noncliff pairs, with a median of 28 cliff pairs per task (range 14-98). No single task dominates the pair distribution: CHEMBL1613777 contributes 24% of total cliff pairs. The 6-task core_relaxed profile contains 325 cliff pairs across 7,168 molecules. Each profile generates 400 standard and 400 adversarial episodes per task per seed (5 seeds), totaling 40,000 episodes for extended_relaxed_10_10.

**Table 1**: Attrition funnel (157 -> 6-10), profile coverage statistics.

---

## R2: Baselines exhibit a ranking-decision mismatch

We evaluated four baseline models — kNN (k=5), randomForest (500 trees), ProtoNet (pretrained on FS-Mol), and kNN-cliff-aware (cliff-augmented support negatives) — on the extended_relaxed_10_10 profile. All models used Morgan fingerprints (2048-bit, radius=2). ProtoNet additionally used the FS-Mol GNN graph feature extractor.

**ProtoNet is the strongest balanced model** (Table 2). On the adversarial split, ProtoNet achieves C-BAcc=0.561 (95% CI [0.518, 0.616]) and SCR=0.843 [0.776, 0.903], representing the best cliff-side classification and the lowest collapse rate among the four baselines. Versus kNN, ProtoNet improves C-BAcc by +0.050 [0.005, 0.106] and reduces SCR by -0.063 [-0.119, -0.021].

**randomForest reveals ranking-decision decoupling**. RF achieves the highest adversarial pair-ranking score (SQ-PSR=0.917 [0.894, 0.941], +0.355 over kNN) but this ranking advantage does not translate into better cliff decisions: C-BAcc=0.524 [0.499, 0.564], only marginally higher than kNN (+0.013 [-0.014, 0.056]). RF's SCR (0.912 [0.867, 0.950]) is the highest among all models, indicating the most severe decision-layer collapse. In other words, RF learns to rank hard pairs correctly but fails to convert that ranking into a usable active/inactive boundary.

**Same-scaffold pairs are systematically harder**. Across all four models, same-scaffold SCR (SS-SCR) meets or exceeds overall SCR. For RF, SS-SCR=0.899; for kNN, SS-SCR=0.901. ProtoNet shows the smallest scaffold penalty (SS-SCR=0.850 vs SCR=0.843), while kNN-cliff-aware achieves the lowest absolute SS-SCR (0.825) at the cost of lower C-BAcc than ProtoNet.

**kNN-cliff-aware provides modest but incomplete improvement**. The cliff-aware augmentation improves SCR over vanilla kNN by -0.061 [-0.108, -0.021] and NC-BAcc by +0.020 [0.002, 0.039] with clean statistical separation. However, the primary cliff classification gain (C-BAcc +0.022) has a 95% CI of [-0.0002, 0.040] which narrowly crosses zero. The intervention signal is real but not strong enough for a formal repair claim.

**The standard split confirms the pattern**. On the standard split (no adversarial cliff injection), ProtoNet again achieves the best balance (C-BAcc=0.528, SCR=0.829), while RF shows the largest Q-PSR/C-BAcc gap (Q-PSR=0.636, C-BAcc=0.520). The ranking-decision mismatch is not an artifact of adversarial episode construction: it is visible under both standard and adversarial evaluation.

**Table 2**: FS-Mol v5 baseline results (4 models, 9 metrics, 10 tasks).  
**Table 3**: Hypothesis and diagnostic evidence summary.

---

## R3: 22 shallow intervention families fail the stronger-baseline gate

We systematically evaluated 22 intervention families spanning six categories: decision repair, support-set modification, episode construction, calibration, training loss modification, and relational learning. Every intervention was evaluated on the relaxed_covext_10_10 intermediate substrate against the kNN and kNN-cliff-aware baselines. The stronger-baseline gate requires a method to surpass kNN-cliff-aware on primary metrics (C-BAcc, SCR) without causing clean negatives on safety metrics (NC-BAcc, NC-PSR, SQ-PSR).

**No intervention family passed the stronger-baseline gate** (Table S1). The most common failure modes were:

1. **Control-side harm** (8 families): Methods such as local-boundary-repair and fixed-support hard-negative replacement improved one primary metric but degraded noncliff control metrics (NC-BAcc, NC-PSR) by statistically significant margins, indicating that the intervention was not cliff-specific.

2. **Vanilla-only wins** (3 families): The corrected query-targeted support negatives variant and related episode-construction approaches showed clean improvements over vanilla kNN but failed to beat the stronger kNN-cliff-aware baseline. Beating the weakest baseline is insufficient for a method claim.

3. **Structural no-effect** (2 families): The boundary_uncertainty calibration and CASE-Net v1 at lambda=0.5 produced predictions identical to ProtoNet on all discrete metrics. The calibration is inherently sign-preserving (margin shrinkage never changes binary predictions), making it structurally incapable of repairing decision-layer collapse.

4. **Insufficient supervision** (1 family): CASE-Net v1's per-episode logistic regression relation head had a median of 2-4 annotated support-support pairs per episode, far too few to learn meaningful cliff-vs-noncliff distinctions. At lambda=0.0 (pure evidence), SCR dropped to 0.510 but C-BAcc also decreased to 0.487, consistent with conservative smoothing rather than cliff-aware reasoning.

5. **No transferable signal** (1 family): CASE-Net v2 trained a cross-task RandomForest relation head on 3,376 pairs from 290 FS-Mol train assays. On 1,858 held-out validation pairs from 19 valid assays, the classifier achieved AUPRC=0.247 (barely above base rate 0.233), AUC-ROC=0.506, and balanced accuracy=0.463. The predicted cliff probabilities on cliff pairs (0.480) and noncliff pairs (0.469) were indistinguishable. Simple 2D pair descriptors (Morgan fingerprint absolute difference, intersection, Tanimoto similarity, scaffold identity, bit counts) were insufficient for transferable cliff-vs-highsim_noncliff relation prediction in this setting.

**Implications**. The systematic failure of shallow interventions — including approaches that directly modify the decision threshold, the support set composition, the episode structure, and the scoring calibration — suggests that the ranking-decision mismatch is not trivially repairable. The consistent pattern of control-side harm when interventions do have an effect indicates that cliff-sensitive repair requires mechanisms that explicitly model the local structure-activity relationship rather than globally re-weighting scores or augmenting negatives.

**Table S1**: 22-family intervention registry with per-family details.

---

## R4: MoleculeACE provides external pair-level supporting evidence

To assess whether the ranking-decision decoupling observed on FS-Mol generalizes beyond the FS-Mol data distribution, we conducted a pair-level diagnostic on 30 MoleculeACE targets (48,714 molecules). Because MoleculeACE does not provide few-shot episode splits, we used the dataset authors' train/test partitions, median-split binarization per target, and pair-level cliff metric computation with matching tau=0.80 and delta=1.0. This protocol differs from FS-Mol's episode-based few-shot evaluation; we therefore interpret MoleculeACE results as external supporting evidence rather than a direct replication.

**Table 4**: MoleculeACE external pair-level diagnostic (25 targets with >=1 test cliff pair, 2000 bootstrap iterations).

**RF ranking advantage is cross-dataset robust**. RF shows higher Q-PSR than kNN in 22 of 25 targets (88%) with test cliff pairs. The macro-averaged RF Q-PSR (0.647 [0.553, 0.735]) substantially exceeds kNN Q-PSR (0.279 [0.193, 0.370]). This consistent ranking advantage mirrors the FS-Mol finding (where RF SQ-PSR exceeds kNN SQ-PSR by +0.355).

**Cliff decision performance is similar between models**. With the corrected pair-balanced C-BAcc metric, kNN (0.522 [0.493, 0.552]) and RF (0.526 [0.509, 0.547]) show nearly identical cliff decision accuracy. This contrasts with FS-Mol, where ProtoNet achieved a wider C-BAcc advantage over kNN. The narrower gap in MoleculeACE may reflect the pair-level protocol (which lacks the support-query structure that amplifies decision-boundary effects in few-shot episodes) or the different activity measurement distributions in ChEMBL-derived data.

**Collapse is modestly elevated in RF relative to kNN**. RF SCR (0.902 [0.859, 0.941]) exceeds kNN SCR (0.875 [0.828, 0.918]) in 16 of 25 targets (64%). This is directionally consistent with the FS-Mol pattern (RF SCR=0.912 vs kNN SCR=0.906 on ext_10_10), although the MoleculeACE gap is narrower. The coexistence of RF's ranking advantage with comparable-or-worse decision performance and modestly higher collapse supports the interpretation that ranking and decision behavior can decouple, but the specific model showing the strongest decoupling differs between substrates.

**Sensitivity to test-pair availability**. Of 30 MoleculeACE targets, 25 have >=1 test cliff pair at tau=0.80, 13 have >=3, and only 8 have >=5. The macro-averaged metrics are therefore driven primarily by 8-13 targets with moderate-to-high cliff pair counts, and the confidence intervals reflect this limited sample size. This sensitivity highlights the practical challenge of constructing external cliff diagnostic sets: even a dataset specifically curated for activity cliff analysis contains many targets with very few cross-threshold high-similarity test pairs.

**Interpretation**. MoleculeACE provides external supporting evidence that the ranking-decision relationship observed on FS-Mol is not purely a FS-Mol-specific artifact. The RF ranking advantage is robust across datasets, and the SCR elevation accompanying this advantage is directionally consistent. However, the model-specific mismatch pattern differs between FS-Mol (where RF shows the widest Q-PSR/C-BAcc gap) and MoleculeACE (where neither model shows a clear C-BAcc advantage). We attribute these differences to the pair-level protocol and the distinct assay composition of MoleculeACE, and we emphasize that the MoleculeACE analysis is an external diagnostic substrate rather than a direct replication of the FS-Mol few-shot protocol.
