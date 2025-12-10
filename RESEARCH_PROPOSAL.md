# Activity Cliff-Aware Meta-Learning for Molecular Property Prediction

## Research Proposal for Doctoral Dissertation

---

## 1. Introduction

### 1.1 Background and Motivation

Molecular property prediction is fundamental to drug discovery and materials science. Traditional machine learning models struggle with **data scarcity** in chemical space—for many properties, labeled data is expensive to obtain through wet-lab experiments. This challenge is exacerbated by **Activity Cliffs (ACs)**: pairs of structurally similar molecules with drastically different property values. ACs represent critical decision boundaries in chemical space but are notoriously difficult to predict.

**Key Challenges:**
1. **Few-Shot Learning**: New molecular properties often have <100 labeled examples
2. **Activity Cliffs**: Structurally similar molecules can have vastly different activities
3. **Task Heterogeneity**: Different properties require different inductive biases
4. **Knowledge Transfer**: How to leverage information from related properties?

### 1.2 Research Objectives

This dissertation proposes **AC-Aware GS-Meta**, a meta-learning framework that:
1. Learns to rapidly adapt to new molecular properties with minimal data
2. Explicitly focuses on Activity Cliff regions during adaptation
3. Leverages heterogeneous property-molecule graphs for task context
4. Implements curriculum learning to progressively focus on harder examples

---

## 2. Related Work

### 2.1 Meta-Learning for Molecular Property Prediction

**MAML (Finn et al., 2017)**: Model-Agnostic Meta-Learning learns good parameter initializations for rapid adaptation. Applied to molecular property prediction by:
- **PAR (Nguyen et al., 2020)**: Pre-training and Adapting Representations
- **Meta-MGNN (Guo et al., 2021)**: Meta-learning with Molecular Graph Neural Networks

**Limitations**: These methods treat all molecules equally, ignoring Activity Cliffs.

### 2.2 Activity Cliff Detection

**Matched Molecular Pairs (MMPs)** (Hussain & Rea, 2010): Pairs of molecules differing by a single structural transformation. Used to identify ACs when paired molecules have large property differences.

**SALI (Structure-Activity Landscape Index)** (Guha & Van Drie, 2008): Quantifies the relationship between structural similarity and activity difference.

**Limitations**: AC detection is typically a post-hoc analysis, not integrated into model training.

### 2.3 Graph-Based Meta-Learning

**GS-Meta (Zhou et al., 2019)**: Uses task-specific subgraphs to provide context for meta-learning. Constructs heterogeneous graphs connecting entities and tasks.

**Adaptation**: We extend this to molecular property prediction by building Molecule-Property Graphs (MPGs) that encode relationships between molecules and properties.

---

## 3. Methodology

### 3.1 Problem Formulation

**Meta-Learning Setup:**
- **Meta-Train**: Properties $\mathcal{P}_{\text{train}} = \{P_1, P_2, ..., P_M\}$
- **Meta-Test**: New properties $\mathcal{P}_{\text{test}}$ with few labeled examples
- **Episode**: For property $P_i$, sample support set $\mathcal{S}_i$ and query set $\mathcal{Q}_i$

**Activity Cliff Definition:**
Given Matched Molecular Pair $(m_1, m_2)$ with Tanimoto similarity $\text{sim}(m_1, m_2) > 0.7$:
$$\text{AC}(m_1, m_2, P_i) = \mathbb{1}[|y_{m_1}^{P_i} - y_{m_2}^{P_i}| > \tau]$$

### 3.2 Core Components

#### 3.2.1 Molecular Graph Encoder

**GNN Architecture**: Graph Isomorphism Network (GIN)
$$h_v^{(k+1)} = \text{MLP}^{(k)}\left((1+\epsilon^{(k)}) \cdot h_v^{(k)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k)}\right)$$

**Featurization**:
- **Atoms**: Atomic number, degree, formal charge, hybridization, aromaticity
- **Bonds**: Bond type, conjugation, ring membership

#### 3.2.2 Molecule-Property Graph (MPG)

**Construction**: For target property $P_{\text{target}}$:
1. **Nodes**: 
   - Molecules in support/query sets
   - Target property $P_{\text{target}}$
   - $k$ auxiliary properties (most correlated with target)
2. **Edges**:
   - $(m, P)$ with label "active" if $y_m^P = 1$
   - $(m, P)$ with label "inactive" if $y_m^P = 0$

**Heterogeneous GNN**: Encodes MPG to produce property embedding $\mathbf{c}_P$

#### 3.2.3 AC-Aware Inner Loop

**Objective**: Adapt classifier $f_\theta$ on support set $\mathcal{S}$ with AC weighting

**Per-Sample Loss**:
$$\ell_i = \text{BCE}(f_\theta(m_i), y_i)$$

**AC Weighting**:
$$w_i = \begin{cases} p & \text{if } m_i \in \mathcal{AC} \\ 1 & \text{otherwise} \end{cases}$$

**Top-R% Curriculum**: Select hardest $R(t)\%$ of weighted samples
$$R(t) = R_{\text{start}} + \frac{t}{T}(R_{\text{end}} - R_{\text{start}})$$

**Fast Weight Update**:
$$\theta' = \theta - \alpha \nabla_\theta \frac{1}{|\mathcal{T}|} \sum_{i \in \mathcal{T}} w_i \ell_i$$
where $\mathcal{T}$ are the top-R% samples.

#### 3.2.4 Meta-Training Objective

**Outer Loop**: Update global parameters $\theta$ to minimize query loss
$$\min_\theta \mathbb{E}_{P_i \sim \mathcal{P}_{\text{train}}} \left[ \mathcal{L}_{\mathcal{Q}_i}(f_{\theta'}) \right]$$

where $\theta'$ are the adapted parameters from the inner loop.

### 3.3 Episode Scheduling (Future Work)

**Motivation**: Not all tasks are equally informative. We want to prioritize tasks that:
1. Have high AC ratios (more challenging)
2. Show high inner-loop loss (model struggles)
3. Provide diverse learning signals

**Scheduler**: Neural network $s_\phi$ that scores candidate tasks
$$\text{score}_i = s_\phi(\text{AC\_ratio}_i, \text{inner\_loss}_i, ...)$$

Tasks sampled proportionally to scores for outer loop update.

---

## 4. Implementation

### 4.1 Architecture

```
project/
├── src/
│   ├── data/
│   │   ├── datasets.py          # Meta-learning dataset loader
│   │   ├── ac_precompute.py     # MMP & AC detection
│   │   ├── mpg_builder.py       # Heterogeneous graph construction
│   │   └── featurization.py     # SMILES → Graph conversion
│   ├── models/
│   │   ├── gnn_encoder.py       # Molecular GNN (GIN)
│   │   ├── mpg_encoder.py       # Heterogeneous GNN for MPG
│   │   ├── gsmeta_core.py       # Main model (GNN + MPG fusion)
│   │   ├── ac_inner_loop.py     # AC-weighted adaptation
│   │   └── ac_scheduler.py      # Episode-level task selection
│   └── training/
│       ├── train_meta.py        # Main entry point
│       ├── loops.py             # Meta-train/test loops
│       └── evaluators.py        # Metrics (ROC-AUC, AC-specific)
└── configs/                     # Hyperparameters
```

### 4.2 Key Design Decisions

1. **Classifier-Only Inner Loop**: Freeze GNN encoder during adaptation for stability
2. **First-Order MAML**: Use detached embeddings to reduce computational cost
3. **Leakage Prevention**: AC computation strictly on train split
4. **Functional Programming**: Use `torch.nn.utils.stateless.functional_call` for fast weights

### 4.3 Baseline Models

1. **Fine-Tuning**: Pre-train on all train properties, fine-tune on support set
2. **MAML**: Standard MAML without AC awareness
3. **AC-Inner**: MAML + AC-weighted inner loop (no MPG, no scheduler)
4. **AC-Full**: Complete system (AC-inner + MPG + scheduler)

---

## 5. Experiments

### 5.1 Datasets

**Tox21**: 12 toxicity assays, ~8k molecules
**SIDER**: 27 side-effect properties, ~1.4k drugs
**MUV**: 17 challenging targets, ~93k molecules

**Evaluation Protocol**:
- Meta-train: 60% of properties
- Meta-val: 20% of properties
- Meta-test: 20% of properties
- Few-shot: K ∈ {1, 5, 10} support examples per property

### 5.2 Metrics

**Global Performance**:
- ROC-AUC (classification)
- RMSE (regression)

**AC-Specific Performance**:
- ROC-AUC on AC subset
- Precision/Recall on AC pairs

### 5.3 Research Questions

**RQ1**: Does AC-aware weighting improve few-shot performance?
- Compare: MAML vs. AC-Inner

**RQ2**: Does MPG context help adaptation?
- Compare: AC-Inner vs. AC-Inner+MPG

**RQ3**: Does episode scheduling improve sample efficiency?
- Compare: AC-Inner+MPG vs. AC-Full

**RQ4**: How does performance scale with K-shot?
- Evaluate K ∈ {1, 5, 10, 20}

---

## 6. Expected Contributions

### 6.1 Methodological Contributions

1. **AC-Aware Meta-Learning**: First framework to explicitly integrate Activity Cliff awareness into meta-learning for molecular property prediction
2. **Curriculum Learning for Meta-Learning**: Novel Top-R% selection strategy for progressive difficulty
3. **Heterogeneous Task Graphs**: Extension of GS-Meta to molecular domain with MPGs

### 6.2 Empirical Contributions

1. Comprehensive evaluation on 3 benchmark datasets
2. Ablation studies isolating contributions of each component
3. Analysis of AC prediction performance vs. global performance

### 6.3 Practical Impact

1. **Drug Discovery**: Faster identification of lead compounds with limited data
2. **Toxicity Prediction**: Better safety assessment for rare endpoints
3. **Materials Science**: Rapid screening of novel materials

---

## 7. Timeline

**Year 1**:
- Q1-Q2: Literature review, baseline implementation
- Q3-Q4: AC-aware inner loop development and evaluation

**Year 2**:
- Q1-Q2: MPG integration and ablation studies
- Q3-Q4: Episode scheduler development

**Year 3**:
- Q1-Q2: Comprehensive experiments and analysis
- Q3-Q4: Dissertation writing and defense preparation

---

## 8. References

1. Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. *ICML*.
2. Zhou, F., Cao, C., Zhang, K., et al. (2019). Meta-GNN: On few-shot node classification in graph meta-learning. *CIKM*.
3. Hussain, J., & Rea, C. (2010). Computationally efficient algorithm to identify matched molecular pairs (MMPs) in large data sets. *J. Chem. Inf. Model.*
4. Nguyen, C. Q., Kreatsoulas, C., & Branson, K. M. (2020). Meta-learning GNN initializations for low-resource molecular property prediction. *ICML Workshop*.
5. Guha, R., & Van Drie, J. H. (2008). Structure–activity landscape index: Identifying and quantifying activity cliffs. *J. Chem. Inf. Model.*

---

## Appendix: Code Availability

**Repository**: `c:\Users\12559\Desktop\project\meta`

**Quick Start**:
```bash
# Install dependencies
pip install -r requirements.txt

# Precompute Activity Cliffs
python src/training/train_meta.py --config configs/model_base.yaml --precompute_ac

# Train AC-aware meta-learner
python src/training/train_meta.py --config configs/exp_ac_full.yaml

# Run baselines
python src/experiments/run_baselines.py
```

**Documentation**: See `README.md`, `ARCHITECTURE.md`, `VERIFICATION.md`
