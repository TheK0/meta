# AC-Aware GS-Meta Project Architecture

This document outlines the current architecture of the project after refactoring to support Activity Cliff (AC) awareness and GS-Meta learning.

## 1. Directory Structure

```text
project_root/
├── configs/                  # Configuration files (YAML)
│   ├── model_base.yaml       # Base hyperparameters
│   └── ...
├── data/                     # Data storage
│   ├── raw/                  # Raw CSV files
│   └── processed/            # Precomputed AC annotations
├── src/                      # Source Code
│   ├── data/                 # Data Handling
│   │   ├── datasets.py       # Dataset loader & Task Sampler (AC-aware)
│   │   ├── ac_precompute.py  # AC/MMP Mining (Leakage-free)
│   │   ├── mpg_builder.py    # Molecule-Property Graph Construction
│   │   └── featurization.py  # SMILES -> Graph conversion
│   ├── models/               # Model Definitions
│   │   ├── gsmeta_core.py    # Main Model (GNN + MPG Fusion)
│   │   ├── ac_inner_loop.py  # AC-Weighted Inner Loop Optimization
│   │   ├── ac_scheduler.py   # Episode-Level Task Scheduler
│   │   ├── gnn_encoder.py    # Molecular Graph Encoder
│   │   └── mpg_encoder.py    # Heterogeneous MPG Encoder
│   ├── training/             # Training Logic
│   │   ├── train_meta.py     # Main Entry Point
│   │   ├── loops.py          # Meta-Train/Test Loops (MAML Logic)
│   │   └── evaluators.py     # Metrics (ROC-AUC, RMSE)
│   └── experiments/          # Experiment Runners
└── scripts/                  # Utility Scripts
```

## 2. Key Components & Data Flow

### A. Data Pipeline
1.  **`datasets.py`**: Loads molecular data. `sample_task` now returns:
    *   Support/Query Molecules (MIDs)
    *   Labels
    *   **AC Masks**: Boolean flags indicating if a molecule is an Activity Cliff node (loaded from precomputation).
2.  **`ac_precompute.py`**: Mines Matched Molecular Pairs (MMP) and identifies Activity Cliffs. **Crucially restricted to the 'train' split** to prevent data leakage.
3.  **`mpg_builder.py`**: Constructs the heterogeneous graph connecting Molecules and Properties. Used to generate task-specific subgraphs for context.

### B. Model Architecture (`gsmeta_core.py`)
*   **`mol_encoder`**: Encodes molecular graphs into embeddings.
*   **`mpg_encoder`**: Encodes the task subgraph (MPG) to generate a "Task Context" embedding.
*   **Fusion**: The Task Context is fused (concatenated/added) with the Query Molecule embeddings before classification.

### C. Meta-Training Loop (`loops.py`)
The training loop implements a **MAML-style** optimization with AC enhancements:

1.  **Task Sampling**:
    *   Candidate tasks are sampled.
    *   **`ACScheduler`** scores candidates based on difficulty (AC ratio), selecting the best tasks for training.
2.  **Inner Loop (`ac_inner_loop.py`)**:
    *   Computes loss on Support Set.
    *   **AC Weighting**: Applies higher weights ($p_i$) to AC samples.
    *   **Curriculum**: Selects Top-R% hardest examples.
    *   Updates **Classifier Weights** (Fast Weights) using gradients. GNN encoder is frozen in the inner loop (First-Order MAML / Classifier-Update).
3.  **Outer Loop**:
    *   Uses updated Classifier weights to predict on Query Set.
    *   Computes loss on Query Set.
    *   Backpropagates through the entire model (GNN + Classifier initialization) to update global parameters.

### D. Evaluation (`meta_test_loop`)
*   Performs test-time adaptation on unseen tasks.
*   Computes standard metrics (ROC-AUC) and **AC-specific metrics** (performance on AC subset) using `evaluators.py`.

## 3. Current Status
*   **Refactoring Complete**: All core components (AC Inner Loop, Scheduler, MPG Context) are implemented and integrated.
*   **Ready for Experimentation**: The framework supports running baselines and full AC-aware experiments.
