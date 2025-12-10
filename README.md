# Meta-Learning for Molecular Property Prediction

This project implements a meta-learning framework for molecular property prediction, incorporating Activity Cliff (AC) awareness and Molecule-Property Graphs (MPG).

## Structure

- `configs/`: Configuration files.
- `data/`: Data storage (raw and processed).
- `src/`: Source code.
  - `data/`: Data loading and processing.
  - `models/`: Model definitions (GNN, GS-Meta, AC components).
  - `training/`: Training loops and utilities.
  - `experiments/`: Experiment runners.
- `scripts/`: Utility scripts.
- `results/`: Output results.

## Setup

### Environment Installation (Conda + Pip with CUDA 12.1)

1. **Create and Activate Conda Environment**
   ```bash
   conda create -n meta_gnn python=3.10 -y
   conda activate meta_gnn
   ```

2. **Install PyTorch (CUDA 12.1)**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install PyTorch Geometric & Dependencies**
   *Note: The URL depends on your installed PyTorch version. Below assumes PyTorch 2.4.x or 2.5.x. If installation fails, check [PyG Installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).*
   ```bash
   pip install torch_geometric
   # Optional dependencies (scatter, sparse, etc.) - highly recommended for GNNs
   # Replace 'torch-2.4.0' with your actual torch version (e.g., torch-2.5.0) if different.
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
   ```

4. **Install Other Dependencies**
   ```bash
   pip install rdkit pandas numpy scikit-learn pyyaml
   ```

5. **Preprocess Data**
   ```bash
   python scripts/preprocess_mmp_ac.py
   ```

## Running Experiments

Run all experiments:
```bash
bash scripts/run_all_experiments.sh
```

Or run individual experiments:
```bash
python src/experiments/run_baselines.py
```
