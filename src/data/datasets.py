import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Generator, Set

class MoleculePropertyDataset:
    """
    Unified dataset loader for molecular property prediction.
    Handles raw data loading, property splitting (meta-train/val/test),
    and provides APIs for accessing molecules and labels.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_path = config.get('data_path', r"C:\Users\yishengyuan\Downloads\ML\meta\src\data\raw\tox21.csv")
        
        # Internal storage
        self.mol_id_to_smiles: Dict[int, str] = {}
        self.prop_id_to_name: Dict[int, str] = {}
        self.labels: Dict[int, Dict[int, float]] = {} # prop_id -> {mol_id -> label}
        
        # Splits: list of property IDs
        self.splits: Dict[str, List[int]] = {
            'train': [],
            'val': [],
            'test': []
        }
        
        # AC annotations
        self.ac_node_indices: Set[int] = set()
        
        self._load_data()

    def _load_data(self):
        """
        Load data from CSV file.
        Assumes columns: 'smiles' (or 'SMILES') and property columns.
        """
        if not os.path.exists(self.data_path):
            print(f"Warning: Data file {self.data_path} not found. Using mock data.")
            self._create_mock_data()
            return

        df = pd.read_csv(self.data_path)
        
        # 1. Determine SMILES column
        if 'smiles' in df.columns:
            smiles_col = 'smiles'
        elif 'SMILES' in df.columns:
            smiles_col = 'SMILES'
        else:
            raise ValueError("No 'smiles' or 'SMILES' column found in data.")
        
        # 2. Property columns = all numeric columns except SMILES, or from config
        default_prop_cols = [c for c in df.columns if c != smiles_col]

        # 只保留数值类型的列，避免 'TOX3021' 这种字符串列
        numeric_cols = [
            c for c in default_prop_cols
            if pd.api.types.is_numeric_dtype(df[c])
        ]

        prop_cols = self.config.get('prop_cols', numeric_cols)

        
        # 3. Populate molecules
        for idx, row in df.iterrows():
            self.mol_id_to_smiles[idx] = row[smiles_col]
            
        # 4. Populate properties and labels
        for p_idx, p_name in enumerate(prop_cols):
            self.prop_id_to_name[p_idx] = p_name
            self.labels[p_idx] = {}
            
            # Get valid labels (not NaN)
            valid_rows = df[~df[p_name].isna()]
            for idx, val in valid_rows[p_name].items():
                self.labels[p_idx][idx] = float(val)

        # 5. Split properties
        self._split_properties(list(self.prop_id_to_name.keys()))

    def _create_mock_data(self):
        """Create dummy data for testing."""
        self.mol_id_to_smiles = {0: 'C', 1: 'CC', 2: 'CCC'}
        self.prop_id_to_name = {0: 'prop_A', 1: 'prop_B'}
        self.labels = {
            0: {0: 1.0, 1: 0.0},
            1: {1: 1.0, 2: 0.0}
        }
        self.splits['train'] = [0]
        self.splits['test'] = [1]

    def _split_properties(self, prop_ids: List[int]):
        """
        Split properties into meta-train/val/test sets.
        """
        # Simple random split for now, or based on config
        ratios = self.config.get('split_ratios', {'train': 0.8, 'val': 0.1, 'test': 0.1})
        np.random.seed(self.config.get('seed', 42))
        np.random.shuffle(prop_ids)
        
        n = len(prop_ids)
        n_train = int(n * ratios['train'])
        n_val = int(n * ratios['val'])
        
        self.splits['train'] = prop_ids[:n_train]
        self.splits['val'] = prop_ids[n_train:n_train+n_val]
        self.splits['test'] = prop_ids[n_train+n_val:]
        
        print(f"Properties split: Train={len(self.splits['train'])}, Val={len(self.splits['val'])}, Test={len(self.splits['test'])}")

    def get_properties_split(self, split: str) -> List[int]:
        """Return the list of property IDs for a given split (train/val/test)."""
        return self.splits.get(split, [])

    def get_molecule_smiles(self, mol_id: int) -> str:
        """Return the SMILES string for a given molecule ID."""
        return self.mol_id_to_smiles.get(mol_id, "")

    def get_label(self, mol_id: int, prop_id: int) -> Optional[float]:
        """Return the label for a molecule-property pair. Returns None if unknown."""
        if prop_id in self.labels:
            return self.labels[prop_id].get(mol_id, None)
        return None

    def iter_labeled_pairs(self, split: str) -> Generator[Tuple[int, int, float], None, None]:
        """
        Iterate over all labeled pairs (mol_id, prop_id, label) in a given split.
        Useful for AC preprocessing.
        """
        prop_ids = self.get_properties_split(split)
        for pid in prop_ids:
            if pid in self.labels:
                for mid, label in self.labels[pid].items():
                    yield mid, pid, label

    def set_ac_annotations(self, ac_node_indices: Set[int]):
        """Set the set of molecule IDs that are identified as Activity Cliff nodes."""
        self.ac_node_indices = ac_node_indices

    def sample_task(self, split: str, k_shot: int, q_query: int) -> Optional[Tuple[int, List[int], List[int], List[float], List[float], List[bool], List[bool]]]:
        """
        Sample a task (property) and its support/query sets.
        
        Args:
            split: 'train', 'val', or 'test'
            k_shot: Number of support samples (per class if classification, total if regression)
            q_query: Number of query samples
            
        Returns:
            Tuple: (prop_id, support_mol_ids, query_mol_ids, support_labels, query_labels, support_ac_mask, query_ac_mask)
            Or None if sampling fails.
        """
        prop_ids = self.get_properties_split(split)
        if not prop_ids:
            return None
            
        # 1. Sample a property
        pid = np.random.choice(prop_ids)
        
        # 2. Get all valid molecules for this property
        valid_mols = list(self.labels[pid].keys())
        if len(valid_mols) < k_shot + q_query:
            return None # Not enough samples
            
        # 3. Sample support and query
        # Simple random sampling (can be improved to stratified sampling for classification)
        selected_mols = np.random.choice(valid_mols, k_shot + q_query, replace=False).tolist()
        support_mols = selected_mols[:k_shot]
        query_mols = selected_mols[k_shot:]
        
        support_labels = [self.labels[pid][m] for m in support_mols]
        query_labels = [self.labels[pid][m] for m in query_mols]
        
        # 4. Generate AC Masks
        support_ac_mask = [m in self.ac_node_indices for m in support_mols]
        query_ac_mask = [m in self.ac_node_indices for m in query_mols]
        
        return pid, support_mols, query_mols, support_labels, query_labels, support_ac_mask, query_ac_mask

