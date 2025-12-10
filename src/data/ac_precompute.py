import pickle
from typing import List, Tuple, Dict, Set
from .datasets import MoleculePropertyDataset

class ACPrecomputer:
    """
    Precomputes Matched Molecular Pairs (MMP) and Activity Cliffs (AC).
    """
    def __init__(self, mol_dataset: MoleculePropertyDataset):
        self.dataset = mol_dataset
        self.mmp_pairs: List[Tuple[int, int]] = [] # List of (mol_id_1, mol_id_2)
        self.ac_pairs: Dict[int, List[Tuple[int, int]]] = {} # prop_id -> list of (mol_id_1, mol_id_2)
        self.is_ac_node: Dict[int, bool] = {} # mol_id -> bool

    def compute_mmp_pairs(self):
        """
        Identify Matched Molecular Pairs (MMP) from the dataset.
        Uses Tanimoto similarity and RDKit MMPA as a heuristic.
        """
        try:
            from rdkit import Chem, DataStructs
            from rdkit.Chem import AllChem, rdMMPA
        except ImportError:
            print("RDKit not installed. Skipping MMP computation.")
            return

        print("Computing MMP pairs...")
        mols = {}
        fps = {}
        
        # 1. Precompute Fingerprints for all molecules
        for mid, smiles in self.dataset.mol_id_to_smiles.items():
            m = Chem.MolFromSmiles(smiles)
            if m:
                mols[mid] = m
                fps[mid] = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
        
        mol_ids = list(mols.keys())
        n_mols = len(mol_ids)
        
        # 2. Compare pairs (Heuristic: High Similarity but not identical)
        # For large datasets, use a library like Faiss or simpler blocking.
        # Here we use a simple O(N^2) loop for demonstration (WARNING: Slow for N > 5000)
        # In production, use self.dataset.iter_labeled_pairs to only check relevant pairs.
        
        # Optimization: Only check pairs that appear in the same property set?
        # Or just global structural analogs.
        
        count = 0
        for i in range(n_mols):
            for j in range(i + 1, n_mols):
                mid1 = mol_ids[i]
                mid2 = mol_ids[j]
                
                # Similarity Check
                sim = DataStructs.TanimotoSimilarity(fps[mid1], fps[mid2])
                
                # MMP Heuristic: High similarity (>0.7) but not identical (<1.0)
                # Or use rdMMPA.FragmentMol to check for single cut difference
                if 0.7 < sim < 1.0:
                    # Optional: Strict MMP check using rdMMPA
                    # frags = rdMMPA.FragmentMol(mols[mid1], minCuts=1, maxCuts=1, maxCutBonds=20)
                    # ... matching logic ...
                    
                    # For this agent task, we stick to the Similarity Heuristic which is robust
                    self.mmp_pairs.append((mid1, mid2))
                    count += 1
                    
        print(f"Found {count} MMP-like pairs based on similarity.")

    def label_ac_pairs(self, threshold: float = 0.0, split: str = 'train'):
        """
        Identify Activity Cliffs (AC) based on label differences.
        Strictly restricted to properties in the specified split (default: 'train').
        
        Args:
            threshold: Threshold for label difference.
            split: Dataset split to use ('train', 'val', 'test'). 
                   CRITICAL: Only use 'train' to avoid leakage.
        """
        print(f"Labeling AC pairs with threshold {threshold} for split '{split}'...")
        
        # Get properties for this split
        prop_ids = self.dataset.get_properties_split(split)
        if not prop_ids:
            print(f"No properties found for split '{split}'. Skipping AC labeling.")
            return

        count = 0
        # Only iterate over properties in the allowed split
        for prop_id in prop_ids:
            if prop_id not in self.dataset.labels:
                continue
                
            labels_map = self.dataset.labels[prop_id]
            self.ac_pairs[prop_id] = []
            
            # Check all MMP pairs
            for (mid1, mid2) in self.mmp_pairs:
                if mid1 in labels_map and mid2 in labels_map:
                    y1 = labels_map[mid1]
                    y2 = labels_map[mid2]
                    
                    # Check for Activity Cliff
                    if abs(y1 - y2) > threshold:
                        self.ac_pairs[prop_id].append((mid1, mid2))
                        self.is_ac_node[mid1] = True
                        self.is_ac_node[mid2] = True
                        count += 1
                        
        print(f"Identified {count} AC pairs across {len(prop_ids)} properties in '{split}'.")

    def save(self, path: str):
        """Save computed annotations to disk."""
        data = {
            'mmp_pairs': self.mmp_pairs,
            'ac_pairs': self.ac_pairs,
            'is_ac_node': self.is_ac_node
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
            
    def load(self, path: str):
        """Load annotations from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.mmp_pairs = data['mmp_pairs']
            self.ac_pairs = data['ac_pairs']
            self.is_ac_node = data['is_ac_node']
