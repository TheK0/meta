try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    Chem = None
    AllChem = None

import numpy as np

def smiles_to_graph(smiles: str):
    """
    Convert a SMILES string to a graph representation.
    
    Args:
        smiles: SMILES string.
        
    Returns:
        A dictionary containing graph data (nodes, edges, features) 
        compatible with PyG/DGL, or None if invalid.
    """
    if Chem is None:
        # In a real environment, we would raise an error or handle this.
        # For now, we allow it to pass if RDKit is missing, but return None.
        print("Warning: RDKit not installed. Cannot convert SMILES to graph.")
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # --- Atom Featurization ---
    atom_features = []
    for atom in mol.GetAtoms():
        features = []
        # 1. Atomic Number (One-hot or integer) - using integer for simplicity with embedding layer
        features.append(atom.GetAtomicNum())
        # 2. Degree
        features.append(atom.GetTotalDegree())
        # 3. Formal Charge
        features.append(atom.GetFormalCharge())
        # 4. Hybridization (Mapped to int)
        features.append(int(atom.GetHybridization()))
        # 5. Aromaticity
        features.append(1 if atom.GetIsAromatic() else 0)
        # 6. Total Num Hs
        features.append(atom.GetTotalNumHs())
        # 7. Chirality (Optional, skipping for simplicity)
        
        atom_features.append(features)
    
    # --- Bond Featurization ---
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        edge_indices.append((i, j))
        edge_indices.append((j, i))
        
        features = []
        # 1. Bond Type (Single, Double, Triple, Aromatic)
        bt = bond.GetBondType()
        bt_val = 0
        if bt == Chem.rdchem.BondType.SINGLE: bt_val = 1
        elif bt == Chem.rdchem.BondType.DOUBLE: bt_val = 2
        elif bt == Chem.rdchem.BondType.TRIPLE: bt_val = 3
        elif bt == Chem.rdchem.BondType.AROMATIC: bt_val = 4
        features.append(bt_val)
        
        # 2. Conjugated
        features.append(1 if bond.GetIsConjugated() else 0)
        
        # 3. In Ring
        features.append(1 if bond.IsInRing() else 0)
        
        edge_features.append(features)
        edge_features.append(features) # Undirected
        
    return {
        'num_nodes': len(atom_features),
        'node_features': np.array(atom_features, dtype=np.float32),
        'edge_index': np.array(edge_indices).T if edge_indices else np.empty((2, 0), dtype=np.int64),
        'edge_features': np.array(edge_features, dtype=np.float32)
    }
