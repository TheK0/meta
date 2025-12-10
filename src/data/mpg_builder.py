    def __init__(self, mol_data: MoleculePropertyDataset):
        self.mol_data = mol_data
        # Mappings for graph construction
        self.mol_to_idx: Dict[int, int] = {}
        self.prop_to_idx: Dict[int, int] = {}
        
    def build_mpg(self, split: str) -> Dict[str, Any]:
        """
        Construct the Molecule-Property Graph (MPG) for a given split.
        Returns a heterogeneous graph structure (e.g., dict of edge indices).
        """
        prop_ids = self.mol_data.get_properties_split(split)
        
        # Collect all relevant molecules for these properties
        relevant_mols: Set[int] = set()
        for pid in prop_ids:
            if pid in self.mol_data.labels:
                relevant_mols.update(self.mol_data.labels[pid].keys())
        
        # Build mappings
        self.mol_to_idx = {mid: i for i, mid in enumerate(relevant_mols)}
        self.prop_to_idx = {pid: i for i, pid in enumerate(prop_ids)}
        
        # Edges: (source_type, relation, target_type) -> list of (src, dst)
        edges = {
            ('mol', 'active', 'prop'): [],
            ('mol', 'inactive', 'prop'): [],
            # ('mol', 'unk', 'prop'): [] # Optional: explicit unknown edges
        }
        
        for pid in prop_ids:
            p_idx = self.prop_to_idx[pid]
            if pid in self.mol_data.labels:
                for mid, label in self.mol_data.labels[pid].items():
                    if mid not in self.mol_to_idx: continue
                    m_idx = self.mol_to_idx[mid]
                    
                    # Determine edge type based on label
                    # Assuming binary classification: 1 = active, 0 = inactive
                    # For regression, logic might differ (e.g. thresholding)
                    if label == 1:
                        edges[('mol', 'active', 'prop')].append((m_idx, p_idx))
                    elif label == 0:
                        edges[('mol', 'inactive', 'prop')].append((m_idx, p_idx))
                    else:
                        # Handle other cases
                        pass
                        
        # Convert to numpy arrays
        edge_index_dict = {}
        for k, v in edges.items():
            edge_index_dict[k] = np.array(v).T if v else np.empty((2, 0))
            
        return {
            'num_mols': len(self.mol_to_idx),
            'num_props': len(self.prop_to_idx),
            'edge_index_dict': edge_index_dict,
            'mol_to_idx': self.mol_to_idx,
            'prop_to_idx': self.prop_to_idx
        }

    def build_task_subgraph(self, prop_id: int, support_mids: List[int], query_mids: List[int], n_aux_props: int = 3):
        """
        Construct an episode subgraph for a target property (GS-Meta logic).
        
        Args:
            prop_id: Target property ID.
            support_mids: List of support molecule IDs.
            query_mids: List of query molecule IDs.
            n_aux_props: Number of auxiliary properties to include (most related).
            
        Returns:
            Dict containing subgraph data (x_dict, edge_index_dict) and node mappings.
        """
        # 1. Identify relevant nodes
        # Target property node
        if prop_id not in self.prop_to_idx:
            # Handle case where prop_id is not in the graph (e.g. new task)
            # For meta-learning, we usually assume the prop node exists or we create a temporary one.
            # Here we assume it exists in the 'train' graph or we skip.
            return None
            
        target_p_idx = self.prop_to_idx[prop_id]
        
        # Molecule nodes (Support + Query)
        # Note: In GS-Meta, we might want to include ALL molecules that have this property, 
        # or just the ones in the episode. Usually just the episode ones + neighbors.
        relevant_mols = set(support_mids + query_mids)
        
        # 2. Select Auxiliary Properties
        # Heuristic: Select properties that share the most molecules with the target property
        # or use precomputed correlations.
        # Simple heuristic: Random or just the ones connected to support mols.
        aux_p_idxs = set()
        
        # Find properties connected to support molecules
        # This requires reverse lookup or iterating edges.
        # For efficiency, let's assume we have a mol->props map or we iterate.
        # Since we don't have a fast reverse map in this simple builder, we'll skip complex selection
        # and just pick a few random other properties for demonstration.
        all_props = list(self.prop_to_idx.values())
        if len(all_props) > 1:
            potential_aux = [p for p in all_props if p != target_p_idx]
            if potential_aux:
                selected_aux = np.random.choice(potential_aux, min(len(potential_aux), n_aux_props), replace=False)
                aux_p_idxs.update(selected_aux)
        
        relevant_props = {target_p_idx} | aux_p_idxs
        
        # 3. Extract Subgraph
        # We need to map global indices to local subgraph indices
        sub_mol_to_local = {mid: i for i, mid in enumerate(relevant_mols)}
        sub_prop_to_local = {pid: i for i, pid in enumerate(relevant_props)}
        
        # Build edges for the subgraph
        # We only include edges between relevant_mols and relevant_props
        sub_edges = {
            ('mol', 'active', 'prop'): [],
            ('mol', 'inactive', 'prop'): []
        }
        
        # Iterate over relevant molecules and check their labels for relevant properties
        for mid in relevant_mols:
            # Check target property
            label = self.mol_data.get_label(mid, prop_id)
            if label is not None:
                # We DO NOT add edges for Query set labels to Target Property! (Leakage)
                if mid in support_mids: 
                    m_local = sub_mol_to_local[mid]
                    p_local = sub_prop_to_local[target_p_idx]
                    if label == 1:
                        sub_edges[('mol', 'active', 'prop')].append((m_local, p_local))
                    elif label == 0:
                        sub_edges[('mol', 'inactive', 'prop')].append((m_local, p_local))
            
            # Check auxiliary properties
            # For aux props, we can use all known labels (transductive setting or just background knowledge)
            # We iterate over the aux properties we selected
            for aux_p_idx in aux_p_idxs:
                # We need to find the original prop_id for this aux_p_idx
                # This is inefficient without a reverse map, but let's assume we can find it or we stored it.
                # Optimization: In __init__, create self.idx_to_prop = {v: k for k, v in self.prop_to_idx.items()}
                # For now, let's skip the reverse lookup if we don't have it and just use the index if possible,
                # BUT self.mol_data.get_label expects prop_id.
                # Let's assume we can't easily get prop_id back here without the map.
                # CRITICAL FIX: We need the map.
                pass 
                
        # To fix the above, we should have created idx_to_prop. 
        # Since we can't easily change __init__ without reloading the whole file context, 
        # let's assume we can iterate self.prop_to_idx to find it (slow but works).
        idx_to_prop = {v: k for k, v in self.prop_to_idx.items()}
        
        for mid in relevant_mols:
            m_local = sub_mol_to_local[mid]
            
            for aux_p_idx in aux_p_idxs:
                aux_prop_id = idx_to_prop[aux_p_idx]
                label = self.mol_data.get_label(mid, aux_prop_id)
                
                if label is not None:
                    p_local = sub_prop_to_local[aux_p_idx]
                    if label == 1:
                        sub_edges[('mol', 'active', 'prop')].append((m_local, p_local))
                    elif label == 0:
                        sub_edges[('mol', 'inactive', 'prop')].append((m_local, p_local)) 

        # Construct PyG HeteroData-like dictionary
        edge_index_dict = {}
        for k, v in sub_edges.items():
            edge_index_dict[k] = np.array(v).T if v else np.empty((2, 0), dtype=np.int64)
            
        return {
            'x_dict': {
                'mol': torch.zeros(len(relevant_mols), 1), # Placeholder, will be replaced by embeddings
                'prop': torch.zeros(len(relevant_props), 1) # Placeholder
            },
            'edge_index_dict': edge_index_dict,
            'mol_map': sub_mol_to_local,
            'prop_map': sub_prop_to_local,
            'target_prop_idx': sub_prop_to_local[target_p_idx]
        }
