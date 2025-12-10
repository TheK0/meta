import torch
import torch.nn as nn
from .gnn_encoder import GNNEncoder
from .mpg_encoder import MPGEncoder

class GSMetaCore(nn.Module):
    """
    Core GS-Meta model.
    Combines GNN encoder for molecules and MPG encoder for task context.
    """
    def __init__(self, config):
        super(GSMetaCore, self).__init__()
        self.config = config
        
        # Molecule Encoder
        self.mol_encoder = GNNEncoder(
            input_dim=config.get('mol_input_dim', 6),
            hidden_dim=config.get('hidden_dim', 128),
            output_dim=config.get('embedding_dim', 128),
            gnn_type=config.get('gnn_type', 'gin')
        )
        
        # MPG Encoder (Context)
        # Metadata (node types, edge types) should be passed from data loader
        self.mpg_encoder = None # Initialized later or passed in
        
        # Prediction Head
        self.classifier = nn.Linear(config.get('embedding_dim', 128), 1)

    def set_mpg_metadata(self, metadata):
        """Initialize MPG encoder with graph metadata."""
        self.mpg_encoder = MPGEncoder(
            mol_dim=self.config.get('embedding_dim', 128), # Input to MPG is mol embedding
            prop_dim=self.config.get('prop_dim', 64),
            hidden_dim=self.config.get('hidden_dim', 128),
            output_dim=self.config.get('embedding_dim', 128),
            metadata=metadata
        )

    def forward(self, support_mols, query_mols, mpg_data=None, params=None):
        """
        Meta-learning forward pass.
        
        Args:
            support_mols: Batch of support molecules (PyG Batch).
            query_mols: Batch of query molecules (PyG Batch).
            mpg_data: MPG graph data for the episode (Dict).
            params: Optional dict of parameters for functional call (fast weights).
        """
        # 1. Encode molecules
        # Output: [batch_size, embedding_dim]
        # GNN encoder is typically shared and frozen during inner loop, so we use self.mol_encoder
        support_emb = self.mol_encoder(support_mols.x, support_mols.edge_index, support_mols.batch)
        query_emb = self.mol_encoder(query_mols.x, query_mols.edge_index, query_mols.batch)
        
        # 2. Encode MPG context (if applicable)
        context_emb = None
        if self.mpg_encoder is not None and mpg_data is not None:
            # Construct MPG node features from GNN embeddings
            mol_map = mpg_data['mol_map']
            num_mpg_mols = len(mol_map)
            mol_emb_dim = support_emb.size(1)
            device = support_emb.device
            
            # Initialize with zeros (or could use a learnable embedding if not present)
            mpg_mol_feats = torch.zeros(num_mpg_mols, mol_emb_dim, device=device)
            
            # Fill support embeddings
            # We assume support_mols.mids is attached (list of original molecule IDs)
            if hasattr(support_mols, 'mids'):
                for i, mid in enumerate(support_mols.mids):
                    if mid in mol_map:
                        mpg_mol_feats[mol_map[mid]] = support_emb[i]
            
            # Fill query embeddings
            if hasattr(query_mols, 'mids'):
                for i, mid in enumerate(query_mols.mids):
                    if mid in mol_map:
                        mpg_mol_feats[mol_map[mid]] = query_emb[i]
            
            # Update x_dict for MPG
            # We clone to avoid modifying the original dict in place if reused
            x_dict = mpg_data['x_dict'].copy()
            x_dict['mol'] = mpg_mol_feats
            # Ensure prop features are on device
            if 'prop' in x_dict:
                x_dict['prop'] = x_dict['prop'].to(device)
                
            edge_index_dict = {k: v.to(device) for k, v in mpg_data['edge_index_dict'].items()}
            
            # Run MPG Encoder
            out_dict = self.mpg_encoder(x_dict, edge_index_dict)
            
            # Get Target Property Embedding
            target_idx = mpg_data['target_prop_idx']
            context_emb = out_dict['prop'][target_idx].unsqueeze(0) # [1, dim]
            
        # 3. Combine and predict
        if context_emb is not None:
            # context_emb: [1, dim]
            # query_emb: [batch, dim]
            context_expanded = context_emb.expand(query_emb.size(0), -1)
            # Simple fusion: Additive (Residual)
            # Ensure dimensions match. If not, we might need a projection.
            # Assuming embedding_dim is consistent.
            final_emb = query_emb + context_expanded
        else:
            final_emb = query_emb
            
        # 4. Classifier (with Fast Weights support)
        if params is not None:
            from torch.nn.utils.stateless import functional_call
            logits = functional_call(self.classifier, params, (final_emb,))
        else:
            logits = self.classifier(final_emb)
        
        return logits
