import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv

class MPGEncoder(nn.Module):
    """
    Encoder for Molecule-Property Graphs (Heterogeneous).
    """
    def __init__(self, mol_dim, prop_dim, hidden_dim, output_dim, metadata):
        super(MPGEncoder, self).__init__()
        
        # metadata = (node_types, edge_types)
        self.node_types, self.edge_types = metadata
        
        # Initial projection for different node types if dimensions differ
        self.lin_mol = nn.Linear(mol_dim, hidden_dim)
        self.lin_prop = nn.Linear(prop_dim, hidden_dim)
        
        # Heterogeneous Convolution
        # Using SAGEConv as a generic message passing layer
        self.conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_dim) 
            for edge_type in self.edge_types
        }, aggr='sum')
        
        self.conv2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_dim) 
            for edge_type in self.edge_types
        }, aggr='sum')
        
        self.lin_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_dict, edge_index_dict):
        """
        Args:
            x_dict: Dict of node features {node_type: features}
            edge_index_dict: Dict of edge indices {edge_type: indices}
        """
        # Project features to common hidden dim
        x_dict_h = {}
        if 'mol' in x_dict:
            x_dict_h['mol'] = self.lin_mol(x_dict['mol'])
        if 'prop' in x_dict:
            x_dict_h['prop'] = self.lin_prop(x_dict['prop'])
            
        # Message Passing
        h1 = self.conv1(x_dict_h, edge_index_dict)
        h1 = {key: x.relu() for key, x in h1.items()}
        
        h2 = self.conv2(h1, edge_index_dict)
        h2 = {key: x.relu() for key, x in h2.items()}
        
        # Output projection (e.g., for property nodes)
        out = {}
        for key, val in h2.items():
            out[key] = self.lin_out(val)
            
        return out
