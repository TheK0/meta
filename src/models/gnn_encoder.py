import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool

class GNNEncoder(nn.Module):
    """
    GNN Backbone for molecule encoding.
    Supports GCN, GIN, etc.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, gnn_type='gin'):
        super(GNNEncoder, self).__init__()
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            
            if gnn_type == 'gin':
                mlp = nn.Sequential(
                    nn.Linear(in_channels, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.convs.append(GINConv(mlp))
            else: # Default to GCN
                self.convs.append(GCNConv(in_channels, hidden_dim))
            
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes]
        """
        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.batch_norms[i](h)
            if i < self.num_layers - 1:
                h = F.relu(h)
        
        # Global pooling
        h_graph = global_mean_pool(h, batch)
        
        # Final projection
        out = self.lin(h_graph)
        return out
