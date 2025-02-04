"""Graph_Transformer.py - Graph Transformer implementation aligned with existing models"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_add
from cognn.models.layers import WeightedGCNConv

class CoGNNLayer(nn.Module):
    """CoGNN layer from existing implementation"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = WeightedGCNConv(in_channels, out_channels)
        self.edge_feature_dim = 16
        
        self.edge_proj = nn.Sequential(
            nn.Linear(self.edge_feature_dim, out_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.edge_gate = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None:
            with torch.no_grad():
                edge_weight = torch.sum(edge_attr * edge_attr, dim=-1).sqrt_()
        else:
            edge_weight = None

        x_conv = self.conv(x, edge_index, edge_weight=edge_weight)

        if edge_attr is not None:
            edge_features = self.edge_proj(edge_attr)
            edge_features = scatter_add(edge_features, edge_index[0], dim=0, dim_size=x.size(0))

            combined = torch.cat([x_conv, edge_features], dim=-1)
            gate = self.edge_gate(combined)
            x = gate * x_conv + (1 - gate) * edge_features
        else:
            x = x_conv

        x = self.norm(x)
        x = self.dropout(x)
        return self.act(x)

class LocalTransformerLayer(nn.Module):
    """Local GNN + transformer layer"""
    def __init__(self, in_channels, hidden_channels, num_heads=8, dropout=0.1, layer_idx=0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.layer_idx = layer_idx

        # Local GNN
        self.cognn = CoGNNLayer(in_channels, hidden_channels)

        # Global transformer
        if layer_idx > 0:  # Only use transformer after first layer
            self.transformer = nn.MultiheadAttention(
                hidden_channels, 
                num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_norm = nn.LayerNorm(hidden_channels)
        else:
            self.transformer = None

        # Residual projection if needed
        self.residual_proj = (nn.Linear(in_channels, hidden_channels) 
                            if in_channels != hidden_channels else None)

        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels)
        )

        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        # Save input for residual
        identity = x if self.residual_proj is None else self.residual_proj(x)

        # Local GNN processing
        x_local = self.cognn(x, edge_index, edge_attr)
        x = x_local + identity
        
        # Global transformer processing (if not first layer)
        if self.transformer is not None:
            x_dense, mask = to_dense_batch(x, batch)
            x_global = self.transformer(
                x_dense, x_dense, x_dense,
                key_padding_mask=~mask,
                need_weights=False
            )[0]
            x_global = x_global[mask]
            x = x + self.dropout(x_global)
            x = self.transformer_norm(x)

        # Final processing
        x = x + self.dropout(self.mlp(x))
        x = self.final_norm(x)
        
        return x

class GraphTransformer(nn.Module):
    """Graph Transformer model for mutation effect prediction"""
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=3,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        
        # Input normalization
        self.input_norm = nn.LayerNorm(in_channels)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            LocalTransformerLayer(
                in_channels if i == 0 else hidden_channels,
                hidden_channels,
                num_heads=num_heads,
                dropout=dropout,
                layer_idx=i
            ) for i in range(num_layers)
        ])
        
        # Output projection
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def process_graph(self, x, edge_index, edge_attr, batch):
        """Process a single graph"""
        # Input normalization
        x = self.input_norm(x)
        
        # Process through transformer layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, batch)
            
        # Graph-level pooling
        return scatter_add(x, batch, dim=0)

    def forward(self, wild_data, mutant_data):
        """Forward pass for mutation effect prediction"""
        # Process wild type and mutant graphs
        wild_out = self.process_graph(
            wild_data.x,
            wild_data.edge_index,
            wild_data.edge_attr,
            wild_data.batch
        )
        
        mutant_out = self.process_graph(
            mutant_data.x,
            mutant_data.edge_index,
            mutant_data.edge_attr,
            mutant_data.batch
        )

        # Calculate difference and predict effect
        diff = mutant_out - wild_out
        output = self.output_layer(diff).squeeze(-1)
        
        return output