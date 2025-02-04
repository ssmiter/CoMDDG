"""Modified cognn_layer.py to properly handle edge features"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor
from torch_scatter import scatter_add

from cognn.models.layers import WeightedGCNConv
from cognn.helpers.classes import GumbelArgs, ActionNetArgs
from cognn.models.fixed_action import ActionNet

class CoGNNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 gumbel_args: GumbelArgs = None, 
                 action_args: ActionNetArgs = None):
        super().__init__()
        
        # Core message passing
        self.conv = WeightedGCNConv(in_channels, out_channels)
        
        # Edge feature projection
        self.edge_feature_dim = 16  # Default edge feature dimension
        self.edge_proj = nn.Sequential(
            nn.Linear(self.edge_feature_dim, out_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Gumbel temperature parameters
        self.learn_temp = gumbel_args.learn_temp if gumbel_args is not None else False
        if self.learn_temp:
            self.temp = nn.Parameter(torch.tensor(gumbel_args.temp))
        else:
            self.register_buffer('temp', torch.tensor(gumbel_args.temp))
        
        # Action networks for in/out probabilities
        if action_args is not None:
            self.in_act_net = ActionNet(action_args=action_args)
            self.out_act_net = ActionNet(action_args=action_args)
        else:
            self.in_act_net = None
            self.out_act_net = None
            
        # Normalization and regularization
        self.layer_norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.ReLU()

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None) -> Tensor:
        """
        Forward pass with edge feature handling
        """
        # Process edge features if available
        if edge_attr is not None:
            edge_features = self.edge_proj(edge_attr)
        else:
            edge_features = None
            
        # Base feature processing
        x = self.layer_norm(x)
        
        # Get action probabilities if action networks are available
        if self.in_act_net is not None and self.out_act_net is not None:
            in_logits = self.in_act_net(x, edge_index)
            out_logits = self.out_act_net(x, edge_index)
            
            in_probs = F.gumbel_softmax(in_logits, tau=self.temp, hard=True)
            out_probs = F.gumbel_softmax(out_logits, tau=self.temp, hard=True)
            
            # Create edge weights
            edge_weight = self.create_edge_weight(
                edge_index=edge_index,
                keep_in_prob=in_probs[:, 0],
                keep_out_prob=out_probs[:, 0]
            )
        else:
            edge_weight = None
            
        # Apply conv with edge weights
        out = self.conv(x, edge_index, edge_weight=edge_weight)
        
        # Combine with edge features if available
        if edge_features is not None:
            # Aggregate edge features to nodes
            src, dst = edge_index
            edge_feat_aggr = scatter_add(edge_features, dst, dim=0, dim_size=x.size(0))
            out = out + edge_feat_aggr
            
        out = self.dropout(out)
        out = self.act(out)
        
        return out

    def create_edge_weight(self, edge_index: Adj, keep_in_prob: Tensor, keep_out_prob: Tensor) -> Tensor:
        """Create edge weights using in/out probabilities"""
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        return edge_in_prob * edge_out_prob