"""Modified cognn_layer.py with fixed edge feature handling"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_scatter import scatter_add

from cognn.models.fixed_action import ActionNet
from cognn.models.layers import _WeightedGCNConv as WeightedGCNConv
from cognn.helpers.classes import GumbelArgs, ActionNetArgs

class CoGNNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 gumbel_args: GumbelArgs = None, 
                 action_args: ActionNetArgs = None):
        super().__init__()
        
        # Use original _WeightedGCNConv
        self.conv = WeightedGCNConv(in_channels, out_channels, bias=True)
        
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
        
        # Action networks
        if action_args is not None:
            self.in_act_net = ActionNet(action_args=action_args)
            self.out_act_net = ActionNet(action_args=action_args)
        else:
            self.in_act_net = None
            self.out_act_net = None
        
        # Edge combination
        self.edge_combine = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.Sigmoid()
        )
            
        # Normalization and regularization
        self.layer_norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.ReLU()

    def prepare_edge_attr(self, edge_index: Adj, edge_attr: OptTensor, num_nodes: int) -> Tuple[Adj, OptTensor, OptTensor]:
        """Helper function to properly handle edge attributes and self loops"""
        if edge_attr is not None:
            # First remove any existing self loops
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            
            # Calculate edge weights from edge attributes if needed
            edge_weight = torch.sum(edge_attr * edge_attr, dim=-1).sqrt()
            
            # Add self loops to edge_index
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            
            # Create new edge_attr tensor with self loops
            num_edges = edge_attr.size(0)
            num_self_loops = num_nodes
            
            # Create self loop edge attributes (initialize as ones)
            self_loop_attr = torch.ones(num_self_loops, self.edge_feature_dim, 
                                      device=edge_attr.device, dtype=edge_attr.dtype)
            
            # Concatenate original edge attributes with self loop attributes
            edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
            
            # Update edge weights with self loops
            self_loop_weights = torch.ones(num_self_loops, device=edge_weight.device)
            edge_weight = torch.cat([edge_weight, self_loop_weights])
            
        else:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            edge_weight = None
            
        return edge_index, edge_attr, edge_weight

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None) -> Tensor:
        """
        Forward pass with fixed edge feature handling
        """
        # Process edge features and handle self loops
        edge_index, edge_attr, edge_weight = self.prepare_edge_attr(
            edge_index, edge_attr, x.size(0)
        )
        
        # Project edge features if available
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
            
            # Combine edge weights
            action_weight = self.create_edge_weight(
                edge_index=edge_index,
                keep_in_prob=in_probs[:, 0],
                keep_out_prob=out_probs[:, 0]
            )
            if edge_weight is not None:
                edge_weight = edge_weight * action_weight
            else:
                edge_weight = action_weight
            
        # Apply conv with edge weights
        out = self.conv(x, edge_index, edge_attr=edge_attr, edge_weight=edge_weight)
        
        # Combine with edge features if available
        if edge_features is not None:
            # Aggregate edge features to nodes
            src, dst = edge_index
            edge_feat_aggr = scatter_add(edge_features, dst, dim=0, dim_size=x.size(0))
            
            # Adaptive combination
            combined = torch.cat([out, edge_feat_aggr], dim=-1)
            gate = self.edge_combine(combined)
            out = gate * out + (1 - gate) * edge_feat_aggr
            
        out = self.dropout(out)
        out = self.act(out)
        
        return out

    def create_edge_weight(self, edge_index: Adj, keep_in_prob: Tensor, keep_out_prob: Tensor) -> Tensor:
        """Create edge weights using in/out probabilities"""
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        return edge_in_prob * edge_out_prob