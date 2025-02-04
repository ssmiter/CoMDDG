"""Fixed action.py with proper edge_attr handling"""
import torch.nn as nn
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor

from cognn.helpers.classes import ActionNetArgs


class ActionNet(nn.Module):
    def __init__(self, action_args: ActionNetArgs):
        """
        Create a model which represents the agent's policy with proper edge feature handling.
        """
        super().__init__()
        self.num_layers = action_args.num_layers
        self.net = action_args.load_net()
        self.dropout = nn.Dropout(action_args.dropout)
        self.act = action_args.act_type.get()

    def forward(self, x: Tensor, edge_index: Adj, env_edge_attr: OptTensor = None,
                act_edge_attr: OptTensor = None) -> Tensor:
        """Modified forward pass to properly handle edge attributes"""
        edge_attrs = [env_edge_attr] + (self.num_layers - 1) * [act_edge_attr]

        h = x
        for idx, (edge_attr, layer) in enumerate(zip(edge_attrs[:-1], self.net[:-1])):
            # Handle different layer types
            if hasattr(layer, 'forward_with_edge_attr'):
                # For layers that explicitly support edge attributes
                h = layer.forward_with_edge_attr(h, edge_index, edge_attr)
            else:
                # For layers that don't use edge attributes
                h = layer(h, edge_index)
            h = self.dropout(h)
            h = self.act(h)

        # Final layer
        if hasattr(self.net[-1], 'forward_with_edge_attr'):
            h = self.net[-1].forward_with_edge_attr(h, edge_index, edge_attrs[-1])
        else:
            h = self.net[-1](h, edge_index)

        return h
