"""
Enhanced CoGNN-only Model
"""
import torch
import torch.nn as nn
from torch_scatter import scatter_add
from cognn.helpers.classes import GumbelArgs, ActionNetArgs, ActivationType
from cognn.helpers.model import ModelType
from cognn.models.cognn_layer import CoGNNLayer


def gin_mlp_func(in_channels, out_channels, bias):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels, bias=bias),
        nn.ReLU(),
        nn.Dropout(0.1)
    )


class IntegratedCoGNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Initialize GumbelArgs with all required parameters
        gumbel_args = GumbelArgs(
            learn_temp=True,
            temp_model_type=ModelType.GCN,
            tau0=0.1,
            temp=1.0,
            gin_mlp_func=gin_mlp_func
        )

        # Initialize ActionNetArgs with all required parameters
        action_args = ActionNetArgs(
            model_type=ModelType.GCN,
            num_layers=2,
            hidden_dim=out_channels,
            dropout=0.1,
            act_type=ActivationType.RELU,
            env_dim=in_channels,
            gin_mlp_func=gin_mlp_func
        )

        # CoGNN layer from cognn package
        self.cognn = CoGNNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            gumbel_args=gumbel_args,
            action_args=action_args
        )

        # Additional processing layers
        self.post_process = nn.Sequential(
            nn.LayerNorm(out_channels),
            nn.Dropout(0.1)
        )

    def forward(self, x, edge_index, edge_attr):
        out = self.cognn(x, edge_index, edge_attr)
        return self.post_process(out)


class EnhancedCoGNNOnly(nn.Module):
    def __init__(self, d_model, feature_dim):
        super().__init__()

        # CoGNN layer
        self.cognn = IntegratedCoGNNLayer(
            in_channels=d_model,
            out_channels=d_model,
        )

        # Feature enhancement components
        self.feature_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)

        self.feature_enhance = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # Process through CoGNN path
        cognn_out = self.cognn(x, edge_index, edge_attr)
        cognn_out = self.feature_norm(cognn_out)

        # Feature enhancement and residual connection
        out = self.feature_enhance(cognn_out) + x
        out = self.output_norm(out)

        return out


class CoGNNOnly(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, gmb_args, num_layers):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_channels)
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.blocks = nn.ModuleList([
            EnhancedCoGNNOnly(
                feature_dim=in_channels if i == 0 else hidden_channels,
                d_model=hidden_channels
            ) for i in range(num_layers)
        ])

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, out_channels)
        )

    def process_subgraph(self, x, edge_index, edge_attr, batch):
        x = self.input_norm(x)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x, edge_index, edge_attr, batch)

        return scatter_add(x, batch, dim=0)

    def forward(self, wild_data, mutant_data):
        wild_out = self.process_subgraph(
            wild_data.x,
            wild_data.edge_index,
            wild_data.edge_attr,
            wild_data.batch
        )

        mutant_out = self.process_subgraph(
            mutant_data.x,
            mutant_data.edge_index,
            mutant_data.edge_attr,
            mutant_data.batch
        )

        diff = mutant_out - wild_out
        output = self.output_layer(diff).squeeze(-1)

        return output
