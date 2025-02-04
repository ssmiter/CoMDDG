"""
Enhanced CoM Bidirectional with CoGNN integration
PCC: S250:0.91
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, lexsort, degree
from torch_scatter import scatter_add
from mamba_ssm import Mamba
from cognn.helpers.classes import GumbelArgs, ActionNetArgs, ActivationType
from cognn.helpers.model import ModelType
from cognn.models.cognn_layer import CoGNNLayer
# from model.CoM_Bidirectional_sorted_2 import EnhancedGMBLayer

class EnhancedGMBLayer(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.forward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.backward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.temp = nn.Parameter(torch.tensor(1.0))
        self.min_temp = 0.1

    def forward(self, x, edge_index, edge_attr, batch):
        # Your existing GMB implementation
        deg = degree(edge_index[0], x.size(0), dtype=torch.float)
        deg_logits = deg.unsqueeze(-1)
        deg_soft = F.gumbel_softmax(deg_logits, tau=torch.clamp(self.temp, min=self.min_temp), hard=False)
        h_ind_perm = lexsort([deg_soft.squeeze(-1), batch])
        h_dense, mask = to_dense_batch(x[h_ind_perm], batch[h_ind_perm])

        forward_out = self.forward_mamba(h_dense)
        x_reverse = torch.flip(h_dense, dims=[1])
        backward_out = self.backward_mamba(x_reverse)
        backward_out = torch.flip(backward_out, dims=[1])

        combined = torch.cat([forward_out, backward_out], dim=-1)
        gate = self.gate(combined)
        y_prime = gate * forward_out + (1 - gate) * backward_out

        y_prime = y_prime * mask.unsqueeze(-1)
        h_ind_perm_reverse = torch.argsort(h_ind_perm)
        out = y_prime[mask][h_ind_perm_reverse]

        return out


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
            temp_model_type=ModelType.GCN,  # Using GCN as temp model
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

        # Define gin_mlp_func

    def forward(self, x, edge_index, edge_attr):
        out = self.cognn(x, edge_index, edge_attr)
        return self.post_process(out)


class EnhancedGMB(nn.Module):
    def __init__(self, d_model, feature_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.edge_feature_dim = 16

        # GMB layer
        self.gmb = EnhancedGMBLayer(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # Initialize GumbelArgs for CoGNN
        gumbel_args = GumbelArgs(
            learn_temp=True,
            temp_model_type=ModelType.GCN,
            tau0=0.1,
            temp=1.0,
            gin_mlp_func=lambda in_dim, out_dim, bias: nn.Sequential(
                nn.Linear(in_dim, out_dim, bias=bias),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        )

        # Initialize ActionNetArgs for CoGNN
        action_args = ActionNetArgs(
            model_type=ModelType.GCN,
            num_layers=2,
            hidden_dim=d_model,
            dropout=0.1,
            act_type=ActivationType.RELU,
            env_dim=d_model,
            gin_mlp_func=lambda in_dim, out_dim, bias: nn.Sequential(
                nn.Linear(in_dim, out_dim, bias=bias),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        )

        # Initialize CoGNN layer with proper arguments
        self.cognn = CoGNNLayer(
            in_channels=d_model,
            out_channels=d_model,
            gumbel_args=gumbel_args,
            action_args=action_args
        )

        # Feature fusion components
        self.channel_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid()
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, 2),
            nn.Softmax(dim=-1)
        )

        self.feature_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)

        self.feature_enhance = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # Process through GMB path
        gmb_out = self.gmb(x, edge_index, edge_attr, batch)
        gmb_out = self.feature_norm(gmb_out)

        # Process through CoGNN path with edge features
        cognn_out = self.cognn(x, edge_index, edge_attr)
        cognn_out = self.feature_norm(cognn_out)

        # Apply channel attention
        gmb_attention = self.channel_attention(gmb_out)
        cognn_attention = self.channel_attention(cognn_out)

        gmb_out = gmb_out * gmb_attention
        cognn_out = cognn_out * cognn_attention

        # Feature fusion
        combined = torch.cat([gmb_out, cognn_out], dim=-1)
        fusion_weights = self.fusion_gate(combined)

        out = fusion_weights[:, 0].unsqueeze(-1) * gmb_out + \
              fusion_weights[:, 1].unsqueeze(-1) * cognn_out

        # Feature enhancement and residual connection
        out = self.feature_enhance(out) + x
        out = self.output_norm(out)

        return out

# Rest of your model remains the same
class CoGNN_GraphMambaSorted(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, gmb_args, num_layers):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_channels)
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        self.blocks = nn.ModuleList([
            EnhancedGMB(
                feature_dim=in_channels if i == 0 else hidden_channels,
                **gmb_args
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

    # process_subgraph and forward methods remain the same
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