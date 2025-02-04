"""
Enhanced CoM Bidirectional Mamba Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, lexsort, degree
from torch_scatter import scatter_add
from mamba_ssm import Mamba

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

class EnhancedMambaOnly(nn.Module):
    def __init__(self, d_model, feature_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        
        # GMB layer
        self.gmb = EnhancedGMBLayer(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
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
        # Process through GMB path
        gmb_out = self.gmb(x, edge_index, edge_attr, batch)
        gmb_out = self.feature_norm(gmb_out)

        # Feature enhancement and residual connection
        out = self.feature_enhance(gmb_out) + x
        out = self.output_norm(out)

        return out

class MambaOnly(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, gmb_args, num_layers):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_channels)
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        self.blocks = nn.ModuleList([
            EnhancedMambaOnly(
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