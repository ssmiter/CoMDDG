"""
Enhanced CoM Bidirectional with CoGNN integration and hybrid sorting/non-sorting adaptive fusion
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


class UnsortedMambaLayer(nn.Module):
    """
    Non-sorting Mamba layer from v1
    """
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

    def forward(self, x, batch):
        # Convert to dense batch directly without sorting
        h_dense, mask = to_dense_batch(x, batch)

        # Process through forward Mamba
        forward_out = self.forward_mamba(h_dense)

        # Process through backward Mamba
        x_reverse = torch.flip(h_dense, dims=[1])
        backward_out = self.backward_mamba(x_reverse)
        backward_out = torch.flip(backward_out, dims=[1])

        # Combine outputs with learned gate
        combined = torch.cat([forward_out, backward_out], dim=-1)
        gate = self.gate(combined)
        y_prime = gate * forward_out + (1 - gate) * backward_out

        # Apply mask
        y_prime = y_prime * mask.unsqueeze(-1)

        # Flatten to match the input format
        out = y_prime[mask]

        return out


class DegreeSortedMambaLayer(nn.Module):
    """
    Degree-sorting Mamba layer from v2
    """
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

        self.temp = nn.Parameter(torch.tensor(0.5))
        self.min_temp = 0.1

    def forward(self, x, edge_index, batch):
        # Get node degrees
        deg = degree(edge_index[0], x.size(0), dtype=torch.float)
        num_graphs = batch.max().item() + 1

        # Initialize tensor for softened degrees
        deg_gumbel = torch.zeros_like(deg)

        # Apply gumbel_softmax separately for each graph
        for g in range(num_graphs):
            # Find nodes in the current graph
            graph_mask = (batch == g)
            graph_indices = torch.where(graph_mask)[0]

            # Get degrees for the current graph
            graph_deg = deg[graph_indices]

            # Apply gumbel_softmax to current graph's degrees
            tau = torch.clamp(self.temp, min=self.min_temp)
            if self.training:  # Use Gumbel-Softmax with randomness during training
                graph_deg_gumbel = F.gumbel_softmax(graph_deg.unsqueeze(-1), tau=tau, hard=False, dim=0).squeeze(-1)
            else:  # Use deterministic sorting during evaluation
                graph_deg_gumbel = graph_deg

            # Store results back
            deg_gumbel[graph_indices] = graph_deg_gumbel

        # Sort using lexsort: first by batch, then by gumbel-softened degrees
        h_ind_perm = lexsort([deg_gumbel, batch])

        # Convert nodes to dense batch
        h_dense, mask = to_dense_batch(x[h_ind_perm], batch[h_ind_perm])

        # Mamba processing
        forward_out = self.forward_mamba(h_dense)
        x_reverse = torch.flip(h_dense, dims=[1])
        backward_out = self.backward_mamba(x_reverse)
        backward_out = torch.flip(backward_out, dims=[1])

        combined = torch.cat([forward_out, backward_out], dim=-1)
        gate = self.gate(combined)
        y_prime = gate * forward_out + (1 - gate) * backward_out

        # Apply mask
        y_prime = y_prime * mask.unsqueeze(-1)

        # Restore original order
        h_ind_perm_reverse = torch.argsort(h_ind_perm)
        out = y_prime[mask][h_ind_perm_reverse]

        return out


def gin_mlp_func(in_channels, out_channels, bias):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels, bias=bias),
        nn.ReLU(),
        nn.Dropout(0.1)
    )


class HybridEnhancedGMB(nn.Module):
    """
    Combined module with both sorted and unsorted pathways
    """
    def __init__(self, d_model, feature_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()

        # Unsorted Mamba pathway
        self.unsorted_mamba = UnsortedMambaLayer(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # Degree-sorted Mamba pathway
        self.sorted_mamba = DegreeSortedMambaLayer(
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

        # Initialize CoGNN layer
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

        # Fusion gate for unsorted and sorted outputs
        self.pathway_fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, 2),
            nn.Softmax(dim=-1)
        )

        # Fusion gate for mamba and cognn outputs
        self.mamba_cognn_fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, 2),
            nn.Softmax(dim=-1)
        )

        # Normalization layers
        self.feature_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)

        # Feature enhancement
        self.feature_enhance = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # Process through unsorted Mamba pathway
        unsorted_out = self.unsorted_mamba(x, batch)
        unsorted_out = self.feature_norm(unsorted_out)

        # Process through sorted Mamba pathway
        sorted_out = self.sorted_mamba(x, edge_index, batch)
        sorted_out = self.feature_norm(sorted_out)

        # Fuse the two Mamba pathways
        mamba_combined = torch.cat([unsorted_out, sorted_out], dim=-1)
        mamba_fusion_weights = self.pathway_fusion_gate(mamba_combined)

        mamba_out = mamba_fusion_weights[:, 0].unsqueeze(-1) * unsorted_out + \
                    mamba_fusion_weights[:, 1].unsqueeze(-1) * sorted_out

        # Process through CoGNN pathway
        cognn_out = self.cognn(x, edge_index, edge_attr)
        cognn_out = self.feature_norm(cognn_out)

        # Apply channel attention
        mamba_attention = self.channel_attention(mamba_out)
        cognn_attention = self.channel_attention(cognn_out)

        mamba_out = mamba_out * mamba_attention
        cognn_out = cognn_out * cognn_attention

        # Fuse Mamba and CoGNN outputs
        combined = torch.cat([mamba_out, cognn_out], dim=-1)
        fusion_weights = self.mamba_cognn_fusion_gate(combined)

        out = fusion_weights[:, 0].unsqueeze(-1) * mamba_out + \
              fusion_weights[:, 1].unsqueeze(-1) * cognn_out

        # Feature enhancement and residual connection
        out = self.feature_enhance(out) + x
        out = self.output_norm(out)

        return out


class HybridCoGNN_GraphMamba(nn.Module):
    """
    Main model integrating hybrid components
    """
    def __init__(self, in_channels, hidden_channels, out_channels, gmb_args, num_layers):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_channels)
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.blocks = nn.ModuleList([
            HybridEnhancedGMB(
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
