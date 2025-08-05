import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..base_gnn import BaseGNN


class Graphormer(BaseGNN):
    """
    Custom Graph-Transformer implementation for the Q-network. The fundamental matrix (influence matrix) used in the
    FJ opinion dynamics is used to inject influence into the attention mechanism.
    Number of parameters: 2 * num_layers * embed_dim^2 + (4+num_layers) * embed_dim
    """

    def __init__(self, embed_dim=64, num_layers=3, num_heads=4, **kwargs):
        super().__init__(embed_dim, **kwargs)

        self.input_proj = nn.Linear(in_features=3, out_features=embed_dim, bias=True)
        self.layers = nn.ModuleList(
            [GraphormerLayer(embed_dim, num_heads) for _ in range(num_layers)]
        )

        self.num_heads = num_heads

        self.to(self.device)

    def prepare_batch(self, raw_states):
        xs = []  # node features
        influences = []  # influence matrices

        for state in raw_states:
            n_nodes = state["graph_data"].num_nodes

            # Build node features
            sigma_tensor = torch.tensor(
                state["sigma"], dtype=torch.float32, device=self.device
            ).view(-1, 1)
            tau_tensor = torch.zeros(
                (n_nodes, 1), dtype=torch.float32, device=self.device
            )
            if state["tau"] is not None:
                tau_tensor[state["tau"]] = 1.0
            l_tensor = torch.full(
                (n_nodes, 1),
                fill_value=state.get("edges_left", 0),
                dtype=torch.float32,
                device=self.device,
            )
            x = torch.cat([sigma_tensor, tau_tensor, l_tensor], dim=1)
            xs.append(x)

            # Influence matrix
            influence_matrix = torch.tensor(
                state["influence_matrix"], device=self.device, dtype=torch.float32
            )
            influences.append(influence_matrix)

        x_batch = torch.stack(xs)  # Shape: [B, N, 3]
        influence_batch = torch.stack(influences)  # Shape: [B, N, N]

        return {"x": x_batch, "influence_matrix": influence_batch}

    def forward_batch(self, batch):
        x = batch["x"]  # [B, N, 3]
        influence_matrix = batch["influence_matrix"]  # [B, N, N]
        B, N, _ = x.shape

        x = self.input_proj(x)  # [B, N, E]
        for layer in self.layers:
            x = layer(x, influence_matrix)  # [B, N, E]

        x_flat = x.view(B * N, -1)  # Flatten x to [B*N, E]

        batch_vector = torch.arange(B, device=x.device).repeat_interleave(N)

        return x_flat, batch_vector


class GraphormerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = GraphormerAttention(embed_dim, num_heads)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # self.dropout1 = nn.Dropout(0.1)
        # self.dropout2 = nn.Dropout(0.1)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, influence_matrix):
        # compute attention (with LN before attention)
        x = self.attn(self.norm1(x), influence_matrix) + x

        # compute feed-forward network (with LN before FFN)
        x = self.ffn(self.norm2(x)) + x
        return x


class GraphormerAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0

        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # scaling of influence matrix before softmax
        self.influence_weight_1 = nn.Parameter(torch.tensor(1.0))
        self.influence_bias_1 = nn.Parameter(torch.tensor(0.0))

        # scaling of influence matrix after softmax
        self.influence_weight_2 = nn.Parameter(torch.tensor(1.0))
        self.influence_bias_2 = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, influence_matrix):

        B, N, _ = x.size()

        # Project input to queries, keys, and values and reshape to [B, H, N, D/H]
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention, shape: [B, H, N, N]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Inject influence matrix (broadcasted over heads)
        influence_matrix_before_softmax = (
            self.influence_weight_1 * influence_matrix + self.influence_bias_1
        )
        attn_scores = attn_scores + influence_matrix_before_softmax.unsqueeze(1)

        # Softmax normalization
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, N, N]

        # Apply influence matrix after softmax
        influence_matrix_after_softmax = (
            self.influence_weight_2 * influence_matrix + self.influence_bias_2
        )
        attn_weights = attn_weights * influence_matrix_after_softmax.unsqueeze(1)

        # Weighted sum of values
        out = torch.matmul(attn_weights, v)  # [B, H, N, D/H]

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)  # [B, N, D]

        # Final linear projection
        return self.out_proj(out)
