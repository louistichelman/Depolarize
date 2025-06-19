import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_architecture import BaseArchitecture
import networkx as nx
import numpy as np

class Graphormer(BaseArchitecture):
    def __init__(self, embed_dim=64, num_layers = 3, num_heads = 4, **kwargs):
        super().__init__(embed_dim, **kwargs)
        self.input_proj = nn.Linear(in_features=3, out_features=embed_dim)
        self.layers = nn.ModuleList([GraphormerLayer(embed_dim, num_heads) for _ in range(num_layers)])

    def prepare_batch(self, raw_states):
        xs = []
        influences = []

        for state in raw_states:
            G, sigma, tau, l, _ = state
            n_nodes = G.number_of_nodes()

            # Node features
            sigma_tensor = torch.tensor(sigma, dtype=torch.float32)
            tau_tensor = torch.zeros((n_nodes, 1))
            if tau is not None:
                tau_tensor[tau] = 1.0
            l_tensor = torch.full((n_nodes, 1), fill_value=l, dtype=torch.float32)
            x = torch.cat([sigma_tensor.view(-1, 1), tau_tensor, l_tensor], dim=1)
            xs.append(x)

            # Influence matrix
            nodelist = sorted(G.nodes())
            L = torch.tensor(nx.laplacian_matrix(G, nodelist=nodelist).toarray(), dtype=torch.float32)
            I = torch.eye(n_nodes, dtype=torch.float32)
            influence_matrix = torch.linalg.inv(I + L)
            influences.append(influence_matrix)

        x_batch = torch.stack(xs)                  # Shape: [B, N, 3]
        influence_batch = torch.stack(influences)  # Shape: [B, N, N]

        return {
            "x": x_batch.to(self.device),
            "influence_matrix": influence_batch.to(self.device)
        }

    def forward_batch(self, batch):
        x = batch["x"]                        # [B, N, 3]
        influence_matrix = batch["influence_matrix"]  # [B, N, N]
        B, N, _ = x.shape

        x = self.input_proj(x)  # [B, N, E]
        for layer in self.layers:
            x = layer(x, influence_matrix)  # [B, N, E]

        x_flat = x.view(B * N, -1) # Flatten x to [B*N, E]

        batch_vector = torch.arange(B, device=x.device).repeat_interleave(N)

        return x_flat, batch_vector

class GraphormerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = GraphormerAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, influence_matrix):
        x = x + self.attn(self.norm1(x), influence_matrix)
        x = x + self.ffn(self.norm2(x))
        return x

class GraphormerAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0

        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, influence_matrix):
        """
        Args:
            x: Tensor of shape [B, N, embed_dim]         - node features
            influence_matrix: Tensor of shape [B, N, N]  - influence matrix per graph in batch

        Returns:
            Tensor of shape [B, N, embed_dim] - updated node features after attention
        """
        B, N, _ = x.size()

        # Project input to queries, keys, and values
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, N, N]

        # Inject influence matrix (broadcasted over heads)
        influence_matrix = influence_matrix.unsqueeze(1)  # [B, 1, N, N]
        attn_scores = attn_scores * (1+influence_matrix)

        # Softmax normalization
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, N, N]

        # Optional: mask or rescale with influence again after softmax
        # attn_weights = attn_weights * influence_matrix  # (Optional, depends on design)

        # Weighted sum of values
        out = torch.matmul(attn_weights, v)  # [B, H, N, D]

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)  # [B, N, E]

        # Final linear projection
        return self.out_proj(out)




