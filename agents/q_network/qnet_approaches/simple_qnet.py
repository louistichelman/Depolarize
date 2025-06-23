from .base_qnet import BaseQNetwork
import torch
import torch.nn.functional as F

class SimpleQNetwork(BaseQNetwork):
    def __init__(self, model):
        super().__init__(model)
        self.device = self.model.device
        embed_dim = self.model.embed_dim
        self.linear = torch.nn.Linear(embed_dim, embed_dim)
        self.q_proj = torch.nn.Linear(embed_dim, 1)
        self.to(self.device)

    def compute_q_values(self, node_embeddings, raw_states, batch):
        x = F.relu(self.linear(node_embeddings))
        q_values = self.q_proj(x).squeeze(-1)  # shape: [num_nodes_in_batch]
        return q_values
