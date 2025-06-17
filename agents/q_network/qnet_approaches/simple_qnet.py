from .base_qnet import BaseQNetwork
import torch

class SimpleQNetwork(BaseQNetwork):
    def __init__(self, model):
        super().__init__(model)
        embed_dim = self.model.embed_dim
        self.q_proj = torch.nn.Linear(embed_dim, 1)

    def compute_q_values(self, node_embeddings, raw_states, batch):
        q_values = self.q_proj(node_embeddings).squeeze(-1)  # shape: [num_nodes_in_batch]
        return q_values
