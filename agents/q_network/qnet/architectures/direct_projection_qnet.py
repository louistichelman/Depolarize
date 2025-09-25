from ..base_qnet import BaseQNetwork, BaseGNN
import torch
import torch.nn.functional as F


class DP_QNetwork(BaseQNetwork):
    """
    Q-network that directly projects node embeddings to Q-values.
    Arguments:
    - model: the GNN backbone (instance of subclass of BaseGNN)
    """
    def __init__(self, model: BaseGNN):
        super().__init__(model)
        self.device = self.model.device
        embed_dim = self.model.embed_dim
        self.linear = torch.nn.Linear(embed_dim, embed_dim)
        self.q_proj = torch.nn.Linear(embed_dim, 1)
        self.to(self.device)

    def compute_q_values(self, node_embeddings: torch.Tensor, raw_states: list[dict], batch: torch.Tensor):
        x = F.relu(self.linear(node_embeddings))
        q_values = self.q_proj(x).squeeze(-1)  # shape: [num_nodes_in_batch]
        return q_values
