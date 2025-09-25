from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from ..gnn.base_gnn import BaseGNN

class BaseQNetwork(nn.Module, ABC):
    """
    Base class for Q-network architectures.
    Combines a GNN backbone with a method to compute Q-values from node embeddings.
    """
    def __init__(self, model: BaseGNN):
        super().__init__()
        self.model = model  # an instance of subclass of BaseModel

    def forward(self, raw_states: list[dict]):
        node_embeddings, batch = self.model(raw_states)  # [num_nodes_in_batch, embed_dim]
        return self.compute_q_values(node_embeddings, raw_states, batch)

    @abstractmethod
    def compute_q_values(self, node_embeddings: torch.Tensor, raw_states: list[dict], batch: torch.Tensor):
        pass

