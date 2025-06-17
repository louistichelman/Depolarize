from abc import ABC, abstractmethod
import torch.nn as nn

class BaseQNetwork(nn.Module, ABC):
    def __init__(self, model):
        super().__init__()
        self.model = model  # an instance of subclass of BaseModel

    def forward(self, raw_states):
        node_embeddings, batch = self.model(raw_states)  # [num_nodes_in_batch, embed_dim]
        return self.compute_q_values(node_embeddings, raw_states, batch)

    @abstractmethod
    def compute_q_values(self, node_embeddings, raw_states, batch):
        pass

