import torch
from abc import ABC, abstractmethod


class BaseGNN(torch.nn.Module, ABC):
    """
    Base class for the underlying GNN architecture used in the Q-network.
    """

    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        self.device = kwargs.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.embed_dim = embed_dim
        self.num_heads = None

    def forward(self, raw_states):
        """
        raw_states: list of environment states (dicts) with keys:
            - "graph": networkx graph object
            - "graph_data": PyG Data object containing graph structure
            - "sigma": opinions of nodes (list)
            - "tau": None or Int indicating which node was chosen before
            - "edges_left": optional integer indicating remaining edges to add
            - "influence_matrix": influence matrix (optional, used in some architectures)
        This is the interface expected by the DQN agent.
        """
        batch = self.prepare_batch(raw_states)
        return self.forward_batch(batch)

    @abstractmethod
    def prepare_batch(self, raw_states):
        """
        Converts a list of environment states into a batched format
        suitable for forward_batch.
        Returns: a PyG Batch object or a dictionary with batched data.
        """
        raise NotImplementedError("Must be implemented by subclass.")

    @abstractmethod
    def forward_batch(self, batch):
        """
        The actual forward pass given a preprocessed batch.
        Returns: node embeddings.
        """
        raise NotImplementedError("Must be implemented by subclass.")
