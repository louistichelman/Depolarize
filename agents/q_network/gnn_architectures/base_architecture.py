import torch
from abc import ABC, abstractmethod


class BaseArchitecture(torch.nn.Module, ABC):
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        self.device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.embed_dim = embed_dim
        self.num_heads = None

    def forward(self, raw_states):
        """
        raw_states: list of environment states (G, sigma, tau, l) 
        This is the interface expected by the DQN agent.
        """
        batch = self.prepare_batch(raw_states)
        return self.forward_batch(batch)

    @abstractmethod
    def prepare_batch(self, raw_states):
        """
        Converts a list of environment states into a batched format 
        suitable for the architecture. Should return a structure 
        (e.g., PyG Batch, tensor dict, etc.) ready for the forward pass.
        """
        raise NotImplementedError("Must be implemented by subclass.")

    @abstractmethod
    def forward_batch(self, batch):
        """
        The actual forward pass given a preprocessed batch.
        Returns: node embeddings.
        """
        raise NotImplementedError("Must be implemented by subclass.")
