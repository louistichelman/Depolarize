"""
FJOpinionDynamics Environment
-----------------------------

This file defines the FJOpinionDynamics environment, an implementation of the
Friedkin–Johnsen (FJ) opinion dynamics model as a Markov Decision Process (MDP).

Key features:
- Represents states as dictionaries containing graph structure, opinions,
  current polarization, and other metadata.
- Supports both randomly initialized states (Watts–Strogatz graphs + random opinions)
  and pre-saved start states loaded from disk.
- Provides reset, step, and terminal condition methods to interact with the environment.
- Computes polarization based on the FJ model, with optional resistance matrix support.

This environment is used for training and evaluating reinforcement learning
agents (e.g., DQN with GNNs) in the Offline Depolarization Problem (OffDP).
"""


import numpy as np
import networkx as nx
import torch
import random
from torch_geometric.utils import from_networkx
from scipy.sparse.csgraph import shortest_path


from ..base_env import BaseEnv


class FJOpinionDynamics(BaseEnv):
    """
    Environment for optimizing polarization reduction under the FJ opinion dynamics model.
    This environment does not generate all unique states

    The state is represented as a dictionary with keys:
    - 'graph': the current graph (networkx.Graph)
    - 'sigma': the current opinions of the nodes (list of floats)
    - 'tau': the node that is currently being considered for rewiring (or None) (int or None)
    - 'edges_left': the number of edges that can still be added (int)
    - 'polarization': the current polarization of the network (float)
    - 'influence_matrix': the fundamental matrix of the network (numpy.ndarray)
    - 'graph_data': the graph data in PyTorch Geometric format (torch_geometric.utils.data.Data)

    Parameters
    ----------
    n : int, optional
        Number of nodes in the graph. Required if start_states is not provided.
    start_states : str, optional
        Path to a file containing pre-saved start states. If provided, n is ignored.
    average_degree : int, optional
        Average degree of the generated Watts-Strogatz graph (default: 6).
    k : int, optional
        Number of edge modifications allowed (budget). Default is n // 10.
    keep_resistance_matrix : bool, optional
        If True, the resistance matrix is kept in the influence_matrix field, used for training Graphormer-GD.
        Default is False.
    **kwargs : additional keyword arguments

    """

    def __init__(self, n: int = None, start_states: str = None, **kwargs):
        super().__init__()

        if start_states is not None:
            self.n = self._load_start_states(start_states)
        elif n is not None:
            self.n = n
        else:
            raise ValueError("Either 'n' or 'start_states' must be provided.")

        self.average_degree = kwargs.get("average_degree", 6)
        self.k = kwargs.get("k", self.n // 10)

        self.keep_resistance_matrix = kwargs.get("keep_resistance_matrix", False)

        self.current_state = None

    def _load_start_states(self, file_path: str):
        """
        Load start states from a file.
        """
        self.start_states = torch.load(file_path, weights_only=False)
        n = len(self.start_states[0]["sigma"])
        return n

    def reset(self):
        """
        Resets the environment to a random state. We use a Watts-Strogatz graph
        and initial opinions are chosen uniformly from [-1, 1], if no start states are provided.
        Returns: current_state
        """
        if hasattr(self, "start_states"):
            # If start states are loaded, randomly select one
            self.current_state = random.choice(self.start_states)
            self.current_state["edges_left"] = self.k
            return self.current_state.copy()

        G = nx.watts_strogatz_graph(self.n, k=self.average_degree, p=0.1)
        sigma = np.random.uniform(-1, 1, size=self.n)

        polarization, influence_matrix = self.polarization(
            G,
            sigma,
            return_influence_matrix=True,
            keep_resistance_matrix=self.keep_resistance_matrix,
        )

        self.current_state = {
            "graph": G,
            "sigma": sigma,
            "tau": None,
            "edges_left": self.k,
            "polarization": polarization,
            "influence_matrix": influence_matrix,
            "graph_data": from_networkx(G),
        }

        return self.current_state.copy()

    def is_terminal(self, state: dict = None):
        """
        Returns True if the state is terminal, False otherwise.
        """
        if state is not None:
            self.current_state = state
        return state["edges_left"] == 0

    def step(self, action: int, state: dict = None):
        """
        Given the current state (or given state) and an action, performs a step in the environment.
        Returns the next state, reward, and whether the state is terminal.
        """
        if state is not None:
            self.current_state = state

        self.current_state = self.current_state.copy()
        if self.current_state["edges_left"] == 0:
            raise ValueError("Cannot take step in terminal state")

        terminal = False

        if self.current_state["tau"] is None:
            self.current_state["tau"] = action
            return self.current_state, 0, terminal
        else:
            self.current_state["edges_left"] -= 1
            if self.current_state["edges_left"] == 0:
                terminal = True
            u, v = self.current_state["tau"], action
            self.current_state["tau"] = None
            if u == v:
                return self.current_state, 0, terminal
            G_new = self.current_state["graph"].copy()
            if G_new.has_edge(u, v):
                G_new.remove_edge(u, v)
            else:
                G_new.add_edge(u, v)
            polarization_old = self.current_state["polarization"]
            (
                self.current_state["polarization"],
                self.current_state["influence_matrix"],
            ) = self.polarization(
                G_new,
                self.current_state["sigma"],
                return_influence_matrix=True,
                keep_resistance_matrix=self.keep_resistance_matrix,
            )
            self.current_state["graph"], self.current_state["graph_data"] = (
                G_new,
                from_networkx(G_new),
            )
            return (
                self.current_state,
                polarization_old - self.current_state["polarization"],
                terminal,
            )

    @staticmethod
    def polarization(
        G: nx.Graph, sigma: np.ndarray, return_influence_matrix: bool = False, keep_resistance_matrix: bool = False
    ):
        """
        Returns the polarization of a network.
        """
        L = nx.laplacian_matrix(G).toarray()
        I = np.eye(L.shape[0])
        influence_matrix = np.linalg.inv(I + L)
        expressed_sigma = influence_matrix @ sigma
        polarization = expressed_sigma @ expressed_sigma

        if return_influence_matrix:
            if keep_resistance_matrix:
                n = len(sigma)
                J = np.ones((n, n)) / n

                M = np.linalg.inv(L + J)

                diag = np.diag(M)
                R = diag[:, None] + diag[None, :] - 2 * M

                return polarization, R
            return polarization, influence_matrix
        return polarization
