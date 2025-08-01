import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx

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
    - 'influence_matrix': the influence matrix of the network (numpy.ndarray)
    - 'graph_data': the graph data in PyTorch Geometric format (torch_geometric.utils.data.Data)

    The actions are node indices.
    """

    def __init__(self, n, **kwargs):
        super().__init__()

        self.n = n
        self.average_degree = kwargs.get("average_degree", 3 + n // 50)
        self.k = kwargs.get("k", n // 10)

        self.current_state = None

    def reset(self):
        """
        Resets the environment to a random state. We use a Watts-Strogatz graph
        and initial opinions are chosen uniformly from [-1, 1].
        Returns: current_state
        """
        G = nx.watts_strogatz_graph(self.n, k=self.average_degree, p=0.2)
        sigma = np.random.uniform(-1, 1, size=self.n)

        polarization, influence_matrix = self.polarization(
            G, sigma, return_influence_matrix=True
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

    def is_terminal(self, state=None):
        """
        Returns True if the state is terminal, False otherwise.
        """
        if state is not None:
            self.current_state = state
        return state["edges_left"] == 0

    def step(self, action, state=None):
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
                G_new, self.current_state["sigma"], return_influence_matrix=True
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
    def polarization(G, sigma, return_influence_matrix=False):
        """
        Returns the polarization of a network.
        """
        L = nx.laplacian_matrix(G).toarray()
        I = np.eye(L.shape[0])
        influence_matrix = np.linalg.inv(I + L)
        expressed_sigma = influence_matrix @ sigma
        polarization = expressed_sigma @ expressed_sigma
        if return_influence_matrix:
            return polarization, influence_matrix
        return polarization
