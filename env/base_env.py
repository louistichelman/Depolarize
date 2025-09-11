from abc import ABC, abstractmethod
import copy
import igraph as ig
import networkx as nx


class BaseEnv(ABC):
    """
    Base class for all environments used in this project.
    """

    def __init__(self):
        super().__init__()
        self.current_state = None

    @abstractmethod
    def step(self, action, state=None):
        """
        Given either the current_state (if state is None) or a provided state,
        performs a step in the environment with the given action.
        Returns: a tuple (next_state, reward, done).
        """
        raise NotImplementedError("Must be implemented by subclass.")

    @abstractmethod
    def reset(self):
        """
        Resets the current_state to a random starting state.
        Returns: current_state.
        """
        raise NotImplementedError("Must be implemented by subclass.")

    def is_terminal(self, state=None):
        """
        Returns True if the current_state (or given state) is terminal, False otherwise.
        """
        return False

    def clone(self):
        """
        Returns a deep copy of the environment.
        This is useful for creating several independent environments for better training.
        """
        return copy.deepcopy(self)

    @staticmethod
    def state_hash(G, sigma, tau, action=None, return_permuted_tuple=False):
        """
        Returns a canonical representation of (G, sigma, tau) using igraph canonical_permutation.
        """
        G_ig = ig.Graph()
        G_ig.add_vertices(sorted(G.nodes()))
        G_ig.add_edges(list(G.edges()))
        G_ig.vs["color"] = [op + 1 if i == tau else op for i, op in enumerate(sigma)]

        perm = G_ig.canonical_permutation(color="color")
        G_perm = G_ig.permute_vertices(perm)
        sigma_perm = [sigma[perm.index(i)] for i in range(len(sigma))]
        tau_perm = None if tau is None else perm[tau]

        adj = tuple(map(tuple, G_perm.get_adjacency().data))

        G_perm_nx = nx.Graph()
        G_perm_nx.add_nodes_from(G_perm.vs["name"])
        G_perm_nx.add_edges_from(G_perm.get_edgelist())
        G_perm = G_perm_nx

        if action is not None:
            action_perm = perm[action]
            return (adj, tuple(sigma_perm), tau_perm), action_perm

        if return_permuted_tuple:
            return (adj, tuple(sigma_perm), tau_perm), G_perm, sigma_perm, tau_perm

        return (adj, tuple(sigma_perm), tau_perm)
