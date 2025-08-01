import numpy as np
import itertools
import networkx as nx
import igraph as ig
from tqdm import tqdm

from ..base_env import BaseEnv


class FJOpinionDynamicsFinite(BaseEnv):
    """
    Environment for optimizing polarization reduction under the FJ opinion dynamics model.
    This environment generates all unique states (G, sigma, tau, l) up to isomorphism, which is
    necessary for dynamic programming approaches.
    For big n however, this becomes computationally expensive, which is why we use 'FJOpinionDynamics' for larger n.

    The state is represented as a tuple (G, sigma, tau, l) where:
    - G is the graph (network) of agents. (networkx.Graph)
    - sigma is the vector of opinions of agents. (list of -1, 1)
    - tau is the node that is currently being considered for rewiring (or None). (int or None)
    - l is the number of edges that we already added in this episode. (int)
    """

    def __init__(self, n, k=2, max_edges=None, generate_states=True):
        super().__init__()
        self.n = n
        self.k = k

        # maximum number of edges that we allow in a graph (before adding edges)
        # mainly used to reduce the number of states
        self.max_edges = (
            max_edges + self.k if max_edges is not None else n * (n - 1) // 2
        )
        if generate_states:
            self.states, self.hash_to_index, self.starting_states = (
                self._generate_states()
            )
            self.state_idxes = list(range(len(self.states)))
        self.actions = list(range(self.n))
        current_state = None

    def _generate_states(self):
        """
        Generates all unique states (G, sigma, tau, l) up to isomorphism and returns
        - states: a list of tuples (G, sigma, tau, l)
        - hash_to_index: a dictionary mapping (state_hash, l) to index in states
        - starting_states: a list of indices of starting states
        """
        all_edges = list(itertools.combinations(range(self.n), 2))

        all_sigmas = [
            sigma  # we only allow sigmas with at least two different opinions
            for sigma in itertools.product([-1, 1], repeat=self.n)
            if len(set(sigma)) > 1
        ]

        all_taus = list(range(self.n)) + [None]

        seen = set()
        states = []
        starting_states = []
        hash_to_index = {}
        state_counter = 0

        for r in tqdm(range(self.max_edges + 1), desc="Generating states"):

            for edge_subset in itertools.combinations(all_edges, r):

                G = nx.Graph()
                G.add_nodes_from(range(self.n))
                G.add_edges_from(edge_subset)

                for sigma in all_sigmas:
                    for tau in all_taus:

                        state_hash, G_perm, sigma_perm, tau_perm = self.state_hash(
                            G, sigma, tau, return_permuted_tuple=True
                        )  # we save the version of the state that is permuted according to the igraph canonical permutation

                        if self.state_hash(G_perm, sigma_perm, tau_perm) != state_hash:
                            raise ValueError("State hash mismatch after permutation.")

                        if state_hash not in seen:
                            seen.add(state_hash)
                            max_edges_left_to_add = min(self.max_edges - r, self.k)

                            for l in range(self.k - max_edges_left_to_add, self.k):
                                states.append((G_perm.copy(), sigma_perm, tau_perm, l))
                                hash_to_index[(state_hash, l)] = state_counter
                                state_counter += 1

                            if (
                                tau is None
                            ):  # the state with l=k is only added for tau=None
                                states.append((G_perm.copy(), sigma_perm, tau, self.k))
                                hash_to_index[(state_hash, self.k)] = state_counter
                                state_counter += 1

                                if (
                                    max_edges_left_to_add == self.k
                                ):  # the states with l=0 and tau=None are starting states
                                    starting_states.append(
                                        hash_to_index[(state_hash, 0)]
                                    )
        return states, hash_to_index, starting_states

    def state_hash(self, G, sigma, tau, action=None, return_permuted_tuple=False):
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

    def reset(self):
        """
        Resets the current state to a random starting state.
        """
        self.current_state = self.starting_states[
            np.random.randint(len(self.starting_states))
        ]
        return self.current_state

    def is_terminal(self, state=None):
        """
        Returns True if the state is terminal, False otherwise.
        """
        if state is None:
            state = self.current_state
        _, _, _, l = self.states[state]
        return l == self.k  # the state is terminal if we have added k edges

    def step(self, action, state=None):
        """
        Given the current state (or given state) and an action, performs a step in the environment.
        Returns the next state, reward, and whether the state is terminal.
        """
        if state is None:
            state = self.current_state
        G, sigma, tau, l = self.states[state]

        if l == self.k:
            raise ValueError("Cannot take step in terminal state")

        terminal = False

        if (
            tau is None
        ):  # in this case the action is the node we want to consider for rewiring
            new_state = self.hash_to_index[(self.state_hash(G, sigma, action), l)]
            self.current_state = new_state
            return (
                new_state,
                0,
                terminal,
            )  # the reward is 0 since we didnt chance an edge yet
        else:
            if l + 1 == self.k:
                terminal = True
            polarization_old = self.polarization(G, sigma)
            G_new = G.copy()
            u, v = tau, action
            if u == v:
                new_state = self.hash_to_index[(self.state_hash(G, sigma, None), l + 1)]
                self.current_state = new_state
                return new_state, 0, terminal
            elif G_new.has_edge(u, v):
                G_new.remove_edge(u, v)
            else:
                G_new.add_edge(u, v)
            polarization_new = self.polarization(G_new, sigma)
            new_state = self.hash_to_index[(self.state_hash(G_new, sigma, None), l + 1)]
            self.current_state = new_state
            return new_state, polarization_old - polarization_new, terminal

    def polarization(self, G, sigma):
        """
        Returns the polarization of a network.
        """
        nodelist = sorted(G.nodes())
        L = nx.laplacian_matrix(G, nodelist=nodelist).toarray()
        I = np.eye(L.shape[0])
        final_opinions = np.linalg.inv(I + L).dot(np.array(sigma))
        return np.linalg.norm(final_opinions)
