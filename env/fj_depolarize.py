import numpy as np
import itertools
import networkx as nx
import igraph as ig
from tqdm import tqdm
from collections import defaultdict

class FJDepolarize:
    def __init__(self, n, k=2, max_edges = None):
        self.n = n
        self.k = k
        self.max_edges = max_edges# + self.k if max_edges is not None else n * (n - 1) // 2 
        self.current_state = None
    
    def reset(self):
        # G = nx.gnm_random_graph(self.n, np.random.randint(self.n / 2, self.max_edges))
        G = nx.watts_strogatz_graph(self.n, k=4, p=0.1)
        sigma = np.random.choice([-1, 1], size=self.n)
        while len(np.unique(sigma)) < 2:
            sigma = np.random.choice([-1, 1], size=self.n)

        self.current_state = (G, sigma, None, self.k)
        return self.current_state
    
    def is_terminal(self, state = None):
        """Returns True if the state is terminal, False otherwise."""
        if state is None:
            state = self.current_state
        _, _, _, l = state
        return l == 0
    
    def step(self, action, state = None):
        """Returns the reward given action and state, aswell as resulting next state"""
        if state is None:
            state = self.current_state
        G, sigma, tau, l = state
        if l == 0:
            raise ValueError("Cannot take step in terminal state")
        terminal = False
        if tau is None:
            self.current_state = (G, sigma, action, l)
            return self.current_state, 0, terminal
        else:
            if l-1 == 0:
                terminal = True
            u, v = tau, action
            if u == v:
                self.current_state = (G, sigma, None, l-1)
                return self.current_state, 0, terminal
            polarization_old = self.polarization(G, sigma)
            G_new = G.copy()
            if G_new.has_edge(u, v):
                G_new.remove_edge(u, v)
            else:
                G_new.add_edge(u, v)
            polarization_new = self.polarization(G_new, sigma)
            self.current_state = (G_new, sigma, None, l-1)
            return self.current_state, polarization_old-polarization_new, terminal
    
    def polarization(self, G, sigma):
        """Returns the polarization of a network."""
        L = nx.laplacian_matrix(G).toarray()
        I = np.eye(L.shape[0])
        final_opinions = np.linalg.inv(I + L).dot(np.array(sigma))
        return np.linalg.norm(final_opinions)