import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx


class FJDepolarize:
    def __init__(self, n, k=2, max_edges = None, **kwargs):
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
        polarization, influence_matrix = self.polarization(G, sigma, return_influence_matrix=True)

        self.current_state = {"graph": G, 
                              "sigma": sigma, 
                              "tau": None, 
                              "edges_left": self.k, 
                              "polarization": polarization, 
                              "influence_matrix": influence_matrix,
                              "graph_data": from_networkx(G)}
        return self.current_state.copy()
    
    def is_terminal(self, state = None):
        """Returns True if the state is terminal, False otherwise."""
        if state is not None:
            self.current_state = state
        return state["edges_left"] == 0
    
    def step(self, action, state = None):
        """Returns the reward given action and state, aswell as resulting next state"""
        if state is not None:
            self.current_state = state

        if self.current_state["edges_left"] == 0:
            raise ValueError("Cannot take step in terminal state")
        
        terminal = False
        
        if self.current_state["tau"] is None:
            self.current_state["tau"] = action
            return self.current_state.copy(), 0, terminal
        else:
            self.current_state["edges_left"] -= 1
            if self.current_state["edges_left"] == 0:
                terminal = True
            u, v = self.current_state["tau"], action
            self.current_state["tau"] = None
            if u == v:
                return self.current_state.copy(), 0, terminal
            G_new = self.current_state["graph"].copy()
            if G_new.has_edge(u, v):
                G_new.remove_edge(u, v)
            else:
                G_new.add_edge(u, v)
            polarization_old = self.current_state["polarization"]
            self.current_state["polarization"], self.current_state["influence_matrix"] = self.polarization(G_new, self.current_state["sigma"], return_influence_matrix=True)
            self.current_state["graph"], self.current_state["graph_data"] = G_new, from_networkx(G_new)
            return self.current_state.copy(), polarization_old-self.current_state["polarization"], terminal
    
    def polarization(self, G, sigma, return_influence_matrix = False):
        """Returns the polarization of a network."""
        L = nx.laplacian_matrix(G).toarray()
        I = np.eye(L.shape[0])
        influence_matrix = np.linalg.inv(I + L)
        polarization = np.linalg.norm(influence_matrix.dot(np.array(sigma)))
        if return_influence_matrix:
            return polarization, influence_matrix
        return polarization