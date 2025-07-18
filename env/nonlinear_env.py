import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
import math
import copy


class NLOpinionDynamics:
    def __init__(self, n, **kwargs):
        self.n = n
        self.average_degree = kwargs.get("average_degree", 6)
        self.n_edge_updates_per_step = kwargs.get("n_edge_updates_per_step", 5)
        self.g = kwargs.get("g", 0.999)
        self.K = kwargs.get("K", 0.01)
        self.alpha = kwargs.get("alpha", 0.2)
        self.beta = kwargs.get("beta", 0.6)
        self.epsilon = kwargs.get("epsilon", 0.1)
        self.current_state = None
    
    def reset(self):
        G = nx.watts_strogatz_graph(self.n, k=self.average_degree, p=0.2)
        sigma = [np.random.uniform(-5, 5) for _ in range(self.n)] 
        self.beliefs = sigma
        while len(np.unique(sigma)) < 2:
            sigma = np.random.choice([-1, 1], size=self.n)
        polarization = self.polarization(sigma)

        self.current_state = {"graph": G, 
                              "sigma": sigma, 
                              "tau": None, 
                              "polarization": polarization, 
                              "graph_data": from_networkx(G)}
        return self.current_state.copy()
    
    # def is_terminal(self, state = None):
    #     """Returns True if the state is terminal, False otherwise."""
    #     if state is not None:
    #         self.current_state = state
    #     return state["edges_left"] == 0
    
    def step(self, action, state = None):
        """Returns the reward given action and state, aswell as resulting next state"""
        if state is not None:
            self.current_state = state

        self.current_state = self.current_state.copy()
        
        if self.current_state["tau"] is None:
            self.current_state["tau"] = action
            return self.current_state, 0, False
        else:
            u, v = self.current_state["tau"], action
            self.current_state["tau"] = None
            G_new = self.current_state["graph"].copy()
            if u != v:
                if G_new.has_edge(u, v):
                    G_new.remove_edge(u, v)
                else:
                    G_new.add_edge(u, v)
            polarization_old = self.current_state["polarization"]
            self.current_state["graph"], self.current_state["sigma"] = self.opinion_dynamics(G_new, self.current_state["sigma"])
            self.current_state["polarization"] = self.polarization(self.current_state["sigma"])
            self.current_state["graph_data"] = from_networkx(self.current_state["graph"])
            return self.current_state, polarization_old-self.current_state["polarization"], False
        
    def opinion_dynamics(self, G, sigma):
        chosen_nodes = np.random.choice(range(self.n), size=self.n_edge_updates_per_step, replace=False)
        for node in chosen_nodes:
            self.social_rewiring(G, sigma, node)
        sigma = self.opinion_updates(G, sigma)
        return G, sigma

    def opinion_updates(self, G, sigma):
        new_sigma = sigma.copy()
        opinion_range = max(sigma) - min(sigma)
        for node in G.nodes():
            neighbor_opinions = [sigma[j] for j in G.neighbors(node)] + [self.beliefs[node]]
            smoothed_opinions = [math.tanh(self.alpha * 
                                           self.influence_on_each_other(sigma[node], opinion, opinion_range)
                                             * opinion) for opinion in neighbor_opinions]
            if neighbor_opinions:
                new_sigma[node] = self.g*sigma[node]+ self.K * sum(smoothed_opinions) / len(neighbor_opinions)
        return new_sigma

    @staticmethod
    def influence_on_each_other(opinion, opinion_neighbor, opinion_range):
        # opposite_sign_ratio = sum(np.sign(opinion1) != np.sign(opinion_other) for opinion_other in opinions) / len(opinions)
        # print(1 - abs(opinion1 - opinion2) / (opinion_range))
        return 1 - abs(opinion - opinion_neighbor) / opinion_range
    
    def social_rewiring(self, G, sigma, node):  
        neighbors = set(G.neighbors(node))
        if np.random.randint(1, 2 * self.average_degree-1) < len(neighbors):
            # Remove a neighbor with probability proportional to difference in opinion
            valid_neighbors = [n for n in neighbors if G.degree(n) >= 2]
            if not valid_neighbors:
                return  # no valid neighbors to remove
            removal_probs = np.array([self.removal_probability(sigma[node], sigma[j]) for j in valid_neighbors], dtype=np.float64)
            removal_probs /= removal_probs.sum()
            k = np.random.choice(valid_neighbors, p=removal_probs)
            G.remove_edge(node, k)
        else:
            # Pick a new node to connect to with probability proportional to similarity in opinion
            candidates = set(np.random.choice(range(self.n), size=min(100, self.n), replace=False)) - neighbors - {node}
            addition_probs = np.array([self.addition_probability(sigma[node], sigma[j]) for j in candidates], dtype=np.float64)
            addition_probs /= addition_probs.sum()
            new_neighbor = np.random.choice(list(candidates), p=addition_probs)
            G.add_edge(node, new_neighbor)
        

    def removal_probability(self, opinion, opinion_neighbor):
            return (abs(opinion - opinion_neighbor)*(1 - 2 * self.epsilon) + self.epsilon) ** self.beta
    
    def addition_probability(self, opinion, opinion_candidate):
            return (abs(opinion - opinion_candidate)*(1 - 2 * self.epsilon) + self.epsilon) ** (-self.beta)
    
    @staticmethod
    def polarization(sigma):
        """Returns the polarization of a network."""
        return np.linalg.norm(sigma) * 100
    
    def clone(self):
        return copy.deepcopy(self)
