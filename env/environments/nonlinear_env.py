import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
import math

from ..base_env import BaseEnv


class NLOpinionDynamics(BaseEnv):
    """
    Nonlinear Opinion Dynamics Environment.
    This environment simulates opinion dynamics on a network using nonlinear opinion dynamics.
    The states are dictionaries with keys:

    - 'graph': the current graph (networkx.Graph)
    - 'sigma': the current opinions of the nodes (list of floats)
    - 'tau': the last action taken (node index, or None if no action has been taken yet)
    - 'polarization': the current polarization of the network (float)
    - 'graph_data': the graph data in PyTorch Geometric format (torch_geometric.utils.data.Data)

    The actions are node indices.
    """

    def __init__(self, n, **kwargs):
        super().__init__()

        # graph parameters
        self.n = n
        self.average_degree = kwargs.get("average_degree", 3 + n // 50)
        self.n_edge_updates_per_step = kwargs.get("n_edge_updates_per_step", n // 40)

        # social influence parameters
        self.lam = kwargs.get("lam", 0.999)
        self.kappa = kwargs.get("kappa", 0.01)

        # polarization parameters
        self.alpha = kwargs.get("alpha", 0.2)
        self.beta = kwargs.get("beta", 1)
        self.gamma = kwargs.get("gamma", 0.6)

        self.current_state = None

    def reset(self):
        """
        Resets the environment to a random state. We use a Watts-Strogatz graph
        and initial opinions are chosen uniformly from [-5, 5].
        Returns: current_state
        """
        G = nx.watts_strogatz_graph(self.n, k=self.average_degree, p=0.2)
        sigma = [np.random.uniform(-5, 5) for _ in range(self.n)]

        polarization = self.polarization(sigma)

        self.current_state = {
            "graph": G,
            "sigma": sigma,
            "tau": None,
            "polarization": polarization,
            "graph_data": from_networkx(G),
        }

        return self.current_state.copy()

    def step(self, action, state=None):
        """
        Given the current state (or given state) and an action, performs a step in the environment.
        Returns the next state, reward, and whether the state is terminal.
        """
        if state is not None:
            self.current_state = state

        self.current_state = self.current_state.copy()

        if self.current_state["tau"] is None:  # if tau is None, action is the new tau
            self.current_state["tau"] = action
            return self.current_state, 0, False
        else:
            u, v = (
                self.current_state["tau"],
                action,
            )  # if tau is a node, (action, tau) is the chosen edge

            self.current_state["tau"] = None
            G_new = self.current_state["graph"].copy()

            if u != v:
                if G_new.has_edge(u, v):
                    G_new.remove_edge(u, v)
                else:
                    G_new.add_edge(u, v)

            polarization_old = self.current_state["polarization"]
            self.current_state["graph"], self.current_state["sigma"] = (
                self.opinion_dynamics(G_new, self.current_state["sigma"])
            )
            self.current_state["polarization"] = self.polarization(
                self.current_state["sigma"]
            )
            self.current_state["graph_data"] = from_networkx(
                self.current_state["graph"]
            )
            return (
                self.current_state,
                polarization_old - self.current_state["polarization"],
                False,  # no states are terminal in this environment
            )

    def opinion_dynamics(self, G, sigma):
        """
        Performs one step of opinion dynamics according to the nonlinear model.
        Returns the updated graph and opinions.
        """
        chosen_nodes = np.random.choice(
            range(self.n), size=self.n_edge_updates_per_step, replace=False
        )
        for node in chosen_nodes:
            self.social_rewiring(G, sigma, node)

        sigma = self.opinion_updates(G, sigma)
        return G, sigma

    def opinion_updates(self, G, sigma):
        """
        Updates the opinions according to the nonlinear opinion dynamics model.
        Returns the updated opinions.
        """
        new_sigma = sigma.copy()
        opinion_range = max(sigma) - min(sigma)

        def influence_on_each_other(opinion, opinion_neighbor):
            return 1 - abs(opinion - opinion_neighbor) / (self.beta * opinion_range)

        for node in G.nodes():
            neighbor_opinions = [sigma[j] for j in G.neighbors(node)] + [
                self.beliefs[node]
            ]
            activations = [
                math.tanh(
                    self.alpha * influence_on_each_other(sigma[node], opinion) * opinion
                )
                for opinion in neighbor_opinions
            ]
            if neighbor_opinions:
                new_sigma[node] = self.lam * sigma[node] + self.kappa * sum(
                    activations
                ) / len(neighbor_opinions)

        return new_sigma

    def social_rewiring(self, G, sigma, node):
        """
        Performs social rewiring for a given node according to the nonlinear model.
        This involves either removing a neighbor or adding a new connection.
        """
        neighbors = set(G.neighbors(node))

        # add or remove neighbor with probability proportional to the current degree
        if np.random.randint(1, 2 * self.average_degree - 1) < len(neighbors):
            # Remove a neighbor with probability proportional to difference in opinion
            valid_neighbors = [n for n in neighbors if G.degree(n) >= 2]

            if not valid_neighbors:
                return  # no valid neighbors to remove

            def removal_probability(opinion, opinion_neighbor):
                return abs(opinion - opinion_neighbor) ** self.gamma

            removal_probs = np.array(
                [removal_probability(sigma[node], sigma[j]) for j in valid_neighbors],
                dtype=np.float64,
            )
            removal_probs /= removal_probs.sum()
            k = np.random.choice(valid_neighbors, p=removal_probs)
            G.remove_edge(node, k)

        else:
            # Pick a new node to connect to with probability proportional to similarity in opinion
            candidates = (
                set(
                    np.random.choice(
                        range(self.n), size=min(100, self.n), replace=False
                    )
                )
                - neighbors
                - {node}
            )

            def addition_probability(opinion, opinion_candidate):
                return abs(opinion - opinion_candidate) ** (-self.gamma)

            addition_probs = np.array(
                [addition_probability(sigma[node], sigma[j]) for j in candidates],
                dtype=np.float64,
            )

            addition_probs /= addition_probs.sum()
            new_neighbor = np.random.choice(list(candidates), p=addition_probs)
            G.add_edge(node, new_neighbor)

    @staticmethod
    def polarization(sigma):
        """
        Returns the polarization of the network (only depends on sigma).
        """
        return np.linalg.norm(sigma)
