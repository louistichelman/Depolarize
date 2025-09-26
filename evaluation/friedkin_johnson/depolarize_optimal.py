"""
Exact Solver for FJ-OffDP (Exhaustive Search)
---------------------------------------------

This module implements an exhaustive search for the Offline Depolarization Problem (OffDP) 
under the Friedkin–Johnsen (FJ) model. It evaluates all possible sequences of up to k edge 
modifications (additions or deletions) and returns the configuration with minimum polarization.

Notes:
- Runtime grows exponentially in k and |V|; practical only for very small graphs.
- Used in the thesis to provide optimal benchmarks for evaluating heuristics 
  and the greedy algorithm on toy-sized instances.
"""

from collections.abc import Iterable
import networkx as nx

def depolarize_optimal(G: nx.Graph, sigma: Iterable, k: int, polarization_function: callable) -> tuple[nx.Graph, float]:
    """
    Finds the optimal graph configuration for polarization minimization under
    given polarization_function after up to k edge modifications using exhaustive search.

    Parameters
    ----------
    G : networkx.Graph
        Input undirected, unweighted graph.
    sigma : np.ndarray
        Initial opinion vector.
    k : int
        Number of edge modifications allowed (budget).
    polarization_function : callable
        Function that computes polarization for (G, sigma).

    Returns
    ----------
    best_G : networkx.Graph
        Graph configuration achieving the lowest polarization.
    min_polarization : float
        Corresponding polarization value.

    """
    min_polarization = float("inf")
    best_G = None

    def backtrack(G, remaining_k):
        nonlocal min_polarization, best_G
        if remaining_k == 0:
            polarization = polarization_function(G=G, sigma=sigma)
            if polarization < min_polarization:
                min_polarization = polarization
                best_G = G.copy()
            return

        nodes = list(G.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                if not G.has_edge(u, v):
                    G.add_edge(u, v)
                    backtrack(G, remaining_k - 1)
                    G.remove_edge(u, v)  # Restore the original graph
                if G.has_edge(u, v):
                    G.remove_edge(u, v)
                    backtrack(G, remaining_k - 1)
                    G.add_edge(u, v)  # Restore the original graph

    backtrack(G, k)
    return best_G, min_polarization
