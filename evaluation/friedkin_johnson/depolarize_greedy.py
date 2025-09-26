"""
Greedy FJ-Depolarize Algorithm
------------------------------

This module implements the exact greedy heuristic for the unweighted Offline Depolarization Problem (OffDP) 
under the Friedkin–Johnsen (FJ) model. We assume no weigts.

Algorithm:
- Iteratively selects up to k edges to add (if absent) or delete (if present),
  choosing the edge that maximally decreases network polarization at each step.
- Polarization change is computed exactly using Lemma 3.1 from the Thesis (Rácz et al. (2021)).

Purpose:
This implementation is used in the thesis to compare greedy performance against 
heuristics (DS, CD) and optimal solutions on small graphs, and to serve as a baseline 
for deep reinforcement learning experiments.
"""

import numpy as np
import networkx as nx


def greedy_fj_depolarize(G: nx.Graph, sigma, k: int) -> nx.Graph:
    """
    Greedy FJ-Depolarize algorithm for the unweighted Offline Depolarization Problem (OffDP)

    Parameters
    ----------
    G : networkx.Graph
        Input undirected, unweighted graph.
    sigma : np.ndarray or list
        Initial opinion vector (values in [-1, 1]).
    k : int
        Number of edge modifications (budget).
    
    Returns
    -------
    networkx.Graph
        Modified graph after k greedy interventions.
    """

    G = G.copy()
    n = G.number_of_nodes()
    nodes = list(G.nodes())

    # Internal weights w_ii = 1
    D = np.eye(n)

    # Initial Laplacian L
    A = nx.to_numpy_array(G, nodelist=nodes)
    L = np.diag(A.sum(axis=1)) - A

    for _ in range(k):
        LD_inv = np.linalg.inv(L + D)
        z = LD_inv @ sigma

        max_delta_pol = -np.inf
        best_edge = None

        for i in range(n):
            for j in range(i + 1, n):
                delta = 1 if A[i, j] == 0 else -1

                x = LD_inv[:, i] - LD_inv[:, j]
                viTx = x[i] - x[j]  # actually v^T (L+D)^{-1} v = x^T v = x_i - x_j
                z_diff = z[i] - z[j]
                denom = 1 + delta * viTx

                delta_pol = (
                    z_diff * (2 * delta * z @ x) / denom
                    - (z_diff**2) * (delta**2 * x @ x) / denom**2
                )

                if delta_pol > max_delta_pol:
                    max_delta_pol = delta_pol
                    best_edge = (i, j)

        i, j = best_edge
        if A[i, j] == 0:
            A[i, j] = A[j, i] = 1
        else:
            A[i, j] = A[j, i] = 0

        # update Laplacian
        L = np.diag(A.sum(axis=1)) - A

    # Rebuild final graph
    G_final = nx.from_numpy_array(A)

    return G_final
