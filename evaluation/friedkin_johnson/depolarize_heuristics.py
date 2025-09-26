"""
Heuristic Algorithms for FJ-OffDP
---------------------------------

This module implements two heuristic strategies for the Offline Depolarization Problem (OffDP) 
under the Friedkin–Johnsen (FJ) model. Both heuristics 
were introduced by Rácz et al. (2021) and serve as baselines for evaluating depolarization methods.

Implemented heuristics:
- Disagreement Seeking (DS):
  Iteratively adds the edge with the largest opinion disagreement (z_i - z_j)^2 between its endpoints.

- Coordinate Descent (CD):
  Iteratively adds the edge that maximizes an approximation of the polarization reduction,
  based on a first-order update of the Laplacian.
"""


import numpy as np
import networkx as nx


def heuristic_ds_fj_depolarize(G: nx.Graph, sigma, k: int):
    """
    "Disagreement Seeking" heuristic for Friedkin-Johnson model.

    Parameters
    ----------
    G : networkx.Graph
        Input undirected, unweighted graph.
    sigma : np.ndarray or list
        Initial opinion vector (values in [-1, 1]).
    k : int
        Number of edge additions allowed (budget).

    Returns
    -------
    G_final : networkx.Graph
        Graph after adding k edges using the DS heuristic.
    """
    G = G.copy()
    n = G.number_of_nodes()
    nodes = list(G.nodes())

    I = np.eye(n)
    A = nx.to_numpy_array(G, nodelist=nodes)

    for _ in range(k):
        L = np.diag(A.sum(axis=1)) - A
        influence_matrix = np.linalg.inv(I + L)
        z = influence_matrix @ sigma

        max_delta_pol = -np.inf
        best_edge = None

        for i in range(n):
            for j in range(i + 1, n):
                if i == j or A[i, j] == 1:
                    continue

                disagreement = (z[i] - z[j]) ** 2

                if disagreement > max_delta_pol:
                    max_delta_pol = disagreement
                    best_edge = (i, j)

        i, j = best_edge
        A[i, j] = A[j, i] = 1

    # Rebuild final graph
    G_final = nx.from_numpy_array(A)

    return G_final


def heuristic_cd_fj_depolarize(G, sigma, k):
    """
    "Coordinate descent" heuristic for Friedkin-Johnson model.

    Parameters
    ----------
    G : networkx.Graph
        Input undirected, unweighted graph.
    sigma : np.ndarray or list
        Initial opinion vector (values in [-1, 1]).
    k : int
        Number of edge additions allowed (budget).

    Returns
    -------
    G_final : networkx.Graph
        Graph after adding k edges using the CD heuristic.
    """
    G = G.copy()
    n = G.number_of_nodes()
    nodes = list(G.nodes())

    I = np.eye(n)
    A = nx.to_numpy_array(G, nodelist=nodes)

    for _ in range(k):
        L = np.diag(A.sum(axis=1)) - A
        influence_matrix = np.linalg.inv(I + L)
        z = influence_matrix @ sigma
        z = z.reshape(-1, 1)

        max_delta_pol = -np.inf
        best_edge = None

        for i in range(n):
            for j in range(i + 1, n):
                if i == j or A[i, j] == 1:
                    continue
                v_ij = np.zeros(n)
                v_ij[i] = 1
                v_ij[j] = -1
                L_ij = np.outer(v_ij, -v_ij)
                del_ij = -2 * (z.T @ (influence_matrix @ (L_ij @ z)))
                del_ij = del_ij.item()

                if del_ij > max_delta_pol:
                    max_delta_pol = del_ij
                    best_edge = (i, j)

        i, j = best_edge
        A[i, j] = A[j, i] = 1

    # Rebuild final graph
    G_final = nx.from_numpy_array(A)

    return G_final
