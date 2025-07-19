import numpy as np
import networkx as nx

def greedy_fj_depolarize(G, sigma, k):
    """
    Input:
    G: networkx.Graph (unweighted, undirected)
    sigma: np.array of internal opinions (shape n,)
    k: number of steps
    
    Output:
    best_edges: list of chosen edges [(i,j), ...]
    G_final: networkx.Graph after modifications
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

        max_delta_pol= -np.inf
        best_edge = None

        for i in range(n):
            for j in range(i + 1, n):
                delta = 1 if A[i, j] == 0 else -1

                x = LD_inv[:, i] - LD_inv[:, j]
                viTx = x[i] - x[j]  # actually v^T (L+D)^{-1} v = x^T v = x_i - x_j
                z_diff = z[i] - z[j]
                denom = 1 + delta * viTx

                delta_pol = (z_diff * (2 * delta * z @ x) / denom -
                          (z_diff ** 2) * (delta ** 2 * x @ x) / denom ** 2)
                
                # print("Checking edge ({}, {}): delta_pol = {}".format(nodes[i], nodes[j], delta_pol))

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
