from env import BaseEnv


def depolarize_optimal(G, sigma, k, polarization_function):
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
