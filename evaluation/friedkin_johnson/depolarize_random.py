import random
import networkx as nx


def depolarize_random_strategy(G: nx.Graph, k: int):

    G_random = G.copy()
    for _ in range(k):
        nodes = list(G_random.nodes())
        while True:
            u, v = random.choice(nodes), random.choice(nodes)
            if u != v and not G_random.has_edge(u, v):
                G_random.add_edge(u, v)
                break
    return G_random
