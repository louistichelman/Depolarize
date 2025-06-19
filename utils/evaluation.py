from tqdm import tqdm
import random

def depolarize_greedy(state, env):
    k = env.k
    if isinstance(state, int):
        G, sigma, _, _ = env.states[state]
    else:
        G, sigma = state["graph"], state["sigma"]
    G_greedy = G.copy()
    for _ in range(k):
        best_edge = optimal_edge_to_add_remove(G_greedy, sigma, env.polarization)
        if best_edge:
            u, v = best_edge
            if G_greedy.has_edge(u, v):
                G_greedy.remove_edge(u, v)
            else:
                G_greedy.add_edge(u, v, weight=1.0)
        else:
            break
    polarization = env.polarization(G = G_greedy, sigma = sigma)
    return G_greedy, polarization

def optimal_edge_to_add_remove(G, sigma, polarization_measure):
    min_polarization = float('inf')
    best_edge = None

    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=1.0)
                polarization = polarization_measure(G = G, sigma = sigma)
                if polarization < min_polarization:
                    min_polarization = polarization
                    best_edge = (u, v)
                G.remove_edge(u, v)  # Restore the original graph
            if G.has_edge(u, v):
                G.remove_edge(u, v)
                polarization = polarization_measure(G = G, sigma = sigma)
                if polarization < min_polarization:
                    min_polarization = polarization
                    best_edge = (u, v)
                G.add_edge(u, v, weight=1.0) # Restore the original graph
    return best_edge

def depolarize_policy(state, env, policy):
    while True:
        action = policy(state)
        next_state, _, terminal = env.step(action, state)
        state = next_state
        if terminal:
            break
    if isinstance(state, int):
        G_policy, sigma, _, _ = env.states[state]
    else:
        G_policy, sigma = state["graph"], state["sigma"]
    polarization = env.polarization(G = G_policy, sigma = sigma)
    return G_policy, polarization

def depolarize_random(state, env):
    k = env.k
    if isinstance(state, int):
        G, sigma, _, _ = env.states[state]
    else:
        G, sigma = state["graph"], state["sigma"]
    G_random = G.copy()
    for _ in range(k):
        nodes = list(G_random.nodes())
        while True:
            u, v = random.choice(nodes), random.choice(nodes)
            if u != v and not G_random.has_edge(u, v):
                G_random.add_edge(u, v)
                break
    polarization = env.polarization(G = G_random, sigma = sigma)
    return G_random, polarization


def depolarize_optimal(state, env):
    if isinstance(state, tuple):
        G, sigma, _, _ = state
    else:
        G, sigma, _, _ = env.states[state]
    k = env.k
    min_polarization = float('inf')
    best_G = None

    def backtrack(G, remaining_k):
        nonlocal min_polarization, best_G
        if remaining_k == 0:
            polarization = env.polarization(G = G, sigma = sigma)
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
                    G.add_edge(u, v)     # Restore the original graph

    backtrack(G, k)
    return best_G, min_polarization

