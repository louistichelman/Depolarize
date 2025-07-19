from tqdm import tqdm
import random
from .greedy_depolarize import greedy_fj_depolarize

def depolarize_using_policy(state, env, policy):
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

def depolarize_random_strategy(state, env):
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

def depolarize_greedy_fj(state, env):
    """
    Input:
    state: dict with keys 'graph' (networkx.Graph), 'sigma' (np.array of internal opinions)
    env: environment object with method depolarization and attribute k
    
    Output:
    G_greedy: networkx.Graph after modifications
    polarization: float, polarization of the final graph
    """
    k = env.k
    if isinstance(state, int):
        G, sigma, _, _ = env.states[state]
    else:
        G, sigma = state["graph"], state["sigma"]

    G_greedy = greedy_fj_depolarize(G, sigma, k)

    return G_greedy, env.polarization(G = G_greedy, sigma = sigma)


def depolarize_optimal(state, env):
    k = env.k
    if isinstance(state, int):
        G, sigma, _, _ = env.states[state]
    else:
        G, sigma = state["graph"], state["sigma"]
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

