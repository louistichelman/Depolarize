"""
Policy-Based Depolarization
-----------------------------------

This module provides a helper function to apply a given policy on an FJ 
opinion dynamics environment until termination. It works with both 
representations of environment states in FJOpinionDynamicsFinite and FJOpinionDynamics.

The function returns the depolarized graph and its final polarization level.
"""

from env import BaseEnv


def depolarize_using_policy(state, env: BaseEnv, policy: callable):
    """
    Applies a given policy on the environment until reaching a terminal state.

    Parameters
    ----------
    state : dict or int
        Initial state of the environment.
    env : BaseEnv
        The environment instance (FJOpinionDynamics or FJOpinionDynamicsFinite).
    policy : callable
        A function that takes a state and returns an action.
        
    Returns
    -------
    G_policy : networkx.Graph
        The depolarized graph after applying the policy.
    polarization : float
        The final polarization level of the graph.
    """
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
    polarization = env.polarization(G=G_policy, sigma=sigma)
    return G_policy, polarization
