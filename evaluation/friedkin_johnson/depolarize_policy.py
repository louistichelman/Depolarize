from env import BaseEnv


def depolarize_using_policy(state: dict, env: BaseEnv, policy: callable):
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
