from tqdm import tqdm
import numpy as np
import random
import os
import json
import torch
from env import NLOpinionDynamics
from agents.dqn import DQN


def test_and_save_policy_dqn(run_name, n_values, n_steps=20000, folder="val"):
    """
    Test a learned DQN policy by simulating opinion dynamics with policy interventions and save the results.
    """

    run_dir = os.path.join("results", "dqn", "nonlinear", "runs", run_name)

    # Load parameters
    with open(os.path.join(run_dir, "params_env.json"), "r") as f:
        params_env = json.load(f)
    with open(os.path.join(run_dir, "params_agent.json"), "r") as f:
        params_agent = json.load(f)

    params_agent["wandb_init"] = False

    # Initialize environment and agent
    env = NLOpinionDynamics(**params_env)
    agent = DQN(**params_agent)

    # Load model weights
    q_net_path = os.path.join(run_dir, "q_network_params.pth")
    target_net_path = os.path.join(run_dir, "target_network_params.pth")

    agent.q_network.load_state_dict(
        torch.load(q_net_path, map_location=torch.device("cpu"))
    )
    agent.target_network.load_state_dict(
        torch.load(target_net_path, map_location=torch.device("cpu"))
    )

    for n in n_values:
        env.n = n
        # Load start states
        states_path = os.path.join(
            "data",
            "nonlinear",
            folder,
            f"start_states_{folder}_n{n}_d{params_env['average_degree']}.pt",
        )
        with open(states_path, "rb") as f:
            states = torch.load(f, weights_only=False)
        recorded_opinions = test_policy(
            env, policy=agent.policy_greedy, n_steps=n_steps, states=states
        )

        # Save results
        save_path = os.path.join(run_dir, folder)
        os.makedirs(save_path, exist_ok=True)
        np.save(
            os.path.join(
                save_path,
                f"recorded_opinions_dqn_n{n}.npy",
            ),
            recorded_opinions,
        )


def test_and_save_baselines(
    params_env, n_values, n_steps=20000, folder="test", force_recomputation=False
):
    """
    Test various baseline policies and save the results.
    """
    for n in n_values:
        params_env["n"] = n
        print(f"Testing baselines for n={n} ...")

        baselines_dir = os.path.join(
            "data",
            "nonlinear",
            "baselines",
            folder,
            f"n_nodes_{params_env['n']}_average_degree_{params_env['average_degree']}_n_updates{params_env['n_edge_updates_per_step']}",
        )

        os.makedirs(baselines_dir, exist_ok=True)

        # Initialize environment
        env = NLOpinionDynamics(**params_env)

        # Load start states
        states_path = os.path.join(
            "data",
            "nonlinear",
            folder,
            f"start_states_{folder}_n{params_env['n']}_d{params_env['average_degree']}.pt",
        )
        with open(states_path, "rb") as f:
            states = torch.load(f, weights_only=False)

        # Test without policy
        save_path = os.path.join(baselines_dir, f"recorded_opinions_no_policy.npy")
        if os.path.exists(save_path) and not force_recomputation:
            print(
                f"Results already exist at {save_path}, skipping test without policy."
            )
        else:
            recorded_opinions_no_policy = test_policy(
                env, n_steps=n_steps, states=states
            )
            np.save(save_path, recorded_opinions_no_policy)
            print(f"Results saved to {save_path}")

        # Test the minmax policy
        save_path = os.path.join(baselines_dir, f"recorded_opinions_minmax.npy")
        if os.path.exists(save_path) and not force_recomputation:
            print(
                f"Results already exist at {save_path}, skipping test with minmax policy."
            )
        else:
            recorded_opinions_min_max = test_policy(
                env, policy=minmax_policy, n_steps=n_steps, states=states
            )
            np.save(save_path, recorded_opinions_min_max)
            print(f"Results saved to {save_path}")

        # Test the deleting policy
        save_path = os.path.join(baselines_dir, f"recorded_opinions_deleting.npy")
        if os.path.exists(save_path) and not force_recomputation:
            print(
                f"Results already exist at {save_path}, skipping test with deleting policy."
            )
        else:
            recorded_opinions_deleting = test_policy(
                env, policy=deleting_policy, n_steps=n_steps, states=states
            )
            np.save(save_path, recorded_opinions_deleting)
            print(f"Results saved to {save_path}")

        # Test the soft minmax policy
        save_path = os.path.join(baselines_dir, f"recorded_opinions_soft_minmax.npy")
        if os.path.exists(save_path) and not force_recomputation:
            print(
                f"Results already exist at {save_path}, skipping test with soft minmax policy."
            )
        else:
            recorded_opinions_greedy = test_policy(
                env, policy=soft_minmax_policy, n_steps=n_steps, states=states
            )
            np.save(save_path, recorded_opinions_greedy)
            print(f"Results saved to {save_path}")


def test_policy(
    env, policy=lambda state, env: 0, n_steps=20000, states=None, n_simpulations=15
):
    """Test a policy by simulating opinion dynamics and recording the opinions.
    Args:
        env (NLOpinionDynamics): The environment to simulate.
        policy (callable): The policy function to use for actions.
        n_steps (int): Number of steps to simulate.
        n_simpulations (int): Number of simulations to run.
    Returns:
        np.ndarray: Recorded opinions from the simulations.
    """
    recorded_opinions = []

    if states is None:
        # Generate random initial states if not provided
        states = [env.reset() for _ in range(n_simpulations)]

    for state in tqdm(states, desc=f"Simulating {len(states)} runs with policy"):
        recorded_opinions_simulation = [state["sigma"].copy()]
        for time_step in range(1, n_steps + 1):
            action = policy(state=state, env=env)
            next_state, _, _ = env.step(action=action, state=state)
            if time_step % 2 == 0:
                recorded_opinions_simulation.append(next_state["sigma"].copy())
            state = next_state
        recorded_opinions.append(np.array(recorded_opinions_simulation))

    return np.array(recorded_opinions)  # Shape: (n_simpulations, n_steps, n_nodes)


# ----baselines: simple strategies----
def random_policy(state, env):
    return random.randint(0, len(state["sigma"]) - 1)


def minmax_policy(state, env):
    if state["tau"] is None:
        return np.argmax(state["sigma"])
    else:
        sigma_values = np.array(state["sigma"])
        return np.argmin(
            sigma_values[
                np.setdiff1d(
                    np.arange(len(sigma_values)), state["graph"].neighbors(state["tau"])
                )
            ]
        )


# def minmax_policy_soft(state, env):
#     def activation(source, target, opinion_range):
#         return np.tanh(0.2 * (1 - abs(source - target) / (opinion_range)) * source)

#     if state["tau"] is None:
#         # sigma_values = np.array(state["sigma"])
#         return random.randint(0, len(state["sigma"]) - 1)
#         # if random.random() < 0.5:
#         #     threshold_top = np.percentile(sigma_values, 75)
#         #     candidates = np.where(sigma_values >= threshold_top)[0]
#         #     return random.choice(candidates)
#         # else:
#         #     threshold_bottom = np.percentile(sigma_values, 25)
#         #     candidates = np.where(sigma_values <= threshold_bottom)[0]
#         #     return random.choice(candidates)
#     else:
#         sigma_values = np.array(state["sigma"])
#         target = state["tau"]
#         target_opinion = sigma_values[target]
#         opinion_range = np.max(sigma_values) - np.min(sigma_values)
#         non_neighbors = np.setdiff1d(
#             np.arange(len(sigma_values)), list(state["graph"].neighbors(target))
#         )
#         return min(
#             non_neighbors,
#             key=lambda node: np.sign(target_opinion)
#             * activation(sigma_values[node], target_opinion, opinion_range),
#         )


def deleting_policy(state, env):
    if state["tau"] is None:
        if random.random() < 0.5:
            return np.argmax(state["sigma"])
        else:
            return np.argmin(state["sigma"])
    else:
        sigma_values = np.array(state["sigma"])
        neighbors = list(state["graph"].neighbors(state["tau"]))
        if not neighbors:
            return 0  # No neighbors available
        return min(
            neighbors,
            key=lambda node: abs(sigma_values[node] - sigma_values[state["tau"]]),
        )


# def greedy_policy(state, env):
#     if state["tau"] is None:
#         sigma_values = np.array(state["sigma"])
#         if random.random() < 0.5:
#             threshold_top = np.percentile(sigma_values, 75)
#             candidates = np.where(sigma_values >= threshold_top)[0]
#             return random.choice(candidates)
#         else:
#             threshold_bottom = np.percentile(sigma_values, 25)
#             candidates = np.where(sigma_values <= threshold_bottom)[0]
#             return random.choice(candidates)
#     else:
#         tau = state["tau"]
#         best_node = None
#         best_value = float("inf")
#         for i in range(len(state["sigma"])):
#             G_new = state["graph"].copy()
#             if G_new.has_edge(tau, i):
#                 G_new.remove_edge(tau, i)
#             else:
#                 G_new.add_edge(tau, i)
#             sigma_new = env.opinion_updates(G_new, state["sigma"])
#             if env.polarization(G_new, sigma_new) < best_value:
#                 best_value = env.polarization(G_new, sigma_new)
#                 best_node = i
#         return best_node


# def minmax_policy(state):
#     if state["tau"] is None:
#         return np.argmax(state["sigma"])
#     else:
#         sigma_values = np.array(state["sigma"])
#         return np.argmin(sigma_values[np.setdiff1d(np.arange(len(sigma_values)), state["graph"].neighbors(state["tau"]))])


def soft_minmax_policy(state, env):
    if state["tau"] is None:
        sigma_values = np.array(state["sigma"])
        threshold_top = np.percentile(sigma_values, 80)
        candidates = np.where(sigma_values <= threshold_top)[0]
        return max(candidates, key=lambda node: state["sigma"][node])
    else:
        sigma_values = np.array(state["sigma"])
        threshold_bottom = np.percentile(sigma_values, 20)
        candidates = set(np.where(sigma_values >= threshold_bottom)[0]) - set(
            state["graph"].neighbors(state["tau"])
        )
        return min(candidates, key=lambda node: state["sigma"][node])


# def deleting_policy(state):
#     if state["tau"] is None:
#         if random.random() < 0.5:
#             return np.argmax(state["sigma"])
#         else:
#             return np.argmin(state["sigma"])
#     else:
#         sigma_values = np.array(state["sigma"])
#         neighbors = list(state["graph"].neighbors(state["tau"]))
#         if not neighbors:
#             return 0  # No neighbors available
#         return min(neighbors, key=lambda node: abs(sigma_values[node] - sigma_values[state["tau"]]))
