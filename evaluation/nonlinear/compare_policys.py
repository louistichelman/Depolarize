from tqdm import tqdm
import numpy as np
import random
import os
import json
import torch
from env import NLOpinionDynamics
from agents.dqn import DQN


def test_and_save_policy_dqn(run_name, n_steps=20000, n_simpulations=15):
    """Test a learned DQN policy by simulating opinion dynamics with policy interventions and save the results.
    Args:
        run_name (str): Name of the run to load.
        n_steps (int): Number of steps to simulate.
        n_simpulations (int): Number of simulations to run.
    """

    run_dir = os.path.join("saved files", "dqn", "nonlinear", "saved_runs", run_name)

    # Load parameters
    with open(os.path.join(run_dir, "params_env.json"), "r") as f:
        params_env = json.load(f)
    with open(os.path.join(run_dir, "params_agent.json"), "r") as f:
        params_agent = json.load(f)

    params_agent["wandb_init"] = False

    # Initialize environment and agent
    env = NLOpinionDynamics(**params_env)
    agent = DQN(env=env, **params_agent)

    # Load model weights
    q_net_path = os.path.join(run_dir, "q_network_params.pth")
    target_net_path = os.path.join(run_dir, "target_network_params.pth")

    agent.q_network.load_state_dict(
        torch.load(q_net_path, map_location=torch.device("cpu"))
    )
    agent.target_network.load_state_dict(
        torch.load(target_net_path, map_location=torch.device("cpu"))
    )

    # Test the policy
    recorded_opinions = test_policy(
        env, policy=agent.policy_greedy, n_steps=n_steps, n_simpulations=n_simpulations
    )

    # Save results
    save_path = os.path.join(run_dir, f"recorded_opinions_dqn.npy")
    np.save(save_path, recorded_opinions)
    print(f"Results saved to {save_path}")


def test_and_save_baselines(params_env, n_steps=20000, n_simpulations=15):
    """Test various baseline policies and save the results.
    Args:
        params_env (dict): Parameters for the environment.
        n_steps (int): Number of steps to simulate.
        n_simpulations (int): Number of simulations to run.
    """

    baselines_dir = os.path.join(
        "saved files",
        "dqn",
        "nonlinear",
        "baselines",
        f"n_nodes_{params_env['n']}_average_degree_{params_env['average_degree']}_n_updates{params_env['n_edge_updates_per_step']}",
    )

    os.makedirs(baselines_dir, exist_ok=True)

    # Initialize environment
    env = NLOpinionDynamics(**params_env)

    # Test without policy
    save_path = os.path.join(baselines_dir, f"recorded_opinions_no_policy.npy")
    if os.path.exists(save_path):
        print(f"Results already exist at {save_path}, skipping test without policy.")
    else:
        recorded_opinions_no_policy = test_policy(
            env, n_steps=n_steps, n_simpulations=n_simpulations
        )
        np.save(save_path, recorded_opinions_no_policy)
        print(f"Results saved to {save_path}")

    # Test the minmax policy
    save_path = os.path.join(baselines_dir, f"recorded_opinions_minmax.npy")
    if os.path.exists(save_path):
        print(
            f"Results already exist at {save_path}, skipping test with minmax policy."
        )
    else:
        recorded_opinions_min_max = test_policy(
            env, policy=minmax_policy, n_steps=n_steps, n_simpulations=n_simpulations
        )
        np.save(save_path, recorded_opinions_min_max)
        print(f"Results saved to {save_path}")

    # Test the softminmax policy
    save_path = os.path.join(baselines_dir, f"recorded_opinions_softminmax.npy")
    if os.path.exists(save_path):
        print(
            f"Results already exist at {save_path}, skipping test with softminmax policy."
        )
    else:
        recorded_opinions_soft_min_max = test_policy(
            env,
            policy=minmax_policy_soft,
            n_steps=n_steps,
            n_simpulations=n_simpulations,
        )
        np.save(save_path, recorded_opinions_soft_min_max)
        print(f"Results saved to {save_path}")

    # Test the deleting policy
    save_path = os.path.join(baselines_dir, f"recorded_opinions_deleting.npy")
    if os.path.exists(save_path):
        print(
            f"Results already exist at {save_path}, skipping test with deleting policy."
        )
    else:
        recorded_opinions_deleting = test_policy(
            env, policy=deleting_policy, n_steps=n_steps, n_simpulations=n_simpulations
        )
        np.save(save_path, recorded_opinions_deleting)
        print(f"Results saved to {save_path}")

    # # Test the greedy policy
    # save_path = os.path.join(baselines_dir, f"recorded_opinions_greedy.npy")
    # if os.path.exists(save_path):
    #     print(
    #         f"Results already exist at {save_path}, skipping test with greedy policy."
    #     )
    # else:
    #     recorded_opinions_greedy = test_policy(
    #         env, policy=greedy_policy, n_steps=n_steps, n_simpulations=n_simpulations
    #     )
    #     np.save(save_path, recorded_opinions_greedy)
    #     print(f"Results saved to {save_path}")


def test_policy(env, policy=lambda x, y: 0, n_steps=20000, n_simpulations=15):
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

    for _ in tqdm(
        range(n_simpulations), desc=f"Simulating {n_simpulations} runs with policy"
    ):
        state = env.reset()
        recorded_opinions_simulation = [state["sigma"].copy()]
        for time_step in range(1, n_steps + 1):
            action = policy(state, env)
            next_state, _, _ = env.step(action)
            if time_step % 2 == 0:
                recorded_opinions_simulation.append(next_state["sigma"].copy())
            state = next_state
        recorded_opinions.append(np.array(recorded_opinions_simulation))

    return np.array(recorded_opinions)  # Shape: (n_simpulations, n_steps, n_nodes)


# ----baselines: simple strategies----
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


def minmax_policy_soft(state, env):
    def activation(source, target, opinion_range):
        return np.tanh(0.2 * (1 - abs(source - target) / (opinion_range)) * source)

    if state["tau"] is None:
        sigma_values = np.array(state["sigma"])
        if random.random() < 0.5:
            threshold_top = np.percentile(sigma_values, 75)
            candidates = np.where(sigma_values >= threshold_top)[0]
            return random.choice(candidates)
        else:
            threshold_bottom = np.percentile(sigma_values, 25)
            candidates = np.where(sigma_values <= threshold_bottom)[0]
            return random.choice(candidates)
    else:
        sigma_values = np.array(state["sigma"])
        target = state["tau"]
        target_opinion = sigma_values[target]
        opinion_range = np.max(sigma_values) - np.min(sigma_values)
        non_neighbors = np.setdiff1d(
            np.arange(len(sigma_values)), list(state["graph"].neighbors(target))
        )
        return min(
            non_neighbors,
            key=lambda node: np.sign(target_opinion)
            * activation(sigma_values[node], target_opinion, opinion_range),
        )


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


def greedy_policy(state, env):
    if state["tau"] is None:
        sigma_values = np.array(state["sigma"])
        if random.random() < 0.5:
            threshold_top = np.percentile(sigma_values, 75)
            candidates = np.where(sigma_values >= threshold_top)[0]
            return random.choice(candidates)
        else:
            threshold_bottom = np.percentile(sigma_values, 25)
            candidates = np.where(sigma_values <= threshold_bottom)[0]
            return random.choice(candidates)
    else:
        tau = state["tau"]
        best_node = None
        best_value = float("inf")
        for i in range(200):
            G_new = state["graph"].copy()
            if G_new.has_edge(tau, i):
                G_new.remove_edge(tau, i)
            else:
                G_new.add_edge(tau, i)
            sigma_new = env.opinion_updates(G_new, state["sigma"])
            if env.polarization(sigma_new) < best_value:
                best_value = env.polarization(sigma_new)
                best_node = i
        return best_node


# def minmax_policy(state):
#     if state["tau"] is None:
#         return np.argmax(state["sigma"])
#     else:
#         sigma_values = np.array(state["sigma"])
#         return np.argmin(sigma_values[np.setdiff1d(np.arange(len(sigma_values)), state["graph"].neighbors(state["tau"]))])

# def softminmax_policy(state):
#     if state["tau"] is None:
#         sigma_values = np.array(state["sigma"])
#         threshold_top  = np.percentile(sigma_values, 80)
#         candidates = np.where(sigma_values <= threshold_top)[0]
#         return max(candidates, key=lambda node: state["sigma"][node])
#     else:
#         sigma_values = np.array(state["sigma"])
#         threshold_bottom  = np.percentile(sigma_values, 20)
#         candidates = set(np.where(sigma_values >= threshold_bottom)[0]) - set(state["graph"].neighbors(state["tau"]))
#         return min(candidates, key=lambda node: state["sigma"][node])

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
