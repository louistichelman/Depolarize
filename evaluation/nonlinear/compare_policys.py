"""
Evaluation for Policy in Nonlinear Opinion Dynamics (NL-OnDP)
-----------------------------------------------------------

This module provides tools to evaluate learned DQN policies and simple
heuristic baselines in the nonlinear opinion dynamics environment
(NLOpinionDynamics). It records opinion trajectories, actions, and
graph metrics, and saves them for later analysis.

Functions:
- test_and_save_policy_dqn(run_name, n_values, n_steps=20000, folder="val"):
    Evaluate a trained DQN policy on validation/test states and save results.

- test_and_save_baselines(params_env, n_values, n_steps=20000, folder="test", force_recomputation=False):
    Evaluate baseline strategies (no policy, min-max, deleting, soft min-max)
    on validation/test states and save results.

- test_policy(env, policy=lambda state, env: 0, n_steps=20000, states=None, n_simpulations=15):
    Core simulation loop. Runs opinion dynamics with a given policy, recording
    opinions, actions, and graph metrics across steps.

- minmax_policy(state):
    Baseline that connects extreme opinions (max with min).

- deleting_policy(state):
    Baseline that tends to remove edges between nodes with similar opinions.

- soft_minmax_policy(state):
    Softer variant of min-max: uses percentile thresholds to select nodes.

Outputs:
- Numpy arrays of recorded opinions, actions, and graph metrics.
- Saved to disk under results/dqn/nonlinear/... (for DQN) or
  data/nonlinear/baselines/... (for baselines).
"""


from tqdm import tqdm
import numpy as np
import random
import os
import json
import torch
from env import NLOpinionDynamics
from agents.dqn import DQN
import networkx as nx


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
        recorded_opinions, recorded_actions, recorded_graph_metrics = test_policy(
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
        np.save(
            os.path.join(
                save_path,
                f"recorded_actions_dqn_n{n}.npy",
            ),
            recorded_actions,
        )
        np.save(
            os.path.join(
                save_path,
                f"recorded_graph_metrics_dqn_n{n}.npy",
            ),
            recorded_graph_metrics,
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
        recorded_opinions_no_policy, _, recorded_graph_metrics_no_policy = test_policy(
            env, n_steps=n_steps, states=states
        )
        np.save(os.path.join(baselines_dir, f"recorded_opinions_no_policy.npy"), recorded_opinions_no_policy)
        np.save(os.path.join(baselines_dir, f"recorded_graph_metrics_no_policy.npy"), recorded_graph_metrics_no_policy)
        print(f"Results saved to {baselines_dir}")

        # Test the minmax policy
        recorded_opinions_min_max, _, recorded_graph_metrics_min_max = test_policy(
            env, policy=minmax_policy, n_steps=n_steps, states=states
        )
        np.save(os.path.join(baselines_dir, f"recorded_opinions_minmax.npy"), recorded_opinions_min_max)
        np.save(os.path.join(baselines_dir, f"recorded_graph_metrics_minmax.npy"), recorded_graph_metrics_min_max)
        print(f"Results saved to {baselines_dir}")

        # Test the deleting policy
        recorded_opinions_deleting, _, recorded_graph_metrics_deleting = test_policy(
            env, policy=deleting_policy, n_steps=n_steps, states=states
        )
        np.save(os.path.join(baselines_dir, f"recorded_opinions_deleting.npy"), recorded_opinions_deleting)
        np.save(os.path.join(baselines_dir, f"recorded_graph_metrics_deleting.npy"), recorded_graph_metrics_deleting)
        print(f"Results saved to {baselines_dir}")

        # Test the soft minmax policy
        recorded_opinions_greedy, _, recorded_graph_metrics_greedy = test_policy(
            env, policy=soft_minmax_policy, n_steps=n_steps, states=states
        )
        np.save(os.path.join(baselines_dir, f"recorded_opinions_soft_minmax.npy"), recorded_opinions_greedy)
        np.save(os.path.join(baselines_dir, f"recorded_graph_metrics_soft_minmax.npy"), recorded_graph_metrics_greedy)
        print(f"Results saved to {baselines_dir}")


def test_policy(
    env, policy=lambda state: 0, n_steps=20000, states=None, n_simpulations=15
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
    recorded_actions = []
    recorded_graph_metrics = []

    if states is None:
        # Generate random initial states if not provided
        states = [env.reset() for _ in range(n_simpulations)]

    for state in tqdm(states, desc=f"Simulating {len(states)} runs with policy"):
        recorded_actions_simulation = []
        recorded_opinions_simulation = [state["sigma"].copy()]
        recorded_graph_metrics_simulation = []
        for _ in range(1, n_steps + 1):
            action = policy(state=state)
            if state["tau"] is not None:
                # opinions
                recorded_opinions_simulation.append(next_state["sigma"].copy())

                # actions
                connected = state["graph"].has_edge(state["tau"], action)
                deg_tau = state["graph"].degree[state["tau"]]
                deg_action = state["graph"].degree[action]

                recorded_actions_simulation.append((
                    state["sigma"][state["tau"]],
                    state["sigma"][action],
                    connected,
                    deg_tau,
                    deg_action
                ))

                # graph metrics
                deg_dict = dict(state["graph"].degree())
                deg_vec = np.array([deg_dict[node] for node in state["graph"].nodes()])
                avg_clustering = nx.average_clustering(state["graph"])
                recorded_graph_metrics_simulation.append((
                    np.mean(deg_vec),
                    np.std(deg_vec),
                    avg_clustering
                ))

            next_state, _, _ = env.step(action=action, state=state)                
            state = next_state
        recorded_opinions.append(np.array(recorded_opinions_simulation))
        recorded_actions.append(np.array(recorded_actions_simulation))
        recorded_graph_metrics.append(np.array(recorded_graph_metrics_simulation))

    return np.array(recorded_opinions), np.array(recorded_actions), np.array(recorded_graph_metrics)  # Shape: (n_simpulations, n_steps, n_nodes), (n_simpulations, n_steps, 3)


# ----baselines: simple strategies----


def minmax_policy(state):
    if state["tau"] is None:
        return np.argmax(state["sigma"])
    else:
        sigma_values = np.array(state["sigma"])
        candidates = np.setdiff1d(np.arange(len(sigma_values)), state["graph"].neighbors(state["tau"]))
        return candidates[np.argmin(sigma_values[candidates])]


def deleting_policy(state):
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


def soft_minmax_policy(state):
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
