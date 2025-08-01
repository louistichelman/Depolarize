import os
import pickle
from env import FJOpinionDynamics, FJOpinionDynamicsFinite
from .depolarizing_functions import (
    depolarize_using_policy,
    depolarize_random_strategy,
    depolarize_greedy_fj,
)
from tqdm import tqdm
from agents.dqn import DQN
import json
import torch


def generate_random_test_states_fj_env(n_values, num_states=150):
    """
    Generate random test states for the FJOpinionDynamics environment.
    """
    save_path = os.path.join(
        "saved files", "dqn", "friedkin_johnson", "evaluation", "test_states"
    )
    os.makedirs(save_path, exist_ok=True)
    all_states = {}

    for n in n_values:
        if os.path.exists(os.path.join(save_path, f"test_states_n{n}.pkl")):
            all_states[n] = pickle.load(
                open(os.path.join(save_path, f"test_states_n{n}.pkl"), "rb")
            )
            continue
        env = FJOpinionDynamics(n=n, k=1)
        # we use the hash function of the FJOpinionDynamicsFinite class to ensure uniqueness
        env_simple = FJOpinionDynamicsFinite(n=n, k=1, generate_states=False)
        states = []
        seen = set()

        while len(states) < num_states:
            state = env.reset()
            # Use the state_hash method from FJOpinionDynamicsFinite to ensure uniqueness
            state_hash = env_simple.state_hash(
                state["graph"], state["sigma"], state["tau"]
            )

            if state_hash not in seen:
                seen.add(state_hash)
                states.append(state)
        all_states[n] = states
        with open(os.path.join(save_path, f"test_states_n{n}.pkl"), "wb") as f:
            pickle.dump(states, f)

    return all_states


def compute_greedy_solutions(k_values, test_states):
    """Compute and save greedy solutions for FJ-Depolarize for given test states and k values.
    Args:
        k_values (list): List of k values for which to compute solutions.
        test_states (dict): Dictionary of test states where keys are n values.
    """
    save_path = os.path.join(
        "saved files", "dqn", "friedkin_johnson", "evaluation", "greedy_solutions"
    )
    os.makedirs(save_path, exist_ok=True)

    for n, states in test_states.items():
        for k in tqdm(
            k_values, desc=f"Computing greedy solutions for n={n} and k in {k_values}"
        ):
            env = FJOpinionDynamics(n=n, k=k)
            greedy_solutions = []
            for state in states:
                _, pol = depolarize_greedy_fj(state, env)
                greedy_solutions.append((state, pol))
            with open(
                os.path.join(save_path, f"greedy_solutions_n{n}_k{k}.pkl"), "wb"
            ) as f:
                pickle.dump(greedy_solutions, f)


def evaluate_dqn_policy_vs_greedy_various_n(
    run_name,
    n_values,
    k_values,
):
    """Compare learned DQN policy to greedy solutions for FJ-Depolarize for various n and k values.
    If reruns are present, they are averaged over.
    Args:
        run_name (str): Name of the run to evaluate.
        n_values (list): List of n values to evaluate.
        k_values (list): List of k values to evaluate.
    """

    run_dir = os.path.join(
        "saved files", "dqn", "friedkin_johnson", "saved_runs", run_name
    )

    results = compare_dqn_vs_greedy_various_n(run_dir, n_values, k_values)

    if os.path.exists(os.path.join(run_dir, "reruns")):
        for folder in os.listdir(os.path.join(run_dir, "reruns")):
            rerun_path = os.path.join(run_dir, "reruns", folder)
            results_rerun = compare_dqn_vs_greedy_various_n(
                rerun_path, n_values, k_values
            )
            results["dqn_better"] += results_rerun["dqn_better"]
            results["greedy_better"] += results_rerun["greedy_better"]
            results["difference"] += results_rerun["difference"]
        number_of_reruns = len(os.listdir(os.path.join(run_dir, "reruns")))
        results["dqn_better"] /= number_of_reruns
        results["greedy_better"] /= number_of_reruns
        results["difference"] /= number_of_reruns

    with open(os.path.join(run_dir, f"evaluation_comparison_to_greedy.pkl"), "wb") as f:
        pickle.dump(results, f)


def compare_dqn_vs_greedy_various_n(run_dir, n_values, k_values):
    """Compare learned DQN policy to greedy solutions for FJ-Depolarize for various n and k values.
    Args:
        run_dir (str): Directory of the run to evaluate.
        n_values (list): List of n values to evaluate.
        k_values (list): List of k values to evaluate.
    Returns:
        dict: A dictionary with results containing the number of states, how many times DQN was better than greedy,
        how many times greedy was better, and the difference in polarization.
    """

    with open(os.path.join(run_dir, "params_env.json"), "r") as f:
        params_env = json.load(f)
    with open(os.path.join(run_dir, "params_agent.json"), "r") as f:
        params_agent = json.load(f)

    params_agent["wandb_init"] = False

    env = FJOpinionDynamics(**params_env)
    agent = DQN(env=env, **params_agent)

    q_net_path = os.path.join(run_dir, "q_network_params.pth")
    target_net_path = os.path.join(run_dir, "target_network_params.pth")

    agent.q_network.load_state_dict(
        torch.load(q_net_path, map_location=torch.device("cpu"))
    )
    agent.target_network.load_state_dict(
        torch.load(target_net_path, map_location=torch.device("cpu"))
    )
    evaluation_dir = os.path.join(
        "saved files", "dqn", "friedkin_johnson", "evaluation", "greedy_solutions"
    )

    epsilon = 1e-4

    results = {}
    for n in n_values:
        for k in k_values:
            with open(
                os.path.join(evaluation_dir, f"greedy_solutions_n{n}_k{k}.pkl"), "rb"
            ) as f:
                greedy_solutions = pickle.load(f)
            env = FJOpinionDynamics(n=n, k=k)
            polarization_diff = 0
            dqn_better = 0
            greedy_better = 0
            for state, greedy_solution in greedy_solutions:
                state["edges_left"] = k
                _, polarization_dqn = depolarize_using_policy(
                    state, env, agent.policy_greedy
                )
                polarization_diff = (
                    polarization_diff + polarization_dqn - greedy_solution
                )
                if abs(polarization_dqn - greedy_solution) > epsilon:
                    if polarization_dqn < greedy_solution:
                        dqn_better += 1
                    else:
                        greedy_better += 1
            results[(n, k)] = {
                "number_states": len(greedy_solutions),
                "dqn_better": dqn_better,
                "greedy_better": greedy_better,
                "difference": polarization_diff,
            }
    return results


def compare_dqn_policy_to_greedy_single_setting(run_name, n=None, k=None):
    """Compare learned DQN policy to greedy solutions for a single setting of FJ-Depolarize.
    Args:
        run_name (str): Name of the run to evaluate.
        n (int, optional): Number of nodes. If None, uses the value from the run parameters.
        k (int, optional): Number of edges. If None, uses the value from the run parameters.
    """

    run_dir = os.path.join(
        "saved files", "dqn", "friedkin_johnson", "saved_runs", run_name
    )
    evaluation_dir = os.path.join(
        "saved files", "dqn", "friedkin_johnson", "evaluation", "greedy_solutions"
    )

    with open(os.path.join(run_dir, "params_env.json"), "r") as f:
        params_env = json.load(f)
    with open(os.path.join(run_dir, "params_agent.json"), "r") as f:
        params_agent = json.load(f)

    params_agent["wandb_init"] = False

    env = FJOpinionDynamics(**params_env)
    agent = DQN(env=env, **params_agent)

    q_net_path = os.path.join(run_dir, "q_network_params.pth")
    target_net_path = os.path.join(run_dir, "target_network_params.pth")

    agent.q_network.load_state_dict(
        torch.load(q_net_path, map_location=torch.device("cpu"))
    )
    agent.target_network.load_state_dict(
        torch.load(target_net_path, map_location=torch.device("cpu"))
    )

    if n is None:
        n = params_env["n"]
    if k is None:
        k = params_env["k"]

    with open(
        os.path.join(evaluation_dir, f"greedy_solutions_n{n}_k{k}.pkl"), "rb"
    ) as f:
        greedy_solutions = pickle.load(f)

    polarization_gains = []
    for state, greedy_solution in greedy_solutions:
        state["edges_left"] = k
        G, sigma = state["graph"], state["sigma"]
        polarization_start = env.polarization(G, sigma)
        _, polarization_dqn = depolarize_using_policy(state, env, agent.policy_greedy)
        _, polarization_random = depolarize_random_strategy(state, env)
        polarization_gains.append(
            (
                polarization_start - polarization_dqn,
                polarization_start - greedy_solution,
                polarization_start - polarization_random,
            )
        )

    with open(os.path.join(run_dir, "evaluation_single_setting.pkl"), "wb") as f:
        pickle.dump(polarization_gains, f)
