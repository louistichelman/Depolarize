import os
import pickle
from env import FJOpinionDynamics
from .depolarize_policy import depolarize_using_policy
from .depolarize_random import depolarize_random_strategy
from tqdm import tqdm
from agents.dqn import DQN
import json
import torch


def evaluate_dqn_policy_vs_greedy_ood_n(
    run_name,
    n_values,
    k_values,
    folder="val",
):
    """
    Compare for a given run the learned policy with greedy solutions on validation or test states for various n and k values.
    The created file can then be used to visualize performance in form of a heatmap.
    If reruns are present, they are averaged over.
    """

    run_dir = os.path.join("results", "dqn", "friedkin-johnson", "runs", run_name)

    results = evaluate_dqn_policy_vs_greedy_ood_n_single_run(
        run_dir, n_values, k_values, folder=folder
    )

    if os.path.exists(os.path.join(run_dir, "reruns")):
        variances = {}
        for run_folder in os.listdir(os.path.join(run_dir, "reruns")):
            rerun_path = os.path.join(run_dir, "reruns", run_folder)
            results_rerun = evaluate_dqn_policy_vs_greedy_ood_n_single_run(
                rerun_path, n_values, k_values, folder=folder
            )
            for key in results_rerun:
                variances.setdefault(key, []).append(results_rerun[key]["difference"])
                results[key]["dqn_better"] += results_rerun[key]["dqn_better"]
                results[key]["greedy_better"] += results_rerun[key]["greedy_better"]
                results[key]["difference"] += results_rerun[key]["difference"]
        number_of_reruns = len(os.listdir(os.path.join(run_dir, "reruns")))
        for key in results:
            variances[key] = torch.var(torch.tensor(variances[key])).item()
            results[key]["dqn_better"] /= number_of_reruns + 1
            results[key]["greedy_better"] /= number_of_reruns + 1
            results[key]["difference"] /= number_of_reruns + 1
        with open(
            os.path.join(
                run_dir, f"evaluation_comparison_to_greedy_variance_{folder}.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(variances, f)

    os.makedirs(os.path.join(run_dir, folder), exist_ok=True)

    with open(
        os.path.join(run_dir, folder, "evaluation_comparison_to_greedy.pkl"), "wb"
    ) as f:
        pickle.dump(results, f)


def evaluate_dqn_policy_vs_greedy_ood_n_single_run(
    run_dir, n_values, k_values, folder="val"
):
    """
    Helper function of evaluate_dqn_policy_vs_greedy_ood_n.
    """

    with open(os.path.join(run_dir, "params_agent.json"), "r") as f:
        params_agent = json.load(f)
    with open(os.path.join(run_dir, "params_env.json"), "r") as f:
        params_env = json.load(f)

    agent = DQN(**params_agent)
    env = FJOpinionDynamics(**params_env)

    q_net_path = os.path.join(run_dir, "q_network_params.pth")
    target_net_path = os.path.join(run_dir, "target_network_params.pth")

    agent.q_network.load_state_dict(
        torch.load(q_net_path, map_location=torch.device("cpu"))
    )
    agent.target_network.load_state_dict(
        torch.load(target_net_path, map_location=torch.device("cpu"))
    )
    evaluation_dir = os.path.join(
        "data", "friedkin-johnson", "greedy_solutions", folder
    )

    epsilon = 1e-4

    results = {}
    for n in tqdm(n_values, desc="Evaluating DQN vs Greedy for various n"):
        for k in k_values:
            with open(
                os.path.join(
                    evaluation_dir,
                    f"greedy_solutions_n{n}_d{params_env['average_degree']}_k{k}.pt",
                ),
                "rb",
            ) as f:
                greedy_solutions = torch.load(f, weights_only=False)
            env.n = n
            env.k = k
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


def evaluate_dqn_policy_vs_greedy(run_name, n, k, folder="val"):
    """
    Compare for a given run the learned policy with greedy solutions on validation or test states for a single n,k combination.
    The created file can then be used to visualize performance in form of a dotted plot.
    """

    run_dir = os.path.join("results", "dqn", "friedkin-johnson", "runs", run_name)
    evaluation_dir = os.path.join(
        "data", "friedkin-johnson", "greedy_solutions", folder
    )

    with open(os.path.join(run_dir, "params_env.json"), "r") as f:
        params_env = json.load(f)
    with open(os.path.join(run_dir, "params_agent.json"), "r") as f:
        params_agent = json.load(f)

    env = FJOpinionDynamics(**params_env)
    agent = DQN(**params_agent)

    q_net_path = os.path.join(run_dir, "q_network_params.pth")
    target_net_path = os.path.join(run_dir, "target_network_params.pth")

    agent.q_network.load_state_dict(
        torch.load(q_net_path, map_location=torch.device("cpu"))
    )
    agent.target_network.load_state_dict(
        torch.load(target_net_path, map_location=torch.device("cpu"))
    )

    with open(
        os.path.join(
            evaluation_dir,
            f"greedy_solutions_n{n}_d{params_env['average_degree']}_k{k}.pt",
        ),
        "rb",
    ) as f:
        greedy_solutions = torch.load(f, weights_only=False)

    polarization_gains = []
    for state, greedy_solution in greedy_solutions:
        state["edges_left"] = k
        G, sigma = state["graph"], state["sigma"]
        polarization_start = env.polarization(G, sigma)
        _, polarization_dqn = depolarize_using_policy(state, env, agent.policy_greedy)
        G_random = depolarize_random_strategy(G, k)
        polarization_random = env.polarization(G=G_random, sigma=sigma)
        polarization_gains.append(
            (
                polarization_start - polarization_dqn,
                polarization_start - greedy_solution,
                polarization_start - polarization_random,
            )
        )

    os.makedirs(os.path.join(run_dir, folder), exist_ok=True)

    with open(
        os.path.join(run_dir, folder, "evaluation_single_setting.pkl"), "wb"
    ) as f:
        pickle.dump(polarization_gains, f)


# def evaluate_heuristics_vs_greedy_vs_optimal(
#     n_values,
#     k_values,
#     folder="test",
# ):
#     """
#     Compare the greedy algorithm with heuristics of Rasz et al. and optimal solutions for FJ-Depolarize for various n and k values.
#     Only works for small n values, as optimal solutions are computed via brute force.
#     """
#     epsilon = 1e-4

#     results = {}
#     for n in tqdm(n_values, desc="Evaluating heuristics vs Greedy vs optimal for various n"):
#         for k in k_values:
#             with open(
#                 os.path.join(evaluation_dir, f"greedy_solutions_n{n}_k{k}.pt"), "rb"
#             ) as f:
#                 greedy_solutions = torch.load(f, weights_only=False)
#             env.n = n
#             env.k = k
#             polarization_diff = 0
#             dqn_better = 0
#             greedy_better = 0
#             for state, greedy_solution in greedy_solutions:
#                 state["edges_left"] = k
#                 _, polarization_dqn = depolarize_using_policy(
#                     state, env, agent.policy_greedy
#                 )
#                 polarization_diff = (
#                     polarization_diff + polarization_dqn - greedy_solution
#                 )
#                 if abs(polarization_dqn - greedy_solution) > epsilon:
#                     if polarization_dqn < greedy_solution:
#                         dqn_better += 1
#                     else:
#                         greedy_better += 1
#             results[(n, k)] = {
#                 "number_states": len(greedy_solutions),
#                 "dqn_better": dqn_better,
#                 "greedy_better": greedy_better,
#                 "difference": polarization_diff,
#             }
#     return results
