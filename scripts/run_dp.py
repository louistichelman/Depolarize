import pickle
from tqdm import tqdm
import argparse
import os
import sys
from pathlib import Path
import time
import json


# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from agents.dynamic_programming import DynamicProgramming
from env import FJOpinionDynamicsFinite
from agents.q_learning import QLearning
from evaluation import depolarize_optimal, greedy_fj_depolarize, depolarize_using_policy


def main():
    parser = argparse.ArgumentParser(
        description="Run dynamic programming for FJ-Depolarize"
    )
    parser.add_argument(
        "--n_nodes", type=int, default=3, help="Number of nodes in the graph"
    )
    parser.add_argument("--k_steps", type=int, default=2, help="Number of edges to add")
    parser.add_argument(
        "--max_edges",
        type=int,
        default=None,
        help="Maximum number of edges in the graph before adding edges",
    )
    parser.add_argument(
        "--training_episodes",
        type=int,
        default=100000,
        help="Number of training episodes for Q-Learning",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1.0, help="Learning rate for Q-Learning"
    )
    args = parser.parse_args()

    # LOAD ENVIRONMENT
    env_path = f"results/dp/environments/env_n{args.n_nodes}_k{args.k_steps}_maxedges{args.max_edges}.pkl"
    if os.path.exists(env_path):
        print(f"Loading environment from '{env_path}'...")
        with open(env_path, "rb") as f:
            env = pickle.load(f)
    else:
        env = FJOpinionDynamicsFinite(
            n=args.n_nodes, k=args.k_steps, max_edges=args.max_edges
        )
        os.makedirs(os.path.dirname(env_path), exist_ok=True)
        with open(env_path, "wb") as f:
            pickle.dump(env, f)
        print(f"Environment saved to '{env_path}'.")

    # DYNCAMIC PROGRAMMING
    path = f"results/dp/dynamic_programming_solutions/dp_n{args.n_nodes}_k{args.k_steps}.pkl"
    if os.path.exists(path):
        print(
            f"Dynamic Programming solution already exists at '{path}'. Skipping training."
        )
        with open(path, "rb") as f:
            data = pickle.load(f)
        V, pi = data["V"], data["pi"]
        agent_dp = DynamicProgramming(env)
        agent_dp.V = V
        agent_dp.pi = pi
    else:
        agent_dp = DynamicProgramming(env)
        print("Running Dynamic Programming...")
        start_time = time.time()
        V, pi = agent_dp.run()
        end_time = time.time()
        print(
            f"Dynamic Programming finished in {(end_time - start_time) / 60:.2f} minutes."
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"V": V, "pi": pi}, f)

    # Q-LEARNING
    path = f"results/dp/q_learning_solutions/q_learning_n{args.n_nodes}_k{args.k_steps}_lr{args.learning_rate}_eps{args.training_episodes}.pkl"
    if os.path.exists(path):
        print(f"Q-Learning solution already exists at '{path}'. Skipping training.")
        with open(path, "rb") as f:
            data = pickle.load(f)
        q_table = data["Q"]
        agent_qlearning = QLearning(env)
        agent_qlearning.q_table = q_table
    else:
        agent_qlearning = QLearning(env)
        print("Running Q-Learning...")
        start_time = time.time()
        q_table = agent_qlearning.train(
            n_training_episodes=args.training_episodes,
            min_epsilon=0.2,
            max_epsilon=1.0,
            decay_rate=0.000005,
            learning_rate=args.learning_rate,
        )
        end_time = time.time()
        print(f"Q-Learning finished in {(end_time - start_time) / 60:.2f} minutes.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"Q": q_table}, f)

    # OPTIMAL SOLUTIONS
    path = f"results/dp/optimal_solutions/optimal_solutions_n{args.n_nodes}_k{args.k_steps}.pkl"
    if os.path.exists(path):
        print(f"Optimal solutions already exist at '{path}'. Skipping computation.")
        with open(path, "rb") as f:
            optimal_solutions = pickle.load(f)
    else:
        optimal_solutions = {}
        print("Computing optimal solutions via brute force...")
        for state in env.starting_states:
            G, sigma, _, _ = env.states[state]
            G_optimal, polarization_optimal = depolarize_optimal(
                G, sigma, args.k_steps, FJOpinionDynamicsFinite.polarization
            )
            optimal_solutions[state] = (G_optimal, polarization_optimal)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(optimal_solutions, f)
        print(f"Optimal solutions have been saved to '{path}'.")

    # EVALUATION
    epsilon = 1e-7
    non_optimal_solutions_dp = 0
    non_optimal_solutions_qlearning = 0
    non_optimal_solutions_greedy = 0
    average_polarization_dp = 0
    average_polarization_qlearning = 0
    average_polarization_greedy = 0

    for state in tqdm(env.starting_states):
        _, polarization_optimal = optimal_solutions[state]
        _, polarization_dp = depolarize_using_policy(state, env, agent_dp.policy_greedy)
        average_polarization_dp += polarization_dp
        _, polarization_qlearning = depolarize_using_policy(
            state, env, agent_qlearning.policy_greedy
        )
        average_polarization_qlearning += polarization_qlearning
        G, sigma, _, _ = env.states[state]
        G_greedy = greedy_fj_depolarize(G, sigma, args.k_steps)
        polarization_greedy = FJOpinionDynamicsFinite.polarization(G_greedy, sigma)
        average_polarization_greedy += polarization_greedy
        if abs(polarization_dp - polarization_optimal) > epsilon:
            non_optimal_solutions_dp += 1
        if abs(polarization_qlearning - polarization_optimal) > epsilon:
            non_optimal_solutions_qlearning += 1
        if abs(polarization_greedy - polarization_optimal) > epsilon:
            non_optimal_solutions_greedy += 1
    average_polarization_dp /= len(env.starting_states)
    average_polarization_qlearning /= len(env.starting_states)
    average_polarization_greedy /= len(env.starting_states)
    results = {
        "non_optimal_solutions_dp": non_optimal_solutions_dp,
        "non_optimal_solutions_qlearning": non_optimal_solutions_qlearning,
        "non_optimal_solutions_greedy": non_optimal_solutions_greedy,
        "average_polarization_dp": average_polarization_dp,
        "average_polarization_qlearning": average_polarization_qlearning,
        "average_polarization_greedy": average_polarization_greedy,
        "n_nodes": args.n_nodes,
        "k_steps": args.k_steps,
        "max_edges": args.max_edges,
        "training_episodes": args.training_episodes,
        "learning_rate": args.learning_rate,
    }

    results_path = f"results/dp/evaluation_results/eval_n{args.n_nodes}_k{args.k_steps}_lr{args.learning_rate}_eps{args.training_episodes}.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f)
    print(f"Evaluation results have been saved to '{results_path}'.")


if __name__ == "__main__":
    main()
