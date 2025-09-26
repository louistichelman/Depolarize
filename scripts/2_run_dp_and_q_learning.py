"""
Dynamic Programming and Q-Learning Experiments for FJ-OffDP
-------------------------------------------------------------------

This script trains and evaluates agents for the Offline Depolarization Problem 
under the Friedkin–Johnsen (FJ) opinion dynamics model.

Pipeline:
1. Environment creation:
   - Loads a precomputed FJOpinionDynamicsFinite environment if available.
   - Otherwise, generates it and saves for reuse.

2. Agents:
   - Dynamic Programming (Policy Iteration).
   - Q-Learning (tabular), trained with configurable episodes and learning rate.

3. Baselines and Optimal Solutions:
   - Greedy FJ-Depolarize heuristic.
   - Brute-force computation of optimal solutions (small n only).

4. Evaluation:
   - Compares DP and Q-Learning policies against the optimal and greedy baselines.
   - Computes average polarization and number of non-optimal solutions.
   - Saves evaluation results to JSON.

5. Visualization:
   - Tracks Q-table snapshots during training.
   - Plots the number of non-optimal solutions over training episodes.

Results:
- All data (policies, evaluation metrics, plots) are saved in `results/dp/`.

Usage:
--n_nodes: Number of nodes in the graph (default: 6)
--k_steps: Number of edges to add (budget) (default: 3)
--max_edges: Maximum number of edges in the graph before adding edges (default: 6)
--training_episodes: Number of training episodes for Q-Learning (default: 4000000)
--learning_rate: Learning rate for Q-Learning (default: 1.0)
--snapshots_qlearning: Snapshot frequency for Q-Learning convergence tracking (default: 50000)

Note:
- Suitable only for small graphs (due to state explosion in DP and brute-force).
- Use this as a controlled benchmark for comparing exact and learning-based solutions.
"""


import pickle
from tqdm import tqdm
import argparse
import os
from pathlib import Path
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add root folder to sys.path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from env import FJOpinionDynamicsFinite
from agents.dynamic_programming import DynamicProgramming
from agents.q_learning import QLearning
from evaluation import depolarize_optimal, greedy_fj_depolarize, depolarize_using_policy


def load_environment(n_nodes: int, k_steps: int, max_edges: int):
    """
    Load or create the FJOpinionDynamicsFinite environment.
    """
    env_path = f"results/dp/environments/env_n{n_nodes}_k{k_steps}_maxedges{max_edges}.pkl"
    if os.path.exists(env_path):
        print(f"Loading environment from '{env_path}'...")
        with open(env_path, "rb") as f:
            env = pickle.load(f)
    else:
        env = FJOpinionDynamicsFinite(
            n=n_nodes, k=k_steps, max_edges=max_edges
        )
        os.makedirs(os.path.dirname(env_path), exist_ok=True)
        with open(env_path, "wb") as f:
            pickle.dump(env, f)
        print(f"Environment saved to '{env_path}'.")
    return env

def dynamic_programming(env, n_nodes, k_steps):
    """
    Load or train a Dynamic Programming agent.
    """
    path = f"results/dp/dynamic_programming_solutions/dp_n{n_nodes}_k{k_steps}.pkl"
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
    return agent_dp

def q_learning(env, n_nodes, k_steps, training_episodes, learning_rate, snapshots_qlearning=50000):
    """
    Load or train a Q-Learning agent.
    """
    path = f"results/dp/q_learning_solutions/q_learning_n{n_nodes}_k{k_steps}_lr{learning_rate}_eps{training_episodes}.pkl"
    if os.path.exists(path):
        print(f"Q-Learning solution already exists at '{path}'. Skipping training.")
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent_qlearning = QLearning(env)
        agent_qlearning.q_table = data["Q"]
        q_table_snapshots = data["snapshots"]
    else:
        agent_qlearning = QLearning(env)
        print("Running Q-Learning...")
        start_time = time.time()
        final_q_table, q_table_snapshots = agent_qlearning.train(
            n_training_episodes=training_episodes,
            min_epsilon=0.2,
            max_epsilon=1.0,
            learning_rate=learning_rate,
            take_snapshots_every=snapshots_qlearning,
        )
        end_time = time.time()
        print(f"Q-Learning finished in {(end_time - start_time) / 60:.2f} minutes.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"Q": final_q_table, "snapshots": q_table_snapshots}, f)
    return agent_qlearning, q_table_snapshots

def compute_optimal_solutions(env, n_nodes, k_steps):
    """
    Compute or load optimal solutions using exhaustive search.
    """
    path = f"results/dp/optimal_solutions/optimal_solutions_n{n_nodes}_k{k_steps}.pkl"
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
                G, sigma, k_steps, FJOpinionDynamicsFinite.polarization
            )
            optimal_solutions[state] = (G_optimal, polarization_optimal)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(optimal_solutions, f)
        print(f"Optimal solutions have been saved to '{path}'.")
    return optimal_solutions

def main():
    parser = argparse.ArgumentParser(
        description="Run dynamic programming for FJ-Depolarize"
    )
    parser.add_argument(
        "--n_nodes", type=int, default=6, help="Number of nodes in the graph"
    )
    parser.add_argument("--k_steps", type=int, default=3, help="Number of edges to add")
    parser.add_argument(
        "--max_edges",
        type=int,
        default=6,
        help="Maximum number of edges in the graph before adding edges",
    )
    parser.add_argument(
        "--training_episodes",
        type=int,
        default=4000000,
        help="Number of training episodes for Q-Learning",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1.0, help="Learning rate for Q-Learning"
    )
    parser.add_argument(
        "--snapshots_qlearning", type=int, default=50000, help="Snapshot frequency for Q-Learning"
    )
    args = parser.parse_args()

    env = load_environment(args.n_nodes, args.k_steps, args.max_edges)
    agent_dp = dynamic_programming(env, args.n_nodes, args.k_steps)
    agent_qlearning, q_table_snapshots = q_learning(
        env, args.n_nodes, args.k_steps, args.training_episodes, args.learning_rate, args.snapshots_qlearning
    )
    optimal_solutions = compute_optimal_solutions(env, args.n_nodes, args.k_steps)
    
    # EVALUATION

    # Compare the policies of DP and Q-Learning with the optimal solutions and the greedy heuristic
    print("Evaluating policies...")
    epsilon = 1e-7
    non_optimal_solutions_dp = 0
    non_optimal_solutions_qlearning = 0
    non_optimal_solutions_greedy = 0
    average_polarization_dp = 0
    average_polarization_qlearning = 0
    average_polarization_greedy = 0
# 
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

    # visualize the Q-table evolution
    non_optimal_solutions_over_time = {}
    for i, q_table in enumerate(q_table_snapshots):
        agent_qlearning.q_table = q_table
        non_optimal_solutions_qlearning = 0
        for state in env.starting_states:
            _, polarization_optimal = optimal_solutions[state]
            _, polarization_qlearning = depolarize_using_policy(
                state, env, agent_qlearning.policy_greedy
            )
            if abs(polarization_qlearning - polarization_optimal) > epsilon:
                non_optimal_solutions_qlearning += 1
        non_optimal_solutions_over_time[i * args.snapshots_qlearning] = non_optimal_solutions_qlearning

    df = pd.DataFrame({
    "Episodes": np.array(list(non_optimal_solutions_over_time.keys())[1:16]) // 1000,
    "Non-optimal Solutions": list(non_optimal_solutions_over_time.values())[1:16]
    })

    save_path = f"results/dp/evaluation_results/qlearning_non_optimal_solutions_n{args.n_nodes}_k{args.k_steps}_lr{args.learning_rate}_eps{args.training_episodes}.png"

    plt.figure(figsize=(10, 3))
    ax = sns.barplot(data=df, x="Episodes", y="Non-optimal Solutions", color="skyblue")
    for i, p in enumerate(ax.patches):
        if i == 0:
            continue 
        height = p.get_height()
        ax.annotate(f'{int(height)}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=10,
                    xytext=(0, 3),  # slight offset above the bar
                    textcoords='offset points')
    plt.title("Non-optimal Solutions Over Time")
    plt.xlabel("Episodes in 1000s")
    plt.ylabel("Number of Non-optimal Solutions")
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    main()
