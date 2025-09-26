"""
Compare Heuristic Algorithms for Offline Depolarization Problem (OffDP) (See Table 3.1)
---------------------------------------------------

This script generates small random graph instances and evaluates different strategies 
for the Offline Depolarization Problem (OffDP) under the Friedkin–Johnsen (FJ) opinion 
dynamics model.

Implemented methods:
- Greedy FJ-Depolarize: exact greedy algorithm based on polarization gain.
- Heuristic DS (Disagreement Seeking) and CD (Coordinate Descent) from Rácz et al. (2021).
- No Intervention baseline.
- Optimal solution via exhaustive search, for very small graphs.

Workflow:
1. Generate Erdős–Rényi graphs with random initial opinions (sigma ∈ [-1, 1]).
2. Apply each depolarization strategy with budget k edge modifications.
3. Evaluate final polarization using the FJ model.
4. Save detailed results as pickle files and summary statistics as JSON in results/heuristics/.

Usage:
--num_graphs: Number of random graphs to generate (default: 1000)
--num_nodes: Number of nodes in each graph (default: 10)
--k: Number of edge modifications allowed (budget) (default: 3)
--edge_prob: Probability of edge creation in the Erdős–Rényi model (default: 0.4)

Purpose:
This code reproduces the experiments described in Chapter 3 of the thesis, 
comparing greedy and heuristic algorithms against optimal solutions on small graphs. 
The results establish Greedy FJ-Depolarize as a strong baseline for later 
deep reinforcement learning experiments.
"""


import networkx as nx
import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path
import argparse
import sys
import os
import json

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from evaluation import (
    greedy_fj_depolarize,
    heuristic_ds_fj_depolarize,
    heuristic_cd_fj_depolarize,
    depolarize_optimal,
)
from env import FJOpinionDynamicsFinite


def generate_graphs_erdos_renyi(num_graphs: int, num_nodes: int, edge_prob: float):
    """
    Generate a list of random Erdős–Rényi graphs with random sigma vectors.
    Ensures uniqueness of (graph, sigma) pairs using state_hash.
    """
    graphs = []
    sigma_vectors = []
    seen = set()

    while len(graphs) < num_graphs:
        # Generate a random graph (Erdos-Renyi)
        G = nx.erdos_renyi_graph(n=num_nodes, p=edge_prob)
        # Generate a random sigma vector with values between -1 and 1
        sigma = np.random.uniform(-1, 1, size=num_nodes)
        while np.all(sigma == 0) or np.all(sigma == 1):
            sigma = np.random.uniform(-1, 1, size=num_nodes)

        # Ensure uniqueness using state_hash
        graph_hash = FJOpinionDynamicsFinite.state_hash(G, sigma)
        if graph_hash in seen:
            continue
        seen.add(graph_hash)

        graphs.append(G)
        sigma_vectors.append(sigma)

    return graphs, sigma_vectors

def compute_greedy_solutions(graphs: list[nx.Graph], sigma_vectors: list[np.ndarray], k: int):
    """
    Compute greedy solutions for a list of graphs and sigma vectors.
    Returns a list of tuples (graph, polarization) after applying greedy FJ-Depolarize.
    """
    greedy_solutions = []
    for G, sigma in tqdm(zip(graphs, sigma_vectors), total=len(graphs), desc="Computing greedy solutions"):
        solution = greedy_fj_depolarize(G, sigma, k=k)
        greedy_solutions.append((G, FJOpinionDynamicsFinite.polarization(solution, sigma)))
    return greedy_solutions

def compute_heuristic_ds_solutions(graphs: list[nx.Graph], sigma_vectors: list[np.ndarray], k: int):
    """
    Compute heuristic DS solutions for a list of graphs and sigma vectors.
    Returns a list of tuples (graph, polarization) after applying heuristic DS.
    """
    heuristic_ds_solutions = []
    for G, sigma in tqdm(zip(graphs, sigma_vectors), total=len(graphs), desc="Computing heuristic DS solutions"):
        solution = heuristic_ds_fj_depolarize(G, sigma, k=k)
        heuristic_ds_solutions.append((G, FJOpinionDynamicsFinite.polarization(solution, sigma)))
    return heuristic_ds_solutions

def compute_heuristic_cd_solutions(graphs: list[nx.Graph], sigma_vectors: list[np.ndarray], k: int):
    """
    Compute heuristic CD solutions for a list of graphs and sigma vectors.
    Returns a list of tuples (graph, polarization) after applying heuristic CD.
    """
    heuristic_cd_solutions = []
    for G, sigma in tqdm(zip(graphs, sigma_vectors), total=len(graphs), desc="Computing heuristic CD solutions"):
        solution = heuristic_cd_fj_depolarize(G, sigma, k=k)
        heuristic_cd_solutions.append((G, FJOpinionDynamicsFinite.polarization(solution, sigma)))
    return heuristic_cd_solutions

def compute_optimal_solutions(graphs: list[nx.Graph], sigma_vectors: list[np.ndarray], k: int):
    """
    Compute optimal solutions for a list of graphs and sigma vectors using exhaustive search.
    """
    optimal_results = []
    for G, sigma in tqdm(zip(graphs, sigma_vectors), total=len(graphs), desc="Computing optimal solutions"):
        optimal_solution = depolarize_optimal(
            G, sigma, k=k, polarization_function=FJOpinionDynamicsFinite.polarization
        )
        optimal_results.append(optimal_solution)
    return optimal_results


def main():
    parser = argparse.ArgumentParser(
            description="Compute greedy solutions for FJ-Depolarize."
        )
    parser.add_argument("--num_graphs", type=int, default=1000, help="Number of graphs to generate")
    parser.add_argument("--num_nodes", type=int, default=10, help="Number of nodes in each graph")
    parser.add_argument("--k", type=int, default=3, help="Number of steps for depolarization")
    parser.add_argument("--edge_prob", type=float, default=0.4, help="Probability of edge creation in Erdos-Renyi graph")

    args = parser.parse_args()

    graphs, sigma_vectors = generate_graphs_erdos_renyi(args.num_graphs, args.num_nodes, args.edge_prob)

    no_heuristic_results = [(G, FJOpinionDynamicsFinite.polarization(G, sigma)) for G, sigma in zip(graphs, sigma_vectors)]
    greedy_results = compute_greedy_solutions(graphs, sigma_vectors, k=args.k)
    heuristic_ds_results = compute_heuristic_ds_solutions(graphs, sigma_vectors, k=args.k)
    heuristic_cd_results = compute_heuristic_cd_solutions(graphs, sigma_vectors, k=args.k)
    optimal_results = compute_optimal_solutions(graphs, sigma_vectors, k=args.k)

    # Save results
    save_path = "results/heuristics/"
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "greedy_results.pkl"), "wb") as f:
        pickle.dump(greedy_results, f)

    with open(os.path.join(save_path, "heuristic_ds_results.pkl"), "wb") as f:
        pickle.dump(heuristic_ds_results, f)

    with open(os.path.join(save_path, "heuristic_cd_results.pkl"), "wb") as f:
        pickle.dump(heuristic_cd_results, f)

    with open(os.path.join(save_path, "optimal_results.pkl"), "wb") as f:
        pickle.dump(optimal_results, f)

    # Summary statistics
    epsilon = 1e-7

    no_heuristic_polarizations = [result[1] for result in no_heuristic_results]
    greedy_polarizations = [result[1] for result in greedy_results]
    heuristic_ds_polarizations = [result[1] for result in heuristic_ds_results]
    heuristic_cd_polarizations = [result[1] for result in heuristic_cd_results]
    optimal_polarizations = [result[1] for result in optimal_results]

    no_heuristic_arr = np.array(no_heuristic_polarizations)
    greedy_arr = np.array(greedy_polarizations)
    ds_arr = np.array(heuristic_ds_polarizations)
    cd_arr = np.array(heuristic_cd_polarizations)
    optimal_arr = np.array(optimal_polarizations)

    no_heuristic_opt = np.sum(optimal_arr + epsilon > no_heuristic_arr)
    greedy_opt = np.sum(optimal_arr + epsilon > greedy_arr)
    ds_opt = np.sum(optimal_arr + epsilon > ds_arr)
    cd_opt = np.sum(optimal_arr + epsilon > cd_arr)

    summary = {
        "no_heuristic_mean": float(np.mean(no_heuristic_arr)),
        "greedy_mean": float(np.mean(greedy_arr)),
        "heuristic_ds_mean": float(np.mean(ds_arr)),
        "heuristic_cd_mean": float(np.mean(cd_arr)),
        "optimal_mean": float(np.mean(optimal_arr)),
        "no_heuristic_optimal": int(no_heuristic_opt),
        "greedy_optimal": int(greedy_opt),
        "heuristic_ds_optimal": int(ds_opt),
        "heuristic_cd_optimal": int(cd_opt),
    }

    with open(os.path.join(save_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()