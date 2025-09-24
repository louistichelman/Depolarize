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


def generate_graphs_erdos_renyi(num_graphs, num_nodes, edge_prob):
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

def compute_greedy_solutions(graphs, sigma_vectors, k):
    greedy_solutions = []
    for G, sigma in tqdm(zip(graphs, sigma_vectors), total=len(graphs), desc="Computing greedy solutions"):
        solution = greedy_fj_depolarize(G, sigma, k=k)
        greedy_solutions.append((G, FJOpinionDynamicsFinite.polarization(solution, sigma)))
    return greedy_solutions

def compute_heuristic_ds_solutions(graphs, sigma_vectors, k):
    heuristic_ds_solutions = []
    for G, sigma in tqdm(zip(graphs, sigma_vectors), total=len(graphs), desc="Computing heuristic DS solutions"):
        solution = heuristic_ds_fj_depolarize(G, sigma, k=k)
        heuristic_ds_solutions.append((G, FJOpinionDynamicsFinite.polarization(solution, sigma)))
    return heuristic_ds_solutions

def compute_heuristic_cd_solutions(graphs, sigma_vectors, k):
    heuristic_cd_solutions = []
    for G, sigma in tqdm(zip(graphs, sigma_vectors), total=len(graphs), desc="Computing heuristic CD solutions"):
        solution = heuristic_cd_fj_depolarize(G, sigma, k=k)
        heuristic_cd_solutions.append((G, FJOpinionDynamicsFinite.polarization(solution, sigma)))
    return heuristic_cd_solutions

def compute_optimal_solutions(graphs, sigma_vectors, k):
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
    # optimal_results = compute_optimal_solutions(graphs, sigma_vectors, k=args.k)

    # Save results
    save_path = "results/heuristics/"
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "greedy_results.pkl"), "wb") as f:
        pickle.dump(greedy_results, f)

    with open(os.path.join(save_path, "heuristic_ds_results.pkl"), "wb") as f:
        pickle.dump(heuristic_ds_results, f)

    with open(os.path.join(save_path, "heuristic_cd_results.pkl"), "wb") as f:
        pickle.dump(heuristic_cd_results, f)

    # with open(os.path.join(save_path, "optimal_results.pkl"), "wb") as f:
    #     pickle.dump(optimal_results, f)

    # Summary statistics
    epsilon = 1e-7

    no_heuristic_polarizations = [result[1] for result in no_heuristic_results]
    greedy_polarizations = [result[1] for result in greedy_results]
    heuristic_ds_polarizations = [result[1] for result in heuristic_ds_results]
    heuristic_cd_polarizations = [result[1] for result in heuristic_cd_results]
    # optimal_polarizations = [result[1] for result in optimal_results]

    no_heuristic_arr = np.array(no_heuristic_polarizations)
    greedy_arr = np.array(greedy_polarizations)
    ds_arr = np.array(heuristic_ds_polarizations)
    cd_arr = np.array(heuristic_cd_polarizations)
    # optimal_arr = np.array(optimal_polarizations)

    # no_heuristic_opt = np.sum(optimal_arr + epsilon > no_heuristic_arr)
    # greedy_opt = np.sum(optimal_arr + epsilon > greedy_arr)
    # ds_opt = np.sum(optimal_arr + epsilon > ds_arr)
    # cd_opt = np.sum(optimal_arr + epsilon > cd_arr)

    summary = {
        "no_heuristic_mean": float(np.mean(no_heuristic_arr)),
        "greedy_mean": float(np.mean(greedy_arr)),
        "heuristic_ds_mean": float(np.mean(ds_arr)),
        "heuristic_cd_mean": float(np.mean(cd_arr)),
        # "optimal_mean": float(np.mean(optimal_arr)),
        # "no_heuristic_optimal": int(no_heuristic_opt),
        # "greedy_optimal": int(greedy_opt),
        # "heuristic_ds_optimal": int(ds_opt),
        # "heuristic_cd_optimal": int(cd_opt),
    }

    with open(os.path.join(save_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()