import networkx as nx
import numpy as np
from evaluation import (
    greedy_fj_depolarize,
    heuristic_ds_fj_depolarize,
    heuristic_cd_fj_depolarize,
    depolarize_optimal,
)
from env import FJOpinionDynamics
from tqdm import tqdm
import pickle


if __name__ == "__main__":

    num_graphs = 1000
    num_nodes = 150

    graphs = []
    sigma_vectors = []

    for _ in range(num_graphs):
        # Generate a random graph (Erdos-Renyi)
        G = nx.erdos_renyi_graph(n=num_nodes, p=0.4)
        graphs.append(G)

        # Generate a random sigma vector with values between -1 and 1
        sigma = np.random.uniform(-1, 1, size=num_nodes)
        while np.all(sigma == 0) or np.all(sigma == 1):
            sigma = np.random.uniform(-1, 1, size=num_nodes)
        sigma_vectors.append(sigma)

    greedy_results = []
    heuristic_ds_results = []
    heuristic_cd_results = []
    # optimal_results = []

    for G, sigma in tqdm(zip(graphs, sigma_vectors)):
        # Compute greedy solution
        greedy_solution = greedy_fj_depolarize(G, sigma, k=10)
        greedy_results.append(
            (G, FJOpinionDynamics.polarization(greedy_solution, sigma))
        )

        # Compute heuristic solutions
        heuristic_ds_solution = heuristic_ds_fj_depolarize(G, sigma, k=10)
        heuristic_ds_results.append(
            (G, FJOpinionDynamics.polarization(heuristic_ds_solution, sigma))
        )
        heuristic_cd_solution = heuristic_cd_fj_depolarize(G, sigma, k=10)
        heuristic_cd_results.append(
            (G, FJOpinionDynamics.polarization(heuristic_cd_solution, sigma))
        )

        # Compute optimal solution
        # optimal_solution = depolarize_optimal(
        #     G, sigma, k=3, polarization_function=FJOpinionDynamics.polarization
        # )
        # optimal_results.append(optimal_solution)

    # --- Save results ---
    with open("greedy_results.pkl", "wb") as f:
        pickle.dump(greedy_results, f)

    with open("heuristic_ds_results.pkl", "wb") as f:
        pickle.dump(heuristic_ds_results, f)

    with open("heuristic_cd_results.pkl", "wb") as f:
        pickle.dump(heuristic_cd_results, f)

    # with open("optimal_results.pkl", "wb") as f:
    #     pickle.dump(optimal_results, f)
