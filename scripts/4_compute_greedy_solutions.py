#!/usr/bin/env python3
"""
Greedy Baseline Computation for FJ-OffDP
--------------------------------------

This script computes greedy baseline solutions for the Friedkin–Johnsen
offline depolarization problem (FJ-OffDP).

Workflow:
1. Loads start states (generated via `3_make_dataset.py`) from the specified
   folder (`val`, or `test`).
2. Applies the FJ Greedy Depolarize algorithm for each state and
   each specified intervention budget k. (Chapter 3.3 in the thesis.)
3. Saves the resulting graphs and polarization scores to
   `data/friedkin-johnson/greedy_solutions`.

Usage:
--n_values: List of graph sizes (n) to process (default: [100, 150, 200, 300, 400]) (must match generated test/val states)
--k_values: List of intervention budgets (k) to process (default: [5, 10, 15, 20])
--folder: Folder containing the start states (`val` or `test`, default: `test`)
--average_degree: Average degree of the graphs to consider (default: 6) (must match generated test/val states)

"""
import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import argparse

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from env import FJOpinionDynamics
from evaluation import greedy_fj_depolarize


def compute_greedy_solutions(
    k_values: list, n_values: list, folder: str, average_degree: int = 6
):
    """
    Compute and save greedy solutions for FJ-OffDP for given n values and k values
    for validation and test sets.
    The states must be generated first via make_dataset.py.
    """
    greedy_solutions_path = os.path.join("data", "friedkin-johnson", "greedy_solutions")

    for n in n_values:
        for k in tqdm(
            k_values,
            desc=f"Computing greedy solutions ({folder} sets) for n={n} and k in {k_values}",
        ):
            with open(
                os.path.join(
                    "data",
                    "friedkin-johnson",
                    folder,
                    f"start_states_{folder}_n{n}_d{average_degree}.pt",
                ),
                "rb",
            ) as f:
                states = torch.load(f, weights_only=False)
            greedy_solutions = []
            for state in states:
                G, sigma = state["graph"], state["sigma"]
                G_greedy = greedy_fj_depolarize(G, sigma, k)
                greedy_solutions.append(
                    (state, FJOpinionDynamics.polarization(G_greedy, sigma))
                )
            save_path = os.path.join(
                greedy_solutions_path,
                folder,
                f"greedy_solutions_n{n}_d{average_degree}_k{k}.pt",
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                torch.save(greedy_solutions, f)


def main():
    parser = argparse.ArgumentParser(
        description="Compute greedy solutions for FJ-Depolarize."
    )
    parser.add_argument(
        "--n_values",
        type=int,
        nargs="+",
        default=[100, 150, 200, 300, 400],
        help="List of n values for which to compute solutions.",
    )
    parser.add_argument(
        "--average_degree",
        type=int,
        default=6,
        help="Average degree of the graphs.",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20],
        help="List of k values for which to compute solutions.",
    )
    parser.add_argument(
        "--folder",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Folder containing the start states.",
    )

    args = parser.parse_args()

    compute_greedy_solutions(
        args.k_values, args.n_values, args.folder, average_degree=args.average_degree
    )


if __name__ == "__main__":
    main()
