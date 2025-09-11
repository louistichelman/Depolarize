#!/usr/bin/env python3
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
    Compute and save greedy solutions for FJ-Depolarize for given n values and k values
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
        default=[50, 100, 150],
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
        default=[1, 2, 3, 4, 5],
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
