#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from evaluation import test_and_save_baselines
from visualization import visualize_polarization_development_multiple_policies


def evaluate_baseline_strategies(
    params_env: dict, n_values: list, n_steps: int = 20000, folder: str = "val"
):
    test_and_save_baselines(
        params_env=params_env, n_values=n_values, n_steps=n_steps, folder=folder
    )
    visualize_polarization_development_multiple_policies(
        params_env=params_env, folder=folder
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline strategies for NL-DepolarizeOnline."
    )

    parser.add_argument(
        "--n_values",
        type=int,
        nargs="+",
        default=[100, 200, 300, 400],
        help="List of node counts to evaluate.",
    )

    parser.add_argument(
        "--average_degree",
        type=int,
        default=6,
        help="Average degree of the network.",
    )

    parser.add_argument(
        "--n_edge_updates_per_step",
        type=int,
        default=5,
        help="Number of edge updates in nonlinear opinion dynamics (only relevant for NL-DepolarizeOnline).",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=20000,
        help="Number of steps to simulate.",
    )

    parser.add_argument(
        "--folder",
        type=str,
        default="test",
        help="Folder of states to evaluate.",
    )

    args = parser.parse_args()

    params_env = {
        "environment": "nonlinear",
        "average_degree": args.average_degree,
        "n_edge_updates_per_step": args.n_edge_updates_per_step,
    }

    evaluate_baseline_strategies(
        params_env, n_values=args.n_values, n_steps=args.n_steps, folder=args.folder
    )


if __name__ == "__main__":
    main()
