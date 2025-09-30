#!/usr/bin/env python3
"""
Evaluate Baseline Strategies for NL-OnDP
---------------------------------------------------

This script computes and evaluates baseline strategies in the depolarizing the nonlinear
opinion dynamics environment (see Chapter 4.2). It performs the following steps:

1. Load validation or test states generated beforehand.
2. Run baseline strategies (no intervention, minmax, soft minmax, deleting)
   on these states over a fixed number of simulation steps.
3. Save the recorded opinions and graph metrics for each baseline.
4. Visualize polarization development over time, comparing the baselines.

Usage:
--n_values: List of graph sizes (n) to evaluate (default: [100, 200, 300, 400]) (must match generated states)
--n_steps: Number of simulation steps to run for each state (default: 20000) (steps/2 equals time horizon T, since one steps chooses one endnode)
--folder: Folder containing the states to evaluate (`val` or `test`, default: `val`)
--average_degree: Average degree of the network (default: 6) (must match generated states)
--n_edge_updates_per_step: Number of edge updates in nonlinear opinion dynamics (default: 4)

Results are stored in the `data/nonlinear/baselines` directory.
"""


import argparse
import sys
from pathlib import Path

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from evaluation import test_and_save_baselines
from visualization import visualize_polarization_development_dqn_and_baselines


def evaluate_baseline_strategies(
    params_env: dict, n_values: list, n_steps: int = 20000, folder: str = "val"
):
    test_and_save_baselines(
        params_env=params_env, n_values=n_values, n_steps=n_steps, folder=folder
    )
    visualize_polarization_development_dqn_and_baselines(
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
        default=4,
        help="Number of edge updates in nonlinear opinion dynamics.",
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
