#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import torch
import os
from tqdm import tqdm
import random

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from env import ENVIRONMENT_REGISTRY


def set_off_start_states(params_env: dict, start_time_values: list):
    """
    Let start states be set off at the given start time values in the nonlinear environment.
    This is used to generate start states that cover later states of the network dynamics.
    """
    train_states_path = os.path.join("data", "nonlinear", "train")
    start_states = torch.load(
        os.path.join(
            train_states_path,
            f"start_states_train_n{params_env['n']}_d{params_env['average_degree']}.pt",
        ),
        weights_only=False,
    )

    env = ENVIRONMENT_REGISTRY[params_env["environment"]](**params_env)

    start_states_off = []
    number_of_nodes = params_env["n"]
    for state in tqdm(
        start_states, desc="Generating start states set off at specific times"
    ):
        # Reset the environment to
        for i in range(max(start_time_values) + 2):
            if i in start_time_values:
                start_states_off.append(state.copy())
            action = random.choice(range(number_of_nodes))
            next_state, _, _ = env.step(action=action, state=state)
            state = next_state

    torch.save(
        start_states_off,
        os.path.join(
            train_states_path,
            f"start_states_off_4_train_n{params_env['n']}_d{params_env['average_degree']}.pt",
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate start states for NL-DepolarizeOnline that are set off at specific start times."
    )

    parser.add_argument(
        "--env",
        type=str,
        choices=["friedkin-johnson", "nonlinear"],
        default="nonlinear",
        help="Environment type: 'fj' for FJ-Depolarize or 'nl' for NL-DepolarizeOnline",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=150,
        help="Graph size (passed to env)",
    )

    parser.add_argument(
        "--average_degree",
        type=int,
        default=6,
        help="Average degree of the graph",
    )

    parser.add_argument(
        "--n_edge_updates_per_step",
        type=int,
        default=5,
        help="Number of edge updates in nonlinear opinion dynamics (only relevant for NL-DepolarizeOnline).",
    )

    parser.add_argument(
        "--start_time_values",
        type=int,
        nargs="+",
        default=[
            0,
            2000,
            4000,
            6000,
            8000,
            10000,
            12000,
            14000,
            16000,
            18000,
            20000,
        ],
        help="List of start time values to set off the start states.",
    )

    args = parser.parse_args()

    params_env = {
        "environment": args.env,
        "n": args.n,
        "average_degree": args.average_degree,
        "n_edge_updates_per_step": args.n_edge_updates_per_step,
    }

    set_off_start_states(
        params_env=params_env, start_time_values=args.start_time_values
    )


if __name__ == "__main__":
    main()
