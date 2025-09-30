#!/usr/bin/env python3

"""
Generate Enhanced Training States for NL-OnDP (Training Method 1)
--------------------------------------------------------

This script augments the training dataset for nonlinear opinion dynamics (NL-OnDP)
by creating *diverse start states*. Instead of only using the original training
graphs, the script simulates opinion dynamics for a number of steps and
records states at specified time offsets. (Training Method 1 in Chapter 5.4)

Workflow:
1. Load the original training start states.
2. Simulate the nonlinear dynamics forward in time for each state.
3. At the specified `start_time_values`, save a copy of the current state
   (capturing networks at different polarization levels).
4. Store all collected states as a new enhanced training set and save at data/nonlinear/train.

Usage:
--env: Environment type, either 'friedkin-johnson' or 'nonlinear' (I only used: 'nonlinear')
--n: Graph size (default: 150) (must match generated train states)
--average_degree: Average degree of the graph (default: 6) (must match generated train states)
--n_edge_updates_per_step: Number of edge updates in nonlinear opinion dynamics (default: 4)
--start_time_values: List of time steps at which to capture states (default: [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000])

The resulting dataset provides a richer variety of polarization levels
and network structures, which improves training robustness for DQN.
"""
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
            f"start_states_diverse_train_n{params_env['n']}_d{params_env['average_degree']}.pt",
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
