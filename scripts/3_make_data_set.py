"""
Dataset Generation for DQN Experiments
--------------------------------------

This script generates training, validation, and test start states for 
depolarization experiments under either:
    - The FJ-OffDP, FJOpinionDynamics environment
    - The NL-OnDP, NonlinearOpinionDynamics environment

Functionality:
- Uses the environment registry (ENVIRONMENT_REGISTRY) to instantiate the chosen env.
- Randomly generates unique start states (graph + opinion vector + other env info).
- Ensures no duplicate states (via state_hash).
- Saves datasets as PyTorch `.pt` files in `data/{env}/train`, `val`, and `test`.
- Optionally generates out-of-distribution (OOD) datasets for different graph sizes.

Dataset structure:
    data/
      friedkin-johnson/ or nonlinear/
        train/start_states_train_n{n}_d{degree}.pt
        val/start_states_val_n{n}_d{degree}.pt
        test/start_states_test_n{n}_d{degree}.pt
        (optional OOD splits saved in val/ and test/)

Usage:
--env: Environment type ("friedkin-johnson" or "nonlinear").
--n: Graph size for in-distribution datasets.
--average_degree: Average degree of generated graphs.
--n_train: Number of training states to generate.
--n_val: Number of validation states to generate.
--n_test: Number of testing states to generate.
--out_of_distribution_n: List of graph sizes for OOD test/val states.
--seed: Random seed for reproducibility.

Notes:
- Files are stored in PyTorch tensor format (`.pt`).
- Generated datasets are used as start states for DQN training and evaluation.
"""


#!/usr/bin/env python3
import argparse
import os
import torch
import random
import sys
from pathlib import Path
import json

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from env import ENVIRONMENT_REGISTRY, BaseEnv


def generate_random_states(env: BaseEnv, num_states: int = 150, seen: set = None):
    """
    Generate random test states for the given environment.
    """
    if seen is None:
        seen = set()

    states = []
    while len(states) < num_states:
        state = env.reset()
        # Use the state_hash method of the Base Environment to ensure uniqueness
        state_hash = BaseEnv.state_hash(state["graph"], state["sigma"], state["tau"])
        if state_hash not in seen:
            seen.add(state_hash)
            states.append(state)

    return states, seen


def main():
    parser = argparse.ArgumentParser(
        description="Generate datasets splits for training and testing FJ-Depolarize or NL-DepolarizeOnline."
    )

    parser.add_argument(
        "--env",
        type=str,
        choices=["friedkin-johnson", "nonlinear"],
        required=True,
        help="Environment type",
    )

    # graph-specific parameters
    parser.add_argument("--n", type=int, default=150, help="Graph size (passed to env)")

    parser.add_argument(
        "--average_degree",
        type=int,
        default=6,
        help="Average degree of the graph",
    )

    # train-val-test split parameters
    parser.add_argument(
        "--n_train",
        type=int,
        default=100,
        help="Number of training start states to generate",
    )
    parser.add_argument(
        "--out_of_distribution_n",
        type=int,
        nargs="+",
        help="List of out-of-distribution n values for generating test states",
        default=None,
    )
    parser.add_argument(
        "--n_val",
        type=int,
        default=100,
        help="Number of validation start states to generate",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=100,
        help="Number of testing start states to generate",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create save path based on environment type
    if args.env == "friedkin-johnson":
        env_dir = os.path.join("data", "friedkin-johnson")
    else:
        env_dir = os.path.join("data", "nonlinear")

    # Create environment
    env = ENVIRONMENT_REGISTRY[args.env](**vars(args))

    start_states_train, seen = generate_random_states(env, num_states=args.n_train)

    save_path = os.path.join(env_dir, "train")
    os.makedirs(save_path, exist_ok=True)
    torch.save(
        start_states_train,
        os.path.join(
            save_path, f"start_states_train_n{args.n}_d{args.average_degree}.pt"
        ),
    )

    print(f"Saved {len(start_states_train)} start states to {save_path}")


    start_states_val, seen = generate_random_states(
        env, num_states=args.n_val, seen=seen
    )

    save_path = os.path.join(env_dir, "val")
    os.makedirs(save_path, exist_ok=True)
    torch.save(
        start_states_val,
        os.path.join(
            save_path, f"start_states_val_n{args.n}_d{args.average_degree}.pt"
        ),
    )

    print(f"Saved {len(start_states_val)} start states to {save_path}")

    start_states_test, _ = generate_random_states(
        env, num_states=args.n_test, seen=seen
    )

    save_path = os.path.join(env_dir, "test")
    os.makedirs(save_path, exist_ok=True)
    torch.save(
        start_states_test,
        os.path.join(
            save_path, f"start_states_test_n{args.n}_d{args.average_degree}.pt"
        ),
    )

    print(f"Saved {len(start_states_test)} start states to {save_path}")

    if args.out_of_distribution_n is not None:
        print(f"Generating OOD test states ...")
        for n in args.out_of_distribution_n:
            env.n = n
            ood_states_val, seen = generate_random_states(env, num_states=args.n_val)

            torch.save(
                ood_states_val,
                os.path.join(
                    env_dir, "val", f"start_states_val_n{n}_d{args.average_degree}.pt"
                ),
            )

            ood_states_test, _ = generate_random_states(
                env, num_states=args.n_test, seen=seen
            )

            torch.save(
                ood_states_test,
                os.path.join(
                    env_dir, "test", f"start_states_test_n{n}_d{args.average_degree}.pt"
                ),
            )

        print(f"Saved OOD val and test states to {env_dir}")


if __name__ == "__main__":
    main()
