#!/usr/bin/env python3
import argparse
import os
import torch
import json
import sys
from pathlib import Path

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from env import ENVIRONMENT_REGISTRY
from agents.dqn import DQN
from evaluation import (
    evaluate_dqn_policy_vs_greedy_ood_n,
    test_and_save_policy_dqn,
)
from visualization import (
    visualize_dqn_vs_greedy_ood_n,
    visualize_dqn_vs_greedy_ood_n_simple,
    visualize_variance_ood_n,
    visualize_dqn_vs_greedy,
    performance_overview,
    visualize_polarization_development_multiple_policies,
)


def run_training(params_env, params_agent):

    env = ENVIRONMENT_REGISTRY[params_env["environment"]](**params_env)

    agent = DQN(env=env, **params_agent)
    agent.train()

    if agent.run_name:
        # Define run-specific folder
        run_dir = os.path.join(
            "results",
            "dqn",
            params_env["environment"],
            "runs",
            agent.run_name,
        )
        os.makedirs(run_dir, exist_ok=True)

        # Save full parameter dict
        with open(os.path.join(run_dir, "params_env.json"), "w") as f:
            json.dump(params_env, f, indent=4)

        with open(os.path.join(run_dir, "params_agent.json"), "w") as f:
            json.dump(params_agent, f, indent=4)

        # Save model parameters
        torch.save(
            agent.q_network.state_dict(), os.path.join(run_dir, "q_network_params.pth")
        )
        torch.save(
            agent.target_network.state_dict(),
            os.path.join(run_dir, "target_network_params.pth"),
        )

        return agent.run_name


def continue_training(run_name, timesteps_train):

    # Find directory of run
    if os.path.exists(
        os.path.join("results", "dqn", "friedkin-johnson", "runs", run_name)
    ):
        run_dir = os.path.join("results", "dqn", "friedkin-johnson", "runs", run_name)
    else:
        run_dir = os.path.join("results", "dqn", "nonlinear", "runs", run_name)

    # Load parameters
    with open(os.path.join(run_dir, "params_env.json"), "r") as f:
        params_env = json.load(f)
    with open(os.path.join(run_dir, "params_agent.json"), "r") as f:
        params_agent = json.load(f)

    env = ENVIRONMENT_REGISTRY[params_env["environment"]](**params_env)
    agent = DQN(env=env, run_name=f"{run_name}_+", **params_agent)
    agent.timesteps_train = timesteps_train

    # Load saved model weights
    q_net_path = os.path.join(run_dir, "q_network_params.pth")
    target_net_path = os.path.join(run_dir, "target_network_params.pth")

    # Load model weights into agent
    agent.q_network.load_state_dict(
        torch.load(q_net_path, map_location=torch.device("cpu"))
    )
    agent.target_network.load_state_dict(
        torch.load(target_net_path, map_location=torch.device("cpu"))
    )
    agent.train()

    if agent.run_name:
        # Define run-specific folder
        run_dir = os.path.join(
            "results",
            "dqn",
            params_env["environment"],
            "runs",
            agent.run_name,
        )
        os.makedirs(run_dir, exist_ok=True)

        # Save full parameter dict
        with open(os.path.join(run_dir, "params_env.json"), "w") as f:
            json.dump(params_env, f, indent=4)

        with open(os.path.join(run_dir, "params_agent.json"), "w") as f:
            json.dump(params_agent, f, indent=4)

        # Save model parameters
        torch.save(
            agent.q_network.state_dict(), os.path.join(run_dir, "q_network_params.pth")
        )
        torch.save(
            agent.target_network.state_dict(),
            os.path.join(run_dir, "target_network_params.pth"),
        )

        return agent.run_name


def rerun_training(run_name, number_of_reruns=1):

    # Find directory of run
    if os.path.exists(
        os.path.join("results", "dqn", "friedkin-johnson", "runs", run_name)
    ):
        run_dir = os.path.join("results", "dqn", "friedkin-johnson", "runs", run_name)
    else:
        run_dir = os.path.join("results", "dqn", "nonlinear", "runs", run_name)

    # Load parameters
    with open(os.path.join(run_dir, "params_env.json"), "r") as f:
        params_env = json.load(f)
    with open(os.path.join(run_dir, "params_agent.json"), "r") as f:
        params_agent = json.load(f)

    for i in range(number_of_reruns):
        # Initialize environment and agent
        env = ENVIRONMENT_REGISTRY[params_env["environment"]](**params_env)
        agent = DQN(env=env, run_name=f"{run_name}_rerun_{i}", **params_agent)
        agent.train()

        # Define run-specific folder
        run_dir = os.path.join(
            "results",
            "dqn",
            params_env["environment"],
            "runs",
            run_name,
            "reruns",
            f"rerun_{i}",
        )
        os.makedirs(run_dir, exist_ok=True)

        # Save full parameter dict
        with open(os.path.join(run_dir, "params_env.json"), "w") as f:
            json.dump(params_env, f, indent=4)
        with open(os.path.join(run_dir, "params_agent.json"), "w") as f:
            json.dump(params_agent, f, indent=4)

        # Save model parameters
        torch.save(
            agent.q_network.state_dict(), os.path.join(run_dir, "q_network_params.pth")
        )
        torch.save(
            agent.target_network.state_dict(),
            os.path.join(run_dir, "target_network_params.pth"),
        )


def evaluate_run_fj_depolarize(run_name, n_values, k_values, n, k, folder="val"):
    evaluate_dqn_policy_vs_greedy_ood_n(
        run_name=run_name,
        n_values=n_values,
        k_values=k_values,
        folder=folder,
    )
    visualize_dqn_vs_greedy(run_name, n=n, k=k, folder=folder)
    visualize_dqn_vs_greedy_ood_n(run_name, folder=folder)
    visualize_dqn_vs_greedy_ood_n_simple(run_name, folder=folder)
    performance_overview(run_name, folder=folder)
    visualize_variance_ood_n(run_name, folder=folder)


def evaluate_run_nonlinear(run_name, n_values, n_steps=20000, folder="val"):
    test_and_save_policy_dqn(run_name, n_values, n_steps=n_steps, folder=folder)
    visualize_polarization_development_multiple_policies(
        run_name=run_name, folder=folder
    )    


def main():
    parser = argparse.ArgumentParser(
        description="Train DQN for FJ-Depolarize or NL-DepolarizeOnline and evaluate on validation sets."
    )

    # Environment parameters
    parser.add_argument(
        "--environment",
        type=str,
        choices=["friedkin-johnson", "nonlinear"],
        default="friedkin-johnson",
        help="Environment to train on.",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of nodes in the graph during training.",
    )

    parser.add_argument(
        "--average_degree",
        type=int,
        default=6,
        help="Average degree of the graph during training.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=15,
        help="Number of edges to change (only relevant for FJ-Depolarize).",
    )

    parser.add_argument(
        "--n_edge_updates_per_step",
        type=int,
        default=5,
        help="Number of edge updates in nonlinear opinion dynamics (only relevant for NL-DepolarizeOnline).",
    )

    parser.add_argument(
        "--keep_resistance_matrix",
        action="store_true",
        help="Whether to keep the resistance matrix instead of the fundamental matrix.",
    )

    parser.add_argument(
        "--keep_influence_matrix",
        action="store_true",
        help="Whether to keep the influence matrix (needed for Graphormer) (in fj environment is automatically true).",
    )

    # Agent parameters
    parser.add_argument(
        "--wandb_init",
        action="store_true",
        help="Whether to initialize wandb for logging.",
    )
    parser.add_argument(
        "--gnn",
        type=str,
        choices=["Global", "GraphSage", "Graphormer", "GCN"],
        default="GraphSage",
        help="Type of GNN to use.",
    )
    parser.add_argument(
        "--qnet",
        type=str,
        choices=["simple", "complex"],
        default="complex",
        help="Type of Q-network to use.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0004,
        help="Learning rate for the agent.",
    )
    parser.add_argument(
        "--embed_dim", type=int, default=128, help="Embedding dimension for the GNN."
    )
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of layers in the GNN."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0, help="Discount factor for DQN."
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Number of attention heads (for Graphormer).",
    )
    parser.add_argument(
        "--reset_probability",
        type=float,
        default=None,
        help="Probability to reset environment (optional).",
    )
    parser.add_argument(
        "--parallel_envs", type=int, default=1, help="Number of parallel environments."
    )

    parser.add_argument(
        "--train_freq",
        type=int,
        default=4,
        help="Frequency of training steps (default is every 4 steps).",
    )

    parser.add_argument(
        "--end_e", type=float, default=1.0, help="End epsilon for exploration."
    )
    parser.add_argument(
        "--target_update_freq",
        type=int,
        default=100000,
        help="Frequency of target network updates.",
    )
    parser.add_argument(
        "--timesteps_train",
        type=int,
        default=300000,
        help="Number of timesteps to train.",
    )

    parser.add_argument(
        "--td_loss_one_edge",
        action="store_true",
        help="Whether to compute the TD loss based on the reward for adding one edge (instead of the full action).",
    )

    parser.add_argument(
        "--record_opinions_while_training",
        action="store_true",
        help="Whether to record opinions while training.",
    )

    # parameters for evaluation
    parser.add_argument(
        "--n_values",
        type=int,
        nargs="+",
        default=[100, 150, 200, 300, 400],
        help="List of n values for which to evaluate.",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[10, 15, 20, 25, 30],
        help="List of k values for which to evaluate (only relevant for FJ-Depolarize).",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="val",
        help="Folder of states to evaluate (default is 'val').",
    )

    # parameters for rerunning training
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name of the run to rerun, if None a new run will be created.",
    )

    parser.add_argument(
        "--number_of_reruns",
        type=int,
        default=None,
        help="Only relevant if rerunning a training.",
    )

    args = parser.parse_args()

    if args.run_name is not None:
        rerun_training(args.run_name, args.number_of_reruns)
        if args.environment == "friedkin-johnson":
            evaluate_run_fj_depolarize(
                run_name=args.run_name,
                n_values=args.n_values,
                k_values=args.k_values,
                n=args.n,
                k=args.k,
                folder=args.folder,
            )
        return

    params_env = {
        "environment": args.environment,
        "start_states": os.path.join(
            "data",
            args.environment,
            "train",
            f"start_states_train_n{args.n}_d{args.average_degree}.pt",
        ),
        "n": args.n,
        "average_degree": args.average_degree,
        "n_edge_updates_per_step": args.n_edge_updates_per_step,
        "keep_resistance_matrix": args.keep_resistance_matrix,
        "keep_influence_matrix": args.keep_influence_matrix,
        "k": args.k,
    }

    params_agent = {
        "gnn": args.gnn,
        "qnet": args.qnet,
        "learning_rate": args.learning_rate,
        "td_loss_one_edge": args.td_loss_one_edge,
        "embed_dim": args.embed_dim,
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "num_heads": args.num_heads,
        "wandb_init": args.wandb_init,
        "reset_probability": args.reset_probability,
        "parallel_envs": args.parallel_envs,
        "end_e": args.end_e,
        "train_freq": args.train_freq,
        "target_update_freq": args.target_update_freq,
        "timesteps_train": args.timesteps_train,
        "record_opinions_while_training": args.record_opinions_while_training,
    }

    # Run training
    run_name = run_training(params_env=params_env, params_agent=params_agent)
    # run_name = "Graphormer-complex-n10-k3-hd128-layers4-lr0.0004-heads4-bs64-p1-g1.0-tuf100000-EQBEB"
    # visualize_variance_ood_n_simple(run_name, folder=args.folder)
    # continue_training(run_name=run_name, timesteps_train=args.timesteps_train)
    if args.number_of_reruns is not None:
        rerun_training(run_name=run_name, number_of_reruns=args.number_of_reruns)
    if args.environment == "friedkin-johnson":
        # Evaluate and visualize results
        evaluate_run_fj_depolarize(
            run_name=run_name,
            n_values=args.n_values,
            k_values=args.k_values,
            n=args.n,
            k=args.k,
            folder=args.folder,
        )

    elif args.environment == "nonlinear":
        # Evaluate and visualize results
        if not args.record_opinions_while_training: # if we recorded opinions while training, no are not interested in evaluation on other graphs
            evaluate_run_nonlinear(
                run_name=run_name,
                n_values=args.n_values,
                folder=args.folder,
            )


if __name__ == "__main__":
    main()
