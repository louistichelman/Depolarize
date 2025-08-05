from env import ENVIRONMENT_REGISTRY
from agents.dqn import DQN
from evaluation import *
from visualization import *
import torch
import os
import json


def run_training(params_env, params_agent):

    env = ENVIRONMENT_REGISTRY[params_env["environment"]](**params_env)

    agent = DQN(env=env, **params_agent)
    agent.train()

    if agent.run_name:
        # Define run-specific folder
        run_dir = os.path.join(
            "saved files",
            "dqn",
            params_env["environment"],
            "saved_runs",
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
        os.path.join("saved files", "dqn", "friedkin_johnson", "saved_runs", run_name)
    ):
        run_dir = os.path.join(
            "saved files", "dqn", "friedkin_johnson", "saved_runs", run_name
        )
    else:
        run_dir = os.path.join(
            "saved files", "dqn", "nonlinear", "saved_runs", run_name
        )

    # Load parameters
    with open(os.path.join(run_dir, "params_env.json"), "r") as f:
        params_env = json.load(f)
    with open(os.path.join(run_dir, "params_agent.json"), "r") as f:
        params_agent = json.load(f)

    # Initialize environment and agent
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

    # Resume training
    agent.train()

    # Define run-specific folder
    if agent.run_name:
        run_dir = os.path.join(
            "saved files",
            "dqn",
            params_env["environment"],
            "saved_runs",
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
        os.path.join("saved files", "dqn", "friedkin_johnson", "saved_runs", run_name)
    ):
        run_dir = os.path.join(
            "saved files", "dqn", "friedkin_johnson", "saved_runs", run_name
        )
    else:
        run_dir = os.path.join(
            "saved files", "dqn", "nonlinear", "saved_runs", run_name
        )

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
            "saved files",
            "dqn",
            params_env["environment"],
            "saved_runs",
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


# ------ Evaluation FJ-Depolarize ------
def generate_test_states_and_greedy_solutions_fj(n_values, k_values):

    test_states = generate_random_test_states_fj_env(n_values=n_values, num_states=200)
    compute_greedy_solutions(
        k_values=k_values,
        test_states=test_states,
    )


def evaluate_run_fj(run_name, n_values, k_values, single_setting=False, n=None, k=None):
    if single_setting:
        compare_dqn_policy_to_greedy_single_setting(run_name, n=n, k=k)
        visualize_comparison_dqn_vs_greedy_single_setting(run_name)
    else:
        evaluate_dqn_policy_vs_greedy_various_n(
            run_name=run_name,
            n_values=n_values,
            k_values=k_values,
        )
        visualize_comparison_dqn_vs_greedy(run_name)
        performance_overview_run(run_name)


# ------ Evaluation NL-DepolarizeOnline ------
def evaluate_baseline_strategies(params_env, n_steps=20000, n_simpulations=15):
    test_and_save_baselines(
        params=params_env, n_steps=n_steps, n_simpulations=n_simpulations
    )
    visualize_polarization_development_multiple_policies(params=params_env)


def evaluate_run_nonlinear(run_name, n_steps=20000, n_simpulations=15):
    test_and_save_policy_dqn(run_name, n_steps=n_steps, n_simpulations=n_simpulations)
    visualize_polarization_development_multiple_policies(run_name=run_name)


if __name__ == "__main__":
    params_env = {
        "environment": "friedkin_johnson",  # friedkin_johnson or nonlinear
        "n": 100,
        "average_degree": 5,  # 6 for 200, 4 for 50, 4 for 25
        "n_edge_updates_per_step": 5,  # 5 for 200, 2 for 50, 1 for 25
        "keep_spd_matrix": True,  # Keep shortest path distance matrix
        "k": 10,
    }
    params_agent = {
        "gnn": "Graphormer",  # "Global", "GraphSage", "Graphormer"
        "qnet": "complex",  # "simple", "complex"
        "learning_rate": 0.0004,
        "embed_dim": 128,
        "num_layers": 4,
        "batch_size": 64,
        "gamma": 1,
        "num_heads": 4,
        "wandb_init": True,
        "reset_probability": None,  # 0.0001
        "parallel_envs": 1,
        "end_e": 1,  # End epsilon for exploration
        "target_update_freq": 100000,  # Frequency of target network updates
        "timesteps_train": 300000,  # Number of timesteps to train
    }

    # generate_test_states_and_greedy_solutions_fj(
    #     n_values=[150], k_values=[5, 10, 15, 20]
    # )

    # visualize_comparison_dqn_vs_greedy_simple(
    #     "GraphSage-complex-n100-k10-hd128-layers4-lr0.0004-heads0-bs64-tuf100000-YGFP9"
    # )
    run_name = (
        "GraphSage-complex-n175-k15-hd128-layers5-lr0.0004-heads0-bs64-tuf100000-3VDDT"
    )
    # visualize_comparison_dqn_vs_greedy_simple(run_name)
    compare_dqn_policy_to_greedy_single_setting(run_name, n=200, k=20)
    visualize_comparison_dqn_vs_greedy_single_setting(run_name)

    # run_name = run_training(params_env, params_agent)
    # run_name = (
    #     "Graphormer-complex-n100-k10-hd128-layers2-lr0.0004-heads4-bs64-tuf100000-XDURS"
    # )
    # rerun_training(run_name=run_name, number_of_reruns=2)

    # evaluate_run_fj(
    #     run_name=run_name,
    #     n_values=[75, 100, 125, 150, 175, 200],
    #     k_values=[5, 10, 15, 20],
    #     single_setting=False,
    # )

    # run_name = run_training(params_env, params_agent)

    # evaluate_run_fj(
    #     run_name=run_name,
    #     n_values=[75, 100, 125, 150, 175, 200],
    #     k_values=[5, 10, 15, 20],
    #     single_setting=False,
    # )

    # run_name = run_training(params_env, params_agent)

    # evaluate_run_fj(
    #     run_name=run_name,
    #     n_values=[75, 100, 125, 150, 175, 200],
    #     k_values=[5, 10, 15, 20],
    #     single_setting=False,
    # )
