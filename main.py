from env import ENVIRONMENT_REGISTRY
from agents.dqn import DQN
from evaluation import *
from visualization import *
import torch
import torch.profiler
import os
import json


def run_training(params, timesteps_train):
    
    env = ENVIRONMENT_REGISTRY[params["environment"]](**params) # initialize environment

    agent = DQN(env = env, timesteps_train = timesteps_train, **params)   

    # with torch.profiler.profile(
    #     schedule=torch.profiler.schedule(
    #         wait=1,        # Skip profiling for the 1st step (warmup)
    #         warmup=1,      # Record but ignore for metrics
    #         active=3,      # Record this many steps
    #         repeat=1       # Repeat once
    #     ),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./logdir'),
    #     record_shapes=True,
    #     with_stack=True,
    #     profile_memory=True,
    #     with_flops=True,
    #     use_cuda=True
    # ) as prof:
    agent.train()

    if agent.run_name:
        # Define run-specific folder
        run_dir = os.path.join("saved files", "dqn", params["environment"], "saved_runs", agent.run_name)
        os.makedirs(run_dir, exist_ok=True)

        # Save full parameter dict
        with open(os.path.join(run_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)

        # Save model parameters
        torch.save(agent.q_network.state_dict(), os.path.join(run_dir, "q_network_params.pth"))
        torch.save(agent.target_network.state_dict(), os.path.join(run_dir, "target_network_params.pth"))

        return agent.run_name


def continue_training(run_name, timesteps_train):
    if os.path.exists(os.path.join("saved files", "dqn", "friedkin_johnson", "saved_runs", run_name)):
        run_dir = os.path.join("saved files", "dqn", "friedkin_johnson", "saved_runs", run_name)
    else:
        run_dir = os.path.join("saved files", "dqn", "nonlinear", "saved_runs", run_name)

    # --- Load saved parameters ---
    with open(os.path.join(run_dir, "params.json"), "r") as f:
        params = json.load(f)

    # --- Initialize environment and agent ---
    env = ENVIRONMENT_REGISTRY[params["environment"]](**params)
    agent = DQN(env = env, timesteps_train=timesteps_train, run_name = f"{run_name}_+", **params)

    # --- Load model weights ---
    q_net_path = os.path.join(run_dir, "q_network_params.pth")
    target_net_path = os.path.join(run_dir, "target_network_params.pth")

    agent.q_network.load_state_dict(torch.load(q_net_path, map_location=torch.device("cpu")))
    agent.target_network.load_state_dict(torch.load(target_net_path, map_location=torch.device("cpu")))

    # --- Resume training ---
    agent.train()

    if agent.run_name:
        # Define run-specific folder
        run_dir = os.path.join("saved files", "dqn", params["environment"], "saved_runs", agent.run_name)
        os.makedirs(run_dir, exist_ok=True)

        # Save full parameter dict
        with open(os.path.join(run_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)

        # Save model parameters
        torch.save(agent.q_network.state_dict(), os.path.join(run_dir, "q_network_params.pth"))
        torch.save(agent.target_network.state_dict(), os.path.join(run_dir, "target_network_params.pth"))

        return agent.run_name

def generate_test_states_and_greedy_polarizations(n_values, k_values, filename_test_states, filename_greedy_polarization):
    test_states = generate_random_states(n_values=n_values, filename=filename_test_states)
    compute_greedy_polarizations(k_values=k_values, test_states = test_states, filename=filename_greedy_polarization)

def evaluate_run_fj(run_name, filename_test_states, filename_greedy_polarization):
    compare_dqn_greedy(run_name=run_name, filename_test_states=filename_test_states, filename_greedy_polarizatios=filename_greedy_polarization)
    visualize_comparison(run_name)
    
def evaluate_single_setting_fj(run_name, n=None, k=None, filename_test_states="test_states_n100_k_10-15-20_27-06", filename_greedy_polarization="greedy_polarizations_n100_k_10-15-20_27-06"):
    visualize_dqn_vs_greedy_single_setting(run_name, n=n, k=k, filename_test_states=filename_test_states, filename_greedy_polarizatios=filename_greedy_polarization)

def evaluate_run_nonlinear(run_name, n_steps=20000, n_simpulations=25):
    # test_and_save_policy_dqn(run_name, n_steps=n_steps, n_simpulations=n_simpulations)
    # test_and_save_baselines(run_name, n_steps=n_steps, n_simpulations=n_simpulations)
    visualize_polarizations_saved_runs(run_name, title="Polarization Over Time")

if __name__ == "__main__":
    params = {
        # envirionment parameters
        "environment": "nonlinear", # friedkin_johnson or nonlinear
        "n": 200,
        "average_degree": 6, #6 for 200, 4 for 50, 4 for 25
        "n_edge_updates_per_step": 5, #5 for 200, 2 for 50, 1 for 25
        "k": 4, 
        # model parameters
        "model_architecture":  "GraphSage",  # GraphSage or GAT
        "qnet_approach": "complex", 
        "learning_rate": 0.0001,
        "embed_dim": 128,
        "num_layers": 6,
        "batch_size": 64,
        "gamma": 0.5,
        "num_heads": 4,
        "wandb_init": True,
        "reset_probability": 0.0001,
        "parallel_envs": 4,  
        "end_e": 0.6,  # End epsilon for exploration
        "target_update_freq": 4000,  # Frequency of target network updates
        }
    
    # Generating test states and greedy polarizations for fj evaluation
    # n_values = [100]
    # k_values = [10, 15, 20]
    # filename_test_states = "test_states_n100_k_10-15-20_27-06"
    # filename_greedy_polarization = "greedy_polarizations_n100_k_10-15-20_27-06"
    # generate_test_states_and_greedy_polarizations(n_values, k_values, filename_test_states, filename_greedy_polarization)
    
    # Run and evaluate fj
    # run_name = run_training(params, timesteps_train= 100000)               
    # evaluate_run(run_name, filename_test_states=filename_test_states, filename_greedy_polarization=filename_greedy_polarization)
    # run_name = continue_training(timesteps_train = 100000, run_name = "GraphSage-complex-n100-k15-hd128-layers4-lr0.0004-heads0-bs64-G30BI_+")  
    # evaluate_run(run_name, filename_test_states=filename_test_states, filename_greedy_polarization=filename_greedy_polarization)
    # run_name = continue_training(timesteps_train = 100000, run_name=run_name) 
    # evaluate_run(run_name, filename_test_states=filename_test_states, filename_greedy_polarization=filename_greedy_polarization)

    # evaluate_single_setting_fj("global-simple-n100-k10-hd128-layers4-lr0.0001-heads0-bs64-IKUIM", n=100, k=10, filename_test_states=filename_test_states, filename_greedy_polarization=filename_greedy_polarization)

    # Run and evaluate nonlinear
    # run_name = run_training(params, timesteps_train= 150000)  
    # run_name = continue_training(run_name="GraphSage-complex-n200-hd63-layers4-lr0.0008-heads0-bs64-G48MQ_+", timesteps_train=100000)             
    # evaluate_run_nonlinear(run_name)
    evaluate_run_nonlinear("GraphSage-complex-n200-hd64-layers4-lr0.0001-heads0-bs64-g0.4-par4-e0.6-tuf4000-MK0XM")
    # evaluate_run_nonlinear("GraphSage-complex-n50-hd64-layers4-lr0.0001-heads0-bs64-g0.8-par4-e0.6-tuf6000-54N38")
    # evaluate_run_nonlinear("GraphSage-complex-n200-hd128-layers6-lr0.0001-heads0-bs64-g0.5-par4-e0.6-tuf4000-MF1WR")
