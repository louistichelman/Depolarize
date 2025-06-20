from env.fj_depolarize import FJDepolarize
from agents.dqn import DQN
from evaluation.greedy_comparison import *
import torch
import torch.profiler
import os
import json


def run_training(params, timesteps_train):
    
    env = FJDepolarize(**params) # initialize environment

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
        run_dir = os.path.join("saved files", "dqn", "saved_runs_dqn", agent.run_name)
        os.makedirs(run_dir, exist_ok=True)

        # Save full parameter dict
        with open(os.path.join(run_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)

        # Save model parameters
        torch.save(agent.q_network.state_dict(), os.path.join(run_dir, "q_network_params.pth"))
        torch.save(agent.target_network.state_dict(), os.path.join(run_dir, "target_network_params.pth"))


def continue_training(run_name, timesteps_train):
    run_dir = os.path.join("saved files", "dqn", "saved_runs_dqn", run_name)

    # --- Load saved parameters ---
    with open(os.path.join(run_dir, "params.json"), "r") as f:
        params = json.load(f)

    # --- Initialize environment and agent ---
    env = FJDepolarize(**params)
    agent = DQN(env = env, timesteps_train=timesteps_train, **params)

    # --- Load model weights ---
    q_net_path = os.path.join(run_dir, "q_network_params.pth")
    target_net_path = os.path.join(run_dir, "target_network_params.pth")

    agent.q_network.load_state_dict(torch.load(q_net_path, map_location=torch.device("cpu")))
    agent.target_network.load_state_dict(torch.load(target_net_path, map_location=torch.device("cpu")))

    # --- Resume training ---
    agent.train()

    # Save model parameters
    torch.save(agent.q_network.state_dict(), q_net_path)
    torch.save(agent.target_network.state_dict(), target_net_path)

def generate_test_states_and_greedy_polarizations(n_values, k_values, filename_test_states, filename_greedy_polarization):
    test_states = generate_random_states(n_values=n_values, filename=filename_test_states)
    compute_greedy_polarizations(k_values=k_values, test_states = test_states, filename=filename_greedy_polarization)

def evaluate_run(run_name, filename_test_states, filename_greedy_polarization):
    compare_dqn_greedy(run_name=run_name, filename_test_states=filename_test_states, filename_greedy_polarizatios=filename_greedy_polarization)
    visualize_comparison(run_name)
    
if __name__ == "__main__":
    # TRAINING
    # params = {"n": 10,
    #           "k": 4, 
    #           "model_architecture":  "GraphSage",
    #           "qnet_approach": "simple", 
    #           "learning_rate": 0.0008,
    #           "embed_dim": 64,
    #           "num_layers": 3,
    #           "wandb_init": True, # if false the run will also not be saved
    #           "run_name": "test_graphsage_simple"}
    
    # run_training(params, timesteps_train= 100000)                 
    # continue_training(timesteps_train = 20000, run_name="test_graphsage_simple") 

    # EVALUATION
    n_values = [6, 8, 10, 12, 14, 16, 18, 20]
    k_values = [2, 3, 4, 5, 6]
    filename_test_states = "test_states_19.06"
    filename_greedy_polarization = "greedy_polarizations_19.06"

    # generate_test_states_and_greedy_polarizations(n_values, k_values, filename_test_states, filename_greedy_polarization)
    evaluate_run(run_name="test_graphsage_simple", filename_test_states=filename_test_states, filename_greedy_polarization=filename_greedy_polarization)

