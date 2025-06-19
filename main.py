from env.fj_depolarize import FJDepolarize
from agents.dqn import DQN
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

    
if __name__ == "__main__":
    params = {"n": 10,
              "k": 4, 
              "model_architecture":  "GraphSage",
              "qnet_approach": "simple", 
              "learning_rate": 0.0008,
              "embed_dim": 64,
              "num_layers": 3,
              "wandb_init": True, # if false the run will also not be saved
              "run_name": "test_graphsage_simple"}
    
    
    # run_training(params, timesteps_train= 100000)                 

    continue_training(timesteps_train = 20000, run_name="test_graphsage_simple") 

