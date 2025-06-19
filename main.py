from env.fj_depolarize import FJDepolarize
from agents.dqn import DQN
import torch
import torch.profiler



def run_training(n_nodes, k_steps, architecture, approach, embed_dim, **kwargs):
    
    env = FJDepolarize(n=n_nodes, k=k_steps) # initialize environment

    agent = DQN(env, model_architecture=architecture, qnet_approach=approach, embed_dim=embed_dim, **kwargs)   

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
        torch.save(agent.q_network.state_dict(), f"saved files/dqn_params/{agent.run_name}_q_network_params.pth")
        torch.save(agent.target_network.state_dict(), f"saved files/dqn_params/{agent.run_name}_target_network_params.pth")
    
    
if __name__ == "__main__":
    run_training(n_nodes=10, 
                k_steps=4, 
                architecture = "global",
                approach = "simple", 
                timesteps_train = 100000,
                embed_dim = 64,
                num_layers = 4,
                run_name = "cluster_test")                 # possible further arguments: num_heads, number_of_layers, learning_rate,...

    

