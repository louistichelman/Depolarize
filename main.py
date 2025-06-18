from env.fj_depolarize import FJDepolarize
from agents.dqn import DQN
import torch


def run_training(n_nodes, k_steps, architecture, approach, embed_dim, **kwargs):
    
    env = FJDepolarize(n=n_nodes, k=k_steps) # initialize environment

    agent = DQN(env, model_architecture=architecture, qnet_approach=approach, embed_dim=embed_dim, **kwargs)   
    agent.train()

    if agent.run_name:
        print("training worked")
        # torch.save(agent.q_network.state_dict(), f"saved files/dqn_params/{agent.run_name}_q_network_params.pth")
        # torch.save(agent.target_network.state_dict(), f"saved files/dqn_params/{agent.run_name}_target_network_params.pth")
    
    
if __name__ == "__main__":
    print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    run_training(n_nodes=10, 
                k_steps=4, 
                architecture = "global",
                approach = "simple", 
                timesteps_train = 100000,
                embed_dim = 64,
                run_name = "cluster_test")                 # possible further arguments: num_heads, number_of_layers, learning_rate,...

    

