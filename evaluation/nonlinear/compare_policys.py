from tqdm import tqdm
import numpy as np
import random
import os
import json
import torch
from env import NLOpinionDynamics
from agents.dqn import DQN

def test_and_save_policy_dqn(run_name, n_steps=20000, n_simpulations=25):
    
    run_dir = os.path.join("saved files", "dqn", "nonlinear", "saved_runs", run_name)

    # Load parameters
    with open(os.path.join(run_dir, "params.json"), "r") as f:
        params = json.load(f)

    params["wandb_init"] = False
    # Initialize environment and agent
    env = NLOpinionDynamics(**params)
    agent = DQN(env=env, **params)
    
    # Load model weights
    q_net_path = os.path.join(run_dir, "q_network_params.pth")
    target_net_path = os.path.join(run_dir, "target_network_params.pth")

    agent.q_network.load_state_dict(torch.load(q_net_path, map_location=torch.device("cpu")))
    agent.target_network.load_state_dict(torch.load(target_net_path, map_location=torch.device("cpu")))
    
    # Test the policy
    recorded_opinions = test_policy(env, policy=agent.policy_greedy, n_steps=n_steps, n_simpulations=n_simpulations)

    # Save results
    save_path = os.path.join(run_dir, f"recorded_opinions_dqn.npy")
    np.save(save_path, recorded_opinions)
    print(f"Results saved to {save_path}")


def test_and_save_baselines(run_name, n_steps=20000, n_simpulations=25):

    run_dir = os.path.join("saved files", "dqn", "nonlinear", "saved_runs", run_name)

    # Load parameters
    with open(os.path.join(run_dir, "params.json"), "r") as f:
        params = json.load(f)

    # Initialize environment
    env = NLOpinionDynamics(**params)

    # Ensure the baselines directory exists
    os.makedirs(os.path.join(run_dir, "baselines"), exist_ok=True)

    # Test without policy
    recorded_opinions_no_policy = test_policy(env, n_steps=n_steps, n_simpulations=n_simpulations)
    save_path = os.path.join(run_dir, "baselines", f"recorded_opinions_no_policy.npy")
    np.save(save_path, recorded_opinions_no_policy)
    print(f"Results saved to {save_path}")

    # Test the minmax policy
    recorded_opinions_min_max = test_policy(env, policy=minmax_policy, n_steps=n_steps, n_simpulations=n_simpulations)
    save_path = os.path.join(run_dir, "baselines", f"recorded_opinions_minmax.npy")
    np.save(save_path, recorded_opinions_min_max)
    print(f"Results saved to {save_path}")

    # Test the softminmax policy
    recorded_opinions_soft_min_max = test_policy(env, policy=softminmax_policy, n_steps=n_steps, n_simpulations=n_simpulations)
    save_path = os.path.join(run_dir, "baselines", f"recorded_opinions_softminmax.npy")
    np.save(save_path, recorded_opinions_soft_min_max)
    print(f"Results saved to {save_path}")

    # Test the deleting policy
    recorded_opinions_deleting = test_policy(env, policy=deleting_policy, n_steps=n_steps, n_simpulations=n_simpulations)
    save_path = os.path.join(run_dir, "baselines", f"recorded_opinions_deleting.npy")
    np.save(save_path, recorded_opinions_deleting)
    print(f"Results saved to {save_path}")  


def test_policy(env, policy=lambda x: 0, n_steps=20000, n_simpulations=25):
    recorded_opinions = []

    for _ in tqdm(range(n_simpulations)):
        state = env.reset()
        recorded_opinions_simulation = [state["sigma"].copy()]
        for time_step in range(1, n_steps):
            action = policy(state)
            next_state, _, _ = env.step(action)
            if time_step % 2 == 0:
                recorded_opinions_simulation.append(next_state["sigma"].copy())
            state = next_state
        recorded_opinions.append(np.array(recorded_opinions_simulation))

    return np.array(recorded_opinions)  # Shape: (n_simpulations, n_steps, n_nodes)

# ----baselines: simple strategies----

def minmax_policy(state):
    if state["tau"] is None:
        return np.argmax(state["sigma"])
    else:
        sigma_values = np.array(state["sigma"])
        return np.argmin(sigma_values[np.setdiff1d(np.arange(len(sigma_values)), state["graph"].neighbors(state["tau"]))])

def softminmax_policy(state):
    if state["tau"] is None:
        sigma_values = np.array(state["sigma"])
        threshold_top  = np.percentile(sigma_values, 80)
        candidates = np.where(sigma_values <= threshold_top)[0]
        return max(candidates, key=lambda node: state["sigma"][node])
    else:
        sigma_values = np.array(state["sigma"])
        threshold_bottom  = np.percentile(sigma_values, 20)
        candidates = set(np.where(sigma_values >= threshold_bottom)[0]) - set(state["graph"].neighbors(state["tau"]))
        return min(candidates, key=lambda node: state["sigma"][node])

def deleting_policy(state):
    if state["tau"] is None:
        if random.random() < 0.5:
            return np.argmax(state["sigma"])  
        else:
            return np.argmin(state["sigma"])
    else:
        sigma_values = np.array(state["sigma"])
        neighbors = list(state["graph"].neighbors(state["tau"]))
        if not neighbors:
            return 0  # No neighbors available
        return min(neighbors, key=lambda node: abs(sigma_values[node] - sigma_values[state["tau"]]))
