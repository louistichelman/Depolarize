import os
import pickle
from env import FJDepolarize, DepolarizeSimple
from .depolarizing_functions import depolarize_greedy, depolarize_policy, depolarize_random
from tqdm import tqdm
from agents.dqn import DQN
import json
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd

def generate_random_states(n_values, num_states=150, filename = "test_states", save_path="saved files/dqn/friedkin_johnson/greedy_comparison"):
    os.makedirs(save_path, exist_ok=True)
    all_states = {}

    for n in n_values:
        env = FJDepolarize(n=n, k=1)
        env_simple = DepolarizeSimple(n=n, k=1, generate_states=False)
        states = []
        seen = set()
        no_new_graphs = 0
        while len(states)<num_states:
            state = env.reset()
            state_hash = env_simple.state_hash(state["graph"], state["sigma"], state["tau"])
            if state_hash in seen:
                if no_new_graphs>10:
                    break
                no_new_graphs += 1
                continue
            no_new_graphs = 0
            seen.add(state_hash)
            states.append(state)
        all_states[n] = states

    with open(os.path.join(save_path, f"{filename}.pkl"), "wb") as f:
        pickle.dump(all_states, f)

    return all_states

def compute_greedy_polarizations(k_values, test_states, filename = "greedy_polarizations", save_path="saved files/dqn/friedkin_johnson/greedy_comparison"):
    os.makedirs(save_path, exist_ok=True)
    results = {}

    for n, states in tqdm(test_states.items()):
        for k in tqdm(k_values):
            env = FJDepolarize(n=n, k=k)
            polarizations = []
            for state in states:
                _, pol = depolarize_greedy(state, env)
                polarizations.append(pol)
            results[(n, k)] = polarizations

    with open(os.path.join(save_path, f"{filename}.pkl"), "wb") as f:
        pickle.dump(results, f)

    return results

def compare_dqn_greedy(run_name, filename_test_states = "test_states_19.06", filename_greedy_polarizatios = "greedy_polarizations_19.06"):

    run_dir = os.path.join("saved files", "dqn", "friedkin_johnson", "saved_runs", run_name)
    comparison_dir = os.path.join("saved files", "dqn", "friedkin_johnson", "greedy_comparison")

    with open(os.path.join(comparison_dir, f"{filename_test_states}.pkl"), "rb") as f:
            test_states = pickle.load(f)

    with open(os.path.join(comparison_dir, f"{filename_greedy_polarizatios}.pkl"), "rb") as f:
            greedy_polarizations = pickle.load(f)

    with open(os.path.join(run_dir, "params.json"), "r") as f:
            params = json.load(f)

    params["wandb_init"] = False

    # --- Initialize environment and agent ---
    env = FJDepolarize(**params)
    agent = DQN(env = env, **params)

    # --- Load model weights ---
    q_net_path = os.path.join(run_dir, "q_network_params.pth")
    target_net_path = os.path.join(run_dir, "target_network_params.pth")

    agent.q_network.load_state_dict(torch.load(q_net_path, map_location=torch.device("cpu")))
    agent.target_network.load_state_dict(torch.load(target_net_path, map_location=torch.device("cpu")))

    epsilon =  1e-4

    results = {}
    for (n, k), pols in greedy_polarizations.items():
        env = FJDepolarize(n=n, k=k)
        polarization_diff = 0
        dqn_better = 0
        greedy_better = 0
        for i, state in enumerate(test_states[n]):
            state["edges_left"] = k
            _, polarization_dqn = depolarize_policy(state, env, agent.policy_greedy)
            polarization_diff = polarization_diff + polarization_dqn - greedy_polarizations[(n,k)][i]
            if abs(polarization_dqn - greedy_polarizations[(n,k)][i]) > epsilon:
                if polarization_dqn < greedy_polarizations[(n,k)][i]:
                    dqn_better += 1
                else:
                    greedy_better += 1
        results[(n,k)] = {"number_states": len(test_states[n]),
                            "dqn_better": dqn_better,
                            "greedy_better": greedy_better,
                            "difference": polarization_diff}

    with open(os.path.join(run_dir, f"greedy_comparison_results.pkl"), "wb") as f:
        pickle.dump(results, f)
    
def visualize_comparison(run_name):
     
    run_dir = os.path.join("saved files", "dqn", "friedkin_johnson", "saved_runs", run_name)

    with open(os.path.join(run_dir, "params.json"), "r") as f:
            params = json.load(f)

    with open(os.path.join(run_dir, "greedy_comparison_results.pkl"), "rb") as f:
            results = pickle.load(f)


    # Define the ranges
    n_values = list({n for n, _ in results.keys()})
    k_values = list({k for _, k in results.keys()})

    # Create matrices for annotations and heatmap coloring
    annot_matrix = []
    color_matrix = []

    for k in k_values:
        row_annot = []
        row_color = []
        for n in n_values:
            key = (n, k)
            if key in results:
                r = results[key]
                num = r["number_states"]
                dqn = r["dqn_better"]
                greedy = r["greedy_better"]
                diff = r["difference"]
                diff_norm = diff / num if num != 0 else 0

                # Annotation text
                annot = f"better: {dqn}\nworse: {greedy}\n equal: {num-dqn-greedy}\n {diff_norm:.4f}"

                row_annot.append(annot)
                row_color.append(diff_norm)
            else:
                row_annot.append("")
                row_color.append(np.nan)
        annot_matrix.append(row_annot)
        color_matrix.append(row_color)

    # Convert to DataFrame for seaborn
    annot_df = pd.DataFrame(annot_matrix, index=k_values, columns=n_values)
    color_df = pd.DataFrame(color_matrix, index=k_values, columns=n_values)

    # Plotting
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(color_df, annot=annot_df, fmt='', cmap="RdYlGn_r", cbar_kws={'label': 'Average Difference in Polarization Gain'}, linewidths=0.5, linecolor='gray', vmin=0, vmax=0.035)

    # Add black rectangle around cell for which we trained
    n_target = params["n"]
    k_target = params["k"]
    row_idx = k_values.index(k_target) # Get the row and column indices in the matrix
    col_idx = n_values.index(n_target)
    rect = patches.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='black', linewidth=3) # Rectangle parameters: (x, y) is the bottom left of the cell
    ax.add_patch(rect)

    plt.xlabel("n")
    plt.ylabel("k")
    plt.title(f"Heatmap of DQN-Agent trained on n={n_target} and k={k_target} vs. Greedy")
    plt.tight_layout()

    # Save the figure to the specified path
    save_path = os.path.join(run_dir, "heatmap_dqn_vs_greedy.png")
    plt.savefig(save_path)
    plt.close()

def visualize_dqn_vs_greedy_single_setting(run_name, n = None, k= None, filename_test_states = "test_states_19.06", filename_greedy_polarizatios = "greedy_polarizations_19.06"):

    run_dir = os.path.join("saved files", "dqn", "friedkin_johnson", "saved_runs", run_name)
    comparison_dir = os.path.join("saved files", "dqn", "friedkin_johnson", "greedy_comparison")

    with open(os.path.join(comparison_dir, f"{filename_test_states}.pkl"), "rb") as f:
            test_states = pickle.load(f)

    with open(os.path.join(comparison_dir, f"{filename_greedy_polarizatios}.pkl"), "rb") as f:
            greedy_polarizations = pickle.load(f)

    with open(os.path.join(run_dir, "params.json"), "r") as f:
            params = json.load(f)

    params["wandb_init"] = False

    # --- Initialize environment and agent ---
    env = FJDepolarize(**params)
    agent = DQN(env = env, **params)

    # --- Load model weights ---
    q_net_path = os.path.join(run_dir, "q_network_params.pth")
    target_net_path = os.path.join(run_dir, "target_network_params.pth")

    agent.q_network.load_state_dict(torch.load(q_net_path, map_location=torch.device("cpu")))
    agent.target_network.load_state_dict(torch.load(target_net_path, map_location=torch.device("cpu")))

    if n is None:
        n = params["n"]
    if k is None:
        k = params["k"] 

    test_states_n = test_states[n]
    greedy_polarizations_nk = greedy_polarizations[(n, k)]

    polarization_gains = []
    for i, state in tqdm(enumerate(test_states_n)):
        state["edges_left"] = k
        G, sigma = state["graph"], state["sigma"]
        polarization_start = env.polarization(G, sigma)
        _, polarization_dqn = depolarize_policy(state, env, agent.policy_greedy)
        _, polarization_random = depolarize_random(state, env)
        polarization_gains.append(( polarization_start - polarization_dqn ,
                                    polarization_start - greedy_polarizations_nk[i],
                                    polarization_start - polarization_random))

    # Step 1: Sort by max(triple)
    polarization_gains_sorted = sorted(polarization_gains, key=lambda t: max(t))

    # Step 2: Prepare data for plotting
    data = {
        'Index': [],
        'Gain': [],
        'Method': []
    }

    for i, (x, y, z) in enumerate(polarization_gains_sorted, start=1):
        data['Index'] += [i, i, i]
        data['Gain'] += [x, y, z]
        data['Method'] += ['DQN', 'Greedy', 'Random']

    df = pd.DataFrame(data)

    # Step 3: Plot with Seaborn
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Index', y='Gain', hue='Method', style='Method', s=30)

    plt.title('Polarization Gains ordered by Maximum Gain')
    plt.xlabel('Sorted Index')
    plt.ylabel('Polarization Gain')
    plt.tight_layout()

    # Save the figure to the specified path
    save_path = os.path.join(run_dir, "evaluation_single_setting.png")
    plt.savefig(save_path)
    plt.close()




