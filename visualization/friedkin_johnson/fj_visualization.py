import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import matplotlib.patches as patches
import pandas as pd


def visualize_comparison_dqn_vs_greedy(run_name):
    """Visualizes results of method compare_dqn_policy_to_greedy_various_n in evalutation/friedkin_johnson/evaluation,
        i.e., visualizes performance differences between dqn solution and greedy solutions for various n and ks.
    Args:
        run_name (str): The name of the run to visualize.
    """

    run_dir = os.path.join(
        "saved files", "dqn", "friedkin_johnson", "saved_runs", run_name
    )

    with open(os.path.join(run_dir, "params_env.json"), "r") as f:
        params_env = json.load(f)

    with open(os.path.join(run_dir, "evaluation_comparison_to_greedy.pkl"), "rb") as f:
        results = pickle.load(f)

    # Define the ranges
    n_values = list({n for n, _ in results.keys()})
    k_values = list({k for _, k in results.keys()})
    n_values.sort()
    k_values.sort()

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
    ax = sns.heatmap(
        color_df,
        annot=annot_df,
        fmt="",
        cmap="RdYlGn_r",
        cbar_kws={"label": "Average Difference in Polarization Gain"},
        linewidths=0.5,
        linecolor="gray",
        vmin=0,
        vmax=0.2,
    )

    # Add black rectangle around cell for which we trained
    n_target = params_env["n"]
    k_target = params_env["k"]
    row_idx = k_values.index(k_target)  # Get the row and column indices in the matrix
    col_idx = n_values.index(n_target)
    rect = patches.Rectangle(
        (col_idx, row_idx), 1, 1, fill=False, edgecolor="black", linewidth=3
    )  # Rectangle parameters: (x, y) is the bottom left of the cell
    ax.add_patch(rect)

    plt.xlabel("n")
    plt.ylabel("k")
    plt.title(
        f"Heatmap of DQN-Agent trained on n={n_target} and k={k_target} vs. Greedy"
    )
    plt.tight_layout()

    # Save the figure to the specified path
    save_path = os.path.join(run_dir, "heatmap_dqn_vs_greedy.png")
    plt.savefig(save_path)
    plt.close()


def visualize_comparison_dqn_vs_greedy_single_setting(run_name):
    """Visualizes the performance of DQN policy compared to greedy solutions for a single setting of FJ-Depolarize.
    Args:
        run_name (str): Name of the run to evaluate.
    """
    run_dir = os.path.join(
        "saved files", "dqn", "friedkin_johnson", "saved_runs", run_name
    )

    with open(os.path.join(run_dir, "evaluation_single_setting.pkl"), "rb") as f:
        polarization_gains = pickle.load(f)

    # Sort the polarization gains by greedy gain
    polarization_gains.sort(key=lambda x: x[1])

    # Prepare data for plotting
    data = {"Index": [], "Gain": [], "Method": []}

    for i, (x, y, z) in enumerate(polarization_gains, start=1):
        data["Index"] += [i, i, i]
        data["Gain"] += [x, y, z]
        data["Method"] += ["DQN", "Greedy", "Random"]

    df = pd.DataFrame(data)

    # Plot with Seaborn
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="Index", y="Gain", hue="Method", s=10)

    # Add horizontal lines for the average gain of each method
    for method, color in zip(["DQN", "Greedy", "Random"], ["C0", "C1", "C2"]):
        avg_gain = df[df["Method"] == method]["Gain"].mean()
        plt.axhline(
            y=avg_gain,
            linestyle="--",
            color=color,
            linewidth=1.5,
            label=f"{method} avg",
        )

    # plt.title("Polarization Gains ordered by Maximum Gain")
    # plt.xlabel("Sorted Index")
    # plt.ylabel("Polarization Gain")
    plt.tight_layout()

    # Save the figure to the specified path
    save_path = os.path.join(run_dir, "evaluation_single_setting.png")
    plt.savefig(save_path)
    plt.close()
