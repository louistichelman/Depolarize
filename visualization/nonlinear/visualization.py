import matplotlib.pyplot as plt
import numpy as np
import os
import json
import seaborn as sns
import csv

def analyze_actions(
    run_name=None, run_folder=None, folder="val"
):
    """
    Analyze recorded actions from one run or a folder of runs.
    
    Handles multiple graph sizes n (dict structure).
    
    For each n:
      - number of adds vs deletes
      - opinion statistics
      - degree statistics
      - plots saved to save_path
    
    Returns:
        dict: metrics per graph size n
    """

    if run_name is not None:
        run_dir = os.path.join("results", "dqn", "nonlinear", "runs", run_name)
        recorded_actions = load_recorded_files(run_dir, aspect="actions", folder=folder)
        save_path = os.path.join(run_dir, folder)
        os.makedirs(save_path, exist_ok=True)
    elif run_folder is not None:
        run_folder_dir = os.path.join("results", "dqn", "nonlinear", "runs", run_folder)
        recorded_actions = {}
        for i, run_name_in_folder in enumerate(os.listdir(run_folder_dir)):
            run_dir_in_folder = os.path.join(run_folder_dir, run_name_in_folder)
            recorded_actions_dqn_i = load_recorded_files(run_dir_in_folder, aspect="actions", folder=folder)
            for n, arr in recorded_actions_dqn_i.items():
                if n not in recorded_actions:
                    recorded_actions[n] = arr
                else:
                    # concatenate simulations
                    recorded_actions[n] = np.concatenate(
                        [recorded_actions[n], arr], axis=0
                    )
        save_path = os.path.join(run_folder_dir, folder)
        os.makedirs(save_path, exist_ok=True)
    else:
        raise ValueError("Either run_name or run_folder must be provided.")

    all_metrics = {}

    for n, actions_arr in recorded_actions.items():

        # --- Overall (time-independent) metrics for this n ---
        actions_flattened = actions_arr.reshape(-1, actions_arr.shape[-1])  # shape (num_sims * T, 5)

        sigma_tau = actions_flattened[:, 0]
        sigma_action = actions_flattened[:, 1]
        connected = actions_flattened[:, 2].astype(bool)
        deg_tau = actions_flattened[:, 3]
        deg_action = actions_flattened[:, 4]

        opinion_diff = np.abs(sigma_tau - sigma_action)

        # masks
        add_mask = ~connected
        del_mask = connected

        metrics = {
            "number_of_recorded_actions": len(actions_flattened),
            "n_adds": int(np.sum(add_mask)),
            "n_deletes": int(np.sum(del_mask)),
            "avg_opinion_tau": float(np.mean(sigma_tau)),
            "avg_opinion_action": float(np.mean(sigma_action)),
            "avg_opinion_diff": float(np.mean(opinion_diff)),
            "avg_opinion_diff_adds": float(np.mean(opinion_diff[add_mask])) if np.any(add_mask) else None,
            "avg_opinion_diff_deletes": float(np.mean(opinion_diff[del_mask])) if np.any(del_mask) else None,
            "avg_deg_tau": float(np.mean(deg_tau)),
            "avg_deg_action": float(np.mean(deg_action)),
            "avg_deg_tau_adds": float(np.mean(deg_tau[add_mask])) if np.any(add_mask) else None,
            "avg_deg_tau_deletes": float(np.mean(deg_tau[del_mask])) if np.any(del_mask) else None,
        }
        all_metrics[n] = metrics

        # plots
        n_save_path = os.path.join(save_path, f"n{n}")
        os.makedirs(n_save_path, exist_ok=True)

        # 1. Adds vs Deletes
        plt.figure()
        plt.bar(["Adds", "Deletes"], [metrics["n_adds"], metrics["n_deletes"]])
        plt.title(f"Adds vs Deletes (n={n})")
        plt.savefig(os.path.join(n_save_path, "adds_vs_deletes.png"))
        plt.close()

        # 2. Opinion differences (boxplot)
        if np.any(add_mask) and np.any(del_mask):
            plt.figure()
            data = [opinion_diff[add_mask], opinion_diff[del_mask]]
            plt.boxplot(data, labels=["Adds", "Deletes"])
            plt.ylabel("Opinion difference |σ_tau - σ_action|")
            plt.title(f"Opinion Differences (n={n})")
            plt.savefig(os.path.join(n_save_path, "opinion_diff_boxplot.png"))
            plt.close()

        # 3. Degree histograms
        plt.figure()
        plt.hist(deg_tau, bins=20, alpha=0.5, label="deg_tau")
        plt.hist(deg_action, bins=20, alpha=0.5, label="deg_action")
        plt.legend()
        plt.title(f"Degree Distributions (n={n})")
        plt.savefig(os.path.join(n_save_path, "degree_histograms.png"))
        plt.close()


        # --- Temporal strategy plot ---
        plot_temporal_strategy(actions_arr, n, n_save_path)

    # --- Save all metrics as JSON ---
    metrics_path = os.path.join(save_path, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)


def plot_temporal_strategy(actions, n, save_path, n_buckets=8):
    """
    Plot temporal strategy with mean and std: adds vs deletes counts and avg opinion difference.
    
    actions: np.array of shape (num_sims, T, 5)
             (sigma_tau, sigma_action, connected, deg_tau, deg_action)
    """
    # sns.set_style("whitegrid")

    # Ensure actions has shape (num_sims, T, 5)
    if actions.ndim == 2:
        actions = actions[None, :, :]  # single run -> batch of 1
    
    num_sims, n_steps, _ = actions.shape
    bucket_size = n_steps // n_buckets
    
    add_counts, del_counts = [], []
    add_diffs, del_diffs = [], []
    add_counts_std, del_counts_std = [], []
    add_diffs_std, del_diffs_std = [], []
    
    for b in range(n_buckets):
        start = b * bucket_size
        end = (b + 1) * bucket_size if b < n_buckets - 1 else n_steps
        
        # Collect metrics per simulation
        adds_per_sim = []
        dels_per_sim = []
        add_diff_per_sim = []
        del_diff_per_sim = []
        
        for sim in range(num_sims):
            sigma_tau = actions[sim, start:end, 0]
            sigma_action = actions[sim, start:end, 1]
            connected = actions[sim, start:end, 2].astype(bool)
            opinion_diff = np.abs(sigma_tau - sigma_action)
            
            adds = ~connected
            dels = connected
            
            adds_per_sim.append(np.sum(adds))
            dels_per_sim.append(np.sum(dels))
            add_diff_per_sim.append(np.mean(opinion_diff[adds]) if np.any(adds) else np.nan)
            del_diff_per_sim.append(np.mean(opinion_diff[dels]) if np.any(dels) else np.nan)
        
        add_counts.append(np.mean(adds_per_sim))
        del_counts.append(np.mean(dels_per_sim))
        add_counts_std.append(np.std(adds_per_sim))
        del_counts_std.append(np.std(dels_per_sim))
        add_diffs.append(np.nanmean(add_diff_per_sim))
        del_diffs.append(np.nanmean(del_diff_per_sim))
        add_diffs_std.append(np.nanstd(add_diff_per_sim))
        del_diffs_std.append(np.nanstd(del_diff_per_sim))
    
    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=(8, 4))
    x = np.arange(n_buckets)

    # Create bucket labels like "0–2k", "2k–4k", ...
    bucket_labels = []
    for b in range(n_buckets):
        start = b * bucket_size
        end = (b + 1) * bucket_size if b < n_buckets - 1 else n_steps
        # Format in thousands
        bucket_labels.append(f"{start//1000}k–{end//1000}k")

    # Use prettier colors:)
    palette = sns.color_palette("deep")

    orange = palette[1] # deep orange
    purple = palette[4]  # deep purple

    
    # Bars with std
    width = 0.35
    ax1.bar(x - width/2, add_counts, width, yerr=add_counts_std, 
            label="Adds", alpha=0.7, color=orange, capsize=3)
    ax1.bar(x + width/2, del_counts, width, yerr=del_counts_std, 
            label="Deletes", alpha=0.7, color=purple, capsize=3)
    ax1.set_xlabel("Time bucket")
    ax1.set_ylabel("Number of actions")
    ax1.set_ylim(0, 1400)
    ax1.set_yticks(np.arange(0, 1300, 200))
    ax1.set_xticks(x)
    ax1.set_xticklabels(bucket_labels) 
    ax1.legend(loc="upper left")
    
    # Second y-axis for opinion differences
    ax2 = ax1.twinx()
    ax2.plot(x, add_diffs, "-o", label="Avg diff (Adds)", color=orange)
    ax2.fill_between(x, np.array(add_diffs) - np.array(add_diffs_std),
                        np.array(add_diffs) + np.array(add_diffs_std),
                        color=orange, alpha=0.2)
    ax2.plot(x, del_diffs, "-o", label="Avg diff (Deletes)", color=purple)
    ax2.fill_between(x, np.array(del_diffs) - np.array(del_diffs_std),
                        np.array(del_diffs) + np.array(del_diffs_std),
                        color=purple, alpha=0.2)
    ax2.set_ylabel("Average opinion difference")
    ax2.set_ylim(0, 15)
    ax2.legend(loc="upper right")
    
    plt.title(f"Temporal Strategy (n={n})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"temporal_strategy_n{n}.png"))
    plt.close()


def visualize_polarization_development_dqn_and_baselines(
    run_name=None, run_folder=None, params_env=None, title="Polarization Over Time", folder="val"
):
    dict_policies_to_visualize = {}

    if run_name is not None:
        run_dir = os.path.join("results", "dqn", "nonlinear", "runs", run_name)
        with open(os.path.join(run_dir, "params_env.json"), "r") as f:
            params_env = json.load(f)
        recorded_opinions_dqn = load_recorded_files(run_dir, aspect="opinions", folder=folder)
        for n in recorded_opinions_dqn:
            dict_policies_to_visualize[n] = [("DQN", recorded_opinions_dqn[n])]

    if run_folder is not None:
        run_folder_dir = os.path.join("results", "dqn", "nonlinear", "runs", run_folder)
        for i, run_name_in_folder in enumerate(os.listdir(run_folder_dir)):
            if i == 0:
                run_dir_in_folder = os.path.join(run_folder_dir, run_name_in_folder)
                with open(os.path.join(run_dir_in_folder, "params_env.json"), "r") as f:
                    params_env = json.load(f)
                recorded_opinions_dqn = load_recorded_files(run_dir_in_folder, aspect="opinions", folder=folder)
                continue
            run_dir_in_folder = os.path.join(run_folder_dir, run_name_in_folder)
            recorded_opinions_dqn_i = load_recorded_files(run_dir_in_folder, aspect="opinions", folder=folder)
            for n in recorded_opinions_dqn:
                recorded_opinions_dqn[n] = np.concatenate(
                    (recorded_opinions_dqn[n], recorded_opinions_dqn_i[n]), axis=0
                )
        # return recorded_opinions_dqn
        for n in recorded_opinions_dqn:
            dict_policies_to_visualize[n] = [("DQN", recorded_opinions_dqn[n])]

    baselines_dir = os.path.join(
        "data",
        "nonlinear",
        "baselines",
        folder,
    )
    if os.path.exists(baselines_dir):
        for fname in os.listdir(baselines_dir):
            if fname.endswith(
                f"average_degree_{params_env['average_degree']}_n_updates{params_env['n_edge_updates_per_step']}"
            ):
                n = int(
                    fname[
                        len("n_nodes_") : -len(
                            f"_average_degree_{params_env['average_degree']}_n_updates{params_env['n_edge_updates_per_step']}"
                        )
                    ]
                )

                if os.path.exists(
                    os.path.join(
                        baselines_dir, fname, "recorded_opinions_no_policy.npy"
                    )
                ):
                    recorded_opinions_no_policy = np.load(
                        os.path.join(
                            baselines_dir, fname, "recorded_opinions_no_policy.npy"
                        )
                    )
                    dict_policies_to_visualize.setdefault(n, []).append(
                        ("No Intervention", recorded_opinions_no_policy)
                    )
                if os.path.exists(
                    os.path.join(baselines_dir, fname, "recorded_opinions_minmax.npy")
                ):
                    recorded_opinions_minmax = np.load(
                        os.path.join(
                            baselines_dir, fname, "recorded_opinions_minmax.npy"
                        )
                    )
                    dict_policies_to_visualize.setdefault(n, []).append(
                        ("MinMax", recorded_opinions_minmax)
                    )
                if os.path.exists(
                    os.path.join(
                        baselines_dir, fname, "recorded_opinions_soft_minmax.npy"
                    )
                ):
                    recorded_opinions_soft_minmax = np.load(
                        os.path.join(
                            baselines_dir, fname, "recorded_opinions_soft_minmax.npy"
                        )
                    )
                    dict_policies_to_visualize.setdefault(n, []).append(
                        ("Soft MinMax", recorded_opinions_soft_minmax)
                    )
                if os.path.exists(
                    os.path.join(baselines_dir, fname, "recorded_opinions_deleting.npy")
                ):
                    recorded_opinions_deleting = np.load(
                        os.path.join(
                            baselines_dir, fname, "recorded_opinions_deleting.npy"
                        )
                    )
                    dict_policies_to_visualize.setdefault(n, []).append(
                        ("Deleting", recorded_opinions_deleting)
                    )
                if os.path.exists(
                    os.path.join(baselines_dir, fname, "recorded_opinions_old_soft.npy")
                ):
                    recorded_opinions_old_soft = np.load(
                        os.path.join(
                            baselines_dir, fname, "recorded_opinions_old_soft.npy"
                        )
                    )
                    dict_policies_to_visualize.setdefault(n, []).append(
                        ("Old Soft MinMax", recorded_opinions_old_soft)
                    )
                if os.path.exists(
                    os.path.join(baselines_dir, fname, "recorded_opinions_random.npy")
                ):
                    recorded_opinions_random = np.load(
                        os.path.join(
                            baselines_dir, fname, "recorded_opinions_random.npy"
                        )
                    )
                    dict_policies_to_visualize.setdefault(n, []).append(
                        ("Random", recorded_opinions_random)
                    )
    else:
        print(f"Baselines directory does not exist. Skipping baseline visualizations.")

    if run_name is not None:
        save_path_figure = os.path.join(run_dir, folder, "polarization_over_time.png")
        save_path_metrics = os.path.join(run_dir, folder, "polarization_metrics.csv")
    elif run_folder is not None:
        os.makedirs(os.path.join(run_folder_dir, folder), exist_ok=True)
        save_path_figure = os.path.join(run_folder_dir, folder, "polarization_over_time.png")
        save_path_metrics = os.path.join(run_folder_dir, folder, "polarization_metrics.csv")
    else:
        save_path_figure = os.path.join(baselines_dir, "polarization_over_time.png")
        save_path_metrics = os.path.join(baselines_dir, "polarization_metrics.csv")

    visualize_polarization_development_various_policies(
        dict_policies_to_visualize,
        plot_file_path=save_path_figure,
        metrics_file_path=save_path_metrics,
        title=title,
    )


def visualize_polarization_development_various_policies(
    dict_policies_to_visualize,
    n=150,
    plot_file_path="polarization_over_time.png",
    metrics_file_path="polarization_metrics.csv",
    title="Polarization Over Time",
    step_to_report=9999
):
    sns.set_theme(style="whitegrid")

    # Define color palette but skip red
    base_palette = sns.color_palette("tab10")
    custom_palette = [base_palette[3]] + [c for i, c in enumerate(base_palette) if i not in [1, 2, 3]]  # reserve red for DQN

    num_plots = len(dict_policies_to_visualize)
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4), sharey=True)

    if num_plots == 1:
        axes = [axes]

    # collect results to save later
    results = []

    for ax, (key, list_policies) in zip(axes, sorted(dict_policies_to_visualize.items())):
        for color, (strat, opinions_array) in zip(custom_palette, list_policies):
            mean_pol, std_pol = average_pol(opinions_array)
            steps = np.arange(opinions_array.shape[1])

            # plot mean
            ax.plot(steps, mean_pol, label=strat, alpha=0.8, color=color)

            # shaded region ±1 std
            ax.fill_between(steps, mean_pol - std_pol, mean_pol + std_pol, alpha=0.2, color=color)

            # collect polarization at step_to_report
            if step_to_report < len(mean_pol):
                results.append({
                    "method": strat,
                    "n": key,
                    "step": step_to_report,
                    "mean": mean_pol[step_to_report],
                    "std": std_pol[step_to_report]
                })
            else:
                results.append({
                    "method": strat,
                    "n": key,
                    "step": step_to_report,
                    "mean": None,
                    "std": None
                })

        ax.set_xlabel("Time Steps")
        ax.set_title(f"{title} (n={key})")
        ax.set_xticks([0, 5000, 10000])
        ax.set_yticks([20, 50, 100, 150, 175]) 
        ax.grid(True)

    axes[0].set_ylabel("Polarization")
    fig.tight_layout()
    plt.savefig(plot_file_path)
    plt.close(fig)

    # Save separate plot for the requested n
    if n in dict_policies_to_visualize:
        fig_single, ax_single = plt.subplots(figsize=(4.8, 3.2))
        for color, (strat, opinions_array) in zip(custom_palette, dict_policies_to_visualize[n]):
            mean_pol, std_pol = average_pol(opinions_array)
            steps = np.arange(opinions_array.shape[1])
            ax_single.plot(steps, mean_pol, label=strat, alpha=0.8, color=color)
            ax_single.fill_between(steps, mean_pol - std_pol, mean_pol + std_pol, alpha=0.2, color=color)

        ax_single.set_xlabel("Time Steps")
        ax_single.set_ylabel("Polarization")

        # # Move y-axis to the right
        # ax_single.yaxis.tick_right()
        # ax_single.yaxis.set_label_position("right")

        # ax_single.set_xticks([0, 5000, 10000])
        ax_single.set_ylim(20, 125)
        # ax_single.set_yticks([50, 100]) 
        ax_single.grid(True)
        # ax_single.legend(fontsize=9)

        single_file_path = plot_file_path.replace(".png", f"_n{n}.png")
        plt.tight_layout()
        plt.savefig(single_file_path)
        plt.close(fig_single)

        if dict_policies_to_visualize[n][0][0] == "DQN":
            for i in range(len(dict_policies_to_visualize[n][0][1]) // 3):
                single_opinions_file_path = single_file_path.replace(".png", f"_n{n}_opinions_{i}.png")
                plot_opinions_with_polarization(dict_policies_to_visualize[n][0][1], run_to_visualize=3 * i, file_path=single_opinions_file_path)


    # save results to file
    with open(metrics_file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "n", "step", "mean", "std"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def load_recorded_files(file_path, aspect="opinions", folder="val"):
        evaluation_dir = os.path.join(file_path, folder)
        recorded_aspect = {}
        if os.path.exists(evaluation_dir):
            for fname in os.listdir(evaluation_dir):
                if fname.startswith(f"recorded_{aspect}_") and fname.endswith(".npy"):
                    n = int(fname[len(f"recorded_{aspect}_dqn_n") : -len(".npy")])
                    recorded_aspect[n] = np.load(os.path.join(evaluation_dir, fname))
        return recorded_aspect


def plot_opinions_with_polarization(recorded_opinions_array, run_to_visualize=0, file_path=None):
    single_run = recorded_opinions_array[run_to_visualize]
    fig, ax1 = plt.subplots(figsize=(5.5, 4))
    time_steps = np.arange(0, len(single_run))
    n_nodes = single_run.shape[1]

    for node in range(n_nodes):
        ax1.plot(time_steps, single_run[:, node], alpha=0.7, color="gray", linewidth=0.3)
    ax1.set_ylim(-10, 10)
    ax1.set_yticks((-10,-5, 0, 5, 10))
    ax1.set_xticks((0,5000, 10000))
    ax1.tick_params(axis='y', labelcolor='gray', labelsize=14)
    ax1.tick_params(axis='x', labelcolor='black', labelsize=14)

    # ax2 = ax1.twinx()
    # ax2.plot(np.arange(0, recorded_opinions_array.shape[1]), average_pol_no_std(recorded_opinions_array), alpha=0.8, linewidth=2.5, color='darkblue')
    # ax2.set_ylim(0, 130)
    # ax2.set_yticks((0, 25, 50, 75, 100, 125))
    # ax2.tick_params(axis='y', labelcolor='darkblue', labelsize=14)
    if file_path:
        plt.savefig(file_path)
    plt.close()


def average_pol(opinions_array):
        pol_per_step = np.array([
            [np.linalg.norm(opinions) for opinions in opinions_array[:, step, :]]
            for step in range(opinions_array.shape[1])
        ]).T
        mean_pol = np.mean(pol_per_step, axis=0)
        std_pol = np.std(pol_per_step, axis=0)
        return mean_pol, std_pol


def average_pol_no_std(opinions_array):
    # Calculate polarization for each time step across all simulations
    polarization_per_step = [
        np.mean([np.linalg.norm(opinions) for opinions in opinions_array[:, step, :]])
        for step in range(opinions_array.shape[1])
    ]
    return np.array(polarization_per_step)