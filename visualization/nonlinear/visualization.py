import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import json
import seaborn as sns
import csv

def visualize_graph(G, opinions, title="notitle", highlight_nodes=None, file_path=None):

    cmap = plt.cm.coolwarm  # Colormap for opinions
    norm = plt.Normalize(vmin=-10, vmax=10)  # Normalize opinions to the range [-10, 10]
    colors = [cmap(norm(opinion)) for opinion in opinions]

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)  # Use a consistent layout for the graph
    nx.draw(
        G,
        pos=pos,
        node_color=colors,
        with_labels=False,
        node_size=100,
        edge_color="gray",
    )

    # Highlight specific nodes if provided
    if highlight_nodes:
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            nodelist=[node for node in highlight_nodes if node is not None],
            node_color="yellow",
            node_size=20,
            # edgecolors='black',
            linewidths=2,
        )
    plt.title(title)
    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()
    plt.close()


def visualize_opinions(
    env, recorded_opinions_one_run, title="Opinion Evolution Over Time", file_path=None
):
    recorded_opinions_array = np.array(
        recorded_opinions_one_run
    )  # shape: (n_times, n_nodes)
    time_steps = np.arange(0, len(recorded_opinions_array))

    plt.figure(figsize=(20, 6))
    for node in range(env.n):
        plt.plot(time_steps, recorded_opinions_array[:, node], alpha=0.7)

    plt.xlabel("Time")
    plt.ylabel("Opinion")
    plt.title(title)
    plt.grid(True)
    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()
    plt.close()


def visualize_polarization_development_multiple_policies(
    run_name=None, run_folder=None, params_env=None, title="Polarization Over Time", folder="val"
):
    dict_policies_to_visualize = {}

    def load_recorded_opinions(file_path):
        evaluation_dir = os.path.join(file_path, folder)
        recorded_opinions_dqn = {}
        if os.path.exists(evaluation_dir):
            for fname in os.listdir(evaluation_dir):
                if fname.startswith("recorded_opinions_") and fname.endswith(".npy"):
                    n = int(fname[len("recorded_opinions_dqn_n") : -len(".npy")])
                    recorded_opinions_dqn[n] = np.load(os.path.join(evaluation_dir, fname))
        return recorded_opinions_dqn

    if run_name is not None:
        run_dir = os.path.join("results", "dqn", "nonlinear", "runs", run_name)
        with open(os.path.join(run_dir, "params_env.json"), "r") as f:
            params_env = json.load(f)
        recorded_opinions_dqn = load_recorded_opinions(run_dir)
        for n in recorded_opinions_dqn:
            dict_policies_to_visualize[n] = [("DQN", recorded_opinions_dqn[n])]

    if run_folder is not None:
        run_folder_dir = os.path.join("results", "dqn", "nonlinear", "runs", run_folder)
        for i, run_name_in_folder in enumerate(os.listdir(run_folder_dir)):
            if i == 0:
                run_dir_in_folder = os.path.join(run_folder_dir, run_name_in_folder)
                with open(os.path.join(run_dir_in_folder, "params_env.json"), "r") as f:
                    params_env = json.load(f)
                recorded_opinions_dqn = load_recorded_opinions(run_dir_in_folder)
                continue
            run_dir_in_folder = os.path.join(run_folder_dir, run_name_in_folder)
            recorded_opinions_dqn_i = load_recorded_opinions(run_dir_in_folder)
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