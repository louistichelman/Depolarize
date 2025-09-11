import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import json


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
    run_name=None, params_env=None, title="Polarization Over Time", folder="val"
):
    dict_policies_to_visualize = {}
    if run_name is not None:
        run_dir = os.path.join("results", "dqn", "nonlinear", "runs", run_name)
        with open(os.path.join(run_dir, "params_env.json"), "r") as f:
            params_env = json.load(f)

        evaluation_dir = os.path.join(run_dir, folder)
        if os.path.exists(evaluation_dir):
            for fname in os.listdir(evaluation_dir):
                if fname.startswith("recorded_opinions_") and fname.endswith(".npy"):
                    n = int(fname[len("recorded_opinions_dqn_n") : -len(".npy")])
                    recorded_opinions_dqn = np.load(os.path.join(evaluation_dir, fname))
                    dict_policies_to_visualize[n] = [("DQN", recorded_opinions_dqn)]

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
                        ("No Policy", recorded_opinions_no_policy)
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
                        baselines_dir, fname, "recorded_opinions_softminmax.npy"
                    )
                ):
                    recorded_opinions_softminmax = np.load(
                        os.path.join(
                            baselines_dir, fname, "recorded_opinions_softminmax.npy"
                        )
                    )
                    dict_policies_to_visualize.setdefault(n, []).append(
                        ("SoftMinMax", recorded_opinions_softminmax)
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
        save_path = os.path.join(run_dir, folder, "polarization_over_time.png")
    else:
        save_path = os.path.join(baselines_dir, "polarization_over_time.png")

    visualize_polarizations(
        dict_policies_to_visualize,
        file_path=save_path,
        title=title,
    )


# def visualize_polarizations(
#     list_policies_to_visualize, file_path, title="Polarization Over Time"
# ):

#     def average_pol(opinions_array):
#         # Calculate polarization for each time step across all simulations
#         polarization_per_step = [
#             np.mean(
#                 [np.linalg.norm(opinions) for opinions in opinions_array[:, step, :]]
#             )
#             for step in range(opinions_array.shape[1])
#         ]
#         return np.array(polarization_per_step)

#     # Visualize the average polarization over time
#     plt.figure(figsize=(20, 6))
#     for strat, opinions_array in list_policies_to_visualize:
#         plt.plot(
#             np.arange(0, opinions_array.shape[1]),
#             average_pol(opinions_array),
#             label=strat,
#             alpha=0.8,
#         )

#     plt.xlabel("Time Steps")
#     plt.ylabel("Polarization")
#     plt.title(title)
#     plt.grid(True)
#     plt.legend(fontsize=18)
#     plt.savefig(file_path)
#     plt.close()


def visualize_polarizations(
    dict_policies_to_visualize, file_path, title="Polarization Over Time"
):
    def average_pol(opinions_array):
        # Calculate polarization for each time step across all simulations
        polarization_per_step = [
            np.mean(
                [np.linalg.norm(opinions) for opinions in opinions_array[:, step, :]]
            )
            for step in range(opinions_array.shape[1])
        ]
        return np.array(polarization_per_step)

    num_plots = len(dict_policies_to_visualize)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), sharey=True)

    # Ensure axes is iterable even if num_plots=1
    if num_plots == 1:
        axes = [axes]

    for ax, (key, list_policies) in zip(
        axes, sorted(dict_policies_to_visualize.items())
    ):
        for strat, opinions_array in list_policies:
            ax.plot(
                np.arange(0, opinions_array.shape[1]),
                average_pol(opinions_array),
                label=strat,
                alpha=0.8,
            )
        ax.set_xlabel("Time Steps")
        ax.set_title(f"{title} (n={key})")
        ax.grid(True)
        ax.legend(fontsize=10)

    axes[0].set_ylabel("Polarization")
    fig.tight_layout()
    plt.savefig(file_path)
    plt.close()
