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
    run_name=None, params_env=None, title="Polarization Over Time"
):
    list_policies_to_visualize = []
    if run_name is not None:
        run_dir = os.path.join(
            "saved files", "dqn", "nonlinear", "saved_runs", run_name
        )
        with open(os.path.join(run_dir, "params_env.json"), "r") as f:
            params_env = json.load(f)

        recorded_opinions_dqn = np.load(
            os.path.join(run_dir, "recorded_opinions_dqn.npy")
        )
        list_policies_to_visualize.append(("DQN", recorded_opinions_dqn))

    baselines_dir = os.path.join(
        "saved files",
        "dqn",
        "nonlinear",
        "baselines",
        f"n_nodes_{params_env['n']}_average_degree_{params_env['average_degree']}_n_updates{params_env['n_edge_updates_per_step']}",
    )

    if os.path.exists(baselines_dir):
        if os.path.exists(
            os.path.join(baselines_dir, "recorded_opinions_no_policy.npy")
        ):
            recorded_opinions_no_policy = np.load(
                os.path.join(baselines_dir, "recorded_opinions_no_policy.npy")
            )
            list_policies_to_visualize.append(
                ("No Policy", recorded_opinions_no_policy)
            )
        if os.path.exists(os.path.join(baselines_dir, "recorded_opinions_minmax.npy")):
            recorded_opinions_minmax = np.load(
                os.path.join(baselines_dir, "recorded_opinions_minmax.npy")
            )
            list_policies_to_visualize.append(("MinMax", recorded_opinions_minmax))
        if os.path.exists(
            os.path.join(baselines_dir, "recorded_opinions_softminmax.npy")
        ):
            recorded_opinions_softminmax = np.load(
                os.path.join(baselines_dir, "recorded_opinions_softminmax.npy")
            )
            list_policies_to_visualize.append(
                ("SoftMinMax", recorded_opinions_softminmax)
            )
        if os.path.exists(
            os.path.join(baselines_dir, "recorded_opinions_deleting.npy")
        ):
            recorded_opinions_deleting = np.load(
                os.path.join(baselines_dir, "recorded_opinions_deleting.npy")
            )
            list_policies_to_visualize.append(("Deleting", recorded_opinions_deleting))
        if os.path.exists(os.path.join(baselines_dir, "recorded_opinions_greedy.npy")):
            recorded_opinions_greedy = np.load(
                os.path.join(baselines_dir, "recorded_opinions_greedy.npy")
            )
            list_policies_to_visualize.append(("Greedy", recorded_opinions_greedy))
    else:
        print(f"Baselines directory does not exist. Skipping baseline visualizations.")

    if run_name is not None:
        save_path = os.path.join(run_dir, "polarization_over_time.png")
    else:
        save_path = os.path.join(baselines_dir, "polarization_over_time.png")

    visualize_polarizations(
        list_policies_to_visualize,
        file_path=save_path,
        title=title,
    )


def visualize_polarizations(
    list_opinions_strategies, file_path, title="Polarization Over Time"
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

    # Visualize the average polarization over time
    plt.figure(figsize=(20, 6))
    for strat, opinions_array in list_opinions_strategies:
        plt.plot(
            np.arange(0, opinions_array.shape[1]),
            average_pol(opinions_array),
            label=strat,
            alpha=0.8,
        )

    plt.xlabel("Time Steps")
    plt.ylabel("Polarization")
    plt.title(title)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.savefig(file_path)
    plt.close()
