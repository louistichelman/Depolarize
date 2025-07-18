import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os


def visualize_graph(G, opinions, title="notitle", highlight_nodes=None, file_path=None):
    # Map opinions to colors using a colormap
    cmap = plt.cm.bwr  # Blue-White-Red colormap
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
            node_color='yellow',
            node_size=20,
            # edgecolors='black',
            linewidths=2
        )
    plt.title(title)
    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()
    plt.close()
    


def visualize_opinions(env, recorded_opinions_one_run, title="Opinion Evolution Over Time", file_path=None):
    recorded_opinions_array = np.array(recorded_opinions_one_run) # shape: (n_times, n_nodes)
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

def visualize_polarizations_saved_runs(run_name, title="Polarization Over Time"):

    run_dir = os.path.join("saved files", "dqn", "nonlinear", "saved_runs", run_name)

    # Load recorded opinions from the saved file
    recorded_opinions_dqn = np.load(os.path.join(run_dir, "recorded_opinions_dqn.npy"))
    recorded_opinions_no_policy = np.load(os.path.join(run_dir, "baselines", "recorded_opinions_no_policy.npy"))
    recorded_opinions_minmax = np.load(os.path.join(run_dir, "baselines", "recorded_opinions_minmax.npy"))
    recorded_opinions_softminmax = np.load(os.path.join(run_dir, "baselines", "recorded_opinions_softminmax.npy"))
    recorded_opinions_deleting = np.load(os.path.join(run_dir, "baselines", "recorded_opinions_deleting.npy"))

    visualize_polarizations([("DQN", recorded_opinions_dqn),
                             ("No Policy", recorded_opinions_no_policy),
                             ("MinMax", recorded_opinions_minmax),
                             ("SoftMinMax", recorded_opinions_softminmax),
                             ("Deleting", recorded_opinions_deleting)],
                            file_path=os.path.join(run_dir, "polarization_over_time.png"),
                            title=title)


def visualize_polarizations(list_opinions_strategies, file_path, title="Polarization Over Time"):

    def average_pol(opinions_array):
        # Calculate polarization for each time step across all simulations
        polarization_per_step = [
            np.mean([np.linalg.norm(opinions) for opinions in opinions_array[:, step, :]])
            for step in range(opinions_array.shape[1])
        ]
        return np.array(polarization_per_step)

    # Visualize the average polarization over time
    plt.figure(figsize=(20, 6))
    for (strat, opinions_array) in list_opinions_strategies:
        plt.plot(np.arange(0, opinions_array.shape[1]), average_pol(opinions_array), label=strat, alpha=0.8)

    plt.xlabel("Time Steps")
    plt.ylabel("Polarization")
    plt.title(title)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.savefig(file_path)
    plt.close()
