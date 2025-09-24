import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import matplotlib.patches as patches
import pandas as pd
import torch


def visualize_dqn_vs_greedy_ood_n(run_name, folder="val"):
    """
    Visualizes results of method compare_dqn_policy_to_greedy_various_n in evalutation/friedkin_johnson/evaluation,
    i.e., visualizes performance differences between dqn solution and greedy solutions for various n and ks.
    Args:
        run_name (str): The name of the run to visualize.
    """

    run_dir = os.path.join("results", "dqn", "friedkin-johnson", "runs", run_name)

    with open(os.path.join(run_dir, "params_env.json"), "r") as f:
        params_env = json.load(f)

    with open(
        os.path.join(run_dir, folder, f"evaluation_comparison_to_greedy.pkl"), "rb"
    ) as f:
        results = pickle.load(f)

    # Define the ranges
    n_values = sorted({n for n, _ in results.keys()})
    k_values = sorted({k for _, k in results.keys()})

    # Create matrices for annotations and heatmap coloring
    annot_matrix = []
    color_matrix = []

    all_dqn_means, all_greedy_means, all_random_means = [], [], []

    for k in k_values:
        row_annot = []
        row_color = []
        for n in n_values:
            key = (n, k)
            if key in results:
                r = results[key]
                dqn_mean = np.mean(r["dqn"])
                greedy_mean = np.mean(r["greedy"])
                random_mean = np.mean(r["random"])

                diff = dqn_mean - greedy_mean

                all_dqn_means.append(dqn_mean)
                all_greedy_means.append(greedy_mean)
                all_random_means.append(random_mean)

                # Multiline annotation text
                annot = (
                    f"Δ={diff:.3f}\n"
                    f"DQN={dqn_mean:.3f}\n"
                    f"G={greedy_mean:.3f}\n"
                    f"R={random_mean:.3f}\n"
                    f"DQN better={r['dqn_better']:.2f}\n"
                    f"G better={r['greedy_better']:.2f}"
                )

                row_annot.append(annot)
                row_color.append(diff)
            else:
                row_annot.append("")
                row_color.append(np.nan)
        annot_matrix.append(row_annot)
        color_matrix.append(row_color)

    # overall averages
    overall_dqn = np.mean(all_dqn_means)
    overall_greedy = np.mean(all_greedy_means)
    overall_random = np.mean(all_random_means)

    # Convert to DataFrame for seaborn
    annot_df = pd.DataFrame(annot_matrix, index=k_values, columns=n_values)
    color_df = pd.DataFrame(color_matrix, index=k_values, columns=n_values)

    # Plotting
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        color_df,
        annot=annot_df,
        fmt="",
        cmap="RdYlGn_r",
        cbar_kws={"label": "Average Difference (DQN − Greedy)"},
        linewidths=0.5,
        linecolor="gray",
        vmin=0,
        vmax=0.5,
        annot_kws={"size": 7, "ha": "center", "va": "center"},
    )

    # Add black rectangle around cell for which we trained
    n_target = params_env["n"]
    k_target = params_env["k"]
    row_idx = k_values.index(k_target)
    col_idx = n_values.index(n_target)
    rect = patches.Rectangle(
        (col_idx, row_idx), 1, 1, fill=False, edgecolor="black", linewidth=3
    )
    ax.add_patch(rect)

    plt.xlabel("n")
    plt.ylabel("k")
    plt.title(
        f"Overall Means → DQN={overall_dqn:.3f}, Greedy={overall_greedy:.3f}, Random={overall_random:.3f}",
        fontsize=12,
    )
    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(run_dir, folder, "heatmap_dqn_vs_greedy.png")
    plt.savefig(save_path, dpi=300)
    plt.close()




# def visualize_dqn_vs_greedy_ood_n(run_name, folder="val"):
#     """
#     Visualizes results of method compare_dqn_policy_to_greedy_various_n in evalutation/friedkin_johnson/evaluation,
#         i.e., visualizes performance differences between dqn solution and greedy solutions for various n and ks.
#     Args:
#         run_name (str): The name of the run to visualize.
#     """

#     run_dir = os.path.join("results", "dqn", "friedkin-johnson", "runs", run_name)

#     with open(os.path.join(run_dir, "params_env.json"), "r") as f:
#         params_env = json.load(f)

#     with open(
#         os.path.join(run_dir, folder, f"evaluation_comparison_to_greedy.pkl"), "rb"
#     ) as f:
#         results = pickle.load(f)

#     # Define the ranges
#     n_values = list({n for n, _ in results.keys()})
#     k_values = list({k for _, k in results.keys()})
#     n_values.sort()
#     k_values.sort()

#     # Create matrices for annotations and heatmap coloring
#     annot_matrix = []
#     color_matrix = []

#     diffs = []
#     for k in k_values:
#         row_annot = []
#         row_color = []
#         for n in n_values:
#             key = (n, k)
#             if key in results:
#                 r = results[key]
#                 # num = r["number_states"]
#                 dqn_mean = np.mean(r["dqn"])
#                 greedy_mean = np.mean(r["greedy"])
#                 random_mean = np.mean(r["random"])
#                 diff = (dqn_mean - greedy_mean)
#                 diffs.append(diff)

#                 # Annotation text
#                 annot = f"{diff:.4f}"

#                 row_annot.append(annot)
#                 row_color.append(diff)
#             else:
#                 row_annot.append("")
#                 row_color.append(np.nan)
#         annot_matrix.append(row_annot)
#         color_matrix.append(row_color)

#     # overall average difference
#     overall_avg_diff = np.nanmean(diffs)

#     # Convert to DataFrame for seaborn
#     annot_df = pd.DataFrame(annot_matrix, index=k_values, columns=n_values)
#     color_df = pd.DataFrame(color_matrix, index=k_values, columns=n_values)

#     # Plotting
#     plt.figure(figsize=(8, 4))
#     ax = sns.heatmap(
#         color_df,
#         annot=annot_df,
#         fmt="",
#         cmap="RdYlGn_r",
#         cbar_kws={"label": "Average Difference in Polarization Gain"},
#         linewidths=0.5,
#         linecolor="gray",
#         vmin=0,
#         vmax=0.5,
#     )

#     # Add black rectangle around cell for which we trained
#     n_target = params_env["n"]
#     k_target = params_env["k"]
#     row_idx = k_values.index(k_target)  # Get the row and column indices in the matrix
#     col_idx = n_values.index(n_target)
#     rect = patches.Rectangle(
#         (col_idx, row_idx), 1, 1, fill=False, edgecolor="black", linewidth=3
#     )  # Rectangle parameters: (x, y) is the bottom left of the cell
#     ax.add_patch(rect)

#     plt.xlabel("n")
#     plt.ylabel("k")
#     plt.title(f"Overall Average: {overall_avg_diff:.4f}")
#     plt.tight_layout()

#     # Save the figure to the specified path
#     save_path = os.path.join(run_dir, folder, "heatmap_dqn_vs_greedy.png")
#     plt.savefig(save_path)
#     plt.close()


def visualize_dqn_vs_greedy_ood_n_simple(run_name, folder="val"):
    """
    Visualizes results of method compare_dqn_policy_to_greedy_various_n in evalutation/friedkin_johnson/evaluation,
        i.e., visualizes performance differences between dqn solution and greedy solutions for various n and ks.
    Args:
        run_name (str): The name of the run to visualize.
    """

    run_dir = os.path.join("results", "dqn", "friedkin-johnson", "runs", run_name)

    with open(os.path.join(run_dir, "params_env.json"), "r") as f:
        params_env = json.load(f)

    with open(
        os.path.join(run_dir, folder, f"evaluation_comparison_to_greedy.pkl"), "rb"
    ) as f:
        results = pickle.load(f)

    # Define the ranges
    n_values = list({n for n, _ in results.keys()})
    k_values = list({k for _, k in results.keys()})
    n_values.sort()
    k_values.sort()

    # Create matrices for annotations and heatmap coloring
    color_matrix = []

    for k in k_values:
        row_color = []
        for n in n_values:
            key = (n, k)
            if key in results:
                r = results[key]
                dqn_mean = np.mean(r["dqn"])
                greedy_mean = np.mean(r["greedy"])
                diff = (dqn_mean - greedy_mean)
                if diff > 0.5:
                    diff = 0.5

                row_color.append(diff)
            else:
                row_color.append(np.nan)
        color_matrix.append(row_color)

    # Convert to DataFrame for seaborn
    color_df = pd.DataFrame(color_matrix, index=k_values, columns=n_values)

    # Plotting
    plt.figure(figsize=(3, 2))
    ax = sns.heatmap(
        color_df,
        fmt="",
        cmap="RdYlGn_r",
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
        vmin=0,
        vmax=0.5,
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

    # Axis labels with larger font
    plt.xlabel("n", fontsize=14.5)
    plt.ylabel("k", fontsize=14.5)

    # Tick labels bigger
    ax.tick_params(axis="both", labelsize=12.5)
    plt.tight_layout()

    # Save the figure to the specified path
    save_path = os.path.join(run_dir, folder, "heatmap_dqn_vs_greedy_simple.png")
    plt.savefig(save_path)
    plt.close()


def visualize_variance_ood_n(run_name, folder="val"):
    """
    Visualizes results of method compare_dqn_policy_to_greedy_various_n in evalutation/friedkin_johnson/evaluation,
        i.e., visualizes performance differences between dqn solution and greedy solutions for various n and ks.
    Args:
        run_name (str): The name of the run to visualize.
    """

    run_dir = os.path.join("results", "dqn", "friedkin-johnson", "runs", run_name)

    with open(os.path.join(run_dir, "params_env.json"), "r") as f:
        params_env = json.load(f)

    with open(
        os.path.join(run_dir, f"evaluation_comparison_to_greedy_variance_{folder}.pkl"),
        "rb",
    ) as f:
        variances = pickle.load(f)

    # Define the ranges
    n_values = list({n for n, _ in variances.keys()})
    k_values = list({k for _, k in variances.keys()})
    n_values.sort()
    k_values.sort()

    # Create matrices for annotations and heatmap coloring
    annot_matrix = []
    color_matrix = []

    variances_list = []
    for k in k_values:
        row_annot = []
        row_color = []
        for n in n_values:
            key = (n, k)
            if key in variances:
                standarddeviation = np.sqrt(variances[key])

                row_annot.append(f"{standarddeviation:.4f}")
                row_color.append(standarddeviation)
                variances_list.append(standarddeviation)
            else:
                row_annot.append("")
                row_color.append(np.nan)
        annot_matrix.append(row_annot)
        color_matrix.append(row_color)

    # Convert to DataFrame for seaborn
    annot_df = pd.DataFrame(annot_matrix, index=k_values, columns=n_values)
    color_df = pd.DataFrame(color_matrix, index=k_values, columns=n_values)

    # overall average variance
    overall_avg_variance = np.nanmean(variances_list)

    # Plotting
    plt.figure(figsize=(8, 4))
    ax = sns.heatmap(
        color_df,
        annot=annot_df,
        fmt="",
        cmap="RdYlGn_r",
        cbar_kws={"label": "Standard Deviations of final Polarizations"},
        linewidths=0.5,
        linecolor="gray",
        vmin=0,
        vmax=0.5,
    )

    plt.xlabel("n")
    plt.ylabel("k")
    plt.title(f"Overall Average Standard Deviation: {overall_avg_variance:.4f}")
    plt.tight_layout()

    # Save the figure to the specified path
    save_path = os.path.join(run_dir, folder, "heatmap_dqn_vs_greedy_variances.png")
    plt.savefig(save_path)
    plt.close()


def performance_overview(run_name, folder="val"):
    """ """

    run_dir = os.path.join("results", "dqn", "friedkin-johnson", "runs", run_name)

    evaluation_dir = os.path.join(
        "data", "friedkin-johnson", "greedy_solutions", folder
    )

    with open(os.path.join(run_dir, "params_env.json"), "r") as f:
        params_env = json.load(f)

    with open(
        os.path.join(run_dir, folder, f"evaluation_comparison_to_greedy.pkl"), "rb"
    ) as f:
        results = pickle.load(f)

    # Define the ranges
    n_values = list({n for n, _ in results.keys()})
    k_values = list({k for _, k in results.keys()})
    n_values.sort()
    k_values.sort()

    # n, k for which we trained
    n_target = params_env["n"]
    k_target = params_env["k"]

    overview = {}
    sum_greedy = 0
    sum_dqn = 0
    sum_random = 0
    sum_settings = 0
    for k in k_values:
        for n in n_values:
            # with open(
            #     os.path.join(
            #         evaluation_dir,
            #         f"greedy_solutions_n{n}_d{params_env['average_degree']}_k{k}.pt",
            #     ),
            #     "rb",
            # ) as f:
            #     greedy_solutions = torch.load(f, weights_only=False)
            # sum_greedy_n_k = np.sum([sol[1] for sol in greedy_solutions])
            # num_states = results[(n, k)]["number_states"]

            greedy_mean = np.mean(results[(n, k)]["greedy"])
            dqn_mean = np.mean(results[(n, k)]["dqn"])
            random_mean = np.mean(results[(n, k)]["random"])
            num_states = results[(n, k)]["number_states"]

            sum_greedy += greedy_mean
            sum_dqn += dqn_mean
            sum_random += random_mean
            sum_settings += 1

            if n == n_target and k == k_target:
                overview["target"] = {
                    "greedy": greedy_mean,
                    "dqn": dqn_mean,
                    "random": random_mean,
                    "num_states": num_states,
                }
    overview["overall"] = {
        "greedy": sum_greedy / sum_settings,
        "dqn": sum_dqn / sum_settings,
        "random": sum_random / sum_settings,
        "num_settings": sum_settings,
    }

    os.makedirs(os.path.join(run_dir, folder), exist_ok=True)

    with open(os.path.join(run_dir, folder, f"performance_overview.json"), "w") as f:
        json.dump(overview, f, indent=4)


def visualize_dqn_vs_greedy(run_name, n, k, folder="val"):
    """Visualizes the performance of DQN policy compared to greedy solutions for a single setting of FJ-Depolarize.
    Args:
        run_name (str): Name of the run to evaluate.
    """
    run_dir = os.path.join("results", "dqn", "friedkin-johnson", "runs", run_name)

    with open(
        os.path.join(run_dir, folder, f"evaluation_comparison_to_greedy.pkl"), "rb"
    ) as f:
        results = pickle.load(f)

    # Sort the polarization gains by greedy gain
    results_single_setting = results[(n, k)]

    polarization_gains = sorted(
        zip(
            results_single_setting["before"],
            results_single_setting["dqn"],
            results_single_setting["greedy"],
            results_single_setting["random"],
        ),
        key=lambda x: x[2],  # Sort by greedy
        reverse=False,
    )

    # Prepare data for plotting
    data = {"Index": [], "Polarization": [], "Method": []}

    for i, (
        no_change,
        polarization_dqn,
        polarization_greedy,
        polarization_random,
    ) in enumerate(polarization_gains, start=1):
        data["Index"] += [i, i, i, i]
        data["Polarization"] += [
            no_change,
            polarization_random,
            polarization_greedy,
            polarization_dqn,
        ]
        data["Method"] += ["No Intervention", "Random", "Greedy", "DQN"]

    df = pd.DataFrame(data)

    # Plot with Seaborn
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(5, 3))
    plt.xlim(0, 4)
    sns.scatterplot(data=df, y="Index", x="Polarization", hue="Method", s=10)

    # plt.title("Polarization Gains ordered by Maximum Gain")
    # plt.xlabel("Sorted Index")
    # plt.ylabel("Polarization Gain")
    plt.tight_layout()

    # Save the figure to the specified path
    save_path = os.path.join(run_dir, folder, "evaluation_single_setting.png")
    plt.savefig(save_path)
    plt.close()
