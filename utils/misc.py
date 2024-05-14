import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ["get_args", "plots"]


def get_args(parser):
    """
    Adds command line arguments to the parser

    :param parser: An instance of : class : ` argparse. ArgumentParser `.
    :returns: A tuple of the same length as the argument list
    """
    parser.add_argument(
        "--game",
        default="Coor",
        type=str,
        choices=["Coor", "CPD", "PG", "CR"],
        help="Coordination (Coor), Continuous Prisoner's Dilemma (CPD), Public Goods (PG) or Collective Risk (CR) (default: Coor)",
    )

    parser.add_argument(
        "--groups",
        default=20,
        type=int,
        help="number of groups (default: 20)",
    )

    parser.add_argument(
        "--group-size",
        default=12,
        type=int,
        help="number of agents per group (default: 12)",
    )

    parser.add_argument(
        "--rounds",
        default=100,
        type=int,
        help="number of simulation rounds (default: 100)",
    )

    parser.add_argument(
        "--pnb",
        default="normal",
        type=str,
        help="pnb distribution (default: normal)",
    )

    parser.add_argument(
        "--B",
        default="normal",
        type=str,
        help="coefficient B distribution (default: normal)",
    )

    parser.add_argument(
        "--C",
        default=0.1,
        type=float,
        help="coefficient C value (default: 0.1)",
    )

    parser.add_argument(
        "--p",
        default=0.5,
        type=float,
        help="probability p (default: 0.5)",
    )

    parser.add_argument(
        "--update-amount",
        default=0.01,
        type=float,
        help="RL update amount (default: 0.01)",
    )

    return parser.parse_args()


def plots(
    output_directory,
    all_coefficients,
    all_indiv_actions,
    all_indiv_actions_last,
    all_group_actions,
    all_pnb,
):
    """
    Plots the results of the simulation

    :param output_directory: The directory to save the plots to
    :param all_coefficients: The data to be plotted are passed in the following parameters as lists or lists of lists
    :param all_indiv_actions
    :param all_indiv_actions_last
    :param all_group_actions
    :param all_pnb
    """
    ## Plots
    plt.style.use("seaborn-v0_8-deep")
    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["font.size"] = 14

    # Convert data to DataFrame
    df = pd.DataFrame(all_coefficients, columns=["B0", "B1", "B2", "B3"])

    # Calculate mean and variance
    means = df.mean()
    variances = df.var()

    # Melt the DataFrame to long format for seaborn plotting
    melted_df = df.melt(var_name="Coefficient", value_name="Value")

    # Plot using seaborn's bar plot with error bars
    sns.barplot(
        x="Coefficient",
        y="Value",
        data=melted_df,
        errorbar="sd",
        palette=["red", "blue", "green", "orange"],
    )
    plt.errorbar(
        x=np.arange(len(means)),
        y=means,
        yerr=np.sqrt(variances),
        fmt="none",
        c="black",
        capsize=5,
        capthick=2,
    )
    plt.xlabel("Coefficient")
    plt.ylabel("Value")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_directory, "coeffs.png"))
    plt.close()

    # Create subplots with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # Adjust figure size if needed

    # Plot histogram of B0 values
    sns.histplot(df["B0"], bins=30, color="red", alpha=0.5, ax=axes[0, 0])
    axes[0, 0].set_title("Frequency of Coefficient B[0] Values")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_xlim(0, 1)

    # Plot histogram of B1 values
    sns.histplot(df["B1"], bins=30, color="blue", alpha=0.5, ax=axes[0, 1])
    axes[0, 1].set_title("Frequency of Coefficient B[1] Values")
    axes[0, 1].set_xlabel("Value")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_xlim(0, 1)

    # Plot histogram of B2 values
    sns.histplot(df["B2"], bins=30, color="green", alpha=0.5, ax=axes[1, 0])
    axes[1, 0].set_title("Frequency of Coefficient B[2] Values")
    axes[1, 0].set_xlabel("Value")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_xlim(0, 1)

    # Plot histogram of B3 values
    sns.histplot(df["B3"], bins=30, color="orange", alpha=0.5, ax=axes[1, 1])
    axes[1, 1].set_title("Frequency of Coefficient B[3] Values")
    axes[1, 1].set_xlabel("Value")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_xlim(0, 1)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "frequency_coeffs.png"))
    plt.close()

    # Calculate the mean along the columns (mean by round)
    mean_values = np.mean(all_group_actions, axis=0)

    # Calculate the variance along the columns (var by round)
    variance_values = np.var(all_group_actions, axis=0)

    # Plot the means with error bars representing the standard deviation (sqrt of variance)
    plt.errorbar(
        x=np.arange(len(mean_values)),
        y=mean_values,
        yerr=np.sqrt(variance_values),
        fmt="o",
        c="black",
        capsize=3,
        capthick=0.5,
        linewidth=0.5,
    )
    plt.xlabel("Round")
    plt.ylabel("Action Mean Value")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_directory, "mean_var_action.png"))
    plt.close()

    # Plotting the histograms side by side
    plt.figure(figsize=(12, 6))

    # Plot histogram for all rounds
    plt.subplot(1, 2, 1)
    plt.hist(
        all_indiv_actions,
        bins=20,
        density=True,
        alpha=0.5,
        color="blue",
        label="All Rounds",
    )
    plt.title("All Rounds")
    plt.xlabel("Action Value")
    plt.ylabel("Density")
    plt.xlim(0, 1)

    # Plot histogram for the final round
    plt.subplot(1, 2, 2)
    plt.hist(
        all_indiv_actions_last,
        bins=20,
        density=True,
        alpha=0.5,
        color="green",
        label="Final Round",
    )
    plt.title("Final Round")
    plt.xlabel("Action Value")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "frequency_actions.png"))
    plt.close()

    """
    # Scatter plot of B0 vs B1
    B0_values = df["B0"]
    B1_values = df["B1"]

    # Fit a line (linear regression)
    coefficients = np.polyfit(B0_values, B1_values, 1)
    m = coefficients[0]  # Slope of the line
    b = coefficients[1]  # Intercept of the line

    # Create scatter plot
    plt.scatter(B0_values, B1_values, color="blue", label="Data Points")

    # Plot the line of best fit
    plt.plot(B0_values, m * B0_values + b, color="red", label="Line of Best Fit")

    plt.xlabel("B0")
    plt.ylabel("B1")
    plt.title("Scatterplot of B0 vs B1")
    plt.grid(True)
    plt.savefig(os.path.join(output_directory, "scatter_B0_B1.png"))
    plt.close()
    """

    plt.figure(figsize=(12, 6))

    # Plot scatterplot for B0 vs individual actions for final round
    plt.subplot(2, 2, 1)
    plt.scatter(
        df["B0"],
        all_indiv_actions_last,
        color="red",
        label="B0 vs Actions (Final Round)",
        alpha=0.5,
    )
    plt.xlabel("B0")
    plt.ylabel("Individual Actions")
    plt.title("Actions for Final Round vs B0")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)

    # Plot scatterplot for B1 vs individual actions for final round
    plt.subplot(2, 2, 2)
    plt.scatter(
        df["B1"],
        all_indiv_actions_last,
        color="blue",
        label="B1 vs Actions (Final Round)",
        alpha=0.5,
    )
    plt.xlabel("B1")
    plt.ylabel("Individual Actions")
    plt.title("Actions for Final Round vs B1")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)

    # Plot scatterplot for B2 vs individual actions for final round
    plt.subplot(2, 2, 3)
    plt.scatter(
        df["B2"],
        all_indiv_actions_last,
        color="green",
        label="B2 vs Actions (Final Round)",
        alpha=0.5,
    )
    plt.xlabel("B2")
    plt.ylabel("Individual Actions")
    plt.title("Actions for Final Round vs B2")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)

    # Plot scatterplot for B3 vs individual actions for final round
    plt.subplot(2, 2, 4)
    plt.scatter(
        df["B3"],
        all_indiv_actions_last,
        color="orange",
        label="B3 vs Actions (Final Round)",
        alpha=0.5,
    )
    plt.xlabel("B3")
    plt.ylabel("Individual Actions")
    plt.title("Actions for Final Round vs B3")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "actions_vs_coeffs.png"))
    plt.close()

    # Plot coefficients when all_indiv_actions_last is 1
    filter_condition0 = lambda x: all_indiv_actions_last[x] == 1
    filtered_coefficients1_list = list(
        filter(filter_condition0, range(len(all_indiv_actions_last)))
    )
    coefficients_action1 = [all_coefficients[i] for i in filtered_coefficients1_list]

    # Convert data to DataFrame
    df_filtered = pd.DataFrame(coefficients_action1, columns=["B0", "B1", "B2", "B3"])

    means = df_filtered.mean()
    variances = df_filtered.var()

    # Melt the DataFrame to long format for seaborn plotting
    melted_df = df_filtered.melt(var_name="Coefficient", value_name="Value")

    # Plot using seaborn's bar plot with error bars
    sns.barplot(
        x="Coefficient",
        y="Value",
        data=melted_df,
        errorbar="sd",
        palette=["red", "blue", "green", "orange"],
    )
    plt.errorbar(
        x=np.arange(len(means)),
        y=means,
        yerr=np.sqrt(variances),
        fmt="none",
        c="black",
        capsize=5,
        capthick=2,
    )
    plt.xlabel("Coefficient")
    plt.ylabel("Value")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_directory, "actions1_vs_coeffs.png"))
    plt.close()

    # Plot coefficients when all_indiv_actions_last is in [0, 1)
    filter_condition0 = lambda x: 0 <= all_indiv_actions_last[x] < 1
    filtered_coefficients0_list = list(
        filter(filter_condition0, range(len(all_indiv_actions_last)))
    )
    coefficients_action0 = [all_coefficients[i] for i in filtered_coefficients0_list]

    # Convert data to DataFrame
    df_filtered = pd.DataFrame(coefficients_action0, columns=["B0", "B1", "B2", "B3"])

    means = df_filtered.mean()
    variances = df_filtered.var()

    # Melt the DataFrame to long format for seaborn plotting
    melted_df = df_filtered.melt(var_name="Coefficient", value_name="Value")

    # Plot using seaborn's bar plot with error bars
    sns.barplot(
        x="Coefficient",
        y="Value",
        data=melted_df,
        errorbar="sd",
        palette=["red", "blue", "green", "orange"],
    )
    plt.errorbar(
        x=np.arange(len(means)),
        y=means,
        yerr=np.sqrt(variances),
        fmt="none",
        c="black",
        capsize=5,
        capthick=2,
    )
    plt.xlabel("Coefficient")
    plt.ylabel("Value")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_directory, "actions0_vs_coeffs.png"))
    plt.close()

    # Plot actions vs. pnb
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.scatter(
        all_indiv_actions_last, all_pnb, color="blue", alpha=0.5
    )  # Scatter plot

    # Adding labels and title
    plt.xlabel("Last Individual Actions")
    plt.ylabel("pnb")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.savefig(os.path.join(output_directory, "actions_vs_pnb.png"))
    plt.close()
