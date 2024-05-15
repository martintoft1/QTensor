import matplotlib.pyplot as plt
import json
import argparse
import pandas as pd
import numpy as np


### ----------- Plotting functions ----------- ###
def parse_arguments():
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Arguments for running an experiment.")
    parser.add_argument('title', type=str, help='Title of plot.')
    parser.add_argument('xaxis', type=str, help='What to show on x-axis. Either iteration or time.')
    parser.add_argument('files', nargs='+', help='List of file-names from simulation_results to plot.')

    return parser.parse_args()


def plot_result_experiments(title, xaxis, files):
    fig, ax1 = plt.subplots()

    for file in files:
        with open(f"experiments/simulation_results/{file}", "r") as f:
            data = json.load(f)

        y_axis_vals = [result["expectation_value"] for result in data["individual_simulation_results"]]
        ansatz_variant = data["metadata"]["ansatz_variant"]

        ax1.set_ylabel("Expectation value")

        if xaxis == "iteration":
            x_axis_vals1 = [result["iteration"] for result in data["individual_simulation_results"]]
            ax1.plot(x_axis_vals1, y_axis_vals, label=ansatz_variant)
            ax1.set_xlabel("Iteration")

        elif xaxis == "time":
            x_axis_vals2 = []
            cumulative_time = 0
            for result in data["individual_simulation_results"]:
                cumulative_time += result["time"]
                x_axis_vals2.append(cumulative_time)
            ax1.plot(x_axis_vals2, y_axis_vals, label=ansatz_variant)
            ax1.set_xlabel("Time (s)")

    plt.title(title)
    fig.tight_layout()
    plt.legend()
    plt.show()


import matplotlib.pyplot as plt

def create_statistics_table(title, files):
    table_data = []

    for file in files:
        with open(f"experiments/simulation_results/{file}", "r") as f:
            data = json.load(f)

        energy_expectations = [result["expectation_value"] for result in data["individual_simulation_results"]]
        mean_energy_expectation = np.mean(energy_expectations)
        median_energy_expectation = np.median(energy_expectations)
        std_dev_energy_expectation = np.std(energy_expectations)
        range_energy_expectation = np.ptp(energy_expectations)

        best_expectation_value = data["global_simulation_results"]["best_expectation_value"]
        average_time = data["global_simulation_results"]["average_time"]

        table_data.append([mean_energy_expectation, median_energy_expectation, best_expectation_value, average_time, std_dev_energy_expectation, range_energy_expectation])

    columns = ["Mean Energy Expectation", "Median Energy Expectation", "Best Expectation Value", "Average Time", "Standard Deviation of Energy Expectation", "Range of Energy Expectation"]
    fig, ax = plt.subplots(1, 1)
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=table_data, colLabels=columns, cellLoc = 'center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(6)

    plt.title(title)
    plt.show()


if __name__ == '__main__':
    #create_statistics_table("QAOA", ["experiment_1_datetime_20240515011309_iteration_1.json"])
    create_statistics_table("ma-QAOA", ["experiment_2_datetime_20240515045634_iteration_1.json"])
    #args = parse_arguments()
    #plot_result_experiments(args.title, args.xaxis, args.files)