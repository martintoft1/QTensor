import matplotlib.pyplot as plt
import json
import argparse



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


if __name__ == '__main__':
    args = parse_arguments()
    plot_result_experiments(args.title, args.xaxis, args.files)