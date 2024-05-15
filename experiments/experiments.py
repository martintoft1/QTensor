from qtensor.FullSimulation import full_sim
import networkx as nx
from datetime import datetime
import argparse
import time
import json
import pandas as pd
import os

### ----------- Convenience functions ----------- ###
def time_from_current_time(days, hours, minutes, seconds):
    """
    Calculates the time in seconds plus the current time.

    Args:
        days (int): The number of days.
        hours (int): The number of hours.
        minutes (int): The number of minutes.
        seconds (int): The number of seconds.

    Returns:
        float: The time in seconds plus the current time.
    """
    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
    return time.time() + total_seconds

def parse_arguments():
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Arguments for running an experiment.")
    parser.add_argument('type', type=str, help='Type of experiment (prelim, main)')
    parser.add_argument('number', type=int, help='Number of experiment')
    parser.add_argument('start_time', type=str, help='Start time in YYYYMMDD_HHMMSS format')
    parser.add_argument('iteration', type=int, help='Iteration of experiment')
    parser.add_argument('max_time', type=str, help='Max runtime in the format DD:HH:MM:SS')
    parser.add_argument('processes', type=int, help='Number of processes to use for experiment')
    parser.add_argument('hardware', type=str, help='Specify the partition or specific computer/hardware used')

    return parser.parse_args()

def experiment(type, number, start_time, iteration = 1, max_time = "00:00:00:10", processes = 1, hardware = "local"):
    """
    Starts the given experiment number, and names it based on the given iteration number and start time.

    Args:
        type (str): The type of experiment ('prelim' or 'main').
        number (int): The number of the experiment.
        start_time (str): The start time in 'YYYYMMDD_HHMMSS' format.
        iteration (int, optional): The iteration of the experiment. Defaults to 1.
        max_time (str, optional): The maximum runtime in 'DD:HH:MM:SS' format. Defaults to '00:00:00:10'.
        processes (int, optional): The number of processes to use for the experiment. Defaults to 1.
    """
    print(f"Running {type} experiment number {number}, iteration {iteration}, at {start_time}, for {max_time} days:hours:minutes:seconds, on hardware {hardware}.")
    filename = f"experiment_{number}_datetime_{start_time}_iteration_{iteration}"

    if type == 'prelim':
        if number == 1:
            preliminary_experiment_1(filename, max_time, processes)
    elif type == 'main':
        if number == 1:
            main_experiment_1(filename, max_time, processes)
        elif number == 2:
            main_experiment_2(filename, max_time, processes)

def load_results(filename):
    """
    Loads the results from a file.

    Args:
        filename (str): The name of the file to load the results from.

    Returns:
        dict: The loaded results.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def load_results_experiment(experiment, datetime, iterations=1):
    """
    Loads the results from a given experiment.

    Args:
        experiment (int): The number of the experiment.
        datetime (str): The datetime when the experiment was run.
        iterations (int, optional): The number of iterations of the experiment. Defaults to 1.

    Returns:
        list: The loaded results.
    """
    results = []
    for i in range(1, iterations+1):
        filename = f"experiment_{experiment}_datetime_{datetime}_iteration_{i}.json"
        results.append(load_results(filename))
    return results

def print_statistics_table(experiment, datetime, iterations=1):
    """
    Prints a statistics table for a given experiment.

    Args:
        experiment (int): The number of the experiment.
        datetime (str): The datetime when the experiment was run.
        iterations (int, optional): The number of iterations of the experiment. Defaults to 1.
    """
    # Load experiment data
    data = load_results_experiment(experiment, datetime, iterations)

    # Extract data from JSON and create a DataFrame
    results = []
    for experiment_id, exp_data in data.items():
        for res in exp_data["individual_simulation_results"]:
            results.append({
                'experiment': experiment_id,
                'iteration': res['iteration'],
                'time': res['time'],
                'expectation_value': res['expectation_value']
            })
    df = pd.DataFrame(results)
    
    # Group by experiment and calculate statistics
    stats = df.groupby('experiment').agg({
        'expectation_value': ['mean', 'median', 'min', 'std', lambda x: x.max() - x.min()],
        'time': ['mean', 'median', 'std', lambda x: x.max() - x.min()]
    })

    # Renaming columns for clarity
    stats.columns = ['Mean Energy', 'Median Energy', 'Best Energy', 'Std Dev Energy', 'Range Energy', 'Mean Time', 'Median Time', 'Std Dev Time', 'Range Time']
    
    print(stats)

### ----------- Experiment-specific functions ----------- ###
def load_graph_and_maxcut():
    # Load the dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base_dir, 'torusg3-8.dat')
    G = nx.Graph()
    with open(data_file, "r") as file:
        for line in file:
            # Extract the first vertex, second vertex, and edge weight
            data = line.strip().split() 
            if len(data) == 3:
                # Convert vertices to appropriate type (e.g., integer) and weight to float or integer
                v1 = int(data[0])
                v2 = int(data[1])
                weight = float(data[2])
                # Add the edge with the specified weight
                G.add_edge(v1, v2, weight=weight)

    # The optimal value for the graph is 457.358179
    optimal_value = 457.358179

    return G, optimal_value

def post_process_result_dimacs(result):
    # Multiply the result by -1 to convert from a minimization problem to a maximization problem, and divide the result by 100,000 as noted on the DIMACS website
    return result * -1 / 100000

### ----------- The experiments ----------- ###
def preliminary_experiment_1(filename, max_time, processes):
    """
        Experiment different preliminary configurations
    """
    G, optimal_value = load_graph_and_maxcut()
    days, hours, minutes, seconds = map(int, max_time.split(':'))

    full_sim(
        p = 2, n_processes = processes, ordering_algo = 'greedy', backend = 'numpy', ansatz_variant='qaoa', param_initializer = 'fourier', param_optimizer = 'differential_evolution',
        weighted = True, G = G, max_time = time_from_current_time(days, hours, minutes, seconds), # max_energy_expectation = 0.7,
        optimal_value = optimal_value, post_process_results=post_process_result_dimacs, # max_epochs=4, # max_tw=25, profile=True
        filename=filename
    )

def main_experiment_1(filename, max_time, processes):
    """
        Experiment with QAOA and final, optimal configuration 
    """
    G, optimal_value = load_graph_and_maxcut()
    days, hours, minutes, seconds = map(int, max_time.split(':'))

    full_sim(
        p = 2, n_processes = processes, ordering_algo = 'greedy', backend = 'numpy', ansatz_variant='qaoa', param_initializer = 'fourier', param_optimizer = 'differential_evolution',
        weighted = True, G = G, max_time = time_from_current_time(days, hours, minutes, seconds), # max_energy_expectation = 0.7,
        optimal_value = optimal_value, post_process_results=post_process_result_dimacs, # max_epochs=4, # max_tw=25, profile=True
        filename=filename
    )
    
    ...

def main_experiment_2(filename, max_time, processes):
    """
        Experiment with ma-QAOA and final, optimal configuration 
    """
    G, optimal_value = load_graph_and_maxcut()
    days, hours, minutes, seconds = map(int, max_time.split(':'))

    full_sim(
        p = 2, n_processes = processes, ordering_algo = 'greedy', backend = 'numpy', ansatz_variant='ma-qaoa', param_initializer = 'fourier', param_optimizer = 'differential_evolution',
        weighted = True, G = G, max_time = time_from_current_time(days, hours, minutes, seconds), # max_energy_expectation = 0.7,
        optimal_value = optimal_value, post_process_results=post_process_result_dimacs, # max_epochs=4, # max_tw=25, profile=True
        filename=filename
    )


### ----------- Run the given experiment ----------- ###
if __name__ == '__main__':
    args = parse_arguments()
    experiment(args.type, args.number, args.iteration, args.start_time, args.max_time, args.processes, args.hardware)