import time
import networkx as nx
import numpy as np
import random
import scipy
import json
from datetime import datetime
import os
import json
import os
import psutil
import GPUtil

from qtensor.toolbox import get_ordering_algo
from qtensor.FeynmanSimulator import FeynmanSimulator
from qtensor.ProcessingFrameworks import PerfBackend
from qtensor import SimpZZQtreeComposer, WeightedSimpZZQtreeQAOAComposer, SimpMaQtreeComposer, WeightedSimpMaQtreeComposer
from qtensor.QAOASimulator import QAOAQtreeSimulator, WeightedQAOASimulator
import qtensor.ProcessingFrameworks as backends


### ----------- Convenience functions for storing available system resources   ----------- ###
def get_cpu_name():
    with open("/proc/cpuinfo", "r") as f:
        for line in f:
            if "model name" in line:
                return line.split(":")[1].strip()
    return "CPU name not found"

def save_cpu_resources():
    """
    Retrieves and formats information about the system's CPU resources.

    Returns:
    dict: A dictionary containing information about the system's CPU resources, including the number of physical and total cores, 
    CPU frequencies, CPU usage per core, and total CPU usage.
    """
    cpu_resources = {}
    cpu_resources["Name"] = get_cpu_name()
    cpu_resources["Physical cores"] = psutil.cpu_count(logical=False)
    cpu_resources["Total cores"] = psutil.cpu_count(logical=True)

    cpufreq = psutil.cpu_freq()
    cpu_resources["Max Frequency"] = f"{cpufreq.max/1000:.2f}Ghz"
    cpu_resources["Min Frequency"] = f"{cpufreq.min/1000:.2f}Ghz"
    cpu_resources["Current Frequency"] = f"{cpufreq.current/1000:.2f}Ghz"

    cpu_resources["CPU Usage Per Core"] = [f"{percentage}%" for percentage in psutil.cpu_percent(percpu=True)]
    cpu_resources["Total CPU Usage"] = f"{psutil.cpu_percent()}%"

    return cpu_resources

def save_memory_resources():
    """
    Retrieves and formats information about the system's memory resources.

    Returns:
    dict: A dictionary containing information about the system's memory resources, including total, available, and used memory, 
    and the percentage of memory used.
    """
    memory_resources = {}
    svmem = psutil.virtual_memory()
    memory_resources["Total"] = f"{svmem.total / (1024 ** 3):.2f} GB"
    memory_resources["Available"] = f"{svmem.available / (1024 ** 3):.2f} GB"
    memory_resources["Used"] = f"{svmem.used / (1024 ** 3):.2f} GB"
    memory_resources["Percentage used"] = f"{svmem.percent}%"

    return memory_resources

def save_gpu_resources():
    """
    Retrieves and formats information about the system's GPU resources.

    Note: Does not work for integrated GPUs like those of Apple M1

    Returns:
    dict: A dictionary containing information about the system's GPU resources, including details for each GPU (ID, name, load, 
    free memory, and used memory), average load, total free memory, and total used memory.
    """

    gpu_resources = []
    total_load = 0
    total_free_memory = 0
    total_used_memory = 0
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        gpu_info = {}
        gpu_info["GPU ID"] = gpu.id
        gpu_info["Name"] = gpu.name
        gpu_info["Load"] = f"{gpu.load*100}%"
        gpu_info["Free Memory"] = f"{gpu.memoryFree / 1024:.2f}GB"
        gpu_info["Used Memory"] = f"{gpu.memoryUsed / 1024:.2f}GB"
        gpu_resources.append(gpu_info)
        total_free_memory += gpu.memoryFree
        total_used_memory += gpu.memoryUsed
    average_load = total_load / len(gpus) if gpus else 0

    return {
        "GPUs": gpu_resources, 
        "Average Load": f"{average_load*100}%", 
        "Total Free Memory": f"{total_free_memory / 1024:.2f}GB", 
        "Total Used Memory": f"{total_used_memory / 1024:.2f}GB"
    }

def save_compute_resources():
    """
    Retrieves and formats information about all of the system's compute resources.

    Returns:
    dict: A dictionary containing information about the system's compute resources, including CPU, GPU, and memory resources.
    """
    resources = {}
    resources["CPU Resources"] = save_cpu_resources()
    resources["GPU Resources"] = save_gpu_resources()
    resources["Memory Resources"] = save_memory_resources()

    return resources

### ----------- Convenience functions for the full simulation ----------- ###
def choose_backend(backend_str):
    """
    Selects the appropriate backend based on the provided string.

    Parameters:
    backend_str (str): The string representing the desired backend. Currently, only 'numpy' is supported.

    Returns:
    Backend: The selected backend. Currently, only NumpyBackend is returned.

    Raises:
    Exception: If an unsupported backend string is provided.
    """
    if backend_str=='numpy':
        return backends.NumpyBackend
    # Note: neither of these currently work
    elif backend_str=='mkl':
        return backends.CMKLExtendedBackend
    elif backend_str=='exatn':
        return backends.ExaTnBackend
    else:
        raise Exception('Unsupported backend')
    
def initialize_composer(ansatz_variant, weighted):
    """
    Initializes the appropriate composer based on the provided ansatz variant and whether the problem is weighted.

    Parameters:
    ansatz_variant (str): The string representing the desired ansatz variant. Currently, 'qaoa' and 'ma-qaoa' are supported.
    weighted (bool): Whether the problem is weighted.

    Returns:
    Composer: The initialized composer.

    Raises:
    Exception: If an unsupported ansatz variant is provided.
    """
    if ansatz_variant=='qaoa':
        if weighted:
            return WeightedSimpZZQtreeQAOAComposer
        else:
            return SimpZZQtreeComposer
    elif ansatz_variant=='ma-qaoa':
        if weighted:
            return WeightedSimpMaQtreeComposer
        else:
            return SimpMaQtreeComposer
    else:
        raise Exception('Unsupported ansatz variant')
    
def initialize_simulator(composer, backend, optimizer, weighted, max_tw):
    """
    Initializes the appropriate simulator based on the provided parameters.

    Parameters:
    composer (Composer): The composer to use in the simulator.
    backend (Backend): The backend to use in the simulator.
    optimizer (Optimizer): The optimizer to use in the simulator.
    weighted (bool): Whether the problem is weighted.
    max_tw (int): The maximum treewidth to use in the simulator.

    Returns:
    Simulator: The initialized simulator.
    """
    if weighted:
        return WeightedQAOASimulator(composer=composer, bucket_backend=backend, optimizer=optimizer, max_tw=max_tw)
    else:
        return QAOAQtreeSimulator(composer=composer, bucket_backend=backend, optimizer=optimizer, max_tw=max_tw)

def initialize_x0(ansatz_variant, p, G, param_initializer, **kwargs):
    """
    Initializes the parameters for the specified ansatz variant, and returns them as a 1D array.

    Parameters:
    ansatz_variant (str): The string representing the desired ansatz variant. Currently, 'qaoa' and 'ma-qaoa' are supported.
    p (int): The depth of the ansatz.
    G (Graph): The graph representing the problem.
    param_initializer (str): The string representing the desired parameter initialization method. Currently, 'random' and 'fourier' are supported.

    Returns:
    list: The initialized parameters.

    Raises:
    Exception: If an unsupported ansatz variant or parameter initialization method is provided.
    """
    if param_initializer=='random' or param_initializer=='fourier': # Same initialization for both random and fourier
        # Initialize parameters in range [-pi, pi], not [0, 2*pi], as the fourier transformation can change their sign 
        if ansatz_variant=='qaoa':
            # Two parameters for each depth p
            return [np.random.uniform(-np.pi, np.pi) for _ in range(p*2)]
        elif ansatz_variant=='ma-qaoa':
            # One parameter for each edge times depth p
            params = [np.random.uniform(-np.pi, np.pi) for _ in range(G.number_of_edges()) for _ in range(p)] 
            # One parameter for each node times depth p
            params.extend([np.random.uniform(-np.pi, np.pi) for _ in range(G.number_of_nodes()) for _ in range(p)])
            return params
        else:
            raise Exception('Unsupported ansatz variant')
    else:
        raise Exception('Unsupported parameter initialization method')

def recreate_gamma_beta_from_x0(ansatz_variant, p, G, x0, param_initializer, **kwargs):
    """
    Recreates gamma and beta parameters for the specified ansatz variant from x0 based on the specified ansatz variant.

    Parameters:
    ansatz_variant (str): The string representing the desired ansatz variant. Currently, 'qaoa' and 'ma-qaoa' are supported.
    p (int): The depth of the ansatz.
    G (Graph): The graph representing the problem.
    x0 (list): The initial parameters.
    param_initializer (str): The string representing the desired parameter initialization method. Currently, 'random' and 'fourier' are supported.

    Returns:
    list: The recreated gamma and beta parameters.

    Raises:
    Exception: If an unsupported ansatz variant or parameter initialization method is provided.
    """
    if param_initializer=='random':
        # Extract gamma and beta directly
        if ansatz_variant=='qaoa':
            return x0[:p], x0[p:]
        elif ansatz_variant=='ma-qaoa':
            return [x0[i*G.number_of_edges():(i+1)*G.number_of_edges()] for i in range(p)], [x0[(p+i)*G.number_of_nodes():(p+(i+1))*G.number_of_nodes()] for i in range(p)]
        else:
            raise Exception('Unsupported ansatz variant')
    
    elif param_initializer=='fourier':
        # Parameters are stored as u and v, need to convert these to 
        # gamma and beta as outlined in the paper 
        # "Quantum Approximate Optimization Algorithm: Performance, Mechanism, 
        # and Implementation on Near-Term Devices"
        if ansatz_variant=='qaoa':
            u, v = x0[:p], x0[p:]
            gamma = [sum(u[k-1] * np.sin((k + 0.5) * ((i - 0.5) * np.pi / p)) for k in range(1, p+1)) for i in range(1, p+1)]
            beta = [sum(v[k-1] * np.cos((k + 0.5) * ((i - 0.5) * np.pi / p)) for k in range(1, p+1)) for i in range(1, p+1)]
            return gamma, beta
        elif ansatz_variant=='ma-qaoa':
            u, v = [x0[i*G.number_of_edges():(i+1)*G.number_of_edges()] for i in range(p)], [x0[(p+i)*G.number_of_nodes():(p+(i+1))*G.number_of_nodes()] for i in range(p)]
            gamma = [[sum(u[k-1][e] * np.sin((k + 0.5) * ((i - 0.5) * np.pi / p)) for k in range(1, p+1)) for e in range(G.number_of_edges())] for i in range(1, p+1)]
            beta = [[sum(u[k-1][e] * np.sin((k + 0.5) * ((i - 0.5) * np.pi / p)) for k in range(1, p+1)) for e in range(G.number_of_nodes())] for i in range(1, p+1)]
            return gamma, beta
        else:
            raise Exception('Unsupported ansatz variant')
    else:
        raise Exception('Unsupported ansatz variant')

def create_stopping_criteria(max_energy_expectation=None, max_time=None, max_epochs=None):
    """
    Creates a stopping criteria function based on the provided maximum values.

    Parameters:
    max_energy_expectation (float, optional): The maximum energy expectation. If None, this criterion is ignored.
    max_time (float, optional): The maximum time. If None, this criterion is ignored.
    max_epochs (int, optional): The maximum number of epochs. If None, this criterion is ignored.

    Returns:
    function: The stopping criteria function. This function takes in the current energy expectation, time, and epoch, and returns False if any of the current values exceed the max values.

    Raises:
    Exception: If no stopping criteria are given.
    """
    if all(val is None for val in [max_energy_expectation, max_time, max_epochs]):
        raise Exception('No stopping criteria given')
    else:
        return lambda energy_expectation, time, epoch: (
            (max_energy_expectation is None or energy_expectation < max_energy_expectation) and
            (max_time is None or time < max_time) and
            (max_epochs is None or epoch < max_epochs)
        )

### ----------- Main functions for the full simulation ----------- ###

def one_sim(x0, n_processes, profile, sim, be_obj, G, ansatz_variant, p, post_process_results, iteration, results_list, param_initializer, continue_running, optimal_value, best_expectation_value, best_gamma, best_beta, param_optimizer): 
    """
    Runs one iteration of the simulation and logs/stores the results.

    Args:
        x0 (array-like): Initial parameters for the simulation.
        n_processes (int): Number of processes to use for parallel computation.
        store_results (bool): Whether to store the results of the simulation.
        profile (bool): Whether to profile the simulation for performance.
        sim (Simulator): The simulator object to use for the simulation.
        be_obj (Backend): The backend object to use for the simulation.
        G (Graph): The graph to simulate.
        ansatz_variant (str): The variant of the ansatz to use.
        p (int): The number of layers in the ansatz.
        post_process_results (callable): Function to post-process the results.
        iteration (list): List with one element, the current iteration number.
        results_list (list): List to store the results of each iteration.
        param_initializer (str): Method to initialize parameters.
        continue_running (callable): Function to determine whether to continue running.
        optimal_value (float): The optimal value for the simulation.
        best_expectation_value (list): List with one element, the best expectation value so far.
        best_gamma (list): List with one element, the best gamma value so far.
        best_beta (list): List with one element, the best beta value so far.
        param_optimizer (str): The optimizer to use for parameter optimization.

    Returns:
        float: The negative of the expectation value, since we are minimizing.
    """
    # Check if stopping criteria is met
    if iteration[0] != 1:
        energy_expectation = 0
        if optimal_value:
            energy_expectation = results_list[-1]['expectation_value'] / optimal_value
        t = time.time()
        i = iteration[0] - 1 # Subtract 1 since the iteration counter is ahead of the actual iteration by 1
        # Stop the optimization if the stopping criteria is met by raising StopIteration
        if not continue_running(energy_expectation=energy_expectation, time=t, epoch=i): 
            # Update the best expectation value and best gamma and beta if the last result is better
            if best_expectation_value[0] < results_list[-1]['expectation_value']:
                best_expectation_value[0] = results_list[-1]['expectation_value']
                best_gamma[0], best_beta[0] = gamma, beta
            raise Exception("Stopping criteria reached.")

    # Extract beta and gamma from x0 based on ansatz variant
    gamma, beta = recreate_gamma_beta_from_x0(ansatz_variant=ansatz_variant, p=p, G=G, x0=x0, param_initializer=param_initializer)

    # Simulate energy expectation
    start = time.time()
    if n_processes==1:
        result = sim.energy_expectation(G, gamma, beta)
        if profile:
            print('Profiling results')
            be_obj.gen_dreport()
    else:
        result = sim.energy_expectation_parallel(G, gamma, beta, n_processes=n_processes) 
    end = time.time()

    # Convert the result from a list to a scalar
    result = result.item()

    # Post process results
    if post_process_results:
        result = post_process_results(result)

    # Log results from sim 
    print(f"Simutation time: {end - start}")
    print("Expectation value: ", result)

    # Store results from sim
    results = {
        'iteration': iteration[0],
        'time': end - start,
        'expectation_value': result
    }
    # Add the results to the list
    results_list.append(results)
    
    # Increment iteration counter
    iteration[0] += 1

    # Update the best expectation value and best gamma and beta if the current result is better
    if best_expectation_value[0] < result:
        best_expectation_value[0] = result
        best_gamma[0], best_beta[0] = gamma, beta

    return -result # Return the negative result since we are minimizing

def parameter_optimization(x0, n_processes, profile, sim, be_obj, G, ansatz_variant, p, post_process_results, param_optimizer, param_initializer, continue_running, optimal_value):
    """
    Optimizes the parameters for the specified ansatz-variant on the specified graph with the specified parameters.

    Args:
        x0 (array-like): Initial parameters for the simulation.
        n_processes (int): Number of processes to use for parallel computation.
        store_results (bool): Whether to store the results of the simulation.
        profile (bool): Whether to profile the simulation for performance.
        sim (Simulator): The simulator object to use for the simulation.
        be_obj (Backend): The backend object to use for the simulation.
        G (Graph): The graph to simulate.
        ansatz_variant (str): The variant of the ansatz to use.
        p (int): The number of layers in the ansatz.
        post_process_results (callable): Function to post-process the results.
        param_optimizer (str): The optimizer to use for parameter optimization.
        param_initializer (str): Method to initialize parameters.
        continue_running (callable): Function to determine whether to continue running.
        optimal_value (float): The optimal value for the simulation.

    Returns:
        tuple: A tuple containing the list of results from each iteration, the best expectation value, and the best gamma and beta values.
    """
    # Prepare the final required vriables for optimization-loop
    bounds = [(-np.pi, np.pi)]*len(x0) # Bounds for the parameters
    iteration = [1] # Iteration counter
    results_list = [] # List to store results
    best_expectation_value = [-np.inf]
    # Store the best gamma and beta achieved so far
    initial_gamma, initial_beta = recreate_gamma_beta_from_x0(ansatz_variant=ansatz_variant, p=p, G=G, x0=x0, param_initializer=param_initializer)
    best_gamma, best_beta = [initial_gamma], [initial_beta]

    # Run the optimization
    if param_optimizer == "differential_evolution":
        # Note that when using differential_evolution, maxiter is the maximum number of generations over which the entire population is evolved, and the actual number is given by formula (maxiter + 1) * popsize * (N - N_equal). Determining the actual number of iterations the parameters are optimized for is not possible with this optimizer
        try:
            result = scipy.optimize.differential_evolution(
                func=one_sim, 
                bounds=bounds, 
                x0=x0,
                args=(n_processes, profile, sim, be_obj, G, ansatz_variant, p, post_process_results, iteration, results_list, param_initializer, continue_running, optimal_value, best_expectation_value, best_gamma, best_beta, param_optimizer),
                atol=1e-12  # Set the absolute tolerance for convergence. Set it to a very low value to ensure that termination is only based on the stopping criteria
            )
        except Exception as e:
            print(e)
        return results_list, best_expectation_value[0], best_gamma[0], best_beta[0]
    else:
        raise Exception('Unsupported parameter optimizer')

def full_sim(p, n_processes, ordering_algo, 
            backend, ansatz_variant, # simulator = 'qtree',
            param_initializer = 'random', param_optimizer = 'differential_evolution',
            profile = False, weighted = False, max_tw = None,
            nodes = None, degree = None, graph_type=None, seed = None, G=None,
            max_energy_expectation = None, max_time = None, max_epochs = None, 
            optimal_value = None, post_process_results = None,
            filename = None):
    """
    Runs a full simulation of the specified QAOA ansatz-variant on the specified graph with the specified parameters.

    Args:
        p (int): The number of layers in the ansatz.
        n_processes (int): Number of processes to use for parallel computation.
        ordering_algo (str): The ordering algorithm to use.
        backend (str): The backend to use for the simulation.
        ansatz_variant (str): The variant of the ansatz to use.
        param_initializer (str): Method to initialize parameters.
        param_optimizer (str): The optimizer to use for parameter optimization.
        profile (bool): Whether to profile the simulation for performance.
        weighted (bool): Whether the graph is weighted.
        max_tw (int): The maximum treewidth for the graph.
        nodes (int): The number of nodes in the graph.
        degree (int): The degree of the graph.
        graph_type (str): The type of the graph.
        seed (int): The seed for the random number generator.
        G (Graph): The graph to simulate.
        max_energy_expectation (float): The maximum energy expectation for the simulation.
        max_time (float): The maximum time for the simulation.
        max_epochs (int): The maximum number of epochs for the simulation.
        optimal_value (float): The optimal value for the simulation.
        store_results (bool): Whether to store the results of the simulation.
        post_process_results (callable): Function to post-process the results.
        filename (str): The filename to store the results.

    Returns:
        None
    """
    # Check if sufficient parameters given
    if G is None and any(val is None for val in [seed, nodes, degree, graph_type]):
        raise Exception('No graph or insufficient graph-parameters given')
    if all(val is None for val in [max_energy_expectation, max_time, max_epochs]):
        raise Exception('No stopping criteria given')
    
    # If graph not given we need to initialize one 
    if G is None:
        np.random.seed(seed)
        random.seed(seed)
        if graph_type=='random_regular':
            G = nx.random_regular_graph(degree, nodes, seed=seed)
        elif graph_type=='erdos_renyi':
            G = nx.erdos_renyi_graph(nodes, degree/(nodes-1), seed=seed)
        else:
            raise Exception('Unsupported graph type')

    # Initialize the optimizer
    opt = get_ordering_algo(ordering_algo=ordering_algo, max_tw=max_tw)

    # Initialize the backend
    be = choose_backend(backend_str=backend)
    be_obj = be()
    if profile:
        be_obj = PerfBackend(print=False)
        be_obj.backend = backend()

    # Initialize the composer
    composer = initialize_composer(ansatz_variant=ansatz_variant, weighted=weighted)

    # Initialize the simulator
    sim = initialize_simulator(composer=composer, backend=be_obj, optimizer=opt, weighted=weighted, max_tw=max_tw)

    # Prepare optimization-parameters 
    x0 = initialize_x0(ansatz_variant=ansatz_variant, p=p, G=G, param_initializer=param_initializer) 

    # Prepare stopping criteria 
    continue_running = create_stopping_criteria(max_energy_expectation=max_energy_expectation, max_time=max_time, max_epochs=max_epochs)

    # Run optimization and return the result
    results_list, best_expectation_value, best_gamma, best_beta = parameter_optimization(x0=x0, n_processes=n_processes, profile=profile, sim=sim, be_obj=be_obj, G=G, ansatz_variant=ansatz_variant, p=p, post_process_results=post_process_results, param_optimizer=param_optimizer, param_initializer=param_initializer, continue_running=continue_running, optimal_value=optimal_value)

    # Store the result-data from the simulation
    # Prepare the metadata
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') # Fomat timestamp on format YYYYMMDD_HHMMSS
    metadata = {
        'timestamp': timestamp,
        'p': p,
        'n_processes': n_processes,
        'ordering_algo': ordering_algo,
        'backend': backend,
        'ansatz_variant': ansatz_variant,
        'profile': profile,
        'weighted': weighted,
        'max_tw': max_tw,
        'nodes': nodes,
        'degree': degree,
        'graph_type': graph_type,
        'seed': seed,
        'G': G is not None,
        'max_energy_expectation': max_energy_expectation,
        'max_time': max_time,
        'max_epochs': max_epochs,
        'true_optimal_value': optimal_value,
        'post_process_results': post_process_results is not None
    }

    # Prepare system resources
    system_resources = save_compute_resources()

    # Prepare simulation results
    gamma, beta = recreate_gamma_beta_from_x0(ansatz_variant=ansatz_variant, p=p, G=G, x0=x0, param_initializer=param_initializer)
    simulation_results = {
        'total_iterations': results_list[-1]['iteration'],
        'total_time': sum(result['time'] for result in results_list),
        'average_time': sum(result['time'] for result in results_list) / len(results_list),
        'final_expectation_value': results_list[-1]['expectation_value'],
        'best_expectation_value': best_expectation_value,
        'final_gamma': gamma,
        'final_beta': beta,
        'best_gamma': best_gamma,
        'best_beta': best_beta
    }

    # Set filename if not given
    if not filename:
        filename = f'simulation_results_{timestamp}'
    # Create the folder for the results if it doesn't already exist
    os.makedirs('experiments/simulation_results', exist_ok=True)
    # Write the metadata and results to a JSON file
    with open(f'experiments/simulation_results/{filename}.json', 'w') as f:
        json.dump({'metadata': metadata, 'system_resources': system_resources, 'global_simulation_results': simulation_results, 'individual_simulation_results': results_list}, f, indent=4)
        print(f"Results stored in 'experiments/simulation_results/{filename}.json'")


### Not used in the current implementation, but could be useful for future work
def compute_exact_gradient(x0, n_processes, store_results, profile, sim, be_obj, G, ansatz_variant, p, post_process_results, iteration, results_list):
    """
    Compute the exact gradient of the expectation value with respect to the parameters.
    """
    epsilon = 1e-8  # small shift
    grad = np.zeros_like(x0)
    for i in range(len(x0)):
        x0_plus = x0.copy()
        x0_plus[i] += epsilon
        x0_minus = x0.copy()
        x0_minus[i] -= epsilon
        print("Computing gradient 'f_plus' for parameter", i+1, "of", len(x0), "parameters.")
        f_plus = one_sim(x0_plus, n_processes, store_results, profile, sim, be_obj, G, ansatz_variant, p, post_process_results, iteration, results_list)
        print("Computing gradient 'f_minus' for parameter", i+1, "of", len(x0), "parameters.")
        f_minus = one_sim(x0_minus, n_processes, store_results, profile, sim, be_obj, G, ansatz_variant, p, post_process_results, iteration, results_list)
        grad[i] = (f_plus - f_minus) / (2 * epsilon)
        print("Gradient for parameter", i+1, "of", len(x0), "parameters:", grad[i])
    return grad