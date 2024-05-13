# QTensor

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python**: This project requires Python 3.9.12. If you do not have Python installed, download and install it from [python.org](https://www.python.org/downloads/) or your system's package manager.

- **pip**: `pip` is a package manager for Python. It's used to install and manage Python packages. You can check if you have `pip` installed by running `pip --version` in your terminal. If you don't have `pip` installed, you can install it by following the instructions on the [pip installation page](https://pip.pypa.io/en/stable/installation/).

- **Homebrew**: macOS users should install Homebrew to manage installation of Python and other necessary libraries. If Homebrew is not installed on your machine, install it by running:
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```

## System-level dependencies
Before installing the QTensor, you need to install some system-level dependencies. Here's how you can do it on different systems:


### mpich

- **MacOS**:
    ```bash
    brew install mpich
    ```

- **Ubuntu**:
    ```bash
    sudo apt-get install mpich
    ```

### Setting up the Environment

After installing `mpich`, you might need to set the `mpicc` path. This step is only necessary if the `mpicc` path is not set correctly.

1. Find the location of `mpicc` using the `which` command:

```bash
which mpicc
```

If the output of this command points to the correct location of `mpicc`, you can skip step 2. If not, proceed with the following step.

2. You have two options to set the `mpicc` path:

   - **Temporary**: Run the following command in your terminal. Replace `/path/to/mpicc` with the path you got from the previous step, excempting the part containing `mpicc`. This will set the `mpicc` path for the current terminal session only.

     ```bash
     export PATH=/path/to/mpicc:$PATH
     ```

   - **Permanent**: If you want to set the `mpicc` path permanently, add the above line to your shell profile file (like `.bashrc`, `.zshrc`, or whatever your system uses). Then, source your profile file to apply the changes:

     ```bash
     source ~/.bashrc  # If you're using bash
     source ~/.zshrc   # If you're using zsh
     ```

     If you're using a different shell, replace `.bashrc` or `.zshrc` with your shell's profile file.

Remember, if you choose the permanent option, you need to source your profile file or open a new terminal window for the changes to take effect.


## Additional dependencies

### Tamaki solver

The tamaki solver repository should be already cloned into
`QTensor/qtree/thirdparty/tamaki_treewidth`. If it is not, it is because you did not clone the QTensor-repository with the `--recurse-submodules` flag. You can clone it manually by navigating to the QTensor/qtree/thirdparty/ repository and using the following command:

```bash
git clone https://github.com/TCS-Meiji/PACE2017-TrackA
```

To compile it, go to the directory and run `make heuristic`.

```bash
> cd QTensor/qtree/thirdparty/tamaki_treewidth
> make heuristic 
javac tw/heuristic/*.java
```

If you have memory errors, modify the `JFLAGS` variable in the bash script `./tw-heuristic`. I use `JFLAGS="-Xmx4g -Xms4g -Xss500m"`.

Before running qtensor with tamaki, make sure `tw-heuristic` resolves as executable. For that, add the `tamaki_treewidth` dir to your `PATH`. Test with `which tw-heuristic`.


## Installation

From the root directory of the repository, run the following commands:

```bash
cd qtree && pip install . && cd ..
pip install .
```

## Usage

### Full simulation
To run an experiment, you need to use the command line and navigate to the directory /experiments containing the experiments.py script. Then, you can run an experiment using the following command:
    
```bash
python experiments.py <type> <number> <start_time> <iteration> <max_time> <processes>
```

Here's what each argument means:

- type: The type of experiment. This can be either 'prelim' or 'main'.
- number: The number of the experiment you want to run.
- start_time: The start time of the experiment in 'YYYYMMDD_HHMMSS' format.
- iteration: The iteration of the experiment.
- max_time: The maximum runtime of the experiment in 'DD:HH:MM:SS' format.
- processes: The number of processes to use for the experiment.

For example, to run a preliminary experiment number 1, you could use the following command:

```bash
python experiments.py prelim 1 20240101_000000 1 00:00:10:00 2
```

If you need help with the command line arguments, you can use the -h or --help option:

```bash
python experiments.py -h
```

### Single energy evaluation

```python
from qtensor import QAOA_energy

G = nx.random_regular_graph(3, 10)
gamma, beta = [np.pi/3], [np.pi/2]

E = QAOA_energy(G, gamma, beta)
```

### Get treewidth

```python
from qtensor.optimisation.Optimizer import OrderingOptimizer
from qtensor.optimisation.TensorNet import QtreeTensorNet
from qtensor import QtreeQAOAComposer

composer = QtreeQAOAComposer(
	graph=G, gamma=gamma, beta=beta)
composer.ansatz_state()


tn = QtreeTensorNet.from_qtree_gates(composer.circuit)

opt = OrderingOptimizer()
peo, tn = opt.optimize(tn)
treewidth = opt.treewidth

```




#### Usage

```python
from qtensor.optimisation.Optimizer import TamakiOptimizer
from qtensor.optimisation.TensorNet import QtreeTensorNet
from qtensor import QtreeQAOAComposer

composer = QtreeQAOAComposer(
	graph=G, gamma=gamma, beta=beta)
composer.ansatz_state()


tn = QtreeTensorNet.from_qtree_gates(composer.circuit)

opt = TamakiOptimizer(wait_time=15) # time in seconds for heuristic algorithm
peo, tn = opt.optimize(tn)
treewidth = opt.treewidth

```
#### Use tamaki for QAOA energy

and also raise an error when treewidth is large.

```python
from qtensor.optimisation.Optimizer import TamakiOptimizer
from qtensor import QAOAQtreeSimulator

class TamakiQAOASimulator(QAOAQtreeSimulator):
    def optimize_buckets(self):
        opt = TamakiOptimizer()
        peo, self.tn = opt.optimize(self.tn)
        if opt.treewidth > 30:
            raise Exception('Treewidth is too large!')
        return peo

sim = TamakiQAOASimulator(QtreeQAOAComposer)

if n_processes:
    res = sim.energy_expectation_parallel(G, gamma=gamma, beta=beta
        ,n_processes=n_processes
    )
else:
    res = sim.energy_expectation(G, gamma=gamma, beta=beta)
return res

```

### Useful features

- raise ValueError if treewidth is too large:
```python
sim = QAOAQtreeSimulator(max_tw=24)
sim.energy_expectation(G, gamma=gamma, beta=beta)
```

- generate graphs

```python
from qtree.toolbox import random_graph

G_reg = random_graph(12, type='random', degree=3, seed=42)
G_er = random_graph(12, type='erdos_renyi', degree=3, seed=42)

```
- get cost estimation

```python
from qtensor.optimisation.Optimizer import TamakiOptimizer
from qtensor.optimisation.TensorNet import QtreeTensorNet
from qtensor import QtreeQAOAComposer

composer = QtreeQAOAComposer(
	graph=G, gamma=gamma, beta=beta)
composer.ansatz_state()

tn = QtreeTensorNet.from_qtree_gates(composer.circuit)

opt = TamakiOptimizer(wait_time=15)
peo, tn = opt.optimize(tn)
treewidth = opt.treewidth
mems, flops = tn.simulation_cost(peo)
print('Max memory=', max(mems), 'Total flops=', sum(flops))
```
- get QAOA cost estimation

```python
from qtensor.toolbox import qaoa_energy_cost_params_from_graph

costs_per_edge = qaoa_energy_cost_params_from_graph(graph, p,
        ordering_algo='greedy', max_time=60)

tws, mems, flops = zip(*costs_per_edge)
print('Max treewidth=', max(tws), 'Max memory=', max(mems), 'Total flops=', sum(flops))
```
