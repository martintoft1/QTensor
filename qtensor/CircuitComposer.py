from loguru import logger as log
from qtensor.utils import get_edge_subgraph
import networkx as nx
from .OpFactory import CircuitBuilder
# import tensorflow as tf
import numpy as np
import cirq

class CircuitComposer():
    """ Director for CircuitBuilder, but with a special way to get the builder"""
    Bulider = CircuitBuilder
    def __init__(self, *args, **params):
        self.params = params
        self.builder = self._get_builder()
        self.n_qubits = self.builder.n_qubits

    #-- Setting up the builder
    def _get_builder_class(self):
        raise NotImplementedError

    def _get_builder(self):
        return self._get_builder_class()()


    #-- Mocking some of bulider behaviour
    @property
    def operators(self):
        return self.builder.operators

    @property
    def circuit(self):
        return self.builder.circuit
    @circuit.setter
    def circuit(self, circuit):
        self.builder.circuit = circuit

    @property
    def qubits(self):
        return self.builder.qubits
    @qubits.setter
    def qubits(self, qubits):
        self.builder.qubits = qubits

    def apply_gate(self, gate, *qubits, **params):
        self.builder.apply_gate(gate, *qubits, **params)

    def conjugate(self):
        # changes builder.circuit, hence self.circuit()
        self.builder.conjugate()
    #--

    def layer_of_Hadamards(self):
        for q in self.qubits:
            self.apply_gate(self.operators.H, q)


class OldQAOAComposer(CircuitComposer):
    """ Abstract base class for QAOA Director """
    def __init__(self, graph, *args, **kwargs):
        self.n_qubits = graph.number_of_nodes()
        super().__init__(*args, **kwargs)

        self.graph = graph

    def _get_builder(self):
        return self._get_builder_class()(self.n_qubits)

    @classmethod
    def _get_of_my_type(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def energy_expectation(self, i, j):
        # Will need to deprecate stateful API and return the circuit
        self.ansatz_state()
        self.energy_edge(i, j)
        first_part = self.builder.circuit

        self.builder.reset()
        self.ansatz_state()
        self.builder.inverse()
        second_part = self.builder.circuit

        self.circuit = first_part + second_part

    def energy_expectation_lightcone(self, edge):
        G = self.graph
        gamma, beta = self.params['gamma'], self.params['beta']
        i,j = edge
        graph = get_edge_subgraph(G, edge, len(gamma))
        log.debug('Subgraph nodes: {}, edges: {}', graph.number_of_nodes(), graph.number_of_edges())
        self.n_qubits = graph.number_of_nodes()
        mapping = {v:i for i, v in enumerate(graph.nodes())}
        graph = nx.relabel_nodes(graph, mapping, copy=True)

        i,j = mapping[i], mapping[j]
        composer = self._get_of_my_type(graph, beta=beta, gamma=gamma)
        composer.energy_expectation(i,j)
        self.circuit = composer.circuit
        # return composer


    def x_term(self, u, beta):
        #self.circuit.append(self.operators.H(u))
        self.apply_gate(self.operators.XPhase, u, alpha=2*beta)
        #self.circuit.append(self.operators.H(u))
    
    def y_term(self, u, beta):
        self.apply_gate(self.operators.YPhase, u, alpha=2*beta)

    def mixer_operator(self, beta, nodes=None):
        if nodes is None: nodes = self.graph.nodes()
        for n in nodes:
            qubit = self.qubits[n]
            self.x_term(qubit, beta)

    def append_zz_term(self, q1, q2, gamma):
        self.apply_gate(self.operators.cX, q1, q2)
        self.apply_gate(self.operators.ZPhase, q2, alpha=2*gamma)
        self.apply_gate(self.operators.cX, q1, q2)
    def cost_operator_circuit(self, gamma, edges=None):
        if edges is None: edges = self.graph.edges()
        for i, j in edges:
            u, v = self.qubits[i], self.qubits[j]
            self.append_zz_term(u, v, gamma)


    def ansatz_state(self):
        beta, gamma = self.params['beta'], self.params['gamma']
        assert(len(beta) == len(gamma))
        p = len(beta) # infering number of QAOA steps from the parameters passed
        self.layer_of_Hadamards()
        # second, apply p alternating operators
        for i in range(p):
            self.cost_operator_circuit(gamma[i])
            self.mixer_operator(beta[i])

    def energy_edge(self, i, j):
        u, v = self.qubits[i], self.qubits[j]
        self.apply_gate(self.operators.Z, u)
        self.apply_gate(self.operators.Z, v)


class QAOAComposer(OldQAOAComposer):
    def cone_ansatz(self, edge):
        beta, gamma = self.params['beta'], self.params['gamma']
        assert(len(beta) == len(gamma))
        p = len(beta) # infering number of QAOA steps from the parameters passed
        self.layer_of_Hadamards()
        # second, apply p alternating operators
        cone_base = self.graph

        for i, g, b in zip(range(p, 0, -1), gamma, beta):
            self.graph = get_edge_subgraph(cone_base, edge, i)
            self.cost_operator_circuit(g)
            self.mixer_operator(b)
        self.graph = cone_base


    def energy_expectation(self, i, j):
        # Will need to deprecate stateful API and return the circuit
        self.cone_ansatz(edge=(i, j))
        self.energy_edge(i, j)
        first_part = self.builder.circuit
        self.builder.reset()

        self.cone_ansatz(edge=(i, j))
        self.builder.inverse()
        second_part = self.builder.circuit

        self.circuit = first_part + second_part

class ZZQAOAComposer(QAOAComposer): # Uses a direct application of the zz gate, otherwise identical to QAOAComposer
    def append_zz_term(self, q1, q2, gamma):
        self.apply_gate(self.operators.ZZ, q1, q2, alpha=2*gamma)

class WeightedZZQAOAComposer(ZZQAOAComposer):
    def cost_operator_circuit(self, gamma, edges=None):
        for i, j, w in self.graph.edges.data('weight', default=1):
            u, v = self.qubits[i], self.qubits[j]
            self.append_zz_term(u, v, gamma*w)


# ------------------ CODE ADDITIONS ------------------
# Multi-angle QAOA composer 
class MaQAOAComposer(ZZQAOAComposer):
    def mixer_operator(self, betas, nodes=None):
        if nodes is None: nodes = self.graph.nodes()
        for n, beta in zip(nodes, betas): # betas should be of same length as nodes
            qubit = self.qubits[n]
            self.x_term(qubit, beta)
    
    def cost_operator_circuit(self, gammas, edges=None):
        if edges is None: edges = self.graph.edges()
        for gamma, (i, j) in zip(gammas, edges): # gammas should be of same length as edges
            u, v = self.qubits[i], self.qubits[j]
            self.append_zz_term(u, v, gamma)

# Weighted version of the MaQAOAComposer
class WeightedMaQAOAComposer(MaQAOAComposer):
    def cost_operator_circuit(self, gammas, edges=None):
        if edges is None: edges = self.graph.edges()
        for gamma, (i, j, w) in zip(gammas, self.graph.edges.data('weight', default=1)): # gammas should be of same length as edges
            u, v = self.qubits[i], self.qubits[j]
            self.append_zz_term(u, v, gamma*w)


# ADAPT-QAOA composer (Note: not fully implemented)
"""
class AdaptQAOASingleComposer(ZZQAOAComposer):
    def __init__(self, graph, *args, **kwargs):
        super().__init__(graph, *args, **kwargs)
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001) # Adam optimizer
        self.ops = self.s_pool() # Pool of single-qubit gates

    def ansatz_state(self):
        beta, gamma = self.params['beta'], self.params['gamma']
        assert(len(beta) == len(gamma))
        p = len(beta) # infering number of QAOA steps from the parameters passed
        self.layer_of_Hadamards()
        if not self.params['mixer']: # First optimization run, need to initialize mixer. Set it to default XPhase
            self.params['mixer'] = [[] for _ in range(p)]
            self.params['mixer'][0] = self.initialize_mixer()
        for i in range(p):
            self.cost_operator_circuit(gamma[i])
            # Dynamically select or construct a new mixer
            self.select_mixer_operator(beta[i], i)
            self.mixer_operator_circuit(self.params['mixer'][i])

    def select_mixer_operator(self, gamma, beta, layer): # TODO: Finish method for selecting the mixer operator. Should maybe be implemented partly with methods in FullSimulation.py
        # Initialize values
        tol = 1e-5 # Threshold for derivative of expectation value of cost operator

        while True: # Run until stopping criteria is met
            # TODO: Construct all candidate circuits 
            
            # TODO: Measure the energy gradient for every candidate circuit 
            
            # TODO: Check if stopping criteria for mixers is met
            if init is None:
                init = max(grads)
            grads_ = [i for i in grads if i >= init]
            if len(grads_) == 0 or counter > (layer * 2 + 25):
                break
            # Stopping criteria is not met, continue

            # TODO: Select the mixer operator associated with the largest component of the gradient, and
            # optimize all parameters currently in the ansatz, βm, γm, m = 1, ..., k,  for the selected mixer operator
            base_circuit = circuits[np.argmax(grads)]
            old = np.inf
            while True:
                with tf.GradientTape() as tape:
                    tape.watch(var)
                    guess = expectation_layer(base_circuit, symbol_names=[s.name for s in symbols], symbol_values=[var], operators=h)
                grads = tape.gradient(guess, var)
                self.opt.apply_gradients(zip([grads], [var]))
                guess = guess.numpy()[0][0]
                # TODO: Check if stopping criteria for parameters is met
                if abs(guess - old) < tol:
                    break
                old = guess
            # TODO: Update parameters
            params = var.numpy().tolist()

    def s_pool(self):
        # Method for creating the pool of single-qubit gates
        # Have 2*nodes + 2 mixers in the pool; one X for every node, one Y for every node, one sum over all X, and one sum over all Y
        pool = []

        # Sum over all X
        mixing_ham = 0
        for n in self.graph.nodes():
            qubit = self.qubits[n]
            mixing_ham += cirq.PauliString(self.operators.X.cirq_op(qubit))
        pool.append(mixing_ham)

        # Sum over all Y
        mixing_ham = 0
        for n in self.graph.nodes():
            qubit = self.qubits[n]
            mixing_ham += cirq.PauliString(self.operators.Y.cirq_op(qubit))
        pool.append(mixing_ham)

        # X for every node
        for n in self.graph.nodes():
            mixing_ham = 0
            qubit = self.qubits[n]
            mixing_ham += cirq.PauliString(self.operators.X.cirq_op(qubit))
            pool.append(mixing_ham)

        # Y for every node
        for n in self.graph.nodes():
            mixing_ham = 0
            qubit = self.qubits[n]
            mixing_ham += cirq.PauliString(self.operators.Y.cirq_op(qubit))
            pool.append(mixing_ham)
        return pool

    
# Weighted version of the AdaptQAOASingleComposer
class WeightedAdaptQAOASingleComposer(AdaptQAOASingleComposer):
    def cost_operator_circuit(self, gammas, edges=None):
        if edges is None: edges = self.graph.edges()
        for gamma, (i, j, w) in zip(gammas, self.graph.edges.data('weight', default=1)): # gammas should be of same length as edges
            u, v = self.qubits[i], self.qubits[j]
            self.append_zz_term(u, v, gamma*w)
"""

# ---------------- END CODE ADDITIONS ----------------