# quantum_neural_network.py

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import TwoLocal
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.algorithms.classifiers import VQC

def build_qnn(num_qubits):
    # Define a variational form for the quantum neural network
    var_form = TwoLocal(num_qubits, ['ry', 'rz'], 'cz', reps=3, entanglement='full')
    
    # Create a Quantum Circuit
    qc = QuantumCircuit(num_qubits)
    qc.compose(var_form, inplace=True)
    
    # Use SamplerQNN for the neural network model
    qnn = SamplerQNN(circuit=qc, input_params=var_form.parameters)
    return qnn

# Example Usage
if __name__ == "__main__":
    num_qubits = 4
    qnn = build_qnn(num_qubits)
    print(f"Quantum Neural Network created with {num_qubits} qubits.")
