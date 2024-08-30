# variational_circuit.py

from qiskit import QuantumCircuit, Aer, transpile
from qiskit.circuit import Parameter
from qiskit.opflow import Z, StateFn, CircuitSampler, AerPauliExpectation

def variational_circuit(num_qubits):
    # Define parameters
    params = [Parameter(f'theta_{i}') for i in range(num_qubits)]
    
    # Create a quantum circuit with parameterized gates
    qc = QuantumCircuit(num_qubits)
    for i, param in enumerate(params):
        qc.ry(param, i)
        if i < num_qubits - 1:
            qc.cz(i, i + 1)
    
    return qc, params

# Example Usage
if __name__ == "__main__":
    num_qubits = 3
    circuit, params = variational_circuit(num_qubits)
    print(f"Variational Circuit created with {num_qubits} qubits and parameters: {params}.")
