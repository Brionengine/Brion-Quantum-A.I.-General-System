# quantum_data_encoding.py

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np

def encode_data(data):
    # Assuming data is a 1D numpy array of length equal to the number of qubits
    num_qubits = len(data)
    qc = QuantumCircuit(num_qubits)
    
    for i, value in enumerate(data):
        # Encode data into the quantum state using Ry rotations
        qc.ry(value, i)
    
    return qc

# Example Usage
if __name__ == "__main__":
    data = np.array([0.5, 0.1, 0.8])
    qc = encode_data(data)
    print("Quantum Circuit for Data Encoding created.")
    print(qc.draw())
