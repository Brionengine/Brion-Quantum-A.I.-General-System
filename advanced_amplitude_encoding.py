# advanced_amplitude_encoding.py

from qiskit import QuantumCircuit
import numpy as np

def amplitude_encoding(data):
    # Ensure data is normalized to sum to 1
    norm_data = data / np.linalg.norm(data)
    num_qubits = int(np.ceil(np.log2(len(norm_data))))
    qc = QuantumCircuit(num_qubits)
    
    # Encode data into amplitude of quantum state
    qc.initialize(norm_data, qc.qubits)
    
    print("Amplitude Encoding Circuit:")
    print(qc.draw())
    return qc

# Example Usage
if __name__ == "__main__":
    data = np.array([0.6, 0.8, 0.5, 0.3])
    qc = amplitude_encoding(data)
