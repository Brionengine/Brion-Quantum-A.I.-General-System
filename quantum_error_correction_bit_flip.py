# quantum_error_correction.py

from qiskit import QuantumCircuit

def three_qubit_bit_flip_code():
    # Encode a logical qubit into three physical qubits
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    
    # Introduce an intentional error on one qubit (optional)
    qc.x(1)
    
    # Decode
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.measure_all()
    
    print("3-Qubit Bit Flip Code Circuit:")
    print(qc.draw())
    return qc

# Example Usage
if __name__ == "__main__":
    qc = three_qubit_bit_flip_code()
