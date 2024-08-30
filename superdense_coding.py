# superdense_coding.py

from qiskit import QuantumCircuit

def superdense_coding():
    # Create a Bell state to share entanglement
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    # Encode two classical bits into quantum state
    qc.x(0)  # Encode classical bit sequence
    qc.z(1)
    
    qc.cx(0, 1)
    qc.h(0)
    qc.measure_all()
    
    print("Superdense Coding Circuit:")
    print(qc.draw())
    return qc

# Example Usage
if __name__ == "__main__":
    qc = superdense_coding()
