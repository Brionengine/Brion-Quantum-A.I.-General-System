from flask import Flask, request, jsonify
from qiskit import Aer, QuantumCircuit, transpile
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.quantum_info import Operator
from qiskit.primitives import Sampler
import numpy as np

# Define the Flask app for API
app = Flask(__name__)

# Sample Hamiltonian for testing purposes
DEFAULT_H = Operator([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.5, 0.0],
    [0.0, 0.5, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

# Define a quantum ansatz for VQE
ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz', reps=2)

# Optimizer setup
optimizer = COBYLA(maxiter=200, tol=1e-3)

# VQE Setup, initialized once
vqe = VQE(ansatz, optimizer=optimizer, sampler=Sampler())

# Function for VQE energy minimization
def run_vqe(hamiltonian):
    try:
        result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)
        return result.eigenvalue.real
    except Exception as e:
        return f"Error in VQE computation: {str(e)}"

# API route for running VQE
@app.route('/run-vqe', methods=['POST'])
def vqe_route():
    try:
        data = request.json
        matrix = np.array(data.get('hamiltonian', DEFAULT_H.data))
        hamiltonian = Operator(matrix)
        energy = run_vqe(hamiltonian)
        return jsonify({"minimum_energy": energy})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Invalid input: {str(e)}"}), 400

# Testing endpoint
@app.route('/test', methods=['GET'])
def test_endpoint():
    try:
        # Test with default Hamiltonian
        energy = run_vqe(DEFAULT_H)
        return jsonify({"status": "success", "energy_test": energy})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Main entry point for running the app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
