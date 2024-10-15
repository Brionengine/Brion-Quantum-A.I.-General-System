# algorithm_registry.py

from quantum_algorithms import grovers_algorithm, shors_algorithm, quantum_phase_estimation

algorithm_registry = {
    'search': {
        'function': grovers_algorithm,
        'args': {'n_qubits': 3},
        'description': "Uses Grover's algorithm to search an unsorted database."
    },
    'factor': {
        'function': shors_algorithm,
        'args': {'N': 15},  # Example number to factor
        'description': "Uses Shor's algorithm for integer factorization."
    },
    'phase estimation': {
        'function': quantum_phase_estimation,
        'args': {},
        'description': "Performs Quantum Phase Estimation."
    },
    # Add new algorithms here
}
