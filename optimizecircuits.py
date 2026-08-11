from __future__ import annotations

try:
    from qiskit.transpiler import PassManager
except ImportError:  # optional dependency: pip install qiskit
    PassManager = None
try:
    from qiskit.transpiler.passes import Optimize1qGates, CommutativeCancellation
except ImportError:  # optional dependency: pip install qiskit
    Optimize1qGates = None
    CommutativeCancellation = None

def optimize_circuit(qc):
    # Define a pass manager with optimization passes
    pass_manager = PassManager([Optimize1qGates(), CommutativeCancellation()])
    optimized_qc = pass_manager.run(qc)
    return optimized_qc
