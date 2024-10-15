from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CommutativeCancellation

def optimize_circuit(qc):
    # Define a pass manager with optimization passes
    pass_manager = PassManager([Optimize1qGates(), CommutativeCancellation()])
    optimized_qc = pass_manager.run(qc)
    return optimized_qc
