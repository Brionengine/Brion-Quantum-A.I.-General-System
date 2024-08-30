# qnn_training_and_evaluation.py

from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit.utils import algorithm_globals

def train_and_evaluate_qnn(qnn):
    # Load a sample dataset
    training_features, training_labels, test_features, test_labels = ad_hoc_data(training_size=20, test_size=10, n=2, gap=0.3)
    
    # Set random seed for reproducibility
    algorithm_globals.random_seed = 42
    
    # Define the Variational Quantum Classifier (VQC) using the QNN
    vqc = VQC(feature_map=qnn.circuit, ansatz=qnn.circuit, optimizer='SPSA')
    
    # Train the model
    vqc.fit(training_features, training_labels)
    
    # Evaluate the model on the test set
    accuracy = vqc.score(test_features, test_labels)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Example Usage
if __name__ == "__main__":
    num_qubits = 4
    qnn = optimized_qnn(num_qubits)
    train_and_evaluate_qnn(qnn)
