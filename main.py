# main.py

from quantum_fusion.core import create_fusion_circuit
from ai_engine.models import AIModel
from security.encryption import EncryptionManager
from utils.helpers import validate_input
from ui.interface import launch_ui

def main():
    # Initialize encryption manager
    encryption_manager = EncryptionManager()

    # Initialize AI model
    ai_model = AIModel()

    # Example input
    plaintext = "What is the status of the server?"
    if validate_input(plaintext):
        # Encrypt the input
        encrypted_input = encryption_manager.encrypt(plaintext)

        # Decrypt the input
        decrypted_input = encryption_manager.decrypt(encrypted_input)

        # Generate command using AI model
        command = ai_model.generate_command(decrypted_input)

        # Encrypt the command
        encrypted_command = encryption_manager.encrypt(command)

        # Decrypt the command for display
        decrypted_command = encryption_manager.decrypt(encrypted_command)
        print(f"Decrypted Command: {decrypted_command}")

        # Create quantum fusion circuit
        fusion_circuit = create_fusion_circuit(num_qubits=4)
        print("Quantum Fusion Circuit:")
        print(fusion_circuit.draw())

        # Launch the user interface
        launch_ui()
    else:
        print("Invalid input.")

if __name__ == "__main__":
    main()
