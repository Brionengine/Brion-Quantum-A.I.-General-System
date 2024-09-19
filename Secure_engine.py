class SecureNLPEngine:
    def __init__(self, encryption_key):
        self.nlp_engine = NLPEngine()
        self.encryption_key = encryption_key

    def secure_parse_input(self, encrypted_input):
        # Decrypt the input
        decrypted_input = decrypt_aes(encrypted_input, self.encryption_key)
        
        # Parse the decrypted input with NLP
        parsed_command = self.nlp_engine.parse_input(decrypted_input)
        
        # Encrypt the output command
        encrypted_output = encrypt_aes(parsed_command, self.encryption_key)
        return encrypted_output
