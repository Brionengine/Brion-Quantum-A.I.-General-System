from cryptography.fernet import Fernet

class EncryptionManager:
    def __init__(self, key=None):
        self.key = key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)

    def encrypt(self, plaintext):
        if not isinstance(plaintext, bytes):
            plaintext = plaintext.encode('utf-8')
        encrypted_text = self.cipher_suite.encrypt(plaintext)
        return encrypted_text

    def decrypt(self, encrypted_text):
        try:
            decrypted_text = self.cipher_suite.decrypt(encrypted_text)
            return decrypted_text.decode('utf-8')
        except Exception as e:
            # Handle decryption error
            raise ValueError("Decryption failed.") from e
