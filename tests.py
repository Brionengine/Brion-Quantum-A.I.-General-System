# tests/test_encryption.py

import unittest
from security.encryption import EncryptionManager

class TestEncryptionManager(unittest.TestCase):
    def test_encryption_decryption(self):
        manager = EncryptionManager()
        plaintext = "Test message"
        encrypted = manager.encrypt(plaintext)
        decrypted = manager.decrypt(encrypted)
        self.assertEqual(plaintext, decrypted)

if __name__ == '__main__':
    unittest.main()
