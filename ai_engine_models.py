from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import base64

def encrypt_aes(data, key):
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(data.encode())
    return base64.b64encode(cipher.nonce + tag + ciphertext).decode()

def decrypt_aes(encrypted_data, key):
    encrypted_data = base64.b64decode(encrypted_data)
    nonce, tag, ciphertext = encrypted_data[:16], encrypted_data[16:32], encrypted_data[32:]
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag).decode()

# Example key generation for AES
aes_key = get_random_bytes(16)
