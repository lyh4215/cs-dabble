from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
)
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import os


#
# 1. ECDH / X25519
#

client_private = X25519PrivateKey.generate()
client_public = client_private.public_key()

server_private = X25519PrivateKey.generate()
server_public = server_private.public_key()


client_shared = client_private.exchange(
    server_public
)

server_shared = server_private.exchange(
    client_public
)


print(
    "shared same:",
    client_shared == server_shared
)

print(
    "shared secret:",
    client_shared.hex()
)


#
# 2. HKDF
#
# raw shared secret을 그대로 AES key로 쓰지 않고
# key derivation function을 거친다.
#

def derive_key(shared_secret):
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,       # AES-256 key = 32 bytes
        salt=None,
        info=b"day4 tls demo",
    ).derive(
        shared_secret
    )


client_key = derive_key(
    client_shared
)

server_key = derive_key(
    server_shared
)


print()
print(
    "client key:",
    client_key.hex()
)

print(
    "server key:",
    server_key.hex()
)

print(
    "key same:",
    client_key == server_key
)


#
# 3. AES-GCM
#

client_aes = AESGCM(
    client_key
)

server_aes = AESGCM(
    server_key
)


message = b"transfer 100"

#
# AES-GCM nonce
# 보통 12 bytes
#
nonce = os.urandom(12)


#
# Client -> Server 암호화
#
ciphertext = client_aes.encrypt(
    nonce,
    message,

    #
    # Additional Authenticated Data
    #
    None,
)


print()
print(
    "plaintext:",
    message
)

print(
    "ciphertext:",
    ciphertext.hex()
)


#
# Server 복호화
#
decrypted = server_aes.decrypt(
    nonce,
    ciphertext,
    None,
)


print(
    "decrypted:",
    decrypted
)