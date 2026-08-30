from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
)


#
# Client ephemeral key pair
#
client_private = X25519PrivateKey.generate()
client_public = client_private.public_key()


#
# Server ephemeral key pair
#
server_private = X25519PrivateKey.generate()
server_public = server_private.public_key()


#
# Client:
# 내 private + Server public
#
client_shared = client_private.exchange(
    server_public
)


#
# Server:
# 내 private + Client public
#
server_shared = server_private.exchange(
    client_public
)


print(
    "client shared:",
    client_shared.hex(),
)

print(
    "server shared:",
    server_shared.hex(),
)

print(
    "same:",
    client_shared == server_shared,
)