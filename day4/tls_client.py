import socket
import ssl


HOST = "127.0.0.1"
PORT = 5443


context = ssl.create_default_context(
    cafile="certs/ca.crt"
)

#context = ssl.create_default_context()


raw_sock = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM,
)

raw_sock.connect(
    (HOST, PORT)
)


sock = context.wrap_socket(
    raw_sock,
    server_hostname="localhost",
)


print(
    "TLS version:",
    sock.version(),
)

print(
    "cipher:",
    sock.cipher(),
)

print(
    "certificate:",
    sock.getpeercert(),
)


sock.sendall(
    b"hello server"
)


response = sock.recv(
    4096
)

print(
    "response:",
    response.decode(),
)


sock.close()