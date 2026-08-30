import socket
from pathlib import Path


HOST = "127.0.0.1"
PORT = 5000

CAPTURE_PATH = Path(
    "dist/captured.bin"
)


frame = CAPTURE_PATH.read_bytes()


sock = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM,
)

sock.connect(
    (HOST, PORT)
)

sock.sendall(
    frame
)

sock.close()

print(
    "captured packet replayed"
)