import json
import socket
import struct
import time


HOST = "127.0.0.1"
PORT = 5000

COUNT = 10000


request = {
    "client_id": "alice",
    "timestamp": int(time.time()),
    "nonce": "fake",
    "message": "hello",
    "mac": "0" * 64,
}

payload = json.dumps(request).encode()

frame = (
    struct.pack("!I", len(payload))
    + payload
)


start = time.perf_counter()

for i in range(COUNT):
    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    )

    try:
        sock.connect((HOST, PORT))
        sock.sendall(frame)
    finally:
        sock.close()

    if i % 1000 == 0:
        print(f"sent={i}")


elapsed = time.perf_counter() - start

print(
    f"{COUNT} requests "
    f"in {elapsed:.3f}s"
)