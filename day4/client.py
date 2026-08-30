import hashlib
import hmac
import json
import socket
import struct
import time
from pathlib import Path
import secrets


HOST = "127.0.0.1"
PORT = 5000

SECRET_KEY = b"day4-secret-key"

CAPTURE_PATH = Path(
    "dist/captured.bin"
)


def calculate_mac(
    client_id,
    timestamp,
    nonce,
    message,
):
    payload = (
        f"{client_id}|"
        f"{timestamp}|"
        f"{nonce}|"
        f"{message}"
    ).encode()

    return hmac.new(
        SECRET_KEY,
        payload,
        hashlib.sha256,
    ).hexdigest()


client_id = "alice"
timestamp = int(time.time())
message = "transfer 100"
nonce = secrets.token_hex(16)


request = {
    "client_id": client_id,
    "timestamp": timestamp,
    "nonce": nonce,
    "message": message,
}

request["mac"] = calculate_mac(
    client_id,
    timestamp,
    nonce,
    message,
)


payload = json.dumps(
    request
).encode()


frame = (
    struct.pack(
        "!I",
        len(payload),
    )
    + payload
)


#
# 공격자가 packet을 복사했다고 가정.
#
CAPTURE_PATH.write_bytes(
    frame
)

print(
    f"captured packet → "
    f"{CAPTURE_PATH}"
)


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
    "request sent"
)