import hashlib
import hmac
import json
import secrets
import socket
import struct
import time


HOST = "127.0.0.1"
PORT = 5000

#
# Bob도 현재 공용 secret을 알고 있다고 가정
#
SECRET_KEY = b"bob-secret-key"


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


#
# 실제로는 Bob인데 Alice라고 주장
#
client_id = "alice"
timestamp = int(time.time())
nonce = secrets.token_hex(16)
message = "I am Alice"


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
    struct.pack("!I", len(payload))
    + payload
)


sock = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM,
)

sock.connect((HOST, PORT))
sock.sendall(frame)
sock.close()

print("spoof request sent")