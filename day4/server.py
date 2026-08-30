import hashlib
import hmac
import json
import socket
import struct

import time
from dataclasses import dataclass


HOST = "127.0.0.1"
PORT = 5000

SECRET_KEY = b"day4-secret-key"
MAX_AGE = 5

SEEN_NONCES = set()

BUCKET_CAPACITY = 20
REFILL_RATE = 10.0

@dataclass
class TokenBucket:
    tokens: float
    last_refill: float


    def allow(self) -> bool:
        now = time.monotonic()

        elapsed = (
            now - self.last_refill
        )

        #
        # 지난 시간만큼 token 충전
        #
        self.tokens = min(
            BUCKET_CAPACITY,
            self.tokens
            + elapsed * REFILL_RATE,
        )

        self.last_refill = now


        #
        # 요청 하나 처리할 token이 있는가?
        #
        if self.tokens < 1:
            return False


        self.tokens -= 1

        return True

BUCKETS = {}

def allow_request(ip):
    bucket = BUCKETS.get(ip)

    if bucket is None:
        bucket = TokenBucket(
            tokens=BUCKET_CAPACITY,
            last_refill=time.monotonic(),
        )

        BUCKETS[ip] = bucket

    return bucket.allow()


def recv_exact(sock, size):
    data = b""

    while len(data) < size:
        chunk = sock.recv(
            size - len(data)
        )

        if not chunk:
            raise ConnectionError(
                "connection closed"
            )

        data += chunk

    return data


def recv_frame(sock):
    header = recv_exact(
        sock,
        4,
    )

    (length,) = struct.unpack(
        "!I",
        header,
    )

    return recv_exact(
        sock,
        length,
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


def verify_request(request):
    expected = calculate_mac(
        request["client_id"],
        request["timestamp"],
        request["nonce"],
        request["message"],
    )

    if not hmac.compare_digest(
        expected,
        request["mac"],
    ):
        print("INVALID MAC")
        return False

    now = int(time.time())
    age = now - request["timestamp"]

    if age < 0:
        print("timestamp is in the future")
        return False

    if age > MAX_AGE:
        print(
            f"request expired: age={age}s"
        )
        return False

    nonce = request["nonce"]

    if nonce in SEEN_NONCES:
        print(
            f"REPLAY DETECTED: nonce={nonce}"
        )
        return False

    SEEN_NONCES.add(nonce)

    return True

server = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM,
)

server.setsockopt(
    socket.SOL_SOCKET,
    socket.SO_REUSEADDR,
    1,
)

server.bind(
    (HOST, PORT)
)

server.listen()

print(
    f"server listening on "
    f"{HOST}:{PORT}"
)


while True:
    conn, addr = server.accept()

    print(
        f"connected: {addr}"
    )

    client_ip = addr[0]


    if not allow_request(client_ip):
        print(
            f"RATE LIMITED: {client_ip}"
        )

        conn.close()
        continue

    try:
        raw = recv_frame(conn)

        request = json.loads(
            raw.decode()
        )

        if not verify_request(request):
            print("INVALID MAC")
            continue

        print(
            "ACCEPT:",
            request["client_id"],
            request["timestamp"],
            request["message"],
        )

    except Exception as e:
        print(
            "error:",
            e,
        )

    finally:
        conn.close()