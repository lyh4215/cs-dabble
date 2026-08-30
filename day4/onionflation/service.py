import hashlib
import json
import queue
import secrets
import threading
import time
import os

from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HOST = "127.0.0.1"
PORT = 8088

ROUND_SECONDS = 5.0

MIN_DIFFICULTY = 10
MAX_DIFFICULTY = 22

# queue가 이 이상 한 번이라도 올라가면
# 다음 round difficulty를 크게 올림.
HIGH_WATER = 20

# congestion이 없으면 천천히 감소.
LOW_WATER = 3

DIFFICULTY_INCREASE = 8
DIFFICULTY_DECREASE = 1
CONTROLLER = os.environ.get(
    "CONTROLLER",
    "peak",
)
HIGH_QUEUE_RATIO = 0.40

IMPROVED_INCREASE = 2
IMPROVED_DECREASE = 2

# service가 초당 약 40 request 처리
SERVICE_RATE = 40.0

CHALLENGE_TTL = 30.0

TRACE_FILE = "dist/service_trace.csv"
TRACE_INTERVAL = 0.1


@dataclass
class WorkItem:
    enqueued_at: float
    done: threading.Event = field(
        default_factory=threading.Event
    )


class ServiceState:
    def __init__(self):
        self.lock = threading.Lock()

        self.difficulty = MIN_DIFFICULTY

        self.round_id = 0
        self.round_start = time.monotonic()
        self.round_end = (
            self.round_start + ROUND_SECONDS
        )

        self.round_peak_queue = 0
        self.last_peak_queue = 0

        self.challenges = {}

        self.round_queue_samples = 0
        self.round_high_samples = 0

        self.work_queue = queue.Queue()


STATE = ServiceState()

def trace_loop():
    import csv

    with open(
        TRACE_FILE,
        "w",
        newline="",
        buffering=1,
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "round",
            "difficulty",
            "queue",
            "round_peak_queue",
        ])

        while True:
            with STATE.lock:
                qsize = STATE.work_queue.qsize()

                STATE.round_queue_samples += 1

                if qsize >= HIGH_WATER:
                    STATE.round_high_samples += 1

                writer.writerow([
                    time.time(),
                    STATE.round_id,
                    STATE.difficulty,
                    qsize,
                    STATE.round_peak_queue,
                ])

            time.sleep(TRACE_INTERVAL)


def has_leading_zero_bits(
    digest: bytes,
    bits: int,
) -> bool:
    value = int.from_bytes(
        digest,
        "big",
    )

    return (
        value >> (256 - bits)
    ) == 0


def verify_puzzle(
    challenge: str,
    nonce: int,
    difficulty: int,
) -> bool:
    message = (
        challenge.encode()
        + b":"
        + str(nonce).encode()
    )

    digest = hashlib.sha256(
        message
    ).digest()

    return has_leading_zero_bits(
        digest,
        difficulty,
    )


def worker_loop():
    delay = 1.0 / SERVICE_RATE

    while True:
        item = STATE.work_queue.get()

        time.sleep(delay)

        item.done.set()

        STATE.work_queue.task_done()


def controller_loop():
    while True:
        with STATE.lock:
            round_end = STATE.round_end

        sleep_for = (
            round_end - time.monotonic()
        )

        if sleep_for > 0:
            time.sleep(sleep_for)

        with STATE.lock:
            peak = STATE.round_peak_queue
            old = STATE.difficulty

            samples = STATE.round_queue_samples
            high_samples = STATE.round_high_samples

            if samples > 0:
                high_ratio = (
                    high_samples / samples
                )
            else:
                high_ratio = 0.0


            if CONTROLLER == "peak":

                #
                # 기존 취약한 controller
                #
                if peak >= HIGH_WATER:
                    STATE.difficulty = min(
                        MAX_DIFFICULTY,
                        STATE.difficulty
                        + DIFFICULTY_INCREASE,
                    )

                elif peak <= LOW_WATER:
                    STATE.difficulty = max(
                        MIN_DIFFICULTY,
                        STATE.difficulty
                        - DIFFICULTY_DECREASE,
                    )


            elif CONTROLLER == "improved":

                #
                # 라운드의 상당 부분에서 congestion이
                # 지속된 경우에만 difficulty 증가
                #
                if high_ratio >= HIGH_QUEUE_RATIO:
                    STATE.difficulty = min(
                        MAX_DIFFICULTY,
                        STATE.difficulty
                        + IMPROVED_INCREASE,
                    )

                else:
                    STATE.difficulty = max(
                        MIN_DIFFICULTY,
                        STATE.difficulty
                        - IMPROVED_DECREASE,
                    )

            STATE.last_peak_queue = peak

            STATE.round_id += 1

            STATE.round_start = (
                STATE.round_end
            )

            STATE.round_end += (
                ROUND_SECONDS
            )

            STATE.round_peak_queue = (
                STATE.work_queue.qsize()
            )
            STATE.round_queue_samples = 0
            STATE.round_high_samples = 0
            new = STATE.difficulty

            # 오래된 challenge 제거
            now = time.monotonic()

            STATE.challenges = {
                c: data
                for c, data
                in STATE.challenges.items()
                if data[1] > now
            }

        print(
            f"[ROUND {STATE.round_id}] "
            f"peak={peak:3d} "
            f"high_ratio={high_ratio:.2f} "
            f"difficulty={old} -> {new}",
            flush=True,
        )


class Handler(BaseHTTPRequestHandler):
    def log_message(
        self,
        format,
        *args,
    ):
        return

    def send_json(
        self,
        status,
        data,
    ):
        body = json.dumps(
            data
        ).encode()

        self.send_response(status)

        self.send_header(
            "Content-Type",
            "application/json",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.end_headers()

        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/challenge":
            self.handle_challenge()
            return

        if self.path == "/stats":
            self.handle_stats()
            return

        self.send_json(
            404,
            {"error": "not found"},
        )

    def do_POST(self):
        if self.path == "/submit":
            self.handle_submit()
            return

        self.send_json(
            404,
            {"error": "not found"},
        )

    def handle_challenge(self):
        challenge = secrets.token_hex(16)

        now = time.monotonic()

        with STATE.lock:
            difficulty = (
                STATE.difficulty
            )

            STATE.challenges[
                challenge
            ] = (
                difficulty,
                now + CHALLENGE_TTL,
            )

            round_id = STATE.round_id

            ends_in = max(
                0.0,
                STATE.round_end - now,
            )

        self.send_json(
            200,
            {
                "challenge": challenge,
                "difficulty": difficulty,
                "round": round_id,
                "round_ends_in": ends_in,
            },
        )

    def handle_stats(self):
        now = time.monotonic()

        with STATE.lock:
            result = {
                "round": STATE.round_id,
                "difficulty":
                    STATE.difficulty,
                "queue":
                    STATE.work_queue.qsize(),
                "round_peak_queue":
                    STATE.round_peak_queue,
                "last_peak_queue":
                    STATE.last_peak_queue,
                "round_ends_in": max(
                    0.0,
                    STATE.round_end - now,
                ),
            }

        self.send_json(
            200,
            result,
        )

    def handle_submit(self):
        try:
            length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )

            body = self.rfile.read(
                length
            )

            request = json.loads(
                body
            )

            challenge = request[
                "challenge"
            ]

            nonce = int(
                request["nonce"]
            )

        except Exception:
            self.send_json(
                400,
                {"error": "bad request"},
            )
            return

        now = time.monotonic()

        #
        # challenge는 1회 사용.
        #
        with STATE.lock:
            data = STATE.challenges.pop(
                challenge,
                None,
            )

        if data is None:
            self.send_json(
                403,
                {"error": "unknown challenge"},
            )
            return

        difficulty, expires = data

        if now > expires:
            self.send_json(
                403,
                {"error": "expired"},
            )
            return

        if not verify_puzzle(
            challenge,
            nonce,
            difficulty,
        ):
            self.send_json(
                403,
                {"error": "invalid puzzle"},
            )
            return

        item = WorkItem(
            enqueued_at=time.monotonic()
        )

        STATE.work_queue.put(item)

        qsize = STATE.work_queue.qsize()

        with STATE.lock:
            STATE.round_peak_queue = max(
                STATE.round_peak_queue,
                qsize,
            )

        if not item.done.wait(
            timeout=20
        ):
            self.send_json(
                503,
                {"error": "timeout"},
            )
            return

        queue_latency = (
            time.monotonic()
            - item.enqueued_at
        )

        self.send_json(
            200,
            {
                "ok": True,
                "difficulty": difficulty,
                "queue_latency":
                    queue_latency,
            },
        )


class LocalServer(
    ThreadingHTTPServer
):
    daemon_threads = True
    request_queue_size = 128


if __name__ == "__main__":
    threading.Thread(
        target=worker_loop,
        daemon=True,
    ).start()

    threading.Thread(
        target=controller_loop,
        daemon=True,
    ).start()

    threading.Thread(
        target=trace_loop,
        daemon=True,
    ).start()

    server = LocalServer(
        (HOST, PORT),
        Handler,
    )

    print(
        f"Toy service running at "
        f"http://{HOST}:{PORT}",
        flush=True,
    )

    server.serve_forever()