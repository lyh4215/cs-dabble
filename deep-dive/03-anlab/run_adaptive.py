
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

HOST = "127.0.0.1"
PORT = 9000
RATE = 120

CASES = [
    ("Fixed-10", "fixed", 10),
    ("Fixed-20", "fixed", 20),
    ("EWMA", "adaptive", 20),
    ("Oracle", "oracle", 20),
]

OUT = Path("results/adaptive")
OUT.mkdir(parents=True, exist_ok=True)


def wait_ready(proc, timeout=5):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("Server exited early")

        try:
            with socket.create_connection(
                (HOST, PORT), timeout=0.2
            ):
                return
        except OSError:
            time.sleep(0.05)

    raise TimeoutError("Server not ready")


def run_case(name, policy, limit):
    env = os.environ.copy()
    env["POLICY"] = policy
    env["QUEUE_LIMIT"] = str(limit)

    log_path = OUT / f"{name}.log"

    with log_path.open("w") as log:
        server = subprocess.Popen(
            [sys.executable, "-u", "adaptive_server.py"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

        try:
            wait_ready(server)

            result = subprocess.run(
                [sys.executable, "client.py", str(RATE)],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            output = result.stdout

        finally:
            server.terminate()
            try:
                server.wait(timeout=3)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()

    return output


def extract(output, label):
    match = re.search(
        rf"^{re.escape(label)}:\s+([\d.]+)",
        output,
        re.MULTILINE,
    )
    return float(match.group(1)) if match else None


def main():
    rows = []

    for name, policy, limit in CASES:
        print(f"\n=== {name} ===")

        output = run_case(name, policy, limit)
        print(output)

        rows.append((
            name,
            extract(output, "Goodput"),
            extract(output, "Rejection rate"),
            extract(output, "P99 latency"),
        ))

    print("\n=== COMPARISON ===")
    print(
        f"{'Policy':12} {'Goodput':>10} "
        f"{'Reject %':>10} {'P99 ms':>10}"
    )

    for name, goodput, rejection, p99 in rows:
        print(
            f"{name:12} {goodput:10.2f} "
            f"{rejection:10.2f} {p99:10.2f}"
        )


if __name__ == "__main__":
    main()
