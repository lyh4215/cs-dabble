
import os
import sys
import time
import socket
import subprocess
from pathlib import Path

QUEUE_LIMITS = [0, 10, 20, 50, 100]
RATE = 120

HOST = "127.0.0.1"
PORT = 9000

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)


def wait_for_server(proc, timeout=5):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Server exited with code {proc.returncode}"
            )

        try:
            with socket.create_connection(
                (HOST, PORT), timeout=0.2
            ):
                return
        except OSError:
            time.sleep(0.05)

    raise TimeoutError("Server failed to start")


def stop_server(proc):
    if proc.poll() is not None:
        return

    proc.terminate()

    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_experiment(queue_limit):
    env = os.environ.copy()
    env["QUEUE_LIMIT"] = str(queue_limit)

    log_path = RESULT_DIR / f"server_q{queue_limit}.log"

    with open(log_path, "w") as log:
        server = subprocess.Popen(
            [sys.executable, "-u", "server.py"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

        try:
            wait_for_server(server)

            result = subprocess.run(
                [sys.executable, "client.py", str(RATE)],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )

            return result.stdout

        finally:
            stop_server(server)


def main():
    summary = []

    for limit in QUEUE_LIMITS:
        print(f"\n{'=' * 45}")
        print(f"QUEUE_LIMIT = {limit}")
        print(f"{'=' * 45}")

        output = run_experiment(limit)

        print(output)
        summary.append(
            f"QUEUE_LIMIT={limit}\n{output}"
        )

    (RESULT_DIR / "experiment2.txt").write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print("\nAll experiments completed.")
    print("Saved to results/experiment2.txt")


if __name__ == "__main__":
    main()
