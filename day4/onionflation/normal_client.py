import argparse
import csv
import hashlib
import json
import statistics
import threading
import time

from urllib.request import (
    Request,
    urlopen,
)


BASE = "http://127.0.0.1:8088"


def get_json(path):
    with urlopen(
        BASE + path,
        timeout=10,
    ) as response:
        return json.load(response)


def post_json(
    path,
    data,
):
    body = json.dumps(
        data
    ).encode()

    request = Request(
        BASE + path,
        data=body,
        headers={
            "Content-Type":
                "application/json"
        },
        method="POST",
    )

    with urlopen(
        request,
        timeout=20,
    ) as response:
        return json.load(response)


def has_leading_zero_bits(
    digest,
    bits,
):
    value = int.from_bytes(
        digest,
        "big",
    )

    return (
        value >> (256 - bits)
    ) == 0


def solve(
    challenge,
    difficulty,
):
    prefix = (
        challenge.encode()
        + b":"
    )

    nonce = 0

    while True:
        digest = hashlib.sha256(
            prefix
            + str(nonce).encode()
        ).digest()

        if has_leading_zero_bits(
            digest,
            difficulty,
        ):
            return nonce

        nonce += 1


def percentile(
    values,
    p,
):
    if not values:
        return 0.0

    values = sorted(values)

    index = int(
        (len(values) - 1) * p
    )

    return values[index]


def run_worker(
    worker_id,
    deadline,
    think_time,
    results,
    lock,
):
    while (
        time.monotonic()
        < deadline
    ):
        try:
            total_start = (
                time.monotonic()
            )

            challenge_data = get_json(
                "/challenge"
            )

            challenge = (
                challenge_data[
                    "challenge"
                ]
            )

            difficulty = (
                challenge_data[
                    "difficulty"
                ]
            )

            solve_start = (
                time.monotonic()
            )

            nonce = solve(
                challenge,
                difficulty,
            )

            solve_time = (
                time.monotonic()
                - solve_start
            )

            response = post_json(
                "/submit",
                {
                    "challenge":
                        challenge,
                    "nonce":
                        nonce,
                },
            )

            total_time = (
                time.monotonic()
                - total_start
            )

            record = {
                "timestamp":
                    time.time(),
                "worker":
                    worker_id,
                "difficulty":
                    difficulty,
                "solve_ms":
                    solve_time * 1000,
                "queue_ms":
                    response[
                        "queue_latency"
                    ] * 1000,
                "total_ms":
                    total_time * 1000,
            }

            with lock:
                results.append(
                    record
                )

            time.sleep(
                think_time
            )

        except Exception as e:
            print(
                f"[worker {worker_id}] "
                f"{e}"
            )

            time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--duration",
        type=float,
        default=20,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--think",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--output",
        default="dist/normal.csv",
    )

    args = parser.parse_args()

    deadline = (
        time.monotonic()
        + args.duration
    )

    results = []

    lock = threading.Lock()

    threads = []

    for worker_id in range(
        args.workers
    ):
        thread = threading.Thread(
            target=run_worker,
            args=(
                worker_id,
                deadline,
                args.think,
                results,
                lock,
            ),
        )

        thread.start()

        threads.append(thread)

    for thread in threads:
        thread.join()

    with open(
        args.output,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "worker",
                "difficulty",
                "solve_ms",
                "queue_ms",
                "total_ms",
            ],
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    if not results:
        print("No results")
        return

    total = [
        r["total_ms"]
        for r in results
    ]

    solve_times = [
        r["solve_ms"]
        for r in results
    ]

    print()
    print(
        f"requests: {len(results)}"
    )

    print(
        "difficulty range:",
        min(
            r["difficulty"]
            for r in results
        ),
        "->",
        max(
            r["difficulty"]
            for r in results
        ),
    )

    print(
        f"mean solve: "
        f"{statistics.mean(solve_times):.2f} ms"
    )

    print(
        f"p50 total: "
        f"{percentile(total, 0.50):.2f} ms"
    )

    print(
        f"p95 total: "
        f"{percentile(total, 0.95):.2f} ms"
    )

    print(
        f"p99 total: "
        f"{percentile(total, 0.99):.2f} ms"
    )

    print(
        f"saved: {args.output}"
    )


if __name__ == "__main__":
    main()