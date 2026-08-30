import csv
import os
import signal
import statistics
import subprocess
import sys
import time

from urllib.request import urlopen


ROOT = os.path.dirname(
    os.path.abspath(__file__)
)


def percentile(
    values,
    p,
):
    values = sorted(values)

    if not values:
        return 0

    index = int(
        (len(values) - 1) * p
    )

    return values[index]


def wait_for_service():
    for _ in range(50):
        try:
            with urlopen(
                "http://127.0.0.1:8088/stats",
                timeout=1,
            ):
                return

        except Exception:
            time.sleep(0.1)

    raise RuntimeError(
        "service did not start"
    )


def start_service(
    log_name,
    controller,
):
    log = open(
        log_name,
        "w",
    )

    env = os.environ.copy()

    env["CONTROLLER"] = controller

    process = subprocess.Popen(
        [
            sys.executable,
            "service.py",
        ],
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )

    wait_for_service()

    return process, log

def stop_service(
    process,
    log,
):
    process.terminate()

    try:
        process.wait(
            timeout=3
        )

    except subprocess.TimeoutExpired:
        process.kill()

    log.close()

    time.sleep(0.5)


def read_results(path):
    rows = []

    with open(path) as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(
                {
                    "difficulty":
                        int(
                            row[
                                "difficulty"
                            ]
                        ),
                    "solve_ms":
                        float(
                            row["solve_ms"]
                        ),
                    "total_ms":
                        float(
                            row["total_ms"]
                        ),
                }
            )

    return rows


def summarize(
    name,
    rows,
):
    total = [
        x["total_ms"]
        for x in rows
    ]

    solve = [
        x["solve_ms"]
        for x in rows
    ]

    difficulty = [
        x["difficulty"]
        for x in rows
    ]

    print()
    print(
        f"=== {name} ==="
    )

    print(
        f"requests      : "
        f"{len(rows)}"
    )

    print(
        f"max difficulty: "
        f"{max(difficulty)}"
    )

    print(
        f"mean solve ms : "
        f"{statistics.mean(solve):.2f}"
    )

    print(
        f"p50 total ms  : "
        f"{percentile(total, .50):.2f}"
    )

    print(
        f"p95 total ms  : "
        f"{percentile(total, .95):.2f}"
    )

    print(
        f"p99 total ms  : "
        f"{percentile(total, .99):.2f}"
    )


def baseline():
    print()
    print(
        "===== BASELINE ====="
    )

    process, log = start_service(
        "dist/service_baseline.log"
    )

    try:
        subprocess.run(
            [
                sys.executable,
                "normal_client.py",
                "--duration",
                "15",
                "--output",
                "dist/baseline.csv",
            ],
            cwd=ROOT,
            check=True,
        )

    finally:
        stop_service(
            process,
            log,
        )

def adversarial(
    controller,
    output,
    log_name,
):
    print()
    print(
        f"===== {controller.upper()} ====="
    )

    process, log = start_service(
        log_name,
        controller,
    )

    try:
        normal = subprocess.Popen(
            [
                sys.executable,
                "normal_client.py",
                "--duration",
                "20",
                "--output",
                output,
            ],
            cwd=ROOT,
        )

        time.sleep(1)

        subprocess.run(
            [
                sys.executable,
                "adversarial_client.py",
            ],
            cwd=ROOT,
            check=True,
        )

        normal.wait()

    finally:
        stop_service(
            process,
            log,
        )

def main():
    adversarial(
        "peak",
        "peak.csv",
        "service_peak.log",
    )

    adversarial(
        "improved",
        "improved.csv",
        "service_improved.log",
    )

    peak_rows = read_results(
        os.path.join(
            ROOT,
            "peak.csv",
        )
    )

    improved_rows = read_results(
        os.path.join(
            ROOT,
            "improved.csv",
        )
    )

    print()
    print("=" * 50)
    print("CONTROLLER COMPARISON")
    print("=" * 50)

    summarize(
        "PEAK CONTROLLER",
        peak_rows,
    )

    summarize(
        "IMPROVED CONTROLLER",
        improved_rows,
    )

if __name__ == "__main__":
    main()