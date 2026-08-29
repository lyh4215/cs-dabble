import csv
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path


REPEATS = 5
HOST = "127.0.0.1"
PORT = 5000

RESULT_DIR = Path("dist/benchmark_results")


SERVERS = {
    "SELECT": ["make", "run-select"],
    "POLL": ["make", "run-poll"],
    "EPOLL-LT": ["make", "run"],
    "EPOLL-ET": ["make", "run-et"],
}


ROW_PATTERN = re.compile(
    r"^\s*"
    r"(\d+)\s+"          # connections
    r"(\d+)\s+"          # requests
    r"([\d.]+)\s+"       # msg/s
    r"([\d.]+)ms\s+"     # mean
    r"([\d.]+)ms\s+"     # p50
    r"([\d.]+)ms\s+"     # p95
    r"([\d.]+)ms"        # p99
)


def wait_for_server(timeout=5.0):
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            with socket.create_connection(
                (HOST, PORT),
                timeout=0.2,
            ):
                return

        except OSError:
            time.sleep(0.05)

    raise RuntimeError(
        "server did not start"
    )


def start_server(mode):
    command = SERVERS[mode]

    safe_name = (
        mode.lower()
        .replace("-", "_")
    )

    log_path = (
        RESULT_DIR
        / f"server_{safe_name}.log"
    )

    log_file = open(
        log_path,
        "w",
    )

    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,

        # make와 실제 server process를
        # 같은 process group으로 묶는다.
        start_new_session=True,
    )

    try:
        wait_for_server()

    except Exception:
        stop_server(process)
        log_file.close()
        raise

    return process, log_file


def stop_server(process):
    if process.poll() is not None:
        return

    try:
        os.killpg(
            process.pid,
            signal.SIGTERM,
        )

        process.wait(timeout=2)

    except subprocess.TimeoutExpired:
        os.killpg(
            process.pid,
            signal.SIGKILL,
        )

        process.wait()


def parse_output(text):
    results = []

    for line in text.splitlines():
        match = ROW_PATTERN.match(line)

        if not match:
            continue

        (
            connections,
            requests,
            throughput,
            mean,
            p50,
            p95,
            p99,
        ) = match.groups()

        results.append({
            "connections": int(connections),
            "requests": int(requests),
            "throughput": float(throughput),
            "mean": float(mean),
            "p50": float(p50),
            "p95": float(p95),
            "p99": float(p99),
        })

    if not results:
        raise RuntimeError(
            "could not parse benchmark output"
        )

    return results


def run_once(mode, run_number):
    print(
        f"[{mode}] "
        f"run {run_number}/{REPEATS}"
    )

    result = subprocess.run(
        [
            sys.executable,
            "load_test.py",
            "--sweep",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    safe_name = (
        mode.lower()
        .replace("-", "_")
    )

    raw_path = (
        RESULT_DIR
        / f"{safe_name}_run_{run_number}.txt"
    )

    raw_path.write_text(
        result.stdout,
        encoding="utf-8",
    )

    return parse_output(
        result.stdout
    )


def run_mode(mode):
    print()
    print(
        f"===== {mode} ====="
    )

    server, log_file = start_server(
        mode
    )

    all_results = []

    try:
        for run_number in range(
            1,
            REPEATS + 1,
        ):
            results = run_once(
                mode,
                run_number,
            )

            for row in results:
                row["mode"] = mode
                row["run"] = run_number

            all_results.extend(
                results
            )

            time.sleep(0.3)

    finally:
        stop_server(server)
        log_file.close()

    return all_results


def save_raw_csv(results):
    path = (
        RESULT_DIR
        / "all_results.csv"
    )

    fields = [
        "mode",
        "run",
        "connections",
        "requests",
        "throughput",
        "mean",
        "p50",
        "p95",
        "p99",
    ]

    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(results)

    return path


def summarize(results):
    grouped = {}

    for row in results:
        key = (
            row["mode"],
            row["connections"],
        )

        grouped.setdefault(
            key,
            [],
        ).append(row)

    rows = []

    for (
        mode,
        connections,
    ), values in grouped.items():

        rows.append({
            "mode": mode,
            "connections": connections,

            "throughput":
                statistics.median(
                    x["throughput"]
                    for x in values
                ),

            "mean":
                statistics.median(
                    x["mean"]
                    for x in values
                ),

            "p50":
                statistics.median(
                    x["p50"]
                    for x in values
                ),

            "p95":
                statistics.median(
                    x["p95"]
                    for x in values
                ),

            "p99":
                statistics.median(
                    x["p99"]
                    for x in values
                ),
        })

    return sorted(
        rows,
        key=lambda x: (
            x["connections"],
            x["mode"],
        ),
    )


def save_summary_csv(summary):
    path = (
        RESULT_DIR
        / "summary_median.csv"
    )

    fields = [
        "mode",
        "connections",
        "throughput",
        "mean",
        "p50",
        "p95",
        "p99",
    ]

    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(summary)

    return path


def print_throughput_table(summary):
    lookup = {
        (
            row["mode"],
            row["connections"],
        ): row
        for row in summary
    }

    connections = sorted({
        row["connections"]
        for row in summary
    })

    print()
    print(
        f"Median throughput of "
        f"{REPEATS} runs"
    )

    print(
        f"{'conn':>6} "
        f"{'select':>10} "
        f"{'poll':>10} "
        f"{'epoll-LT':>10} "
        f"{'epoll-ET':>10}"
    )

    print("-" * 54)

    for conn in connections:
        values = []

        for mode in SERVERS:
            row = lookup[
                (mode, conn)
            ]

            values.append(
                row["throughput"]
            )

        print(
            f"{conn:>6} "
            f"{values[0]:>10.0f} "
            f"{values[1]:>10.0f} "
            f"{values[2]:>10.0f} "
            f"{values[3]:>10.0f}"
        )


def print_p99_table(summary):
    lookup = {
        (
            row["mode"],
            row["connections"],
        ): row
        for row in summary
    }

    connections = sorted({
        row["connections"]
        for row in summary
    })

    print()
    print(
        f"Median p99 latency of "
        f"{REPEATS} runs"
    )

    print(
        f"{'conn':>6} "
        f"{'select':>10} "
        f"{'poll':>10} "
        f"{'epoll-LT':>10} "
        f"{'epoll-ET':>10}"
    )

    print("-" * 62)

    for conn in connections:
        values = []

        for mode in SERVERS:
            row = lookup[
                (mode, conn)
            ]

            values.append(
                row["p99"]
            )

        print(
            f"{conn:>6} "
            f"{values[0]:>9.3f}ms "
            f"{values[1]:>9.3f}ms "
            f"{values[2]:>9.3f}ms "
            f"{values[3]:>9.3f}ms"
        )


def main():
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    for mode in SERVERS:
        all_results.extend(
            run_mode(mode)
        )

        #
        # 다음 server가 같은 port를
        # bind하기 전에 약간 대기
        #
        time.sleep(0.5)

    raw_path = save_raw_csv(
        all_results
    )

    summary = summarize(
        all_results
    )

    summary_path = save_summary_csv(
        summary
    )

    print_throughput_table(
        summary
    )

    print_p99_table(
        summary
    )

    print()

    print(
        f"raw results : "
        f"{raw_path}"
    )

    print(
        f"summary     : "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()