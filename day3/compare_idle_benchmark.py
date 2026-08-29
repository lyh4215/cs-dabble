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

RESULT_DIR = Path(
    "dist/benchmark_results/idle"
)


SERVERS = {
    "SELECT": [
        "make",
        "run-select",
    ],

    "POLL": [
        "make",
        "run-poll",
    ],

    "EPOLL-LT": [
        "make",
        "run",
    ],

    "EPOLL-ET": [
        "make",
        "run-et",
    ],
}


ROW_PATTERN = re.compile(
    r"^\s*"
    r"(\d+)\s+"          # total
    r"(\d+)\s+"          # active
    r"(\d+)\s+"          # requests
    r"([\d.]+)\s+"       # throughput
    r"([\d.]+)ms\s+"     # mean
    r"([\d.]+)ms\s+"     # p50
    r"([\d.]+)ms\s+"     # p95
    r"([\d.]+)ms"        # p99
)


def wait_for_server(
    timeout=5.0,
):
    deadline = (
        time.time()
        + timeout
    )

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


def stop_server(process):
    if process.poll() is not None:
        return

    try:
        os.killpg(
            process.pid,
            signal.SIGTERM,
        )

        process.wait(
            timeout=2
        )

    except subprocess.TimeoutExpired:
        os.killpg(
            process.pid,
            signal.SIGKILL,
        )

        process.wait()


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

        start_new_session=True,
    )

    try:
        wait_for_server()

    except Exception:
        stop_server(
            process
        )

        log_file.close()

        raise

    return (
        process,
        log_file,
    )


def parse_output(text):
    results = []

    for line in text.splitlines():
        match = (
            ROW_PATTERN.match(line)
        )

        if not match:
            continue

        (
            total,
            active,
            requests,
            throughput,
            mean,
            p50,
            p95,
            p99,
        ) = match.groups()

        results.append({
            "total": int(total),
            "active": int(active),
            "requests": int(requests),

            "throughput": float(
                throughput
            ),

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


def run_once(
    mode,
    run_number,
):
    print(
        f"[{mode}] "
        f"run {run_number}/{REPEATS}"
    )

    result = subprocess.run(
        [
            sys.executable,
            "idle_load_test.py",
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
        / (
            f"{safe_name}"
            f"_run_{run_number}.txt"
        )
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

    server, log_file = (
        start_server(mode)
    )

    results = []

    try:
        for run_number in range(
            1,
            REPEATS + 1,
        ):
            rows = run_once(
                mode,
                run_number,
            )

            for row in rows:
                row["mode"] = mode
                row["run"] = run_number

            results.extend(rows)

            time.sleep(0.3)

    finally:
        stop_server(
            server
        )

        log_file.close()

    return results


def summarize(results):
    groups = {}

    for row in results:
        key = (
            row["mode"],
            row["total"],
        )

        groups.setdefault(
            key,
            [],
        ).append(row)

    summary = []

    for (
        mode,
        total,
    ), rows in groups.items():

        summary.append({
            "mode": mode,
            "total": total,

            "throughput":
                statistics.median(
                    row["throughput"]
                    for row in rows
                ),

            "mean":
                statistics.median(
                    row["mean"]
                    for row in rows
                ),

            "p50":
                statistics.median(
                    row["p50"]
                    for row in rows
                ),

            "p95":
                statistics.median(
                    row["p95"]
                    for row in rows
                ),

            "p99":
                statistics.median(
                    row["p99"]
                    for row in rows
                ),
        })

    return summary


def save_csv(
    results,
    filename,
):
    path = (
        RESULT_DIR
        / filename
    )

    if not results:
        return path

    fields = list(
        results[0].keys()
    )

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


def print_throughput(
    summary,
):
    lookup = {
        (
            row["mode"],
            row["total"],
        ): row
        for row in summary
    }

    totals = sorted({
        row["total"]
        for row in summary
    })


    print()
    print(
        f"Mostly-idle throughput "
        f"(median of {REPEATS})"
    )

    print(
        f"{'total':>7} "
        f"{'select':>10} "
        f"{'poll':>10} "
        f"{'epoll-LT':>10} "
        f"{'epoll-ET':>10}"
    )

    print("-" * 55)

    for total in totals:
        print(
            f"{total:>7} "
            f"{lookup[('SELECT', total)]['throughput']:>10.0f} "
            f"{lookup[('POLL', total)]['throughput']:>10.0f} "
            f"{lookup[('EPOLL-LT', total)]['throughput']:>10.0f} "
            f"{lookup[('EPOLL-ET', total)]['throughput']:>10.0f}"
        )


def print_p99(
    summary,
):
    lookup = {
        (
            row["mode"],
            row["total"],
        ): row
        for row in summary
    }

    totals = sorted({
        row["total"]
        for row in summary
    })


    print()
    print(
        f"Mostly-idle p99 latency "
        f"(median of {REPEATS})"
    )

    print(
        f"{'total':>7} "
        f"{'select':>10} "
        f"{'poll':>10} "
        f"{'epoll-LT':>10} "
        f"{'epoll-ET':>10}"
    )

    print("-" * 63)

    for total in totals:
        print(
            f"{total:>7} "
            f"{lookup[('SELECT', total)]['p99']:>9.3f}ms "
            f"{lookup[('POLL', total)]['p99']:>9.3f}ms "
            f"{lookup[('EPOLL-LT', total)]['p99']:>9.3f}ms "
            f"{lookup[('EPOLL-ET', total)]['p99']:>9.3f}ms"
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

        time.sleep(0.5)


    summary = summarize(
        all_results
    )


    raw_path = save_csv(
        all_results,
        "all_results.csv",
    )

    summary_path = save_csv(
        summary,
        "summary_median.csv",
    )


    print_throughput(
        summary
    )

    print_p99(
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