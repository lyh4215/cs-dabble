import re
import statistics
import subprocess


ALPHAS = [
    0.05,
    0.25,
    0.8,
]

RUNS = 10


def run_once(alpha):
    result = subprocess.run(
        [
            "python",
            "scheduler_sim.py",
            "adaptive",
            str(alpha),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    output = result.stdout

    mean_match = re.search(
        r"mean latency: ([\d.]+)s",
        output,
    )

    median_match = re.search(
        r"median latency: ([\d.]+)s",
        output,
    )

    max_match = re.search(
        r"max latency: ([\d.]+)s",
        output,
    )

    a_match = re.search(
        r"A: jobs=(\d+), mean_latency=([\d.]+)s",
        output,
    )

    if not all([
        mean_match,
        median_match,
        max_match,
        a_match,
    ]):
        raise RuntimeError(
            f"failed to parse output:\n{output}"
        )

    return {
        "mean": float(
            mean_match.group(1)
        ),
        "median": float(
            median_match.group(1)
        ),
        "max": float(
            max_match.group(1)
        ),
        "a_jobs": int(
            a_match.group(1)
        ),
        "a_mean": float(
            a_match.group(2)
        ),
    }


def summarize(values):
    return (
        statistics.mean(values),
        statistics.stdev(values),
    )


def main():
    all_results = {}

    for alpha in ALPHAS:
        print(
            f"\n=== alpha={alpha} ==="
        )

        runs = []

        for i in range(RUNS):
            result = run_once(
                alpha
            )

            runs.append(
                result
            )

            print(
                f"{i + 1:02d}/{RUNS} "
                f"mean={result['mean']:.3f}s "
                f"max={result['max']:.3f}s "
                f"A_jobs={result['a_jobs']}"
            )

        all_results[alpha] = runs

    print()
    print(
        "=== FINAL SUMMARY ==="
    )

    for alpha, runs in all_results.items():
        mean_values = [
            r["mean"]
            for r in runs
        ]

        median_values = [
            r["median"]
            for r in runs
        ]

        max_values = [
            r["max"]
            for r in runs
        ]

        a_jobs_values = [
            r["a_jobs"]
            for r in runs
        ]

        mean_avg, mean_std = (
            summarize(mean_values)
        )

        median_avg, median_std = (
            summarize(median_values)
        )

        max_avg, max_std = (
            summarize(max_values)
        )

        a_jobs_avg, a_jobs_std = (
            summarize(a_jobs_values)
        )

        print()
        print(
            f"alpha = {alpha}"
        )

        print(
            f"mean latency  : "
            f"{mean_avg:.3f} "
            f"± {mean_std:.3f}s"
        )

        print(
            f"median latency: "
            f"{median_avg:.3f} "
            f"± {median_std:.3f}s"
        )

        print(
            f"max latency   : "
            f"{max_avg:.3f} "
            f"± {max_std:.3f}s"
        )

        print(
            f"A jobs        : "
            f"{a_jobs_avg:.1f} "
            f"± {a_jobs_std:.1f}"
        )


if __name__ == "__main__":
    main()