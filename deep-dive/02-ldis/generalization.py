
import random
import statistics
from collections import Counter
from itertools import combinations

from heavy_hitter import TopKHistogram
from histogram import build_histogram
from optimizer import actual_cardinality

DOMAIN = 100
WIDTHS = [5, 10, 20]
TOP_K = [0, 1, 2, 3, 5]
N_TRIALS = 200

PAIRS = list(combinations("ABC", 2))


def generate_tables(rng):
    tables = {}

    for name in "ABC":
        # Nonuniform background frequencies
        freq = Counter({
            key: rng.randint(1, 20)
            for key in range(DOMAIN)
        })

        # Multiple heavy hitters
        n_heavy = rng.randint(2, 5)

        for key in rng.sample(range(DOMAIN), n_heavy):
            freq[key] += rng.randint(100, 1000)

        tables[name] = freq

    return tables


def build_topk(table, width, k):
    hist = build_histogram(table, width, DOMAIN)

    heavy = dict(sorted(
        table.items(),
        key=lambda item: (-item[1], item[0])
    )[:k])

    return TopKHistogram(
        buckets=hist.buckets,
        heavy=heavy
    )


def evaluate(tables, width, k):
    hists = {
        name: build_topk(table, width, k)
        for name, table in tables.items()
    }

    actual = {}
    estimated = {}

    for a, b in PAIRS:
        pair = a + b

        actual[pair] = actual_cardinality(
            tables[a], tables[b]
        )

        estimated[pair] = sum(
            hists[a].estimate_frequency(key)
            * hists[b].estimate_frequency(key)
            for key in range(DOMAIN)
        )

    selected = min(estimated, key=estimated.get)
    optimal = min(actual, key=actual.get)

    return {
        "regret": actual[selected] / actual[optimal],
        "correct": selected == optimal,
    }


def percentile(values, q):
    values = sorted(values)
    index = int((len(values) - 1) * q)
    return values[index]


def main():
    rng = random.Random(42)

    # Pre-generate identical workloads for all policies.
    datasets = [
        generate_tables(rng)
        for _ in range(N_TRIALS)
    ]

    for width in WIDTHS:
        print(f"\n=== Bucket Width: {width} ===")
        print(
            f"{'Top-K':>5} {'Mean Regret':>12} "
            f"{'P95 Regret':>12} {'Max Regret':>12} "
            f"{'Optimal %':>10}"
        )

        for k in TOP_K:
            results = [
                evaluate(tables, width, k)
                for tables in datasets
            ]

            regrets = [r["regret"] for r in results]
            correct = [r["correct"] for r in results]

            print(
                f"{k:>5} "
                f"{statistics.mean(regrets):>12.3f} "
                f"{percentile(regrets, 0.95):>12.3f} "
                f"{max(regrets):>12.3f} "
                f"{100 * statistics.mean(correct):>9.1f}%"
            )


if __name__ == "__main__":
    main()
