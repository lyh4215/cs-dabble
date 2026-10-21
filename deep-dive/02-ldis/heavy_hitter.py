
from dataclasses import dataclass
from collections import Counter

from adversarial import (
    search, evaluate, DOMAIN, BUCKET_WIDTH, PAIRS
)
from histogram import build_histogram
from optimizer import actual_cardinality


@dataclass
class TopKHistogram:
    buckets: list[tuple[int, int, int]]
    heavy: dict[int, int]

    def estimate_frequency(self, key):
        if key in self.heavy:
            return self.heavy[key]

        for low, high, total in self.buckets:
            if low <= key <= high:
                saved = sum(
                    count
                    for k, count in self.heavy.items()
                    if low <= k <= high
                )
                heavy_keys = sum(
                    1 for k in self.heavy
                    if low <= k <= high
                )
                remaining_keys = high - low + 1 - heavy_keys

                if remaining_keys == 0:
                    return 0.0

                return (total - saved) / remaining_keys

        return 0.0


def build_topk(table: Counter, k: int):
    hist = build_histogram(
        table, BUCKET_WIDTH, DOMAIN
    )

    heavy = dict(
        sorted(
            table.items(),
            key=lambda item: (-item[1], item[0])
        )[:k]
    )

    return TopKHistogram(
        buckets=hist.buckets,
        heavy=heavy,
    )


def estimate_pair(h1, h2):
    return sum(
        h1.estimate_frequency(key)
        * h2.estimate_frequency(key)
        for key in range(DOMAIN)
    )


def compare_topk(tables, k):
    hists = {
        name: build_topk(table, k)
        for name, table in tables.items()
    }

    actual = {}
    estimated = {}

    for a, b in PAIRS:
        name = a + b

        actual[name] = actual_cardinality(
            tables[a], tables[b]
        )
        estimated[name] = estimate_pair(
            hists[a], hists[b]
        )

    selected = min(estimated, key=estimated.get)
    optimal = min(actual, key=actual.get)

    regret = actual[selected] / actual[optimal]

    return selected, optimal, regret, estimated


if __name__ == "__main__":
    (_, tables) = search(n=10000, seed=42)

    baseline, *_ = evaluate(tables)
    print(f"Baseline regret: {baseline:.3f}x")

    for k in [0, 1, 2, 3]:
        selected, optimal, regret, estimates = (
            compare_topk(tables, k)
        )

        print(f"\nTop-{k}")
        print(f"Selected: {selected}")
        print(f"Optimal:  {optimal}")
        print(f"Regret:   {regret:.3f}x")
        print(f"Estimates: {estimates}")
