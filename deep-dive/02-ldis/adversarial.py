
import random
from collections import Counter
from itertools import combinations

from histogram import (
    build_histogram,
    histogram_join_estimate,
)
from optimizer import actual_cardinality


DOMAIN = 20
BUCKET_WIDTH = 5
PAIRS = list(combinations("ABC", 2))


def make_random_tables(rng):
    tables = {}

    for name in "ABC":
        # All keys start with a small base frequency.
        freq = Counter({k: 1 for k in range(DOMAIN)})

        # Add a randomly located heavy hitter.
        heavy_key = rng.randrange(DOMAIN)
        heavy_size = rng.randint(100, 1000)

        freq[heavy_key] += heavy_size
        tables[name] = freq

    return tables


def evaluate(tables):
    hist = {
        name: build_histogram(
            table,
            BUCKET_WIDTH,
            DOMAIN,
        )
        for name, table in tables.items()
    }

    actual = {}
    estimated = {}

    for a, b in PAIRS:
        pair = a + b
        actual[pair] = actual_cardinality(
            tables[a], tables[b]
        )
        estimated[pair] = histogram_join_estimate(
            hist[a], hist[b], DOMAIN
        )

    selected = min(estimated, key=estimated.get)
    optimal = min(actual, key=actual.get)

    regret = actual[selected] / actual[optimal]

    return regret, selected, optimal, actual, estimated


def search(n=10000, seed=42):
    rng = random.Random(seed)
    best = None

    for _ in range(n):
        tables = make_random_tables(rng)
        result = evaluate(tables)

        if best is None or result[0] > best[0][0]:
            best = (result, tables)

    return best


if __name__ == "__main__":
    result, tables = search()

    regret, selected, optimal, actual, estimated = result

    print("=== WORST CASE FOUND ===")
    print(f"Plan Regret: {regret:.3f}x")
    print(f"Selected: {selected}")
    print(f"Optimal:  {optimal}")

    print("\nPlan Costs:")
    for pair in actual:
        print(
            f"{pair}: estimated={estimated[pair]:,.0f}, "
            f"actual={actual[pair]:,.0f}"
        )

    print("\nHeavy Hitters:")
    for name, freq in tables.items():
        key, count = max(
            freq.items(),
            key=lambda item: item[1]
        )
        print(f"{name}: key={key}, count={count}")
