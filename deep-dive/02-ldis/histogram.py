
from collections import Counter
from dataclasses import dataclass
from optimizer import (
    make_tables,
    actual_cardinality,
    estimate_join,
)


@dataclass
class Histogram:
    buckets: list[tuple[int, int, int]]
    # (minimum key, maximum key, total frequency)

    def estimate_frequency(self, key):
        for low, high, count in self.buckets:
            if low <= key <= high:
                return count / (high - low + 1)
        return 0.0


def build_histogram(
    table: Counter,
    bucket_width: int,
    domain_size: int,
):
    buckets = []

    for low in range(0, domain_size, bucket_width):
        high = min(low + bucket_width - 1, domain_size - 1)

        count = sum(
            table.get(k, 0)
            for k in range(low, high + 1)
        )

        buckets.append((low, high, count))

    return Histogram(buckets)


def histogram_join_estimate(
    hist_r: Histogram,
    hist_s: Histogram,
    domain_size: int,
):
    return sum(
        hist_r.estimate_frequency(k)
        * hist_s.estimate_frequency(k)
        for k in range(domain_size)
    )


def compare(tables, bucket_width, domain_size):
    a, b, c = (tables[name] for name in "ABC")

    hist = {
        name: build_histogram(
            table,
            bucket_width,
            domain_size,
        )
        for name, table in tables.items()
    }

    for left, right in [("A", "B"), ("B", "C")]:
        r, s = tables[left], tables[right]

        actual = actual_cardinality(r, s)
        uniform = estimate_join(r, s)
        histogram = histogram_join_estimate(
            hist[left],
            hist[right],
            domain_size,
        )

        print(f"\n{left} JOIN {right}")
        print(f"Actual:    {actual:,.0f}")
        print(f"Uniform:   {uniform:,.0f}")
        print(f"Histogram: {histogram:,.0f}")
        print(f"Q-error:   {q_error(histogram, actual):.3f}")


def make_skewed_tables():
    a = Counter()
    b = Counter()
    c = Counter()

    for k in range(100):
        # Base frequencies
        a[k] = 10
        b[k] = 10
        c[k] = 10

    # Correlated heavy hitters.
    a[10] = 500
    b[10] = 500

    # C has its heavy hitter elsewhere.
    c[90] = 500

    return {"A": a, "B": b, "C": c}

def q_error(estimated, actual):
    if actual == 0 and estimated == 0:
        return 1.0
    if actual == 0 or estimated == 0:
        return float("inf")

    return max(
        estimated / actual,
        actual / estimated,
    )


if __name__ == "__main__":
    print("\n=== SKEWED DISTRIBUTION ===")

    tables = make_skewed_tables()

    for width in [1, 2, 5, 10, 20, 50, 100]:
        print(f"\nBucket width = {width}")
        compare(tables, width, domain_size=100)