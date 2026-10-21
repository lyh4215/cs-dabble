
from itertools import combinations
from math import prod

from histogram import (
    make_skewed_tables,
    build_histogram,
)


DOMAIN_SIZE = 100


def make_four_tables():
    tables = make_skewed_tables()

    # D has a heavy hitter at key=90.
    tables["D"] = {
        k: (500 if k == 90 else 10)
        for k in range(DOMAIN_SIZE)
    }

    return tables


def frequencies(tables, bucket_width=None):
    result = {}

    for name, table in tables.items():
        if bucket_width is None:
            result[name] = [
                float(table.get(k, 0))
                for k in range(DOMAIN_SIZE)
            ]
        else:
            hist = build_histogram(
                table,
                bucket_width,
                DOMAIN_SIZE,
            )
            result[name] = [
                hist.estimate_frequency(k)
                for k in range(DOMAIN_SIZE)
            ]

    return result


def cardinality(subset, freq):
    # All tables join on the same key.
    return sum(
        prod(freq[name][k] for name in subset)
        for k in range(DOMAIN_SIZE)
    )


def optimize_dp(freq):
    names = sorted(freq)
    dp = {}

    # Base case: one relation, no join cost.
    for name in names:
        dp[frozenset([name])] = (0.0, (name,))

    for size in range(2, len(names) + 1):
        for subset_tuple in combinations(names, size):
            subset = frozenset(subset_tuple)
            output_size = cardinality(subset, freq)

            candidates = []

            for last in sorted(subset):
                previous = subset - {last}
                previous_cost, previous_order = dp[previous]

                cost = previous_cost + output_size
                order = previous_order + (last,)

                candidates.append((cost, order))

            dp[subset] = min(candidates)

    return dp[frozenset(names)]


def actual_plan_cost(order, actual_freq):
    cost = 0.0
    prefix = []

    for name in order:
        prefix.append(name)

        if len(prefix) >= 2:
            cost += cardinality(prefix, actual_freq)

    return cost


def run():
    tables = make_four_tables()
    actual_freq = frequencies(tables)

    oracle_cost, oracle_order = optimize_dp(actual_freq)

    print("=== ORACLE OPTIMIZER ===")
    print("Order:", " -> ".join(oracle_order))
    print("Cost: ", f"{oracle_cost:,.0f}")

    print("\n=== HISTOGRAM OPTIMIZER ===")

    for width in [1, 2, 5, 10, 20, 50, 100]:
        estimated_freq = frequencies(tables, width)

        estimated_cost, selected_order = optimize_dp(
            estimated_freq
        )

        real_cost = actual_plan_cost(
            selected_order,
            actual_freq,
        )

        regret = real_cost / oracle_cost

        print(
            f"Width={width:3d} "
            f"Plan={'->'.join(selected_order):12s} "
            f"Est={estimated_cost:14,.0f} "
            f"Actual={real_cost:14,.0f} "
            f"Regret={regret:.3f}x"
        )


if __name__ == "__main__":
    run()
