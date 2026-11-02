
import argparse
import random
import statistics
import time
import tracemalloc

from exp6_adblock_engine import (
    IndexedMatcher,
    SelectiveIndexedMatcher,
    naive_match,
)


def build_rules(count, kind):
    if kind == "normal":
        return [
            f"/track/{i:08x}/resource"
            for i in range(count)
        ]

    if kind == "similar":
        return [
            f"/track/{'a' * 24}{i:08x}/resource"
            for i in range(count)
        ]

    raise ValueError(kind)


def measure_build(cls, rules):
    # Peak Python allocations during construction.
    tracemalloc.start()
    start = time.perf_counter_ns()

    matcher = cls(rules)

    elapsed_ms = (
        time.perf_counter_ns() - start
    ) / 1e6

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return matcher, elapsed_ms, peak / (1024 * 1024)


def normal_requests(rules, count, seed):
    rng = random.Random(seed)
    result = []

    for _ in range(count):
        if rng.random() < 0.25:
            rule = rng.choice(rules)
            result.append(
                f"https://example.test/assets{rule}"
            )
        else:
            result.append(
                "https://example.test/assets/"
                f"{rng.getrandbits(96):024x}.js"
            )

    return result


def adversarial_requests(matcher, fraction, count, seed):
    rng = random.Random(seed)
    keys = list(matcher.index.keys())

    k = max(1, round(len(keys) * fraction))
    urls = []

    for _ in range(count):
        selected = rng.sample(keys, k)

        # Separator prevents construction of a full
        # '/track/...' rule in this workload.
        url = "https://example.test/?keys=" + "|".join(selected)
        urls.append(url)

    return urls


def percentile(values, p):
    values = sorted(values)
    idx = round((len(values) - 1) * p)
    return values[idx]


def benchmark(fn, urls):
    durations = []
    outputs = []

    for url in urls:
        start = time.perf_counter_ns()
        output = fn(url)
        durations.append(
            (time.perf_counter_ns() - start) / 1000
        )
        outputs.append(output)

    return {
        "mean_us": statistics.mean(durations),
        "p95_us": percentile(durations, 0.95),
        "p99_us": percentile(durations, 0.99),
        "outputs": outputs,
    }


def candidate_stats(matcher, urls):
    counts = [
        len(matcher.candidates(url))
        for url in urls
    ]

    return {
        "mean": statistics.mean(counts),
        "max": max(counts),
        "p99": percentile(counts, 0.99),
    }


def run_workload(name, rules, urls, indexed, selective):
    engines = {
        "naive": lambda url: naive_match(url, rules),
        "prefix": indexed.match,
        "selective": selective.match,
    }

    results = {
        key: benchmark(fn, urls)
        for key, fn in engines.items()
    }

    reference = results["naive"]["outputs"]

    assert results["prefix"]["outputs"] == reference
    assert results["selective"]["outputs"] == reference

    prefix_candidates = candidate_stats(indexed, urls)
    selective_candidates = candidate_stats(selective, urls)

    print(f"\n=== {name} ===")
    print("Requests:", len(urls))
    print("Mean URL length:",
          round(statistics.mean(map(len, urls)), 1))
    print("Blocked requests:", sum(reference))

    print(
        f"{'Engine':<12} {'Mean us':>12}"
        f" {'P95 us':>12} {'P99 us':>12}"
    )

    for key, result in results.items():
        print(
            f"{key:<12}"
            f" {result['mean_us']:12.3f}"
            f" {result['p95_us']:12.3f}"
            f" {result['p99_us']:12.3f}"
        )

    print("\nCandidate counts:")
    for label, stats in (
        ("prefix", prefix_candidates),
        ("selective", selective_candidates),
    ):
        print(
            f"  {label:<10}"
            f" mean={stats['mean']:.2f}"
            f" p99={stats['p99']}"
            f" max={stats['max']}"
        )

    ratio = (
        results["prefix"]["mean_us"]
        / results["selective"]["mean_us"]
    )
    print(f"Prefix / Selective: {ratio:.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", type=int, default=5000)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.rules < 1 or args.requests < 1:
        parser.error("rules and requests must be positive")

    for kind in ("normal", "similar"):
        print(f"\n{'=' * 60}")
        print(f"RULE SET: {kind.upper()}")
        print(f"{'=' * 60}")

        rules = build_rules(args.rules, kind)

        indexed, prefix_build, prefix_peak = measure_build(
            IndexedMatcher, rules
        )
        selective, selective_build, selective_peak = measure_build(
            SelectiveIndexedMatcher, rules
        )

        print("\n=== INDEX BUILD ===")
        print(
            f"Prefix:    {prefix_build:.3f} ms, "
            f"peak allocated {prefix_peak:.2f} MiB"
        )
        print(
            f"Selective: {selective_build:.3f} ms, "
            f"peak allocated {selective_peak:.2f} MiB"
        )

        normal = normal_requests(
            rules, args.requests, args.seed
        )

        run_workload(
            "NORMAL",
            rules,
            normal,
            indexed,
            selective,
        )

        # Deliberately construct URLs containing
        # increasingly many selective index keys.
        for fraction in (0.01, 0.10, 1.0):
            urls = adversarial_requests(
                selective,
                fraction=fraction,
                count=30,
                seed=args.seed + 1,
            )

            # These are intended to be nonmatching.
            assert not any(
                naive_match(url, rules)
                for url in urls
            )

            run_workload(
                f"ADVERSARIAL KEYS {fraction:.0%}",
                rules,
                urls,
                indexed,
                selective,
            )


if __name__ == "__main__":
    main()
