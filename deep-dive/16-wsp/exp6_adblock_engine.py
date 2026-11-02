
import argparse
import random
import statistics
import string
import time
from collections import OrderedDict, defaultdict, Counter



class SelectiveIndexedMatcher:
    def __init__(self, rules):
        self.rules = list(rules)
        self.index = defaultdict(list)

        frequencies = Counter()

        # Number of rules containing each trigram.
        for rule in self.rules:
            if len(rule) < 3:
                raise ValueError("Rules must have length >= 3")

            for gram in set(
                rule[i:i+3] for i in range(len(rule)-2)
            ):
                frequencies[gram] += 1

        # Index each rule under its rarest trigram.
        for rule_id, rule in enumerate(self.rules):
            grams = [
                rule[i:i+3]
                for i in range(len(rule)-2)
            ]

            best = min(
                grams,
                key=lambda gram: (frequencies[gram], gram),
            )
            self.index[best].append(rule_id)

    def candidates(self, url):
        seen = set()

        for i in range(len(url) - 2):
            gram = url[i:i+3]

            for rule_id in self.index.get(gram, ()):
                seen.add(rule_id)

        return seen

    def match(self, url):
        for rule_id in self.candidates(url):
            if self.rules[rule_id] in url:
                return True

        return False



def naive_match(url, rules):
    return any(rule in url for rule in rules)


class IndexedMatcher:
    """
    Index each rule using its first 3 characters.

    If a rule appears in a URL, its first trigram
    must also appear in that URL.
    """

    def __init__(self, rules):
        self.rules = list(rules)
        self.index = defaultdict(list)

        for i, rule in enumerate(self.rules):
            if len(rule) < 3:
                raise ValueError("Rules must have length >= 3")
            self.index[rule[:3]].append(i)

    def candidates(self, url):
        seen = set()

        for i in range(len(url) - 2):
            trigram = url[i:i + 3]

            for rule_id in self.index.get(trigram, ()):
                seen.add(rule_id)

        return seen

    def match(self, url):
        for rule_id in self.candidates(url):
            if self.rules[rule_id] in url:
                return True
        return False


class CachedMatcher:
    def __init__(self, matcher, capacity=512):
        self.matcher = matcher
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def match(self, url):
        if url in self.cache:
            self.hits += 1
            self.cache.move_to_end(url)
            return self.cache[url]

        self.misses += 1
        result = self.matcher.match(url)

        if self.capacity > 0:
            self.cache[url] = result

            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

        return result


def random_token(rng, length=12):
    alphabet = string.ascii_lowercase + string.digits
    return "".join(rng.choices(alphabet, k=length))


def generate_workload(rule_count, request_count, seed):
    rng = random.Random(seed)

    # Every rule is a plain substring.
    rules = [
        "/track/" + random_token(rng)
        for _ in range(rule_count)
    ]

    urls = []

    for _ in range(request_count):
        # Approx. 25% of unique URLs contain a rule.
        if rng.random() < 0.25:
            rule = rng.choice(rules)
            url = f"https://example.test/assets{rule}/data.js"
        else:
            token = random_token(rng, 24)
            url = f"https://example.test/assets/{token}.js"

        urls.append(url)

    return rules, urls


def build_requests(unique_urls, request_count, reuse, seed):
    rng = random.Random(seed)

    requests = []
    previous = []

    for i in range(request_count):
        if previous and rng.random() < reuse:
            # Reuse a URL from a recent working set.
            url = rng.choice(previous[-64:])
        else:
            url = unique_urls[i % len(unique_urls)]

        requests.append(url)
        previous.append(url)

    return requests


def benchmark(fn, requests):
    times_ns = []
    decisions = []

    start_all = time.perf_counter_ns()

    for url in requests:
        start = time.perf_counter_ns()
        result = fn(url)
        times_ns.append(time.perf_counter_ns() - start)
        decisions.append(result)

    total_ns = time.perf_counter_ns() - start_all

    sorted_times = sorted(times_ns)

    def percentile(p):
        index = min(
            len(sorted_times) - 1,
            int((len(sorted_times) - 1) * p)
        )
        return sorted_times[index] / 1000

    return {
        "decisions": decisions,
        "total_ms": total_ns / 1e6,
        "mean_us": statistics.mean(times_ns) / 1000,
        "p95_us": percentile(0.95),
        "p99_us": percentile(0.99),
        "throughput": len(requests) / (total_ns / 1e9),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", type=int, default=5000)
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--reuse", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()


    rules, unique_urls = generate_workload(
        args.rules,
        args.requests,
        args.seed,
    )

    selective = SelectiveIndexedMatcher(rules)

    requests = build_requests(
        unique_urls,
        args.requests,
        args.reuse,
        args.seed + 1,
    )

    start = time.perf_counter()
    indexed = IndexedMatcher(rules)
    build_ms = (time.perf_counter() - start) * 1000

    cached = CachedMatcher(indexed, capacity=512)

    # Verify that all engines have equivalent semantics.
    # Test the full generated workload before benchmarking.
    for url in requests:
        expected = naive_match(url, rules)

        assert indexed.match(url) == expected
        assert selective.match(url) == expected

    # Include a fixed rule and a negative URL sanity check.
    assert naive_match(
        "https://example.test/assets" + rules[0],
        rules,
    )
    assert indexed.match(
        "https://example.test/assets" + rules[0]
    )
    assert not indexed.match(
        "https://example.test/no-matching-token"
    )

    methods = {
        "naive": lambda u: naive_match(u, rules),
        "indexed": indexed.match,
        "selective": selective.match,
        "cached": cached.match,
    }

    results = {}

    for name, fn in methods.items():
        results[name] = benchmark(fn, requests)

    reference = results["naive"]["decisions"]

    for name in ("indexed", "selective", "cached"):
        assert results[name]["decisions"] == reference

    print("\n=== CONFIG ===")
    print("Rules:", args.rules)
    print("Requests:", args.requests)
    print("Reuse probability:", args.reuse)
    print("Index build:", f"{build_ms:.3f} ms")

    print("\n=== RESULTS ===")
    print(
        f"{'Engine':<10} "
        f"{'Total ms':>12} "
        f"{'Mean us':>12} "
        f"{'P95 us':>12} "
        f"{'P99 us':>12} "
        f"{'Req/s':>12}"
    )

    for name, result in results.items():
        print(
            f"{name:<10} "
            f"{result['total_ms']:12.3f} "
            f"{result['mean_us']:12.3f} "
            f"{result['p95_us']:12.3f} "
            f"{result['p99_us']:12.3f} "
            f"{result['throughput']:12.1f}"
        )

    for name, matcher in [
        ("indexed", indexed),
        ("selective", selective),
    ]:
        candidate_counts = [
            len(matcher.candidates(url))
            for url in requests
        ]

        print(f"\n=== {name.upper()} CANDIDATES ===")
        print("Mean:", statistics.mean(candidate_counts))
        print("Max:", max(candidate_counts))
        print(
            "P99:",
            sorted(candidate_counts)[
                int(0.99 * (len(candidate_counts) - 1))
            ]
        )

    print("\n=== SPEEDUP ===")
    baseline = results["naive"]["total_ms"]

    for name in ("indexed", "cached"):
        speedup = baseline / results[name]["total_ms"]
        print(f"{name}: {speedup:.2f}x")

    print("\n=== CACHE ===")
    print("Hits:", cached.hits)
    print("Misses:", cached.misses)
    print(
        "Hit rate:",
        f"{cached.hits / max(1, cached.hits + cached.misses):.2%}"
    )

    print("\nAll decisions matched.")


if __name__ == "__main__":
    main()
