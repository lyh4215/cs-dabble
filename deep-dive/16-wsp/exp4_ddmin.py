
import argparse
import math
from functools import lru_cache
from typing import Callable


ORIGINAL = [
    "default-src 'none'",
    "script-src 'unsafe-inline'",
    "script-src-elem 'self'",
    "script-src-attr 'none'",
    "object-src 'none'",
    "base-uri 'self'",
]


def parse_csp(directives: list[str]) -> dict:
    result = {}

    for directive in directives:
        parts = directive.split()
        if not parts:
            continue

        name = parts[0]
        sources = parts[1:]
        result[name] = sources

    return result


def allowed_by_source_list(sources):
    # Simplified evaluator for inline event handlers.
    if sources is None:
        return True  # No applicable CSP restriction.

    return "'unsafe-inline'" in sources


def correct_evaluator(directives):
    policy = parse_csp(directives)

    # CSP fallback for event-handler attributes.
    sources = policy.get(
        "script-src-attr",
        policy.get(
            "script-src",
            policy.get("default-src"),
        ),
    )
    return allowed_by_source_list(sources)


def buggy_evaluator(directives):
    policy = parse_csp(directives)

    # Deliberately injected bug:
    # if script-src-elem is present, incorrectly
    # ignore script-src-attr for event handlers.
    if "script-src-elem" in policy:
        sources = policy.get(
            "script-src",
            policy.get("default-src"),
        )
    else:
        sources = policy.get(
            "script-src-attr",
            policy.get(
                "script-src",
                policy.get("default-src"),
            ),
        )

    return allowed_by_source_list(sources)


class Oracle:
    def __init__(self, verbose=False):
        self.calls = 0
        self.verbose = verbose

    @lru_cache(maxsize=None)
    def evaluate(self, candidate):
        self.calls += 1

        correct = correct_evaluator(candidate)
        buggy = buggy_evaluator(candidate)

        mismatch = correct != buggy

        if self.verbose:
            print(
                f"  TEST #{self.calls:03d} "
                f"n={len(candidate)} "
                f"correct={correct} "
                f"buggy={buggy} "
                f"DIFF={mismatch}"
            )

        return mismatch


def ddmin(items, test: Callable[[tuple[str, ...]], bool]):
    """
    Find a 1-minimal failure-inducing subset.

    test(tuple(items)) must initially be True.
    """
    current = list(items)

    if not test(tuple(current)):
        raise ValueError(
            "Initial input does not reproduce a difference."
        )

    granularity = 2
    round_no = 0

    while len(current) >= 2:
        round_no += 1
        chunk_size = math.ceil(
            len(current) / granularity
        )

        chunks = [
            current[i:i + chunk_size]
            for i in range(0, len(current), chunk_size)
        ]

        print(
            f"\n[Round {round_no}] "
            f"size={len(current)}, "
            f"granularity={granularity}"
        )

        reduced = False

        # Try each chunk by itself.
        for chunk in chunks:
            if test(tuple(chunk)):
                print(
                    "  KEEP SUBSET:",
                    len(current),
                    "->",
                    len(chunk),
                )
                current = chunk
                granularity = max(
                    granularity - 1, 2
                )
                reduced = True
                break

        if reduced:
            continue

        # Try removing each chunk.
        for index, chunk in enumerate(chunks):
            complement = [
                item
                for j, part in enumerate(chunks)
                if j != index
                for item in part
            ]

            if test(tuple(complement)):
                print(
                    "  REMOVE CHUNK:",
                    chunk,
                    "|",
                    len(current),
                    "->",
                    len(complement),
                )
                current = complement
                granularity = max(
                    granularity - 1, 2
                )
                reduced = True
                break

        if reduced:
            continue

        if granularity >= len(current):
            break

        granularity = min(
            len(current),
            granularity * 2,
        )

    return current


def verify_one_minimal(candidate, test):
    if not test(tuple(candidate)):
        return False

    for i in range(len(candidate)):
        smaller = candidate[:i] + candidate[i + 1:]
        if test(tuple(smaller)):
            print(
                "NOT 1-minimal, removable:",
                candidate[i],
            )
            return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose", action="store_true"
    )
    args = parser.parse_args()

    oracle = Oracle(verbose=args.verbose)

    print("=== INITIAL CSP ===")
    for item in ORIGINAL:
        print(" ", item)

    print("\nCorrect:", correct_evaluator(ORIGINAL))
    print("Buggy:  ", buggy_evaluator(ORIGINAL))
    print("Mismatch:", oracle.evaluate(tuple(ORIGINAL)))

    minimized = ddmin(
        ORIGINAL,
        oracle.evaluate,
    )

    print("\n=== MINIMIZED CSP ===")
    print("; ".join(minimized))

    print("\n=== VERIFICATION ===")
    print("Correct:", correct_evaluator(minimized))
    print("Buggy:  ", buggy_evaluator(minimized))
    print(
        "1-minimal:",
        verify_one_minimal(
            minimized,
            oracle.evaluate,
        ),
    )

    print("\nOriginal directives:", len(ORIGINAL))
    print("Final directives:", len(minimized))
    print("Unique oracle calls:", oracle.calls)

    assert len(minimized) < len(ORIGINAL)
    assert verify_one_minimal(minimized, oracle.evaluate)


if __name__ == "__main__":
    main()
