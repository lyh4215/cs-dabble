
"""InfoLab Exp 6: Sampling + GPU memory cap + Overflow Recovery.

Strategies:
  grow_then_split:
    Sample first, then expand capacity after overflow.
    Split input if the required output exceeds the GPU buffer budget.

  prepartition:
    Estimate output density and partition the input in advance.
    Fall back to safe chunks if a partition overflows.

  safe_fixed:
    Split input into chunks of at most 'budget' elements.
    Since each input emits at most one output, overflow is impossible.

Python 3.11+, NumPy, optional CuPy.
"""

from __future__ import annotations

import argparse
import csv
import math
import time

from pathlib import Path

import numpy as np

from sampling_adaptive_gpu import (
    CUDA,
    reference,
    threshold_for,
    choose_capacity,
    wilson_upper,
)


CHUNK_CUDA = r"""
extern "C" __global__ void append_range(
    int start,
    int length,
    int n,
    unsigned int seed,
    unsigned long long threshold,
    int distribution,
    int capacity,
    int* count,
    int* overflow,
    int* output
) {
    int local = blockIdx.x * blockDim.x + threadIdx.x;

    if (local >= length)
        return;

    int i = start + local;

    if (!accept(i, n, seed, threshold, distribution))
        return;

    int slot = atomicAdd(count, 1);

    if (slot < capacity) {
        output[slot] = i;
    } else if (slot == capacity) {
        atomicExch(overflow, 1);
    }
}
"""


class Runtime:
    def __init__(self, args):
        self.args = args
        self.cp = None

        if args.device == "cuda":
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() == 0:
                raise RuntimeError("No CUDA GPU available")

            self.cp = cp

            self.module = cp.RawModule(
                code=CUDA + "\n" + CHUNK_CUDA,
                options=("-std=c++11",),
                name_expressions=(
                    "sample_count",
                    "append_range",
                ),
            )

            self.kernel = self.module.get_function(
                "append_range"
            )

            self.sampler = self.module.get_function(
                "sample_count"
            )

    def sample(self, p, dist, seed, m, rng):
        a = self.args

        if self.cp is None:
            expected = reference(
                a.n, seed, p, dist
            )

            mask = np.zeros(a.n, dtype=np.bool_)
            mask[expected] = True

            # Stratified sampling
            edges = np.linspace(
                0, a.n, m + 1, dtype=np.int64
            )

            idx = edges[:-1] + np.array([
                rng.integers(0, int(b - e))
                for e, b in zip(
                    edges[:-1], edges[1:]
                )
            ])

            return int(mask[idx].sum())

        cp = self.cp
        block = a.block

        count = cp.zeros(1, dtype=cp.int32)

        offset = int(
            rng.integers(0, 2**30)
        )

        self.sampler(
            ((m + block - 1) // block,),
            (block,),
            (
                np.int32(a.n),
                np.uint32(seed),
                np.uint64(threshold_for(p)),
                np.int32(dist == "clustered"),
                np.int32(m),
                np.int32(offset),
                count,
            ),
        )

        return int(count.get()[0])

    def append(
        self,
        start,
        end,
        capacity,
        p,
        dist,
        seed,
        cpu_ref=None,
    ):
        a = self.args
        length = end - start

        if self.cp is None:
            lo = np.searchsorted(
                cpu_ref, start
            )

            hi = np.searchsorted(
                cpu_ref, end
            )

            count = int(hi - lo)

            output = (
                cpu_ref[lo:hi]
                if count <= capacity
                else None
            )

            return count, output

        cp = self.cp

        count = cp.zeros(
            1, dtype=cp.int32
        )

        overflow = cp.zeros(
            1, dtype=cp.int32
        )

        output = cp.empty(
            capacity, dtype=cp.int32
        )

        self.kernel(
            ((length + a.block - 1) // a.block,),
            (a.block,),
            (
                np.int32(start),
                np.int32(length),
                np.int32(a.n),
                np.uint32(seed),
                np.uint64(threshold_for(p)),
                np.int32(dist == "clustered"),
                np.int32(capacity),
                count,
                overflow,
                output,
            ),
        )

        total = int(count.get()[0])
        over = bool(overflow.get()[0])

        if over != (total > capacity):
            raise RuntimeError(
                "Overflow flag mismatch"
            )

        if over:
            return total, None

        return total, cp.asnumpy(
            output[:total]
        )


def trial(
    runtime,
    strategy,
    p,
    dist,
    seed,
    expected,
    trial_seed,
):
    a = runtime.args

    rng = np.random.default_rng(
        trial_seed
    )

    start_time = time.perf_counter()

    samples = 0
    sample_hits = 0

    if strategy != "safe_fixed":
        samples = a.samples

        sample_hits = runtime.sample(
            p, dist, seed, samples, rng
        )

        initial = choose_capacity(
            a.n,
            sample_hits,
            samples,
            a.z,
            a.slack,
        )

        expected_density = (
            wilson_upper(
                sample_hits, samples, a.z
            )
            * (1 + a.slack)
        )

    else:
        initial = a.budget
        expected_density = 1.0

    attempts = 0
    overflows = 0
    splits = 0
    max_alloc = 0

    copies = []
    stack = []

    # Strategy 1: Predictively partition input
    if strategy == "prepartition":
        width = max(
            1,
            min(
                a.n,
                int(
                    a.budget
                    / max(expected_density, 1e-9)
                ),
            ),
        )

        for lo in range(0, a.n, width):
            hi = min(a.n, lo + width)

            cap = max(
                1,
                min(
                    a.budget,
                    math.ceil(
                        (hi - lo)
                        * expected_density
                    ),
                ),
            )

            stack.append((lo, hi, cap))

        stack.reverse()

    # Strategy 2: Guaranteed-safe fixed chunks
    elif strategy == "safe_fixed":
        for lo in range(0, a.n, a.budget):
            hi = min(
                a.n, lo + a.budget
            )

            stack.append(
                (lo, hi, hi - lo)
            )

        stack.reverse()

    # Strategy 3: Grow first, split if necessary
    else:
        stack = [
            (
                0,
                a.n,
                min(a.budget, initial),
            )
        ]

    while stack:
        lo, hi, cap = stack.pop()

        cap = max(
            1,
            min(a.budget, cap),
        )

        max_alloc = max(
            max_alloc, cap
        )

        attempts += 1

        count, output = runtime.append(
            lo,
            hi,
            cap,
            p,
            dist,
            seed,
            expected,
        )

        # Successful chunk
        if output is not None:
            copies.append(
                np.asarray(
                    output, dtype=np.int32
                )
            )
            continue

        # Overflow
        overflows += 1

        if (
            strategy == "grow_then_split"
            and count <= a.budget
        ):
            # The exact output count is known
            # from the atomic counter.
            # Retry the same range with enough capacity.
            stack.append(
                (lo, hi, count)
            )

        else:
            if hi - lo <= 1:
                raise RuntimeError(
                    "Cannot split further"
                )

            splits += 1

            if strategy == "prepartition":
                # Conservative fallback:
                # at most one output per input.
                segments = []

                for x in range(
                    lo, hi, a.budget
                ):
                    y = min(
                        hi, x + a.budget
                    )

                    segments.append(
                        (x, y, y - x)
                    )

                stack.extend(
                    reversed(segments)
                )

            else:
                # Recursive bisection
                mid = (lo + hi) // 2

                for x, y in (
                    (mid, hi),
                    (lo, mid),
                ):
                    c = max(
                        1,
                        min(
                            a.budget,
                            math.ceil(
                                (y - x)
                                * expected_density
                            ),
                        ),
                    )

                    stack.append(
                        (x, y, c)
                    )

    # GPU timing includes sample, allocation,
    # kernel launches and device-to-host copies.
    # Sorting and validation are excluded.

    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1000

    got = (
        np.concatenate(copies)
        if copies
        else np.empty(
            0, dtype=np.int32
        )
    )

    passed = np.array_equal(
        np.sort(got),
        expected,
    )

    return {
        "ms": elapsed_ms,
        "first_capacity": min(
            a.budget, initial
        ),
        "max_buffer": max_alloc,
        "attempts": attempts,
        "overflows": overflows,
        "splits": splits,
        "samples": samples,
        "sample_hits": sample_hits,
        "correctness": (
            "pass" if passed else "FAIL"
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=600000,
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=150000,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--z",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--slack",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--block",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--selectivities",
        type=float,
        nargs="+",
        default=[0.02, 0.2, 0.8],
    )

    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=["uniform", "clustered"],
        default=["uniform", "clustered"],
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "overflow_recovery_results.csv"
        ),
    )

    args = parser.parse_args()

    if (
        args.n < 1
        or args.n >= 2**31
        or not 1 <= args.budget <= args.n
        or not 1 <= args.samples <= args.n
    ):
        parser.error(
            "Require 1 <= budget,samples <= n < 2^31"
        )

    if (
        args.trials < 1
        or not 0 <= args.slack <= 1
        or args.z < 0
    ):
        parser.error(
            "Invalid trials/slack/z"
        )

    if any(
        not 0 <= p <= 1
        for p in args.selectivities
    ):
        parser.error(
            "Selectivity must be between 0 and 1"
        )

    runtime = Runtime(args)

    rows = []

    for dist in args.distributions:
        for p in args.selectivities:
            expected = reference(
                args.n,
                args.seed,
                p,
                dist,
            )

            for method in (
                "grow_then_split",
                "prepartition",
                "safe_fixed",
            ):
                results = [
                    trial(
                        runtime,
                        method,
                        p,
                        dist,
                        args.seed,
                        expected,
                        args.seed
                        + 10007 * i
                        + int(p * 1000),
                    )
                    for i in range(args.trials)
                ]

                row = {
                    "distribution": dist,
                    "selectivity": p,
                    "method": method,
                    "actual_count": len(expected),
                    "budget": args.budget,
                    "samples": args.samples,
                    "z": args.z,
                    "slack": args.slack,
                    "median_ms": round(
                        float(
                            np.median([
                                r["ms"]
                                for r in results
                            ])
                        ),
                        4,
                    ),
                    "median_max_buffer": int(
                        np.median([
                            r["max_buffer"]
                            for r in results
                        ])
                    ),
                    "mean_attempts": round(
                        float(
                            np.mean([
                                r["attempts"]
                                for r in results
                            ])
                        ),
                        2,
                    ),
                    "mean_overflows": round(
                        float(
                            np.mean([
                                r["overflows"]
                                for r in results
                            ])
                        ),
                        2,
                    ),
                    "mean_splits": round(
                        float(
                            np.mean([
                                r["splits"]
                                for r in results
                            ])
                        ),
                        2,
                    ),
                    "overflow_trials": sum(
                        r["overflows"] > 0
                        for r in results
                    ),
                    "trials": args.trials,
                    "correctness": (
                        "pass"
                        if all(
                            r["correctness"] == "pass"
                            for r in results
                        )
                        else "FAIL"
                    ),
                }

                print(row)
                rows.append(row)

    with args.output.open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", args.output)


if __name__ == "__main__":
    main()
