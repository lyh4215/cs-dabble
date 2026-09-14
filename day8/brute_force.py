import time
import numpy as np


DIM = 128
TOP_K = 10

SIZES = [
    10_000,
    50_000,
    100_000,
    500_000,
]

rng = np.random.default_rng(42)


def normalize(x):
    norm = np.linalg.norm(
        x,
        axis=-1,
        keepdims=True,
    )

    return x / norm


def brute_force_search(
    vectors,
    query,
    k,
):
    # vectors와 query가 이미
    # L2 normalization 되어 있음
    #
    # cosine similarity
    # = dot product

    scores = vectors @ query

    # 전체 sort를 하지 않고
    # top-k 후보만 먼저 찾음
    candidate_idx = np.argpartition(
        scores,
        -k,
    )[-k:]

    # top-k 내부 정렬
    candidate_scores = scores[
        candidate_idx
    ]

    order = np.argsort(
        candidate_scores
    )[::-1]

    top_idx = candidate_idx[
        order
    ]

    return (
        top_idx,
        scores[top_idx],
    )


def benchmark(n):
    vectors = rng.standard_normal(
        (n, DIM),
        dtype=np.float32,
    )

    query = rng.standard_normal(
        DIM,
        dtype=np.float32,
    )

    vectors = normalize(vectors)
    query = normalize(query)

    # warmup
    brute_force_search(
        vectors,
        query,
        TOP_K,
    )

    times = []

    for _ in range(10):
        start = time.perf_counter()

        idx, scores = (
            brute_force_search(
                vectors,
                query,
                TOP_K,
            )
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        times.append(
            elapsed * 1000
        )

    return {
        "n": n,
        "memory_mb":
            vectors.nbytes
            / 1024
            / 1024,

        "median_ms":
            np.median(times),

        "mean_ms":
            np.mean(times),

        "min_ms":
            np.min(times),

        "max_ms":
            np.max(times),

        "top1_score":
            scores[0],
    }


print(
    f"DIM={DIM}, TOP_K={TOP_K}"
)

print()

print(
    f"{'N':>10} "
    f"{'memory(MB)':>12} "
    f"{'median(ms)':>12} "
    f"{'mean(ms)':>10} "
    f"{'min(ms)':>10} "
    f"{'max(ms)':>10}"
)

print("-" * 72)

for n in SIZES:
    r = benchmark(n)

    print(
        f"{r['n']:>10,d} "
        f"{r['memory_mb']:>12.1f} "
        f"{r['median_ms']:>12.3f} "
        f"{r['mean_ms']:>10.3f} "
        f"{r['min_ms']:>10.3f} "
        f"{r['max_ms']:>10.3f}"
    )