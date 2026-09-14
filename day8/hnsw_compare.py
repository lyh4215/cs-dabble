from pathlib import Path
import time

import hnswlib
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DIST_DIR = BASE_DIR / "dist"
DIST_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = DIST_DIR / "hnsw.bin"


# ============================================================
# 설정
# ============================================================

N = 100_000
DIM = 128
N_QUERIES = 50
TOP_K = 10

M = 16
EF_CONSTRUCTION = 200

EF_VALUES = [
    10,
    20,
    50,
    100,
    200,
    400,
]

SEED = 42

rng = np.random.default_rng(SEED)


# ============================================================
# Normalize
# ============================================================

def normalize(x):
    norms = np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    return x / norms


# ============================================================
# 데이터 생성
# ============================================================

print("=== GENERATING DATA ===")

vectors = rng.standard_normal(
    (N, DIM),
    dtype=np.float32,
)

queries = rng.standard_normal(
    (N_QUERIES, DIM),
    dtype=np.float32,
)

vectors = normalize(vectors)
queries = normalize(queries)

print(
    f"vectors : {vectors.shape}"
)

print(
    f"queries : {queries.shape}"
)

print(
    f"memory  : "
    f"{vectors.nbytes / 1024 / 1024:.1f} MiB"
)


# ============================================================
# Exact brute-force ground truth
# ============================================================

def exact_search(
    vectors,
    query,
    k,
):
    scores = vectors @ query

    idx = np.argpartition(
        scores,
        -k,
    )[-k:]

    idx = idx[
        np.argsort(
            scores[idx]
        )[::-1]
    ]

    return idx


print()
print("=== EXACT SEARCH ===")

exact_results = []

exact_times = []

for query in queries:

    start = time.perf_counter()

    result = exact_search(
        vectors,
        query,
        TOP_K,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    exact_results.append(
        result
    )

    exact_times.append(
        elapsed * 1000
    )


exact_results = np.array(
    exact_results
)

print(
    f"median latency : "
    f"{np.median(exact_times):.3f} ms"
)

print(
    f"mean latency   : "
    f"{np.mean(exact_times):.3f} ms"
)


# ============================================================
# HNSW index build
# ============================================================

print()
print("=== BUILD HNSW ===")

index = hnswlib.Index(
    space="cosine",
    dim=DIM,
)

index.init_index(
    max_elements=N,
    ef_construction=EF_CONSTRUCTION,
    M=M,
    random_seed=SEED,
)


start = time.perf_counter()

index.add_items(
    vectors,
    np.arange(N),
)

build_time = (
    time.perf_counter()
    - start
)


print(
    f"build time : "
    f"{build_time:.3f} s"
)

print(
    f"M          : {M}"
)

print(
    f"ef_build   : "
    f"{EF_CONSTRUCTION}"
)


# ============================================================
# index disk size
# ============================================================

index.save_index(
    str(INDEX_PATH)
)

index_size_mb = (
    INDEX_PATH.stat().st_size
    / 1024
    / 1024
)

print(
    f"index size : "
    f"{index_size_mb:.1f} MiB"
)


# ============================================================
# Recall
# ============================================================

def recall_at_k(
    exact,
    approximate,
):
    total = 0

    for true_ids, approx_ids in zip(
        exact,
        approximate,
    ):
        overlap = len(
            set(true_ids)
            & set(approx_ids)
        )

        total += (
            overlap
            / len(true_ids)
        )

    return (
        total
        / len(exact)
    )


# ============================================================
# HNSW query benchmark
# ============================================================

print()
print("=== HNSW SEARCH ===")

print(
    f"{'ef':>6} "
    f"{'recall@10':>12} "
    f"{'median(ms)':>12} "
    f"{'mean(ms)':>12} "
    f"{'min(ms)':>10} "
    f"{'max(ms)':>10}"
)

print("-" * 70)


for ef in EF_VALUES:

    index.set_ef(ef)

    hnsw_results = []

    times = []

    # warmup
    index.knn_query(
        queries[0],
        k=TOP_K,
        num_threads=1,
    )

    for query in queries:

        start = time.perf_counter()

        labels, distances = (
            index.knn_query(
                query,
                k=TOP_K,
                num_threads=1,
            )
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        hnsw_results.append(
            labels[0]
        )

        times.append(
            elapsed * 1000
        )


    hnsw_results = np.array(
        hnsw_results
    )

    recall = recall_at_k(
        exact_results,
        hnsw_results,
    )


    print(
        f"{ef:>6} "
        f"{recall:>12.3f} "
        f"{np.median(times):>12.4f} "
        f"{np.mean(times):>12.4f} "
        f"{np.min(times):>10.4f} "
        f"{np.max(times):>10.4f}"
    )