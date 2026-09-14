from pathlib import Path
import csv
import time

import hnswlib
import numpy as np


# ============================================================
# Path
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DIST_DIR = BASE_DIR / "dist"
DIST_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DIST_DIR / "hnsw_sweep.csv"


# ============================================================
# Experiment settings
# ============================================================

N = 100_000
DIM = 128
N_QUERIES = 50
TOP_K = 10

M_VALUES = [
    8,
    16,
    32,
    48,
]

EF_VALUES = [
    20,
    50,
    100,
    200,
    400,
]

# M의 영향만 보고 싶으므로 일단 고정
EF_CONSTRUCTION = 200

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
# Data
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
    f"vectors      : {vectors.shape}"
)

print(
    f"queries      : {queries.shape}"
)

print(
    f"vector memory: "
    f"{vectors.nbytes / 1024 / 1024:.1f} MiB"
)


# ============================================================
# Exact ground truth
# ============================================================

def exact_search(
    vectors,
    query,
    k,
):
    scores = vectors @ query

    candidate_idx = np.argpartition(
        scores,
        -k,
    )[-k:]

    candidate_scores = scores[
        candidate_idx
    ]

    order = np.argsort(
        candidate_scores
    )[::-1]

    return candidate_idx[
        order
    ]


print()
print("=== EXACT GROUND TRUTH ===")

exact_results = []
exact_times = []


# warmup
exact_search(
    vectors,
    queries[0],
    TOP_K,
)


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


exact_results = np.asarray(
    exact_results
)

exact_median = float(
    np.median(exact_times)
)

exact_mean = float(
    np.mean(exact_times)
)


print(
    f"median latency : "
    f"{exact_median:.3f} ms"
)

print(
    f"mean latency   : "
    f"{exact_mean:.3f} ms"
)


# ============================================================
# Recall@K
# ============================================================

def recall_at_k(
    exact_results,
    approx_results,
):
    recalls = []

    for exact, approx in zip(
        exact_results,
        approx_results,
    ):
        correct = len(
            set(exact)
            & set(approx)
        )

        recalls.append(
            correct / len(exact)
        )

    return float(
        np.mean(recalls)
    )


# ============================================================
# Results
# ============================================================

results = []


# ============================================================
# Sweep M
# ============================================================

for M in M_VALUES:

    print()
    print("=" * 80)

    print(
        f"BUILDING INDEX: M={M}"
    )

    print("=" * 80)

    index = hnswlib.Index(
        space="cosine",
        dim=DIM,
    )

    index.init_index(
        max_elements=N,
        M=M,
        ef_construction=EF_CONSTRUCTION,
        random_seed=SEED,
    )


    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    start = time.perf_counter()

    index.add_items(
        vectors,
        np.arange(N),
    )

    build_time = (
        time.perf_counter()
        - start
    )


    # --------------------------------------------------------
    # Save to measure index size
    # --------------------------------------------------------

    index_path = (
        DIST_DIR
        / f"hnsw_M{M}.bin"
    )

    index.save_index(
        str(index_path)
    )

    index_size_mb = (
        index_path.stat().st_size
        / 1024
        / 1024
    )


    print(
        f"build time : "
        f"{build_time:.3f} s"
    )

    print(
        f"index size : "
        f"{index_size_mb:.1f} MiB"
    )

    print(
        f"ef_build   : "
        f"{EF_CONSTRUCTION}"
    )


    # ========================================================
    # Sweep ef_search
    # ========================================================

    print()

    print(
        f"{'M':>4} "
        f"{'ef':>6} "
        f"{'recall@10':>12} "
        f"{'median(ms)':>12} "
        f"{'mean(ms)':>12} "
        f"{'speedup':>10}"
    )

    print("-" * 62)


    for ef in EF_VALUES:

        index.set_ef(
            ef
        )


        # ----------------------------------------------------
        # Warmup
        # ----------------------------------------------------

        index.knn_query(
            queries[0],
            k=TOP_K,
            num_threads=1,
        )


        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        approx_results = []
        query_times = []


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

            approx_results.append(
                labels[0]
            )

            query_times.append(
                elapsed * 1000
            )


        approx_results = np.asarray(
            approx_results
        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        recall = recall_at_k(
            exact_results,
            approx_results,
        )

        median_ms = float(
            np.median(
                query_times
            )
        )

        mean_ms = float(
            np.mean(
                query_times
            )
        )

        min_ms = float(
            np.min(
                query_times
            )
        )

        max_ms = float(
            np.max(
                query_times
            )
        )

        speedup = (
            exact_median
            / median_ms
        )


        print(
            f"{M:>4} "
            f"{ef:>6} "
            f"{recall:>12.3f} "
            f"{median_ms:>12.4f} "
            f"{mean_ms:>12.4f} "
            f"{speedup:>9.1f}x"
        )


        results.append(
            {
                "M": M,
                "ef_search": ef,

                "ef_construction":
                    EF_CONSTRUCTION,

                "build_time_s":
                    build_time,

                "index_size_mb":
                    index_size_mb,

                "recall_at_10":
                    recall,

                "median_ms":
                    median_ms,

                "mean_ms":
                    mean_ms,

                "min_ms":
                    min_ms,

                "max_ms":
                    max_ms,

                "speedup_vs_exact":
                    speedup,

                "exact_median_ms":
                    exact_median,
            }
        )


    # index 메모리 해제
    del index


# ============================================================
# Save CSV
# ============================================================

with open(
    CSV_PATH,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys(),
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 80)

print(
    "FINAL SUMMARY"
)

print("=" * 80)

print(
    f"{'M':>4} "
    f"{'ef':>6} "
    f"{'size(MB)':>10} "
    f"{'build(s)':>10} "
    f"{'recall':>10} "
    f"{'median(ms)':>12} "
    f"{'speedup':>10}"
)

print("-" * 78)


for r in results:

    print(
        f"{r['M']:>4} "
        f"{r['ef_search']:>6} "
        f"{r['index_size_mb']:>10.1f} "
        f"{r['build_time_s']:>10.2f} "
        f"{r['recall_at_10']:>10.3f} "
        f"{r['median_ms']:>12.4f} "
        f"{r['speedup_vs_exact']:>9.1f}x"
    )


print()
print(
    f"CSV saved to:"
)

print(
    CSV_PATH
)