from pathlib import Path
import time

import faiss
import numpy as np


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DIST_DIR = BASE_DIR / "dist"
DIST_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Settings
# ============================================================

N = 100_000
DIM = 128

N_QUERIES = 50
TOP_K = 10

NLIST = 256
NPROBE = 128

PQ_M_VALUES = [
    8,
    16,
    32,
    64,
]

PQ_NBITS = 8

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
# Generate data
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

vectors = normalize(
    vectors
).astype(
    np.float32
)

queries = normalize(
    queries
).astype(
    np.float32
)


print(
    f"vectors : {vectors.shape}"
)

print(
    f"queries : {queries.shape}"
)

print(
    f"raw vector memory : "
    f"{vectors.nbytes / 1024 / 1024:.2f} MiB"
)


# ============================================================
# Exact ground truth
# ============================================================

print()
print("=== EXACT GROUND TRUTH ===")

exact_index = faiss.IndexFlatIP(
    DIM
)

exact_index.add(
    vectors
)


exact_results = []
exact_times = []


# warmup
exact_index.search(
    queries[:1],
    TOP_K,
)


for query in queries:

    q = query.reshape(
        1,
        -1,
    )

    start = time.perf_counter()

    _, labels = exact_index.search(
        q,
        TOP_K,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    exact_results.append(
        labels[0]
    )

    exact_times.append(
        elapsed * 1000
    )


exact_results = np.asarray(
    exact_results
)

exact_median = float(
    np.median(
        exact_times
    )
)


print(
    f"exact median : "
    f"{exact_median:.4f} ms"
)


# ============================================================
# Recall@K
# ============================================================

def recall_at_k(
    exact_results,
    approx_results,
):
    recalls = []

    for exact_ids, approx_ids in zip(
        exact_results,
        approx_results,
    ):

        correct = len(
            set(exact_ids)
            & set(approx_ids)
        )

        recalls.append(
            correct
            / len(exact_ids)
        )

    return float(
        np.mean(recalls)
    )


# ============================================================
# Benchmark one PQ configuration
# ============================================================

def benchmark_pq(
    pq_m,
):
    print()
    print("=" * 72)

    print(
        f"PQ_M = {pq_m}"
    )

    print("=" * 72)


    # --------------------------------------------------------
    # DIM must be divisible by PQ_M
    # --------------------------------------------------------

    if DIM % pq_m != 0:
        raise ValueError(
            f"DIM={DIM} must be "
            f"divisible by PQ_M={pq_m}"
        )


    subvector_dim = (
        DIM // pq_m
    )


    print(
        f"subvector dim : "
        f"{subvector_dim}"
    )

    print(
        f"PQ code bytes : "
        f"{pq_m * PQ_NBITS // 8}"
    )


    # --------------------------------------------------------
    # Build IVFPQ
    # --------------------------------------------------------

    quantizer = faiss.IndexFlatIP(
        DIM
    )

    index = faiss.IndexIVFPQ(
        quantizer,
        DIM,
        NLIST,
        pq_m,
        PQ_NBITS,
        faiss.METRIC_INNER_PRODUCT,
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    start = time.perf_counter()

    index.train(
        vectors
    )

    train_time = (
        time.perf_counter()
        - start
    )


    # --------------------------------------------------------
    # Add
    # --------------------------------------------------------

    start = time.perf_counter()

    index.add(
        vectors
    )

    add_time = (
        time.perf_counter()
        - start
    )


    index.nprobe = NPROBE


    # --------------------------------------------------------
    # Save index
    # --------------------------------------------------------

    index_path = (
        DIST_DIR
        / f"ivfpq_M{pq_m}.index"
    )

    faiss.write_index(
        index,
        str(index_path),
    )

    index_size_mb = (
        index_path.stat().st_size
        / 1024
        / 1024
    )


    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    index.search(
        queries[:1],
        TOP_K,
    )


    # --------------------------------------------------------
    # Search benchmark
    # --------------------------------------------------------

    approx_results = []
    query_times = []


    for query in queries:

        q = query.reshape(
            1,
            -1,
        )

        start = time.perf_counter()

        _, labels = index.search(
            q,
            TOP_K,
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


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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


    # theoretical code size:
    #
    # PQ code:
    # M * nbits / 8
    #
    # + vector id:
    # about 8 bytes
    #

    theoretical_bytes_per_vector = (
        pq_m
        * PQ_NBITS
        / 8
        + 8
    )


    return {
        "pq_m":
            pq_m,

        "subvector_dim":
            subvector_dim,

        "code_bytes":
            pq_m
            * PQ_NBITS
            // 8,

        "theoretical_bytes_per_vector":
            theoretical_bytes_per_vector,

        "train_time_s":
            train_time,

        "add_time_s":
            add_time,

        "index_size_mb":
            index_size_mb,

        "recall":
            recall,

        "median_ms":
            median_ms,

        "mean_ms":
            mean_ms,
    }


# ============================================================
# Sweep
# ============================================================

results = []

for pq_m in PQ_M_VALUES:

    result = benchmark_pq(
        pq_m
    )

    results.append(
        result
    )


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 100)

print(
    "FINAL SUMMARY"
)

print("=" * 100)


print(
    f"{'PQ_M':>6} "
    f"{'subdim':>8} "
    f"{'code(B)':>9} "
    f"{'size(MB)':>10} "
    f"{'train(s)':>10} "
    f"{'recall@10':>12} "
    f"{'median(ms)':>12}"
)

print("-" * 100)


for r in results:

    print(
        f"{r['pq_m']:>6} "
        f"{r['subvector_dim']:>8} "
        f"{r['code_bytes']:>9} "
        f"{r['index_size_mb']:>10.2f} "
        f"{r['train_time_s']:>10.3f} "
        f"{r['recall']:>12.3f} "
        f"{r['median_ms']:>12.4f}"
    )