from pathlib import Path
import time

import faiss
import numpy as np


# ============================================================
# Settings
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DIST_DIR = BASE_DIR / "dist"
DIST_DIR.mkdir(parents=True, exist_ok=True)

N = 100_000
DIM = 128
N_QUERIES = 50
TOP_K = 10

NLIST = 256

NPROBE_VALUES = [
    8,
    16,
    32,
    64,
    128,
]

# PQ
PQ_M = 16
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

vectors = normalize(vectors).astype(
    np.float32
)

queries = normalize(queries).astype(
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
    f"{vectors.nbytes / 1024 / 1024:.1f} MiB"
)


# ============================================================
# Exact ground truth
# ============================================================

print()
print("=== EXACT ===")

exact = faiss.IndexFlatIP(
    DIM
)

exact.add(
    vectors
)

exact_results = []
exact_times = []

for query in queries:

    q = query.reshape(
        1,
        -1,
    )

    start = time.perf_counter()

    _, labels = exact.search(
        q,
        TOP_K,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    exact_times.append(
        elapsed * 1000
    )

    exact_results.append(
        labels[0]
    )


exact_results = np.asarray(
    exact_results
)

print(
    f"exact median : "
    f"{np.median(exact_times):.4f} ms"
)


# ============================================================
# Recall
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
        overlap = len(
            set(exact_ids)
            & set(approx_ids)
        )

        recalls.append(
            overlap
            / len(exact_ids)
        )

    return float(
        np.mean(recalls)
    )


# ============================================================
# Index builder
# ============================================================

def build_ivf_flat():
    quantizer = faiss.IndexFlatIP(
        DIM
    )

    index = faiss.IndexIVFFlat(
        quantizer,
        DIM,
        NLIST,
        faiss.METRIC_INNER_PRODUCT,
    )

    start = time.perf_counter()

    index.train(
        vectors
    )

    train_time = (
        time.perf_counter()
        - start
    )

    start = time.perf_counter()

    index.add(
        vectors
    )

    add_time = (
        time.perf_counter()
        - start
    )

    return (
        index,
        train_time,
        add_time,
    )


def build_ivf_pq():
    quantizer = faiss.IndexFlatIP(
        DIM
    )

    index = faiss.IndexIVFPQ(
        quantizer,
        DIM,
        NLIST,
        PQ_M,
        PQ_NBITS,
        faiss.METRIC_INNER_PRODUCT,
    )

    start = time.perf_counter()

    index.train(
        vectors
    )

    train_time = (
        time.perf_counter()
        - start
    )

    start = time.perf_counter()

    index.add(
        vectors
    )

    add_time = (
        time.perf_counter()
        - start
    )

    return (
        index,
        train_time,
        add_time,
    )


# ============================================================
# Build
# ============================================================

print()
print("=== BUILD IVFFLAT ===")

ivf_flat, flat_train, flat_add = (
    build_ivf_flat()
)

print(
    f"train: {flat_train:.3f} s"
)

print(
    f"add  : {flat_add:.3f} s"
)


print()
print("=== BUILD IVFPQ ===")

ivf_pq, pq_train, pq_add = (
    build_ivf_pq()
)

print(
    f"train: {pq_train:.3f} s"
)

print(
    f"add  : {pq_add:.3f} s"
)


# ============================================================
# Save and measure size
# ============================================================

flat_path = (
    DIST_DIR
    / "ivf_flat.index"
)

pq_path = (
    DIST_DIR
    / "ivf_pq.index"
)

faiss.write_index(
    ivf_flat,
    str(flat_path),
)

faiss.write_index(
    ivf_pq,
    str(pq_path),
)


flat_size = (
    flat_path.stat().st_size
    / 1024
    / 1024
)

pq_size = (
    pq_path.stat().st_size
    / 1024
    / 1024
)


print()
print("=== INDEX SIZE ===")

print(
    f"IVFFlat : "
    f"{flat_size:.2f} MiB"
)

print(
    f"IVFPQ   : "
    f"{pq_size:.2f} MiB"
)

print(
    f"compression ratio : "
    f"{flat_size / pq_size:.2f}x"
)


# ============================================================
# Benchmark function
# ============================================================

def benchmark(
    index,
    nprobe,
):
    index.nprobe = nprobe

    results = []
    times = []

    # warmup
    index.search(
        queries[:1],
        TOP_K,
    )

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

        times.append(
            elapsed * 1000
        )

        results.append(
            labels[0]
        )

    results = np.asarray(
        results
    )

    recall = recall_at_k(
        exact_results,
        results,
    )

    return {
        "recall":
            recall,

        "median":
            float(
                np.median(times)
            ),

        "mean":
            float(
                np.mean(times)
            ),
    }


# ============================================================
# Sweep
# ============================================================

print()
print("=== SEARCH COMPARISON ===")

print(
    f"{'type':>10} "
    f"{'nprobe':>8} "
    f"{'recall@10':>12} "
    f"{'median(ms)':>12} "
    f"{'mean(ms)':>12}"
)

print("-" * 60)


for nprobe in NPROBE_VALUES:

    flat_result = benchmark(
        ivf_flat,
        nprobe,
    )

    pq_result = benchmark(
        ivf_pq,
        nprobe,
    )


    print(
        f"{'IVFFlat':>10} "
        f"{nprobe:>8} "
        f"{flat_result['recall']:>12.3f} "
        f"{flat_result['median']:>12.4f} "
        f"{flat_result['mean']:>12.4f}"
    )

    print(
        f"{'IVFPQ':>10} "
        f"{nprobe:>8} "
        f"{pq_result['recall']:>12.3f} "
        f"{pq_result['median']:>12.4f} "
        f"{pq_result['mean']:>12.4f}"
    )

    print()