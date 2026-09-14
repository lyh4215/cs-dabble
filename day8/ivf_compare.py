from pathlib import Path
import time

import faiss
import numpy as np


# ============================================================
# 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

N = 100_000
DIM = 128
N_QUERIES = 50
TOP_K = 10

NLIST = 256

NPROBE_VALUES = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
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
# 데이터
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
    f"memory  : "
    f"{vectors.nbytes / 1024 / 1024:.1f} MiB"
)


# ============================================================
# Exact search
#
# cosine similarity
# normalized vector에서는
# inner product와 동일
# ============================================================

print()
print("=== EXACT INDEX ===")

exact_index = faiss.IndexFlatIP(
    DIM
)

exact_index.add(
    vectors
)


# warmup
exact_index.search(
    queries[:1],
    TOP_K,
)


exact_times = []
exact_results = []


for query in queries:

    q = query.reshape(
        1,
        -1,
    )

    start = time.perf_counter()

    scores, labels = (
        exact_index.search(
            q,
            TOP_K,
        )
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

exact_median = float(
    np.median(
        exact_times
    )
)


print(
    f"median latency : "
    f"{exact_median:.4f} ms"
)

print(
    f"mean latency   : "
    f"{np.mean(exact_times):.4f} ms"
)


# ============================================================
# IVF index 생성
# ============================================================

print()
print("=== BUILD IVF ===")

#
# quantizer 자체는
# centroid 검색용 exact index
#
quantizer = faiss.IndexFlatIP(
    DIM
)


ivf_index = faiss.IndexIVFFlat(
    quantizer,
    DIM,
    NLIST,
    faiss.METRIC_INNER_PRODUCT,
)


# ------------------------------------------------------------
# train
#
# centroid를 k-means로 학습
# ------------------------------------------------------------

start = time.perf_counter()

ivf_index.train(
    vectors
)

train_time = (
    time.perf_counter()
    - start
)


# ------------------------------------------------------------
# vector를 각 inverted list에 배치
# ------------------------------------------------------------

start = time.perf_counter()

ivf_index.add(
    vectors
)

add_time = (
    time.perf_counter()
    - start
)


print(
    f"nlist      : {NLIST}"
)

print(
    f"train time : "
    f"{train_time:.3f} s"
)

print(
    f"add time   : "
    f"{add_time:.3f} s"
)

print(
    f"ntotal     : "
    f"{ivf_index.ntotal}"
)


# ============================================================
# Recall
# ============================================================

def recall_at_k(
    exact,
    approximate,
):
    recalls = []

    for true_ids, approx_ids in zip(
        exact,
        approximate,
    ):
        overlap = len(
            set(true_ids)
            & set(approx_ids)
        )

        recalls.append(
            overlap
            / len(true_ids)
        )

    return float(
        np.mean(recalls)
    )


# ============================================================
# nprobe sweep
# ============================================================

print()
print("=== IVF SEARCH ===")

print(
    f"{'nprobe':>8} "
    f"{'recall@10':>12} "
    f"{'median(ms)':>12} "
    f"{'mean(ms)':>12} "
    f"{'speedup':>10}"
)

print("-" * 60)


for nprobe in NPROBE_VALUES:

    ivf_index.nprobe = nprobe

    # warmup
    ivf_index.search(
        queries[:1],
        TOP_K,
    )

    approx_results = []
    times = []


    for query in queries:

        q = query.reshape(
            1,
            -1,
        )

        start = time.perf_counter()

        scores, labels = (
            ivf_index.search(
                q,
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

        approx_results.append(
            labels[0]
        )


    approx_results = np.asarray(
        approx_results
    )


    recall = recall_at_k(
        exact_results,
        approx_results,
    )

    median_ms = float(
        np.median(
            times
        )
    )

    mean_ms = float(
        np.mean(
            times
        )
    )

    speedup = (
        exact_median
        / median_ms
    )


    print(
        f"{nprobe:>8} "
        f"{recall:>12.3f} "
        f"{median_ms:>12.4f} "
        f"{mean_ms:>12.4f} "
        f"{speedup:>9.2f}x"
    )