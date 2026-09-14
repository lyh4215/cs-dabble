from pathlib import Path
import time

import faiss
import numpy as np


# ============================================================
# Settings
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

N = 100_000
DIM = 128

N_QUERIES = 50
TOP_K = 10

NLIST = 256
NPROBE = 128

# 직전 실험에서 가장 accuracy가 좋았던 설정
PQ_M = 64
PQ_NBITS = 8

CANDIDATE_SIZES = [
    10,
    50,
    100,
    500,
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

vectors = normalize(
    vectors
).astype(np.float32)

queries = normalize(
    queries
).astype(np.float32)


print(
    f"vectors : {vectors.shape}"
)

print(
    f"queries : {queries.shape}"
)

print(
    f"raw memory : "
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


# warmup
exact_index.search(
    queries[:1],
    TOP_K,
)


exact_results = []
exact_times = []


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


print(
    f"exact median : "
    f"{np.median(exact_times):.4f} ms"
)


# ============================================================
# Build IVFPQ
# ============================================================

print()
print("=== BUILD IVFPQ ===")

quantizer = faiss.IndexFlatIP(
    DIM
)

pq_index = faiss.IndexIVFPQ(
    quantizer,
    DIM,
    NLIST,
    PQ_M,
    PQ_NBITS,
    faiss.METRIC_INNER_PRODUCT,
)


start = time.perf_counter()

pq_index.train(
    vectors
)

train_time = (
    time.perf_counter()
    - start
)


start = time.perf_counter()

pq_index.add(
    vectors
)

add_time = (
    time.perf_counter()
    - start
)


pq_index.nprobe = NPROBE


print(
    f"PQ_M       : {PQ_M}"
)

print(
    f"nprobe     : {NPROBE}"
)

print(
    f"train time : {train_time:.3f} s"
)

print(
    f"add time   : {add_time:.3f} s"
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
# Exact reranking
# ============================================================

def exact_rerank(
    query,
    candidate_ids,
    k,
):
    """
    PQ가 뽑은 candidate들만
    원본 float32 vector로 다시 정확히 계산.
    """

    # -1 방어
    candidate_ids = candidate_ids[
        candidate_ids >= 0
    ]

    candidate_vectors = vectors[
        candidate_ids
    ]

    # cosine similarity
    # normalized 되어 있으므로 dot product
    scores = (
        candidate_vectors
        @ query
    )

    # candidate 내부 top-k
    if len(scores) <= k:
        order = np.argsort(
            scores
        )[::-1]
    else:
        local_idx = np.argpartition(
            scores,
            -k,
        )[-k:]

        order = local_idx[
            np.argsort(
                scores[local_idx]
            )[::-1]
        ]

    return candidate_ids[
        order
    ]


# ============================================================
# Direct PQ top-10 baseline
# ============================================================

print()
print("=== DIRECT PQ TOP-10 ===")

direct_results = []
direct_times = []


# warmup
pq_index.search(
    queries[:1],
    TOP_K,
)


for query in queries:

    q = query.reshape(
        1,
        -1,
    )

    start = time.perf_counter()

    _, labels = pq_index.search(
        q,
        TOP_K,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    direct_results.append(
        labels[0]
    )

    direct_times.append(
        elapsed * 1000
    )


direct_results = np.asarray(
    direct_results
)

direct_recall = recall_at_k(
    exact_results,
    direct_results,
)


print(
    f"recall@10 : "
    f"{direct_recall:.3f}"
)

print(
    f"median    : "
    f"{np.median(direct_times):.4f} ms"
)


# ============================================================
# Rerank sweep
# ============================================================

print()
print("=== PQ + EXACT RERANK ===")

print(
    f"{'candidate':>10} "
    f"{'candidate recall':>18} "
    f"{'final recall@10':>17} "
    f"{'pq(ms)':>10} "
    f"{'rerank(ms)':>12} "
    f"{'total(ms)':>11}"
)

print("-" * 88)


for candidate_k in CANDIDATE_SIZES:

    final_results = []

    candidate_recalls = []

    pq_times = []
    rerank_times = []
    total_times = []


    # warmup
    pq_index.search(
        queries[:1],
        candidate_k,
    )


    for query_index, query in enumerate(
        queries
    ):

        q = query.reshape(
            1,
            -1,
        )


        # ----------------------------------------------------
        # Stage 1:
        # PQ candidate generation
        # ----------------------------------------------------

        total_start = time.perf_counter()

        pq_start = time.perf_counter()

        _, labels = pq_index.search(
            q,
            candidate_k,
        )

        pq_elapsed = (
            time.perf_counter()
            - pq_start
        )

        candidate_ids = labels[0]


        # ----------------------------------------------------
        # Candidate recall
        #
        # 진짜 top-10 중 몇 개가
        # candidate set 안에 들어왔는지
        # ----------------------------------------------------

        true_top10 = exact_results[
            query_index
        ]

        candidate_set = set(
            candidate_ids
        )

        candidate_recall = (
            len(
                set(true_top10)
                & candidate_set
            )
            / TOP_K
        )

        candidate_recalls.append(
            candidate_recall
        )


        # ----------------------------------------------------
        # Stage 2:
        # exact float32 rerank
        # ----------------------------------------------------

        rerank_start = (
            time.perf_counter()
        )

        reranked = exact_rerank(
            query,
            candidate_ids,
            TOP_K,
        )

        rerank_elapsed = (
            time.perf_counter()
            - rerank_start
        )


        total_elapsed = (
            time.perf_counter()
            - total_start
        )


        final_results.append(
            reranked
        )

        pq_times.append(
            pq_elapsed * 1000
        )

        rerank_times.append(
            rerank_elapsed * 1000
        )

        total_times.append(
            total_elapsed * 1000
        )


    final_results = np.asarray(
        final_results
    )


    final_recall = recall_at_k(
        exact_results,
        final_results,
    )


    print(
        f"{candidate_k:>10} "
        f"{np.mean(candidate_recalls):>18.3f} "
        f"{final_recall:>17.3f} "
        f"{np.median(pq_times):>10.4f} "
        f"{np.median(rerank_times):>12.4f} "
        f"{np.median(total_times):>11.4f}"
    )