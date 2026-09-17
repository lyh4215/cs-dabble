import heapq
import numpy as np


SEED = 42

N_VERTICES = 200_000
N_QUERY = 8_192
N_WORKERS = 40

MEAN_DEGREE = 32
MAX_DEGREE = 8192

# task 하나를 scheduler가 처리할 때 드는 가상 overhead
# work unit과 같은 단위라고 생각
SCHEDULING_OVERHEAD = 100

rng = np.random.default_rng(SEED)


# ============================================================
# Power-law graph
# ============================================================

def make_powerlaw_degrees():

    raw = (
        rng.pareto(
            1.25,
            N_VERTICES
        )
        + 1.0
    )

    raw *= (
        MEAN_DEGREE
        / raw.mean()
    )

    deg = np.clip(
        np.rint(raw),
        1,
        MAX_DEGREE
    )

    return deg.astype(np.int64)


# ============================================================
# 2-hop query work
# ============================================================

def make_query_work(degrees):

    query_vertices = rng.choice(
        N_VERTICES,
        N_QUERY,
        replace=False
    )

    source_degrees = degrees[
        query_vertices
    ]

    total_edges = int(
        source_degrees.sum()
    )

    neighbors = rng.integers(
        0,
        N_VERTICES,
        size=total_edges
    )

    neighbor_degrees = degrees[
        neighbors
    ]

    offsets = np.concatenate([
        [0],
        np.cumsum(
            source_degrees
        )
    ])

    actual_work = []

    for i in range(N_QUERY):

        start = offsets[i]
        end = offsets[i + 1]

        work = neighbor_degrees[
            start:end
        ].sum()

        actual_work.append(
            float(work)
        )

    return np.array(actual_work)


# ============================================================
# Split large task into chunks
#
# 예:
#
# work = 5000
# chunk_size = 1000
#
# ->
# [1000, 1000, 1000, 1000, 1000]
# ============================================================

def split_into_chunks(
    actual_work,
    chunk_size
):

    chunks = []

    for work in actual_work:

        remaining = work

        while remaining > chunk_size:

            chunks.append(
                float(chunk_size)
            )

            remaining -= (
                chunk_size
            )

        if remaining > 0:

            chunks.append(
                float(remaining)
            )

    return np.array(
        chunks
    )


# ============================================================
# Dynamic scheduler
# ============================================================

def dynamic_schedule(
    tasks,
    scheduling_overhead=0
):

    # (finish_time, worker)
    heap = [
        (0.0, worker)
        for worker in range(
            N_WORKERS
        )
    ]

    heapq.heapify(heap)

    for work in tasks:

        finish_time, worker = (
            heapq.heappop(heap)
        )

        finish_time += (
            work
            + scheduling_overhead
        )

        heapq.heappush(
            heap,
            (
                finish_time,
                worker
            )
        )

    loads = np.zeros(
        N_WORKERS
    )

    for finish_time, worker in heap:

        loads[worker] = (
            finish_time
        )

    return loads


# ============================================================
# Metrics
# ============================================================

def metrics(loads):

    maximum = loads.max()
    mean = loads.mean()

    imbalance = (
        maximum / mean
    )

    utilization = (
        loads.sum()
        /
        (
            N_WORKERS
            * maximum
        )
    )

    return (
        imbalance,
        utilization,
        maximum,
    )


# ============================================================
# Main
# ============================================================

degrees = (
    make_powerlaw_degrees()
)

actual_work = (
    make_query_work(
        degrees
    )
)


print("=== POWER-LAW GRAPH ===")

print(
    f"queries         : "
    f"{len(actual_work):,}"
)

print(
    f"mean work       : "
    f"{actual_work.mean():,.1f}"
)

print(
    f"median work     : "
    f"{np.median(actual_work):,.1f}"
)

print(
    f"p99 work        : "
    f"{np.percentile(actual_work, 99):,.1f}"
)

print(
    f"max work        : "
    f"{actual_work.max():,.1f}"
)

print()


chunk_sizes = [
    None,       # vertex-level
    100_000,
    50_000,
    20_000,
    10_000,
    5_000,
    2_000,
    1_000,
    500,
    250,
]


print(
    f"{'chunk':>12} "
    f"{'tasks':>10} "
    f"{'imbalance':>12} "
    f"{'util':>10} "
    f"{'makespan':>14} "
    f"{'overhead%':>12}"
)

print("-" * 76)


base_work = (
    actual_work.sum()
)


for chunk_size in chunk_sizes:

    if chunk_size is None:

        tasks = (
            actual_work.copy()
        )

        name = "vertex"

    else:

        tasks = split_into_chunks(
            actual_work,
            chunk_size
        )

        name = str(
            chunk_size
        )


    loads = dynamic_schedule(
        tasks,
        SCHEDULING_OVERHEAD
    )


    (
        imbalance,
        utilization,
        makespan,
    ) = metrics(loads)


    overhead_total = (
        len(tasks)
        * SCHEDULING_OVERHEAD
    )


    overhead_percent = (
        overhead_total
        /
        base_work
        * 100
    )


    print(
        f"{name:>12} "
        f"{len(tasks):>10,} "
        f"{imbalance:>11.2f}x "
        f"{utilization * 100:>9.1f}% "
        f"{makespan:>14,.0f} "
        f"{overhead_percent:>11.2f}%"
    )