import heapq
import numpy as np


SEED = 42

N_VERTICES = 200_000
N_QUERY = 8_192
N_WORKERS = 40

MEAN_DEGREE = 32
MAX_DEGREE = 8192

SCHEDULING_OVERHEAD = 100

rng = np.random.default_rng(SEED)


def make_powerlaw_degrees():

    raw = (
        rng.pareto(1.25, N_VERTICES)
        + 1.0
    )

    raw *= (
        MEAN_DEGREE
        / raw.mean()
    )

    return np.clip(
        np.rint(raw),
        1,
        MAX_DEGREE
    ).astype(np.int64)


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
        np.cumsum(source_degrees)
    ])

    actual_work = np.empty(
        N_QUERY,
        dtype=np.float64
    )

    for i in range(N_QUERY):

        start = offsets[i]
        end = offsets[i + 1]

        actual_work[i] = (
            neighbor_degrees[start:end]
            .sum()
        )

    return actual_work


# ============================================================
# Fixed chunking
# ============================================================

def fixed_chunking(
    work,
    chunk_size
):

    tasks = []

    for w in work:

        while w > chunk_size:

            tasks.append(
                float(chunk_size)
            )

            w -= chunk_size

        if w > 0:
            tasks.append(float(w))

    return np.array(tasks)


# ============================================================
# Adaptive splitting
# ============================================================

def adaptive_split(
    work,
    threshold,
    target_chunk
):

    tasks = []

    split_vertices = 0


    for w in work:

        # 작은 vertex:
        # 그냥 task 하나
        if w <= threshold:

            tasks.append(
                float(w)
            )

            continue


        # 큰 vertex만 분할
        split_vertices += 1

        remaining = w

        while remaining > target_chunk:

            tasks.append(
                float(target_chunk)
            )

            remaining -= target_chunk

        if remaining > 0:

            tasks.append(
                float(remaining)
            )


    return (
        np.array(tasks),
        split_vertices
    )


# ============================================================
# Dynamic scheduler
# ============================================================

def schedule(tasks):

    heap = [
        (0.0, worker)
        for worker in range(
            N_WORKERS
        )
    ]

    heapq.heapify(heap)


    for work in tasks:

        finish, worker = (
            heapq.heappop(heap)
        )

        finish += (
            work
            + SCHEDULING_OVERHEAD
        )

        heapq.heappush(
            heap,
            (
                finish,
                worker
            )
        )


    loads = np.zeros(
        N_WORKERS
    )

    for finish, worker in heap:
        loads[worker] = finish

    return loads


def metrics(
    loads,
    n_tasks,
    base_work
):

    maximum = loads.max()
    mean = loads.mean()

    imbalance = (
        maximum / mean
    )

    util = (
        loads.sum()
        /
        (
            N_WORKERS
            * maximum
        )
    )

    overhead = (
        n_tasks
        * SCHEDULING_OVERHEAD
        /
        base_work
        * 100
    )

    return (
        imbalance,
        util,
        maximum,
        overhead
    )


# ============================================================
# Main
# ============================================================

degrees = (
    make_powerlaw_degrees()
)

work = (
    make_query_work(
        degrees
    )
)

base_work = work.sum()


print("=== WORKLOAD ===")

print(
    f"queries      : {len(work):,}"
)

print(
    f"mean work    : {work.mean():,.1f}"
)

print(
    f"median work  : {np.median(work):,.1f}"
)

print(
    f"p99 work     : "
    f"{np.percentile(work, 99):,.1f}"
)

print(
    f"max work     : {work.max():,.1f}"
)

print()


# ============================================================
# Baselines
# ============================================================

print("=== BASELINES ===")

print(
    f"{'policy':>18} "
    f"{'tasks':>10} "
    f"{'imbalance':>12} "
    f"{'util':>10} "
    f"{'makespan':>14} "
    f"{'overhead%':>12}"
)

print("-" * 82)


# vertex-level

tasks = work.copy()

loads = schedule(tasks)

(
    imbalance,
    util,
    makespan,
    overhead
) = metrics(
    loads,
    len(tasks),
    base_work
)

print(
    f"{'vertex':>18} "
    f"{len(tasks):>10,} "
    f"{imbalance:>11.2f}x "
    f"{util * 100:>9.1f}% "
    f"{makespan:>14,.0f} "
    f"{overhead:>11.2f}%"
)


# fixed 10k

tasks = fixed_chunking(
    work,
    10_000
)

loads = schedule(tasks)

(
    imbalance,
    util,
    makespan,
    overhead
) = metrics(
    loads,
    len(tasks),
    base_work
)

print(
    f"{'fixed 10k':>18} "
    f"{len(tasks):>10,} "
    f"{imbalance:>11.2f}x "
    f"{util * 100:>9.1f}% "
    f"{makespan:>14,.0f} "
    f"{overhead:>11.2f}%"
)


# ============================================================
# Adaptive
# ============================================================

print()
print(
    "=== ADAPTIVE SPLITTING "
    "(target chunk = 10,000) ==="
)

print(
    f"{'threshold':>12} "
    f"{'split verts':>12} "
    f"{'tasks':>10} "
    f"{'imbalance':>12} "
    f"{'util':>10} "
    f"{'makespan':>14} "
    f"{'overhead%':>12}"
)

print("-" * 92)


TARGET_CHUNK = 10_000

thresholds = [
    10_000,
    20_000,
    30_000,
    50_000,
    75_000,
    100_000,
    150_000,
    200_000,
]


for threshold in thresholds:

    (
        tasks,
        split_vertices
    ) = adaptive_split(
        work,
        threshold,
        TARGET_CHUNK
    )


    loads = schedule(
        tasks
    )


    (
        imbalance,
        util,
        makespan,
        overhead
    ) = metrics(
        loads,
        len(tasks),
        base_work
    )


    print(
        f"{threshold:>12,} "
        f"{split_vertices:>12,} "
        f"{len(tasks):>10,} "
        f"{imbalance:>11.2f}x "
        f"{util * 100:>9.1f}% "
        f"{makespan:>14,.0f} "
        f"{overhead:>11.2f}%"
    )