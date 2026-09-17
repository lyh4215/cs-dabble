import heapq
import numpy as np


SEED = 42

N_VERTICES = 200_000
N_QUERY = 8_192

N_WORKERS = 40

MEAN_DEGREE = 32
MAX_DEGREE = 8192

rng = np.random.default_rng(SEED)


# ============================================================
# Graph generation
# ============================================================

def make_uniform_degrees():
    deg = rng.poisson(
        MEAN_DEGREE,
        N_VERTICES
    )

    return np.maximum(
        deg,
        1
    ).astype(np.int64)


def make_powerlaw_degrees():

    # Pareto heavy-tail
    raw = (
        rng.pareto(
            1.25,
            N_VERTICES
        )
        + 1.0
    )

    # 평균 degree를 대략 맞춘다.
    raw *= (
        MEAN_DEGREE
        / raw.mean()
    )

    deg = np.clip(
        np.rint(raw),
        1,
        MAX_DEGREE
    )

    return deg.astype(
        np.int64
    )


# ============================================================
# Simulate 2-hop expansion
#
# source v:
#
# v
# ├─ u1 ─ degree(u1)개
# ├─ u2 ─ degree(u2)개
# └─ ...
#
# 실제 work:
#
# sum(degree(u))
#
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

    # 각 query source의 1-hop neighbor들을 생성
    neighbors = rng.integers(
        0,
        N_VERTICES,
        size=total_edges
    )

    neighbor_degrees = degrees[
        neighbors
    ]


    # source별로
    # sum(neighbor degree)
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
            neighbor_degrees[
                start:end
            ].sum()
        )


    # query planner가 실제 neighbor는 아직 모른다고 가정.
    #
    # E[2-hop work]
    # ≈ degree(source) × average_degree

    estimated_work = (
        source_degrees
        * degrees.mean()
    )


    return (
        source_degrees,
        estimated_work.astype(
            np.float64
        ),
        actual_work,
    )


# ============================================================
# 1. Naive static partition
#
# vertex 수만 동일하게 나눔
# ============================================================

def static_partition(actual_work):

    chunks = np.array_split(
        actual_work,
        N_WORKERS
    )

    return np.array([
        chunk.sum()
        for chunk in chunks
    ])


# ============================================================
# 2. Query-planner style partition
#
# estimated output이 큰 작업부터 배치
#
# 실제 work를 미리 아는 것은 아님.
# ============================================================

def estimated_balanced_partition(
    estimated_work,
    actual_work
):

    order = np.argsort(
        -estimated_work
    )

    # (estimated load, worker)
    heap = [
        (0.0, worker)
        for worker in range(
            N_WORKERS
        )
    ]

    heapq.heapify(heap)


    actual_loads = np.zeros(
        N_WORKERS
    )


    for task in order:

        estimated_load, worker = (
            heapq.heappop(heap)
        )

        actual_loads[worker] += (
            actual_work[task]
        )

        estimated_load += (
            estimated_work[task]
        )

        heapq.heappush(
            heap,
            (
                estimated_load,
                worker
            )
        )


    return actual_loads


# ============================================================
# 3. Dynamic work queue
#
# worker가 끝나는 즉시
# 다음 vertex를 가져간다고 가정.
#
# future task cost를 미리 알 필요가 없다.
# ============================================================

def dynamic_queue(actual_work):

    # (finish_time, worker)
    heap = [
        (0.0, worker)
        for worker in range(
            N_WORKERS
        )
    ]

    heapq.heapify(heap)


    for work in actual_work:

        finish_time, worker = (
            heapq.heappop(heap)
        )

        finish_time += work

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
        loads[worker] = finish_time


    return loads


# ============================================================
# Metrics
# ============================================================

def metrics(loads):

    mean = loads.mean()
    maximum = loads.max()

    imbalance = (
        maximum
        / mean
    )

    utilization = (
        loads.sum()
        /
        (
            N_WORKERS
            * maximum
        )
    )

    return {
        "mean": mean,
        "max": maximum,
        "imbalance": imbalance,
        "util": utilization,
    }


def print_scheduler_result(
    name,
    loads
):

    m = metrics(loads)

    print(
        f"{name:<20}"
        f"{m['imbalance']:>12.2f}x"
        f"{m['util'] * 100:>12.1f}%"
        f"{m['max']:>16,.0f}"
    )


# ============================================================
# Output size estimation
# ============================================================

def print_estimation_stats(
    estimated,
    actual
):

    ratio = (
        actual
        /
        np.maximum(
            estimated,
            1
        )
    )


    print()
    print(
        "Actual / estimated output ratio"
    )

    for p in [
        50,
        90,
        95,
        99,
        99.9,
    ]:

        print(
            f"p{p:<5}: "
            f"{np.percentile(ratio, p):.2f}x"
        )


    print(
        f"max   : "
        f"{ratio.max():.2f}x"
    )


    # planner가 estimate보다
    # 25% 여유 있게 buffer를 잡았다고 가정

    buffer = (
        estimated
        * 1.25
    )

    overflow_rate = (
        actual > buffer
    ).mean()


    print()
    print(
        "If buffer = "
        "1.25 × estimated output:"
    )

    print(
        f"overflow queries: "
        f"{overflow_rate * 100:.2f}%"
    )


# ============================================================
# Experiment
# ============================================================

def run_experiment(
    name,
    degrees
):

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    print(
        f"vertices        : "
        f"{len(degrees):,}"
    )

    print(
        f"mean degree     : "
        f"{degrees.mean():.2f}"
    )

    print(
        f"median degree   : "
        f"{np.median(degrees):.0f}"
    )

    print(
        f"p99 degree      : "
        f"{np.percentile(degrees, 99):.0f}"
    )

    print(
        f"max degree      : "
        f"{degrees.max():,}"
    )


    (
        source_degree,
        estimated_work,
        actual_work,
    ) = make_query_work(
        degrees
    )


    print()
    print(
        f"query vertices  : "
        f"{N_QUERY:,}"
    )

    print(
        f"largest query degree: "
        f"{source_degree.max():,}"
    )


    static_loads = (
        static_partition(
            actual_work
        )
    )

    estimated_loads = (
        estimated_balanced_partition(
            estimated_work,
            actual_work
        )
    )

    dynamic_loads = (
        dynamic_queue(
            actual_work
        )
    )


    print()
    print(
        f"{'scheduler':<20}"
        f"{'imbalance':>12}"
        f"{'utilization':>12}"
        f"{'max work':>16}"
    )

    print("-" * 60)


    print_scheduler_result(
        "static vertices",
        static_loads
    )

    print_scheduler_result(
        "estimated balance",
        estimated_loads
    )

    print_scheduler_result(
        "dynamic queue",
        dynamic_loads
    )


    print_estimation_stats(
        estimated_work,
        actual_work
    )


# ============================================================
# Main
# ============================================================

uniform = (
    make_uniform_degrees()
)

powerlaw = (
    make_powerlaw_degrees()
)


run_experiment(
    "UNIFORM GRAPH",
    uniform
)

run_experiment(
    "POWER-LAW GRAPH",
    powerlaw
)