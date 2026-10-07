import heapq
from collections import defaultdict


# ============================================================
# Tiny compiler ICFG
#
# 같은 함수 안 edge      : cost 1
# 함수 경계를 넘는 edge : cost 10
# ============================================================

edges = [
    ("main:A", "main:B", 1),

    # main -> optimization pass
    ("main:B", "pass:P0", 10),

    ("pass:P0", "pass:P1", 1),

    # pass -> concrete rewrite rule
    ("pass:P1", "rule:R0", 10),

    ("rule:R0", "rule:R1", 1),
    ("rule:R1", "rule:T", 1),

    # target과 관계없는 compiler code
    ("main:A", "noise:N0", 1),
    ("noise:N0", "noise:N1", 1),
]


TARGET = "rule:T"

# ============================================================
# Reverse graph
# ============================================================

reverse_graph = defaultdict(list)

for u, v, w in edges:
    reverse_graph[v].append((u, w))


# ============================================================
# Slice:
# target으로 도달 가능한 모든 node
# ============================================================

def compute_slice(target):

    stack = [target]
    visited = {target}

    while stack:

        v = stack.pop()

        for u, _ in reverse_graph[v]:

            if u not in visited:
                visited.add(u)
                stack.append(u)

    return visited

slice_nodes = compute_slice(TARGET)

print("SLICE:")
for n in sorted(slice_nodes):
    print(" ", n)


def distances_to_target(target, slice_nodes):

    dist = {
        target: 0
    }

    pq = [
        (0, target)
    ]

    while pq:

        d, v = heapq.heappop(pq)

        if d != dist[v]:
            continue

        for u, weight in reverse_graph[v]:

            if u not in slice_nodes:
                continue

            nd = d + weight

            if (
                u not in dist
                or nd < dist[u]
            ):
                dist[u] = nd

                heapq.heappush(
                    pq,
                    (nd, u)
                )

    return dist


node_dist = distances_to_target(
    TARGET,
    slice_nodes
)

print("\nNODE DISTANCE:")

for node, d in sorted(
    node_dist.items(),
    key=lambda x: -x[1]
):
    print(
        f"{node:10s} -> {d}"
    )


seeds = {

    "seed_far": {
        "main:A",
        "main:B",
    },

    "seed_mid": {
        "main:B",
        "pass:P0",
        "pass:P1",
    },

    "seed_near": {
        "pass:P1",
        "rule:R0",
        "rule:R1",
    },

    # irrelevant coverage가 많은 seed
    "seed_noise": {
        "main:A",
        "noise:N0",
        "noise:N1",
    },
}

def seed_distance(coverage):

    relevant = (
        coverage
        & slice_nodes
    )

    if not relevant:
        return float("inf")

    return sum(
        node_dist[n]
        for n in relevant
    ) / len(relevant)


print("\nSEED DISTANCE:")

results = []

for name, coverage in seeds.items():

    d = seed_distance(
        coverage
    )

    results.append(
        (d, name)
    )

    print(
        f"{name:12s} -> {d:.2f}"
    )


print("\nPRIORITY:")

for d, name in sorted(results):

    print(
        f"{name:12s} distance={d:.2f}"
    )

import math


def energy(distance):
    return max(
        1,
        math.ceil(
            10 / (1 + distance)
        )
    )


print("\nENERGY:")

for d, name in sorted(results):

    e = energy(d)

    print(
        f"{name:12s} "
        f"distance={d:5.2f} "
        f"energy={e}"
    )