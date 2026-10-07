import math
import random
import heapq

from dataclasses import dataclass, replace
from collections import defaultdict

from z3 import (
    BitVec,
    BitVecVal,
    Solver,
    sat,
)


random.seed(0)


# ============================================================
# 1. INPUT PROGRAM
#
# fuzzer가 mutate하는 "입력 프로그램의 특징"을
# 간단하게 구조화한 것.
#
# target optimization이 발동하려면:
#
#   op == DIV
#   signed == True
#   divisor > 0
#   divisor is power of two
#   divisor == 2
#   consumer == ADD
#
# 여야 한다고 가정.
# ============================================================

@dataclass(frozen=True)
class Program:
    op: str
    signed: bool
    divisor: int
    consumer: str

    noise1: int = 10
    noise2: int = 20
    noise3: int = 30
    noise4: int = 40
    noise5: int = 50
    noise6: int = 60
    noise7: int = 70
    noise8: int = 80


# ============================================================
# 2. TOY COMPILER ICFG
#
# 같은 함수 내부 edge       cost = 1
# 함수 경계를 넘는 edge    cost = 10
# ============================================================

edges = [
    ("main:A", "main:B", 1),

    ("main:B", "pass:P0", 10),

    ("pass:P0", "pass:P1", 1),

    ("pass:P1", "rule:R0", 10),

    ("rule:R0", "rule:R1", 1),

    ("rule:R1", "rule:T", 1),

    # target과 무관한 compiler code
    ("main:A", "noise:N0", 1),
    ("noise:N0", "noise:N1", 1),
]


TARGET = "rule:T"


# ============================================================
# reverse graph
# ============================================================

reverse_graph = defaultdict(list)

for u, v, w in edges:
    reverse_graph[v].append(
        (u, w)
    )


# ============================================================
# 3. TARGET SLICE
#
# target으로 갈 수 있는 node만 남긴다.
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


SLICE = compute_slice(TARGET)


# ============================================================
# 4. NODE -> TARGET DISTANCE
#
# reverse Dijkstra
# ============================================================

def distances_to_target(
    target,
    slice_nodes,
):

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


NODE_DIST = distances_to_target(
    TARGET,
    SLICE,
)


# ============================================================
# 5. TOY COMPILER EXECUTION
#
# 입력 프로그램을 compiler가 처리한다고 생각.
#
# 조건을 하나씩 만족할수록 target optimization에
# 가까운 compiler node까지 실행한다.
# ============================================================

def is_power_of_two(x):

    return (
        x > 0
        and (x & (x - 1)) == 0
    )


def compile_program(p):

    coverage = {
        "main:A"
    }

    # noise compiler code
    #
    # selective coverage가 이걸 무시하는지도 보여주기 위해
    # 일부 seed에서 실행되도록 한다.
    if p.noise1 % 2 == 0:

        coverage.add(
            "noise:N0"
        )

        coverage.add(
            "noise:N1"
        )

    # --------------------------------------------------------
    # condition 1
    # --------------------------------------------------------

    if p.op != "DIV":
        return coverage, False

    coverage.add(
        "main:B"
    )

    # --------------------------------------------------------
    # condition 2
    # --------------------------------------------------------

    if not p.signed:
        return coverage, False

    coverage.add(
        "pass:P0"
    )

    # --------------------------------------------------------
    # condition 3
    # --------------------------------------------------------

    if p.divisor <= 0:
        return coverage, False

    coverage.add(
        "pass:P1"
    )

    # --------------------------------------------------------
    # condition 4
    # --------------------------------------------------------

    if not is_power_of_two(
        p.divisor
    ):
        return coverage, False

    coverage.add(
        "rule:R0"
    )

    # --------------------------------------------------------
    # condition 5
    # --------------------------------------------------------

    if p.divisor != 2:
        return coverage, False

    coverage.add(
        "rule:R1"
    )

    # --------------------------------------------------------
    # condition 6
    # --------------------------------------------------------

    if p.consumer != "ADD":
        return coverage, False

    coverage.add(
        "rule:T"
    )

    return coverage, True


# ============================================================
# 6. SEED DISTANCE
#
# seed가 cover한 slice node들의 평균 target distance
#
# noise:N0 / noise:N1은 자동으로 제외된다.
# ============================================================

def seed_distance(coverage):

    relevant = (
        coverage
        & SLICE
    )

    if not relevant:
        return float("inf")

    total = sum(
        NODE_DIST[n]
        for n in relevant
    )

    return (
        total
        / len(relevant)
    )


# ============================================================
# 7. ENERGY
#
# 가까운 seed일수록 더 많은 mutation 기회
# ============================================================

def energy(distance):

    return max(
        1,
        math.ceil(
            10
            / (1 + distance)
        ),
    )


# ============================================================
# 8. DEF-USE GRAPH OF INPUT PROGRAM
#
# c1:
#   d = divisor
#
# c2:
#   q = x / d
#
# c3:
#   out = q + ...
#
#
# c1 --d--> c2 --q--> c3
#
# delta = c2
#
# 즉 target optimization과 관계있는 data-flow 주변만
# targeted mutation 대상으로 사용한다.
# ============================================================

DEF_USE = {

    "c1": [
        ("d", "c2"),
    ],

    "c2": [
        ("q", "c3"),
    ],
}


DELTA = "c2"


def targeted_nodes(delta):

    result = {
        delta
    }

    # definitions feeding delta
    for src, edges in DEF_USE.items():

        for var, dst in edges:

            if dst == delta:
                result.add(src)

    # users of delta
    for var, dst in DEF_USE.get(
        delta,
        []
    ):
        result.add(dst)

    return sorted(result)


TARGETED_NODES = targeted_nodes(
    DELTA
)


# ============================================================
# 9. TARGETED MUTATION
#
# c1 -> divisor
#
# c2 -> division 자체의 속성
#       op / signed
#
# c3 -> consumer
#
# noise command는 targeted mutation 후보에 없다.
# ============================================================

def mutate_node(p, node):

    # --------------------------------------------------------
    # c1: divisor definition
    # --------------------------------------------------------

    if node == "c1":

        mode = random.choice([
            "minus1",
            "plus1",
            "half",
            "double",
        ])

        d = p.divisor

        if mode == "minus1":
            d -= 1

        elif mode == "plus1":
            d += 1

        elif mode == "half":

            if d > 1:
                d //= 2

        elif mode == "double":
            d *= 2

        return replace(
            p,
            divisor=d,
        )

    # --------------------------------------------------------
    # c2: division expression
    # --------------------------------------------------------

    if node == "c2":

        field = random.choice([
            "op",
            "signed",
        ])

        if field == "op":

            new_op = random.choice([
                "ADD",
                "SUB",
                "MUL",
                "DIV",
            ])

            return replace(
                p,
                op=new_op,
            )

        else:

            return replace(
                p,
                signed=not p.signed,
            )

    # --------------------------------------------------------
    # c3: q의 user
    # --------------------------------------------------------

    if node == "c3":

        new_consumer = random.choice([
            "ADD",
            "SUB",
            "MUL",
        ])

        return replace(
            p,
            consumer=new_consumer,
        )

    raise ValueError(
        f"unknown node: {node}"
    )


def targeted_mutation(p):

    node = random.choice(
        TARGETED_NODES
    )

    mutant = mutate_node(
        p,
        node,
    )

    return mutant, node


# ============================================================
# 10. TRANSLATION VALIDATION
#
# Target optimization:
#
#     signed x / 2
#
# 를 compiler가 잘못
#
#     x >> 1
#
# 로 바꿨다고 가정.
#
# target이 실제로 hit된 뒤 이 optimization을 검증한다.
# ============================================================

def to_signed(v):

    if v >= (1 << 31):

        return (
            v - (1 << 32)
        )

    return v


def translation_validate():

    x = BitVec(
        "runtime_x",
        32,
    )

    two = BitVecVal(
        2,
        32,
    )

    source = x / two

    optimized = x >> 1

    solver = Solver()

    # 두 프로그램의 결과가 다른 runtime input이 있는가?
    solver.add(
        source != optimized
    )

    result = solver.check()

    if result == sat:

        model = solver.model()

        xv = model.eval(
            x,
            model_completion=True,
        ).as_long()

        src = model.eval(
            source,
            model_completion=True,
        ).as_long()

        tgt = model.eval(
            optimized,
            model_completion=True,
        ).as_long()

        return {
            "valid": False,

            "x": to_signed(xv),

            "source": to_signed(src),

            "optimized": to_signed(tgt),
        }

    return {
        "valid": True
    }


# ============================================================
# 11. SEED OBJECT
# ============================================================

@dataclass
class Seed:
    program: Program
    coverage: set
    distance: float


def make_seed(program):

    coverage, hit = compile_program(
        program
    )

    distance = seed_distance(
        coverage
    )

    return (
        Seed(
            program=program,
            coverage=coverage,
            distance=distance,
        ),
        hit,
    )


# ============================================================
# 12. INITIAL CORPUS
#
# 일부러 far / mid / near seed를 넣는다.
#
# directed fuzzing이 near seed를 우선 선택하는지 확인.
# ============================================================

INITIAL_PROGRAMS = [

    # far
    Program(
        op="MUL",
        signed=False,
        divisor=-3,
        consumer="SUB",
    ),

    # mid
    Program(
        op="DIV",
        signed=False,
        divisor=3,
        consumer="ADD",
    ),

    # near
    Program(
        op="DIV",
        signed=True,
        divisor=3,
        consumer="ADD",
    ),
]


# ============================================================
# 13. MINI OPTIMUZZ LOOP
# ============================================================

def fuzz(max_rounds=100):

    corpus = []

    seen = set()

    print(
        "================================"
    )

    print(
        "INITIAL CORPUS"
    )

    print(
        "================================"
    )

    for program in INITIAL_PROGRAMS:

        seed, hit = make_seed(
            program
        )

        corpus.append(seed)
        seen.add(program)

        print(
            f"{program}"
        )

        print(
            f"  coverage={sorted(seed.coverage)}"
        )

        print(
            f"  distance={seed.distance:.2f}"
        )

        if hit:

            print(
                "INITIAL TARGET HIT"
            )

            return


    # --------------------------------------------------------
    # Fuzz loop
    # --------------------------------------------------------

    for round_no in range(
        1,
        max_rounds + 1,
    ):

        # ----------------------------------------------------
        # Seed selection
        #
        # 가장 target distance가 작은 seed 선택
        # ----------------------------------------------------

        corpus.sort(
            key=lambda s: s.distance
        )

        parent = corpus[0]

        e = energy(
            parent.distance
        )


        print(
            "\n--------------------------------"
        )

        print(
            f"ROUND {round_no}"
        )

        print(
            "--------------------------------"
        )

        print(
            "selected seed:"
        )

        print(
            " ",
            parent.program,
        )

        print(
            f"distance={parent.distance:.2f}"
        )

        print(
            f"energy={e}"
        )


        # ----------------------------------------------------
        # Energy만큼 mutants 생성
        # ----------------------------------------------------

        for i in range(e):

            child_program, mutated_node = (
                targeted_mutation(
                    parent.program
                )
            )

            if child_program in seen:
                continue

            seen.add(
                child_program
            )

            child, hit = make_seed(
                child_program
            )


            print(
                f"\n mutant {i + 1}"
            )

            print(
                f"  mutated node = "
                f"{mutated_node}"
            )

            print(
                f"  program = "
                f"{child.program}"
            )

            print(
                f"  coverage = "
                f"{sorted(child.coverage)}"
            )

            print(
                f"  distance = "
                f"{child.distance:.2f}"
            )


            # ================================================
            # TARGET OPTIMIZATION HIT
            # ================================================

            if hit:

                print(
                    "\n================================"
                )

                print(
                    "TARGET OPTIMIZATION HIT"
                )

                print(
                    "================================"
                )

                print(
                    child.program
                )


                # ============================================
                # Translation validation
                # ============================================

                print(
                    "\nTRANSLATION VALIDATION"
                )

                result = (
                    translation_validate()
                )


                if result["valid"]:

                    print(
                        "VALID OPTIMIZATION"
                    )

                else:

                    print(
                        "INVALID OPTIMIZATION"
                    )

                    print(
                        "counterexample:"
                    )

                    print(
                        " runtime x =",
                        result["x"],
                    )

                    print(
                        " source x/2 =",
                        result["source"],
                    )

                    print(
                        " optimized x>>1 =",
                        result["optimized"],
                    )

                    print(
                        "\n*** MISCOMPILATION BUG FOUND ***"
                    )

                return


            # ------------------------------------------------
            # Corpus update
            #
            # parent보다 target에 가까워진 경우 보존
            # ------------------------------------------------

            if (
                child.distance
                < parent.distance
            ):

                print(
                    "  -> NEW PROMISING SEED"
                )

                corpus.append(
                    child
                )


    print(
        "\nTARGET NOT FOUND"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "================================"
    )

    print(
        "SLICE"
    )

    print(
        "================================"
    )

    for node in sorted(SLICE):

        print(
            node
        )


    print(
        "\n================================"
    )

    print(
        "NODE DISTANCE"
    )

    print(
        "================================"
    )

    for node, d in sorted(
        NODE_DIST.items(),
        key=lambda x: -x[1],
    ):

        print(
            f"{node:10s} -> {d}"
        )


    print(
        "\n================================"
    )

    print(
        "DEF-USE TARGETED NODES"
    )

    print(
        "================================"
    )

    print(
        "delta:",
        DELTA
    )

    print(
        "nodes:",
        TARGETED_NODES
    )


    print()

    fuzz()