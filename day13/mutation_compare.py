import random
from dataclasses import dataclass, replace


# ============================================================
# Toy input program
#
# 핵심 data flow:
#
# c1: Y = y
# c2: Z = z
# c3: X = Y + Z + k      <- delta
# c4: V = X * scale
#
# c5 ~ c12:
# target과 아무 관계 없는 noise commands
#
# 초기값:
#   X = 1 + 30 + 0 = 31
#   V = 31 * 2 = 62
#
# target:
#   X == 32 and V == 64
# ============================================================


@dataclass(frozen=True)
class Program:
    y: int = 1
    z: int = 30
    k: int = 0
    scale: int = 2

    noise1: int = 10
    noise2: int = 20
    noise3: int = 30
    noise4: int = 40
    noise5: int = 50
    noise6: int = 60
    noise7: int = 70
    noise8: int = 80


# ============================================================
# Program execution
# ============================================================

def execute(p):
    Y = p.y
    Z = p.z

    X = Y + Z + p.k

    V = X * p.scale

    return X, V


# ============================================================
# Compiler optimization target
#
# 이런 IR pattern일 때 특정 optimization이 발동한다고 가정
#
# X == 32
# V == 64
# ============================================================

def hits_target(p):
    X, V = execute(p)

    return (
        X == 32
        and V == 64
    )


# ============================================================
# Def-Use Graph
#
# c1 --Y--> c3
# c2 --Z--> c3
# c3 --X--> c4
#
# delta = 마지막 mutation으로
# target 쪽 progress를 만든 command
# ============================================================

def_use = {
    "c1": [
        ("Y", "c3"),
    ],

    "c2": [
        ("Z", "c3"),
    ],

    "c3": [
        ("X", "c4"),
    ],
}


delta = "c3"


# ============================================================
# delta와 직접적인 Def-Use 관계를 가진
# targeted mutation 후보 계산
#
# 1. delta 자체
# 2. delta에 값을 공급하는 definitions
# 3. delta가 정의한 값을 사용하는 users
# ============================================================

def targeted_nodes(delta):
    result = {delta}

    # --------------------------------------------------------
    # Definitions feeding delta
    #
    # src --var--> delta
    # --------------------------------------------------------

    for src, edges in def_use.items():

        for var, dst in edges:

            if dst == delta:
                result.add(src)

    # --------------------------------------------------------
    # Users of delta
    #
    # delta --var--> dst
    # --------------------------------------------------------

    for var, dst in def_use.get(delta, []):
        result.add(dst)

    return sorted(result)


TARGETED = targeted_nodes(delta)


# ============================================================
# 전체 program nodes
#
# c1 ~ c4  : target과 관련 있음
# c5 ~ c12 : noise
# ============================================================

ALL_NODES = [
    "c1",
    "c2",
    "c3",
    "c4",

    "c5",
    "c6",
    "c7",
    "c8",
    "c9",
    "c10",
    "c11",
    "c12",
]


# ============================================================
# 한 node mutation
#
# 단순화를 위해 해당 값에
# -1 또는 +1
#
# c1 -> y
# c2 -> z
# c3 -> k
# c4 -> scale
# c5~c12 -> noise
# ============================================================

def mutate_node(p, node):

    step = random.choice([
        -1,
        +1,
    ])

    # c1: Y = y
    if node == "c1":
        return replace(
            p,
            y=p.y + step,
        )

    # c2: Z = z
    if node == "c2":
        return replace(
            p,
            z=p.z + step,
        )

    # c3: X = Y + Z + k
    if node == "c3":
        return replace(
            p,
            k=p.k + step,
        )

    # c4: V = X * scale
    if node == "c4":
        return replace(
            p,
            scale=p.scale + step,
        )

    # --------------------------------------------------------
    # Noise commands
    #
    # c5  -> noise1
    # c6  -> noise2
    # ...
    # c12 -> noise8
    # --------------------------------------------------------

    cmd_number = int(
        node[1:]
    )

    noise_number = (
        cmd_number - 4
    )

    field = (
        f"noise{noise_number}"
    )

    return replace(
        p,
        **{
            field:
            getattr(p, field) + step
        },
    )


# ============================================================
# Random mutation
#
# 모든 node 중 아무 곳이나 선택
# ============================================================

def random_mutation(p):

    node = random.choice(
        ALL_NODES
    )

    mutant = mutate_node(
        p,
        node,
    )

    return mutant, node


# ============================================================
# Def-Use targeted mutation
#
# delta의 data dependency 주변에서만 선택
# ============================================================

def targeted_mutation(p):

    node = random.choice(
        TARGETED
    )

    mutant = mutate_node(
        p,
        node,
    )

    return mutant, node


# ============================================================
# Experiment
#
# 각 trial마다 항상 동일한 base seed에서
# mutation을 딱 한 번 적용
#
# 이렇게 해야
#
# "어느 위치를 선택해서 mutate하는가?"
#
# 라는 mutation strategy 자체만 비교할 수 있음.
# ============================================================

def experiment(
    strategy,
    trials=10000,
):

    hits = 0
    chosen = {}

    base = Program()

    for _ in range(trials):

        mutant, node = strategy(
            base
        )

        chosen[node] = (
            chosen.get(node, 0)
            + 1
        )

        if hits_target(mutant):
            hits += 1

    return hits, chosen


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    TRIALS = 10000

    base = Program()

    X, V = execute(base)

    print(
        "================================"
    )
    print(
        "BASE PROGRAM"
    )
    print(
        "================================"
    )

    print(
        base
    )

    print(
        f"X={X}, V={V}"
    )

    print(
        "TARGET HIT:",
        hits_target(base)
    )


    # --------------------------------------------------------
    # Targeted node selection
    # --------------------------------------------------------

    print(
        "\n================================"
    )
    print(
        "DEF-USE TARGET"
    )
    print(
        "================================"
    )

    print(
        "delta:",
        delta
    )

    print(
        "targeted nodes:",
        TARGETED
    )


    # --------------------------------------------------------
    # Random strategy
    # --------------------------------------------------------

    random.seed(0)

    (
        random_hits,
        random_chosen,
    ) = experiment(
        random_mutation,
        TRIALS,
    )


    # --------------------------------------------------------
    # Targeted strategy
    # --------------------------------------------------------

    random.seed(0)

    (
        targeted_hits,
        targeted_chosen,
    ) = experiment(
        targeted_mutation,
        TRIALS,
    )


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(
        "\n================================"
    )
    print(
        "RANDOM MUTATION"
    )
    print(
        "================================"
    )

    print(
        "target hits:",
        random_hits
    )

    print(
        "hit ratio:",
        random_hits / TRIALS
    )

    print(
        "\nchosen nodes:"
    )

    for node in ALL_NODES:

        count = random_chosen.get(
            node,
            0
        )

        print(
            f"{node:4s}: {count}"
        )


    print(
        "\n================================"
    )
    print(
        "TARGETED MUTATION"
    )
    print(
        "================================"
    )

    print(
        "target hits:",
        targeted_hits
    )

    print(
        "hit ratio:",
        targeted_hits / TRIALS
    )

    print(
        "\nchosen nodes:"
    )

    for node in TARGETED:

        count = targeted_chosen.get(
            node,
            0
        )

        print(
            f"{node:4s}: {count}"
        )


    print(
        "\n================================"
    )
    print(
        "IMPROVEMENT"
    )
    print(
        "================================"
    )

    if random_hits > 0:

        print(
            "targeted / random =",
            targeted_hits / random_hits
        )

    else:

        print(
            "random had zero hits"
        )


    # ========================================================
    # Theoretical probability
    #
    # Random:
    #
    # useful node = c1, c2, c3
    #
    # P(useful node) = 3 / 12
    # P(+1)          = 1 / 2
    #
    # P(hit) = 3/12 * 1/2
    #        = 1/8
    #        = 0.125
    #
    #
    # Targeted:
    #
    # candidates = c1,c2,c3,c4
    # useful     = c1,c2,c3
    #
    # P(useful node) = 3 / 4
    # P(+1)          = 1 / 2
    #
    # P(hit) = 3/4 * 1/2
    #        = 3/8
    #        = 0.375
    #
    # Expected improvement:
    #
    # 0.375 / 0.125 = 3x
    # ========================================================

    print(
        "\n================================"
    )
    print(
        "THEORETICAL"
    )
    print(
        "================================"
    )

    random_probability = (
        3 / 12
    ) * (
        1 / 2
    )

    targeted_probability = (
        3 / 4
    ) * (
        1 / 2
    )

    print(
        "random expected hit ratio:",
        random_probability
    )

    print(
        "targeted expected hit ratio:",
        targeted_probability
    )

    print(
        "expected improvement:",
        targeted_probability
        / random_probability
    )