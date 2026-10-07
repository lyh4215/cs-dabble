import math


# ============================================================
# Original test results
#
# TC1 FAIL
# TC2 FAIL
# TC3 PASS
# TC4 PASS
# TC5 PASS
# ============================================================

NUM_FAIL = 2
NUM_PASS = 3


# ============================================================
# Spectrum-Based Fault Localization
#
# 각 statement를:
#
# failing tests 몇 개가 실행했는가?
# passing tests 몇 개가 실행했는가?
#
# paper Figure 1의 값
# ============================================================

coverage = {
    "s1": (2, 3),   # actual faulty statement
    "s2": (2, 3),
    "s3": (2, 2),
    "s4": (2, 2),
    "s5": (1, 1),
    "s6": (2, 3),
}


def ochiai(failed_cov, passed_cov):

    if failed_cov == 0:
        return 0.0

    return (
        failed_cov
        /
        math.sqrt(
            NUM_FAIL
            * (failed_cov + passed_cov)
        )
    )


# ============================================================
# Mutation outcomes
#
# Each tuple:
#
#     (F -> P count, P -> F count)
#
# 논문 Figure 1의 mutant 결과를 그대로 옮김.
# ============================================================

mutants = {

    "s1": [
        # m1
        (0, 1),

        # m2
        (2, 0),
    ],

    "s2": [
        # m3
        (0, 3),

        # m4
        (1, 1),
    ],

    "s3": [
        # m5
        (0, 2),

        # m6
        (0, 2),
    ],

    "s4": [
        # m7
        (0, 2),

        # m8
        (0, 1),
    ],

    "s5": [
        # m9
        (0, 1),

        # m10
        (0, 1),
    ],

    "s6": [
        # m11
        (0, 2),

        # m12
        (0, 3),
    ],
}


# ============================================================
# Compute alpha
#
# MUSE:
#
#            f2p              |mut(P)| * |P|
# alpha = -----------   *   -----------------
#         |mut(P)|*|F|             p2f
#
# |mut(P)| cancels:
#
#         f2p * |P|
# alpha = ----------
#         p2f * |F|
# ============================================================

all_mutants = [
    result
    for stmt_mutants in mutants.values()
    for result in stmt_mutants
]


f2p = sum(
    fp
    for fp, pf in all_mutants
)

p2f = sum(
    pf
    for fp, pf in all_mutants
)


alpha = (
    f2p * NUM_PASS
    /
    (p2f * NUM_FAIL)
)


print(
    "================================"
)

print(
    "GLOBAL MUTATION STATISTICS"
)

print(
    "================================"
)

print(
    "F -> P:",
    f2p
)

print(
    "P -> F:",
    p2f
)

print(
    "alpha:",
    alpha
)


# ============================================================
# MUSE suspiciousness
# ============================================================

def muse_score(stmt):

    stmt_mutants = mutants[stmt]

    scores = []

    for f_to_p, p_to_f in stmt_mutants:

        positive = (
            f_to_p / NUM_FAIL
        )

        negative = (
            alpha
            * p_to_f
            / NUM_PASS
        )

        scores.append(
            positive - negative
        )

    return (
        sum(scores)
        / len(scores)
    )


# ============================================================
# Ochiai ranking
# ============================================================

ochiai_results = []

for stmt, (
    failed_cov,
    passed_cov,
) in coverage.items():

    score = ochiai(
        failed_cov,
        passed_cov
    )

    ochiai_results.append(
        (score, stmt)
    )


ochiai_results.sort(
    reverse=True
)


print(
    "\n================================"
)

print(
    "SBFL — OCHIAI"
)

print(
    "================================"
)

for rank, (
    score,
    stmt,
) in enumerate(
    ochiai_results,
    1
):

    marker = (
        "  <-- BUG"
        if stmt == "s1"
        else ""
    )

    print(
        f"{rank}. "
        f"{stmt} "
        f"score={score:.3f}"
        f"{marker}"
    )


# ============================================================
# MUSE ranking
# ============================================================

muse_results = []

for stmt in mutants:

    score = muse_score(
        stmt
    )

    muse_results.append(
        (score, stmt)
    )


muse_results.sort(
    reverse=True
)


print(
    "\n================================"
)

print(
    "MBFL — MUSE"
)

print(
    "================================"
)

for rank, (
    score,
    stmt,
) in enumerate(
    muse_results,
    1
):

    marker = (
        "  <-- BUG"
        if stmt == "s1"
        else ""
    )

    print(
        f"{rank}. "
        f"{stmt} "
        f"score={score:.3f}"
        f"{marker}"
    )


# ============================================================
# Inspect mutants of faulty statement
# ============================================================

print(
    "\n================================"
)

print(
    "WHY IS s1 SUSPICIOUS?"
)

print(
    "================================"
)


for i, (
    f_to_p,
    p_to_f,
) in enumerate(
    mutants["s1"],
    1
):

    score = (
        f_to_p / NUM_FAIL
        -
        alpha
        * p_to_f / NUM_PASS
    )

    print(
        f"m{i}: "
        f"F->P={f_to_p}, "
        f"P->F={p_to_f}, "
        f"score={score:.3f}"
    )