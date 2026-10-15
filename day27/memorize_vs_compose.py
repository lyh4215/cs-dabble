# ============================================
# world state
#
# (x, y, painted, rotation)
# ============================================

start = (
    0,      # x
    0,      # y
    False,  # painted
    0,      # rotation
)


# ============================================
# reusable primitives
# ============================================

def left(s):
    x, y, paint, rot = s

    return (
        x - 1,
        y,
        paint,
        rot,
    )


def down(s):
    x, y, paint, rot = s

    return (
        x,
        y - 1,
        paint,
        rot,
    )


def paint(s):
    x, y, _, rot = s

    return (
        x,
        y,
        True,
        rot,
    )


def rotate(s):
    x, y, paint, rot = s

    return (
        x,
        y,
        paint,
        (rot + 90) % 360,
    )


OPS = {
    "L": left,
    "D": down,
    "P": paint,
    "R": rotate,
}


def execute(
    state,
    program,
):

    for op in program:

        state = OPS[op](
            state
        )

    return state


# ============================================
# training programs
# ============================================

train_programs = [
    ("L", "D"),
    ("L", "P"),
    ("D", "R"),
    ("P", "R"),
]


# ============================================
# 1. memorizer
#
# 전체 combination을 하나로 외운다.
# ============================================

memorizer = {}

for program in train_programs:

    memorizer[program] = execute(
        start,
        program,
    )


# ============================================
# 2. compositional executor
#
# primitive를 순서대로 적용한다.
# ============================================

def compositional_predict(
    program,
):

    return execute(
        start,
        program,
    )


# ============================================
# test
# ============================================

tests = [
    ("L", "D"),          # seen
    ("P", "R"),          # seen

    ("D", "P", "R"),     # unseen
    ("L", "D", "R"),     # unseen
]


print(
    "=== TEST ==="
)

for program in tests:

    true_result = execute(
        start,
        program,
    )

    memorized = memorizer.get(
        program,
        "UNKNOWN",
    )

    composed = compositional_predict(
        program
    )


    print(
        "\nprogram:",
        " -> ".join(program),
    )

    print(
        "true      :",
        true_result,
    )

    print(
        "memorizer :",
        memorized,
    )

    print(
        "composed   :",
        composed,
    )