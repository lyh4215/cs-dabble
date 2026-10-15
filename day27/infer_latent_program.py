from itertools import product


# ============================================
# state = (x, y, direction)
#
# direction:
# 0 = North
# 1 = East
# 2 = South
# 3 = West
# ============================================


def forward(state):

    x, y, d = state

    if d == 0:
        y += 1

    elif d == 1:
        x += 1

    elif d == 2:
        y -= 1

    elif d == 3:
        x -= 1

    return (
        x,
        y,
        d,
    )


def rotate_right(state):

    x, y, d = state

    return (
        x,
        y,
        (d + 1) % 4,
    )


def rotate_left(state):

    x, y, d = state

    return (
        x,
        y,
        (d - 1) % 4,
    )


OPS = {
    "F": forward,
    "R": rotate_right,
    "L": rotate_left,
}


# ============================================
# execute program
# ============================================

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
# infer hidden program
#
# model knows primitive operators,
# but not which sequence was executed.
#
# brute-force all programs up to max_len.
# ============================================

def infer_program(
    before,
    after,
    max_len=4,
):

    solutions = []

    names = list(
        OPS.keys()
    )

    for length in range(
        1,
        max_len + 1,
    ):

        for program in product(
            names,
            repeat=length,
        ):

            predicted = execute(
                before,
                program,
            )

            if predicted == after:

                solutions.append(
                    program
                )

        # shortest explanation 먼저
        if solutions:
            break

    return solutions


# ============================================
# hidden world
# ============================================

tests = [

    (
        (0, 0, 0),
        ("F", "R", "F"),
    ),

    (
        (2, 3, 1),
        ("R", "F", "F"),
    ),

    (
        (-1, 2, 2),
        ("L", "F", "R"),
    ),
]


print(
    "=== LATENT PROGRAM INFERENCE ==="
)


for before, hidden_program in tests:

    after = execute(
        before,
        hidden_program,
    )


    solutions = infer_program(
        before,
        after,
        max_len=4,
    )


    print(
        "\nbefore:",
        before,
    )

    print(
        "after :",
        after,
    )

    print(
        "hidden:",
        " -> ".join(
            hidden_program
        ),
    )


    print(
        "inferred:"
    )

    for p in solutions:

        print(
            " ",
            " -> ".join(p),
        )