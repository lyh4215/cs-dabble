from z3 import (
    BitVec,
    Int,
    Bool,
    Solver,
    And,
    sat,
)


# ============================================================
# Symbolic program input
# ============================================================

op = Int("op")

signed = Bool("signed")

divisor = BitVec(
    "divisor",
    32
)


# op encoding
ADD = 0
SUB = 1
MUL = 2
DIV = 3
MOD = 4


# ============================================================
# Target path condition
#
# if op == DIV
#   if signed
#     if divisor > 0
#       if power_of_two(divisor)
#           TARGET
# ============================================================

target_condition = And(

    op == DIV,

    signed == True,

    divisor > 0,

    (divisor & (divisor - 1)) == 0,
)


solver = Solver()

solver.add(target_condition)


print(
    "===== SOLVING TARGET PATH ====="
)


if solver.check() == sat:

    model = solver.model()

    print(
        "TARGET is reachable"
    )

    print(
        "op      =",
        model[op]
    )

    print(
        "signed  =",
        model[signed]
    )

    print(
        "divisor =",
        model[divisor].as_long()
    )

else:

    print(
        "TARGET is unreachable"
    )