from z3 import (
    BitVec,
    BitVecVal,
    Solver,
    sat,
    unsat,
)


BITS = 32


def signed(v):
    """
    unsigned 32-bit integer representation
    -> Python signed integer
    """
    if v >= (1 << 31):
        return v - (1 << 32)
    return v


def validate(name, src, tgt, x):
    s = Solver()

    # optimization이 틀리는 입력이 존재하는가?
    s.add(src != tgt)

    print(f"\n===== {name} =====")

    result = s.check()

    if result == sat:
        m = s.model()

        xv = m.eval(x, model_completion=True).as_long()
        sv = m.eval(src, model_completion=True).as_long()
        tv = m.eval(tgt, model_completion=True).as_long()

        print("INVALID")
        print("counterexample found")
        print("x      =", signed(xv))
        print("source =", signed(sv))
        print("target =", signed(tv))

    elif result == unsat:
        print("VALID")
        print("no 32-bit counterexample exists")

    else:
        print("UNKNOWN")


x = BitVec("x", BITS)

# --------------------------------------------------
# Source semantics
#
# signed division by 2
# Z3 BitVec / performs signed division.
# --------------------------------------------------

src = x / BitVecVal(2, BITS)


# --------------------------------------------------
# Candidate optimization 1
#
# x / 2  ->  x >> 1
#
# >> on a BitVec is arithmetic right shift.
# --------------------------------------------------

bad_target = x >> 1

validate(
    "x / 2  ->  x >> 1",
    src,
    bad_target,
    x
)


# --------------------------------------------------
# Candidate optimization 2
#
# Negative odd numbers need a bias before shifting.
#
# sign correction:
# ((x >> 31) & 1)
#
# positive -> 0
# negative -> 1
# --------------------------------------------------

bias = (x >> 31) & 1

fixed_target = (x + bias) >> 1

validate(
    "x / 2  ->  (x + bias) >> 1",
    src,
    fixed_target,
    x
)