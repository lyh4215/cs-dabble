from dataclasses import dataclass
from z3 import (
    Int,
    Bool,
    BitVec,
    BitVecVal,
    Solver,
    And,
    Not,
    sat,
)


# ============================================================
# Concrete program representation
# ============================================================

@dataclass
class Program:
    op: int
    signed: bool
    divisor: int


ADD = 0
SUB = 1
MUL = 2
DIV = 3
MOD = 4


# ============================================================
# Symbolic variables
# ============================================================

sym_op = Int("op")
sym_signed = Bool("signed")
sym_divisor = BitVec("divisor", 32)


def bv32(x):
    return BitVecVal(
        x & 0xffffffff,
        32
    )


# ============================================================
# Concrete helpers
# ============================================================

def is_power_of_two(x):

    return (
        x > 0
        and (x & (x - 1)) == 0
    )


# ============================================================
# One concolic execution
# ============================================================

def execute_concolic(p):

    path = []

    # --------------------------------------------------------
    # branch 1: op == DIV
    # --------------------------------------------------------

    cond = (sym_op == DIV)

    taken = (p.op == DIV)

    path.append(
        (cond, taken, "op == DIV")
    )

    if not taken:
        return path, False


    # --------------------------------------------------------
    # branch 2: signed
    # --------------------------------------------------------

    cond = sym_signed

    taken = p.signed

    path.append(
        (cond, taken, "signed")
    )

    if not taken:
        return path, False


    # --------------------------------------------------------
    # branch 3: divisor > 0
    # --------------------------------------------------------

    cond = (
        sym_divisor > bv32(0)
    )

    taken = (p.divisor > 0)

    path.append(
        (cond, taken, "divisor > 0")
    )

    if not taken:
        return path, False


    # --------------------------------------------------------
    # branch 4: power of two
    # --------------------------------------------------------

    cond = And(
        sym_divisor > bv32(0),

        (
            sym_divisor
            & (sym_divisor - bv32(1))
        ) == bv32(0)
    )

    taken = is_power_of_two(
        p.divisor
    )

    path.append(
        (
            cond,
            taken,
            "power_of_two(divisor)"
        )
    )

    if not taken:
        return path, False


    return path, True

def print_path(path):

    print("\nPATH")

    for i, (
        cond,
        taken,
        name
    ) in enumerate(path):

        print(
            f"{i}: "
            f"{name:25s} "
            f"taken={taken}"
        )

def solve_flipped_path(path, flip_index):

    solver = Solver()

    for i, (
        cond,
        taken,
        _
    ) in enumerate(path):

        # 우리가 뒤집으려는 branch
        if i == flip_index:

            solver.add(
                Not(cond)
                if taken
                else cond
            )

            break

        # 이전 branch들은
        # 현재 실행과 동일하게 유지
        if taken:
            solver.add(cond)
        else:
            solver.add(Not(cond))

    if solver.check() != sat:
        return None

    m = solver.model()

    op_value = m.eval(
        sym_op,
        model_completion=True
    ).as_long()

    signed_value = bool(
        m.eval(
            sym_signed,
            model_completion=True
        )
    )

    divisor_raw = m.eval(
        sym_divisor,
        model_completion=True
    ).as_long()

    # unsigned representation -> signed Python int
    if divisor_raw >= (1 << 31):
        divisor_value = (
            divisor_raw - (1 << 32)
        )
    else:
        divisor_value = divisor_raw

    return Program(
        op=op_value,
        signed=signed_value,
        divisor=divisor_value
    )

start = Program(
    op=DIV,
    signed=True,
    divisor=33
)

print(
    "START:",
    start
)

path, hit = execute_concolic(
    start
)

print_path(path)

print(
    "\nTARGET HIT:",
    hit
)

new_input = solve_flipped_path(
    path,
    flip_index=3
)

print(
    "\nNEW INPUT:",
    new_input
)

new_path, hit = execute_concolic(
    new_input
)

print_path(new_path)

print(
    "\nTARGET HIT:",
    hit
)