import random
from dataclasses import dataclass


random.seed(0)


# ============================================================
# Tiny input program
# ============================================================

@dataclass(frozen=True)
class Program:
    op: str
    signed: bool
    divisor: int


OPS = [
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "MOD",
]


# ============================================================
# Compiler optimization CFG
#
# TARGET:
#
# DIV
#  ↓
# signed
#  ↓
# positive divisor
#  ↓
# power of two
#  ↓
# TARGET
# ============================================================

def is_power_of_two(x):
    return (
        x > 0
        and (x & (x - 1)) == 0
    )


def compile_program(p):
    """
    return:
        reached_depth
        coverage
        hit_target
    """

    coverage = ["ENTRY"]

    # condition 1
    if p.op != "DIV":
        coverage.append("NOT_DIV_EXIT")
        return 0, coverage, False

    coverage.append("DIV")

    # condition 2
    if not p.signed:
        coverage.append("UNSIGNED_EXIT")
        return 1, coverage, False

    coverage.append("SIGNED")

    # condition 3
    if p.divisor <= 0:
        coverage.append("NONPOSITIVE_EXIT")
        return 2, coverage, False

    coverage.append("POSITIVE")

    # condition 4
    if not is_power_of_two(p.divisor):
        coverage.append("NOT_POW2_EXIT")
        return 3, coverage, False

    coverage.append("POWER_OF_TWO")
    coverage.append("TARGET")

    return 4, coverage, True

def random_program():

    return Program(
        op=random.choice(OPS),
        signed=random.choice([
            True,
            False
        ]),
        divisor=random.randint(
            -1000,
            1000
        )
    )


def random_fuzz(max_trials=100000):

    for trial in range(1, max_trials + 1):

        p = random_program()

        depth, coverage, hit = compile_program(p)

        if hit:
            return trial, p, coverage

    return None

def mutate(p):

    field = random.choice([
        "op",
        "signed",
        "divisor"
    ])

    if field == "op":

        return Program(
            op=random.choice(OPS),
            signed=p.signed,
            divisor=p.divisor
        )

    elif field == "signed":

        return Program(
            op=p.op,
            signed=not p.signed,
            divisor=p.divisor
        )

    else:

        # divisor mutation
        mode = random.choice([
            "random",
            "small",
            "bit"
        ])

        if mode == "random":

            d = random.randint(
                -1000,
                1000
            )

        elif mode == "small":

            d = p.divisor + random.choice([
                -2, -1, 1, 2
            ])

        else:

            bit = 1 << random.randint(0, 9)

            d = p.divisor ^ bit

        return Program(
            op=p.op,
            signed=p.signed,
            divisor=d
        )

def directed_fuzz(max_trials=100000):

    current = random_program()

    best_depth, _, hit = compile_program(
        current
    )

    if hit:
        return 1, current, best_depth

    print(
        f"start: {current} "
        f"depth={best_depth}"
    )

    for trial in range(2, max_trials + 1):

        candidate = mutate(current)

        depth, coverage, hit = compile_program(
            candidate
        )

        # target hit
        if hit:

            print(
                f"trial {trial:4d}: "
                f"{candidate} "
                f"depth={depth} TARGET!"
            )

            return trial, candidate, depth

        # target에 더 가까워졌으면 채택
        if depth > best_depth:

            current = candidate
            best_depth = depth

            print(
                f"trial {trial:4d}: "
                f"{candidate} "
                f"depth={depth}"
            )

    return None

print(
    "\n================================"
)
print("RANDOM FUZZING")
print(
    "================================"
)

result = random_fuzz()

print(result)


print(
    "\n================================"
)
print("DIRECTED FUZZING")
print(
    "================================"
)

random.seed(0)

result = directed_fuzz()

print("\nRESULT:")
print(result)