import random


N = 10000

READ = 0
WRITE = 1


def trace_without_decoy(op):
    if op == READ:
        return ("PIM->HOST",)

    if op == WRITE:
        return ("HOST->PIM",)


def trace_with_decoy(op):
    if op == READ:
        # dummy write + real read
        return (
            "HOST->PIM",
            "PIM->HOST",
        )

    if op == WRITE:
        # real write + dummy read
        return (
            "HOST->PIM",
            "PIM->HOST",
        )


def attacker(trace):
    """
    아주 단순한 공격자.
    bus 방향만 보고 READ/WRITE 추측.
    """

    if trace == ("PIM->HOST",):
        return READ

    if trace == ("HOST->PIM",):
        return WRITE

    # 둘 다 보이면 구분 불가 → random guess
    return random.choice([READ, WRITE])


def evaluate(trace_func):
    correct = 0

    for _ in range(N):
        op = random.choice([READ, WRITE])

        trace = trace_func(op)

        guess = attacker(trace)

        if guess == op:
            correct += 1

    return correct / N


random.seed(0)


print("=== Example ===")

print(
    "READ without decoy :",
    trace_without_decoy(READ),
)

print(
    "WRITE without decoy:",
    trace_without_decoy(WRITE),
)

print(
    "READ with decoy    :",
    trace_with_decoy(READ),
)

print(
    "WRITE with decoy   :",
    trace_with_decoy(WRITE),
)


print("\n=== Attacker accuracy ===")

print(
    "Without decoy:",
    f"{evaluate(trace_without_decoy):.2%}",
)

print(
    "With decoy   :",
    f"{evaluate(trace_with_decoy):.2%}",
)