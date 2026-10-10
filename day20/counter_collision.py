import random
import math


def has_collision(q, bits):
    seen = set()

    for _ in range(q):
        x = random.getrandbits(bits)

        if x in seen:
            return True

        seen.add(x)

    return False


def experiment(q, bits, trials=1000):
    collisions = 0

    for _ in range(trials):
        if has_collision(q, bits):
            collisions += 1

    return collisions / trials


def theoretical(q, bits):
    N = 2 ** bits

    return 1 - math.exp(
        -q * (q - 1) / (2 * N)
    )


n = 12

queries = [
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
]


print("=== n-bit state ===")

for q in queries:
    empirical = experiment(q, n)
    theory = theoretical(q, n)

    print(
        f"q={q:4d}",
        f"empirical={empirical:6.2%}",
        f"theory={theory:6.2%}",
    )


print("\n=== 2n-bit state ===")

for q in queries:
    empirical = experiment(q, 2 * n)
    theory = theoretical(q, 2 * n)

    print(
        f"q={q:4d}",
        f"empirical={empirical:6.2%}",
        f"theory={theory:6.2%}",
    )