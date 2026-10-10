import random
import math


BITS = 16
SPACE = 2 ** BITS


def trial(q):
    seen = set()

    for _ in range(q):
        tag = random.randrange(SPACE)

        if tag in seen:
            return True

        seen.add(tag)

    return False


def collision_probability(q, trials=5000):
    collisions = 0

    for _ in range(trials):
        if trial(q):
            collisions += 1

    return collisions / trials


print("Tag bits:", BITS)
print("Tag space:", SPACE)

print(
    "Birthday scale:",
    int(math.sqrt(SPACE)),
)


for q in [
    32,
    64,
    128,
    256,
    512,
    1024,
]:

    p = collision_probability(q)

    print(
        f"q={q:4d}",
        f"collision={p:.2%}",
    )