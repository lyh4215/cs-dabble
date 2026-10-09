import random
import math
from collections import Counter


NUM_BANKS = 4
NUM_BLOCKS = 64
N = 10000


def logical_bank(block_id):
    return block_id % NUM_BANKS


def naive_access(block_id):
    """
    Naive distributed memory:
    block이 들어있는 bank 하나만 활성화
    """
    return (logical_bank(block_id),)


def split_access(block_id):
    """
    PIM-ORAM의 핵심 아이디어를 축약:
    모든 bank를 동시에 활성화
    """
    return tuple(range(NUM_BANKS))


def mutual_information(pairs):
    """
    I(secret ; observation)
    """

    total = len(pairs)

    joint = Counter(pairs)
    xs = Counter(x for x, _ in pairs)
    ys = Counter(y for _, y in pairs)

    mi = 0.0

    for (x, y), count in joint.items():
        pxy = count / total
        px = xs[x] / total
        py = ys[y] / total

        mi += pxy * math.log2(
            pxy / (px * py)
        )

    return mi


random.seed(0)

naive_pairs = []
split_pairs = []

for _ in range(N):
    block = random.randrange(NUM_BLOCKS)

    secret = logical_bank(block)

    naive_obs = naive_access(block)
    split_obs = split_access(block)

    naive_pairs.append(
        (secret, naive_obs)
    )

    split_pairs.append(
        (secret, split_obs)
    )


print("=== Example accesses ===")

for block in [5, 14, 23, 32]:

    print(
        f"block={block:2d}",
        f"secret-bank={logical_bank(block)}",
        f"naive={naive_access(block)}",
        f"split={split_access(block)}",
    )


print("\n=== Leakage ===")

print(
    "Naive mutual information:",
    f"{mutual_information(naive_pairs):.4f} bits",
)

print(
    "Split mutual information:",
    f"{mutual_information(split_pairs):.4f} bits",
)