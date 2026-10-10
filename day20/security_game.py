import random


BITS = 16
SPACE = 2 ** BITS
TRIALS = 5000


def play_game(q):
    """
    b = 0: random permutation
    b = 1: random function

    attacker:
      output collision이 있으면 function이라고 추측.
    """

    b = random.choice([0, 1])

    if b == 0:
        # permutation:
        # 서로 다른 input의 output은 중복되지 않음
        outputs = random.sample(
            range(SPACE),
            q,
        )

    else:
        # random function:
        # output 중복 가능
        outputs = [
            random.randrange(SPACE)
            for _ in range(q)
        ]

    collision = (
        len(set(outputs)) < len(outputs)
    )

    guess = 1 if collision else 0

    return guess == b


def experiment(q):
    correct = 0

    for _ in range(TRIALS):
        if play_game(q):
            correct += 1

    return correct / TRIALS


print(
    f"{'q':>6}",
    f"{'attacker accuracy':>20}",
)


for q in [
    32,
    64,
    128,
    256,
    512,
    1024,
]:

    acc = experiment(q)

    print(
        f"{q:6d}",
        f"{acc:20.2%}",
    )