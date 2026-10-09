import random


NUM_BLOCKS = 64
NUM_PATHS = 64
N = 10000

random.seed(0)


def fixed_mapping(block):
    """
    모든 bank를 활성화하더라도,
    logical block이 항상 같은 physical 위치에 있다고 가정
    """
    return block


class ToyORAM:
    def __init__(self):
        # 각 block에 random path 배정
        self.path = {
            block: random.randrange(NUM_PATHS)
            for block in range(NUM_BLOCKS)
        }

    def access(self, block):
        # 현재 path가 외부에서 관측됨
        observed = self.path[block]

        # 접근 후 새로운 random path로 remap
        self.path[block] = random.randrange(NUM_PATHS)

        return observed


def make_pairs():
    pairs = []

    for _ in range(N):
        a = random.randrange(NUM_BLOCKS)

        # 절반은 같은 block 반복
        if random.random() < 0.5:
            b = a
            same = True

        else:
            choices = list(range(NUM_BLOCKS))
            choices.remove(a)
            b = random.choice(choices)
            same = False

        pairs.append((a, b, same))

    return pairs


pairs = make_pairs()


def evaluate_fixed():
    correct = 0

    for a, b, actually_same in pairs:

        obs1 = fixed_mapping(a)
        obs2 = fixed_mapping(b)

        # 공격자:
        # physical address가 같으면 같은 block이라고 추측
        guess_same = (obs1 == obs2)

        if guess_same == actually_same:
            correct += 1

    return correct / len(pairs)


def evaluate_oram():
    correct = 0
    oram = ToyORAM()

    for a, b, actually_same in pairs:

        obs1 = oram.access(a)
        obs2 = oram.access(b)

        guess_same = (obs1 == obs2)

        if guess_same == actually_same:
            correct += 1

    return correct / len(pairs)


print("Attacker goal:")
print("  Were two consecutive accesses to the same logical block?")

print()

print(
    "Fixed physical mapping accuracy:",
    f"{evaluate_fixed():.2%}",
)

print(
    "ORAM-style remapping accuracy:",
    f"{evaluate_oram():.2%}",
)