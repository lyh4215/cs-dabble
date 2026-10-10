import random


BITS = 16
TRIALS = 10000


def bad_prg():
    """
    16-bit처럼 보이지만
    마지막 bit는 항상 0.
    """
    x = random.randrange(2 ** (BITS - 1))

    return x << 1


def uniform_random():
    return random.randrange(2 ** BITS)


def adversary(ciphertext):
    """
    A:
    ciphertext의 마지막 bit를 보고
    어떤 message가 암호화됐는지 추측.
    """

    return ciphertext & 1


def encryption_game(pad_generator):
    """
    m0 = ...000
    m1 = ...001

    challenger가 랜덤으로 하나 선택해 암호화.
    """

    m0 = 0
    m1 = 1

    b = random.choice([0, 1])

    message = m0 if b == 0 else m1

    pad = pad_generator()

    ciphertext = message ^ pad

    guess = adversary(ciphertext)

    return guess == b


def test_encryption(pad_generator):
    wins = 0

    for _ in range(TRIALS):
        if encryption_game(pad_generator):
            wins += 1

    return wins / TRIALS


print("=== Encryption attacker A ===")

real_success = test_encryption(bad_prg)
ideal_success = test_encryption(uniform_random)

print(
    "With bad PRG :",
    f"{real_success:.2%}",
)

print(
    "With uniform :",
    f"{ideal_success:.2%}",
)


# --------------------------------------------------
# Reduction
#
# B는 challenge value z를 받는다.
#
# z가:
#   bad_prg() 출력인지
#   uniform random인지
#
# 구별해야 한다.
#
# B는 직접 분석하지 않고
# attacker A를 내부에서 사용한다.
# --------------------------------------------------

def reduction_B(z):
    m0 = 0
    m1 = 1

    b = random.choice([0, 1])

    message = m0 if b == 0 else m1

    ciphertext = message ^ z

    guess = adversary(ciphertext)

    # A가 challenge bit를 맞혔다면
    # "이 z는 PRG 출력 같다"고 판단
    return guess == b


def test_B(source):
    outputs_one = 0

    for _ in range(TRIALS):

        z = source()

        if reduction_B(z):
            outputs_one += 1

    return outputs_one / TRIALS


print("\n=== Reduction B ===")

p_prg = test_B(bad_prg)
p_random = test_B(uniform_random)

print(
    "Pr[B says PRG | PRG]    :",
    f"{p_prg:.2%}",
)

print(
    "Pr[B says PRG | Random] :",
    f"{p_random:.2%}",
)

print(
    "Distinguishing advantage:",
    f"{abs(p_prg - p_random):.2%}",
)