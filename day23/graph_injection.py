import torch


# ==================================================
# GNN fraud detector
#
# 한 layer짜리 mean aggregation
#
# score_i =
#   mean(
#       self feature
#       + neighbor features
#   )
#
# score > 0 -> fraud
# score < 0 -> benign
# ==================================================

def fraud_scores(X, A):

    N = len(X)

    # self-loop 추가
    A_hat = A + torch.eye(N)

    degree = A_hat.sum(dim=1)

    # neighbor mean
    H = A_hat @ X

    H = H / degree[:, None]

    return H.squeeze(1)


def connect(A, i, j):

    A[i, j] = 1
    A[j, i] = 1


# ==================================================
# 1. Original graph
#
# target fraud nodes:
# 0, 1, 2
#
# benign nodes:
# 3, 4, 5
# ==================================================

X = torch.tensor([
    [+1.0],    # 0 fraud
    [+1.0],    # 1 fraud
    [+1.0],    # 2 fraud

    [-1.0],    # 3 benign
    [-1.0],    # 4 benign
    [-1.0],    # 5 benign
])


A = torch.zeros(6, 6)


# fraud gang
connect(A, 0, 1)
connect(A, 1, 2)
connect(A, 2, 0)


# benign community
connect(A, 3, 4)
connect(A, 4, 5)
connect(A, 5, 3)


# ==================================================
# Original prediction
# ==================================================

before = fraud_scores(
    X,
    A,
)

print("=== BEFORE ATTACK ===")

for i in range(3):

    print(
        f"fraud target {i}:",
        f"score={before[i]:+.3f}",
        "->",
        "FRAUD" if before[i] > 0 else "BENIGN"
    )


# ==================================================
# 2. Inject attack nodes
#
# attack nodes는 benign처럼 보이는 feature
#
# node 6, 7, 8, 9
# ==================================================

num_attack = 4

X_attack = torch.full(
    (num_attack, 1),
    -1.0,
)


X2 = torch.cat(
    [
        X,
        X_attack,
    ],
    dim=0,
)


A2 = torch.zeros(
    6 + num_attack,
    6 + num_attack,
)

A2[:6, :6] = A


# ==================================================
# 3. multi-target injection
#
# 동일한 attack nodes가
# fraud target 0,1,2에 동시에 연결
# ==================================================

for attack_node in range(
    6,
    6 + num_attack,
):

    for target in [0, 1, 2]:

        connect(
            A2,
            attack_node,
            target,
        )


# ==================================================
# After attack
# ==================================================

after = fraud_scores(
    X2,
    A2,
)

print("\n=== AFTER ATTACK ===")

for i in range(3):

    print(
        f"fraud target {i}:",
        f"score={after[i]:+.3f}",
        "->",
        "FRAUD" if after[i] > 0 else "BENIGN"
    )