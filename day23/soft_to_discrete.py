import torch

torch.manual_seed(0)


def connect(A, i, j):
    A[i, j] = 1
    A[j, i] = 1


def fraud_scores(X, A):

    N = len(X)

    A_hat = A + torch.eye(N)

    degree = A_hat.sum(dim=1)

    H = A_hat @ X
    H = H / degree[:, None]

    return H.squeeze(1)


# ==================================================
# 1. original graph
# ==================================================

X = torch.tensor([
    [+1.0],   # 0 fraud
    [+1.0],   # 1 fraud
    [+1.0],   # 2 fraud

    [-1.0],   # 3 benign
    [-1.0],   # 4 benign
    [-1.0],   # 5 benign
])


A = torch.zeros(6, 6)

connect(A, 0, 1)
connect(A, 1, 2)
connect(A, 2, 0)

connect(A, 3, 4)
connect(A, 4, 5)
connect(A, 5, 3)


targets = [0, 1, 2]

K = 4
N_original = 6
EDGE_BUDGET = 3


# ==================================================
# 2. learn soft attack
# ==================================================

feature_raw = torch.nn.Parameter(
    torch.zeros(K, 1)
)

edge_logits = torch.nn.Parameter(
    torch.zeros(K, N_original)
)


optimizer = torch.optim.Adam(
    [
        feature_raw,
        edge_logits,
    ],
    lr=0.1,
)


for step in range(401):

    attack_feature = torch.tanh(
        feature_raw
    )


    edge_weight = (
        EDGE_BUDGET
        * torch.softmax(
            edge_logits,
            dim=1,
        )
    )


    X2 = torch.cat([
        X,
        attack_feature,
    ])


    A2 = torch.zeros(
        N_original + K,
        N_original + K,
    )

    A2[:N_original, :N_original] = A


    for k in range(K):

        attack_idx = N_original + k

        A2[
            attack_idx,
            :N_original
        ] = edge_weight[k]

        A2[
            :N_original,
            attack_idx
        ] = edge_weight[k]


    scores = fraud_scores(
        X2,
        A2,
    )


    loss = scores[targets].mean()


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ==================================================
# 3. soft result
# ==================================================

with torch.no_grad():

    attack_feature = torch.tanh(
        feature_raw
    )

    edge_weight = (
        EDGE_BUDGET
        * torch.softmax(
            edge_logits,
            dim=1,
        )
    )


print("=== SOFT ATTACK ===")

print(
    "attack features:"
)

print(
    attack_feature.squeeze().tolist()
)

print()

print(
    "attack node 0 edge weights:"
)

print(
    edge_weight[0].tolist()
)


# ==================================================
# 4. convert soft edge → discrete edge
#
# 각 attack node마다
# top-k node만 실제 edge로 선택
# ==================================================

A_discrete = torch.zeros(
    N_original + K,
    N_original + K,
)

A_discrete[
    :N_original,
    :N_original
] = A


for k in range(K):

    # 가장 높은 EDGE_BUDGET개 선택
    topk = torch.topk(
        edge_weight[k],
        k=EDGE_BUDGET,
    ).indices

    attack_idx = (
        N_original + k
    )

    for node in topk:

        node = int(node)

        A_discrete[
            attack_idx,
            node
        ] = 1

        A_discrete[
            node,
            attack_idx
        ] = 1


# ==================================================
# 5. evaluate discrete attack
# ==================================================

X_discrete = torch.cat([
    X,
    attack_feature,
])


scores_discrete = fraud_scores(
    X_discrete,
    A_discrete,
)


print(
    "\n=== DISCRETE EDGES ==="
)


for k in range(K):

    attack_idx = (
        N_original + k
    )

    neighbors = torch.where(
        A_discrete[
            attack_idx,
            :N_original
        ] > 0
    )[0]

    print(
        f"attack {k}:",
        neighbors.tolist()
    )


print(
    "\n=== FRAUD SCORES AFTER "
    "DISCRETE ATTACK ==="
)


for target in targets:

    score = scores_discrete[
        target
    ].item()

    print(
        f"target {target}: "
        f"{score:+.3f}",
        "->",
        "FRAUD"
        if score > 0
        else "BENIGN"
    )