import torch

torch.manual_seed(0)


def connect(A, i, j):

    A[i, j] = 1
    A[j, i] = 1


def fraud_scores(X, A):

    N = len(X)

    # self-loop
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


# fraud triangle
connect(A, 0, 1)
connect(A, 1, 2)
connect(A, 2, 0)


# benign triangle
connect(A, 3, 4)
connect(A, 4, 5)
connect(A, 5, 3)


targets = [0, 1, 2]


# ==================================================
# 2. attack parameters
#
# attack nodes 4개
# ==================================================

K = 4
N_original = 6


# attack node feature
#
# 실제 feature는 tanh를 통과시켜
# [-1, +1] 범위로 제한
feature_raw = torch.nn.Parameter(
    torch.zeros(K, 1)
)


# 각 attack node가
# 기존 6개 node 중 어디에 연결될지
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


# ==================================================
# 3. optimize attack
# ==================================================

for step in range(401):

    # ----------------------------------------------
    # attack feature
    # ----------------------------------------------

    attack_feature = torch.tanh(
        feature_raw
    )


    # ----------------------------------------------
    # soft edge selection
    #
    # 각 attack node의 총 edge budget = 3
    #
    # softmax라서 처음에는
    # 6개 node에 골고루 퍼짐
    # ----------------------------------------------

    edge_weight = (
        3.0
        * torch.softmax(
            edge_logits,
            dim=1,
        )
    )


    # ----------------------------------------------
    # expanded graph
    # ----------------------------------------------

    X2 = torch.cat(
        [
            X,
            attack_feature,
        ],
        dim=0,
    )


    A2 = torch.zeros(
        N_original + K,
        N_original + K,
    )

    A2[
        :N_original,
        :N_original
    ] = A


    # ----------------------------------------------
    # attack node ↔ original node
    #
    # weighted edges
    # ----------------------------------------------

    for k in range(K):

        attack_idx = (
            N_original + k
        )

        A2[
            attack_idx,
            :N_original
        ] = edge_weight[k]

        A2[
            :N_original,
            attack_idx
        ] = edge_weight[k]


    # ----------------------------------------------
    # victim GNN
    # ----------------------------------------------

    scores = fraud_scores(
        X2,
        A2,
    )


    # ----------------------------------------------
    # attacker objective
    #
    # fraud target score를
    # 최대한 낮추고 싶다
    # ----------------------------------------------

    attack_loss = (
        scores[targets].mean()
    )


    # edge가 너무 골고루 퍼지지 않고
    # 몇 군데 집중되도록 약한 entropy penalty
    p = edge_weight / 3.0

    entropy = -(
        p
        * torch.log(
            p + 1e-8
        )
    ).sum(
        dim=1
    ).mean()


    loss = (
        attack_loss
        + 0.02 * entropy
    )


    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    if step % 100 == 0:

        print(
            f"\nstep={step}"
        )

        print(
            "fraud scores:",
            scores[targets]
            .detach()
            .tolist()
        )

        print(
            "attack features:",
            attack_feature
            .detach()
            .squeeze()
            .tolist()
        )

        print(
            "edge weights of attack node 0:"
        )

        print(
            edge_weight[0]
            .detach()
            .tolist()
        )