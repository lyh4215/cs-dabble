import torch
import math

torch.manual_seed(0)

n = 4000


# --------------------------------------------------
# data
# --------------------------------------------------

A = torch.randn(n // 2, 2) * 0.4
A[:, 0] += 2.0

B = torch.randn(n // 2, 2) * 0.4
B[:, 0] -= 2.0

X = torch.cat([A, B])

semantic = torch.cat([
    torch.zeros(n // 2, dtype=torch.long),
    torch.ones(n // 2, dtype=torch.long),
])


# --------------------------------------------------
# tiny learned router
# A/B를 supervision으로 직접 주지 않고,
# 여기서는 routing boundary 학습만 간단히 재현
# --------------------------------------------------

router = torch.nn.Linear(2, 2)

optimizer = torch.optim.Adam(
    router.parameters(),
    lr=0.03,
)

# 이전 MoE에서 학습된 expert assignment를
# routing target이라고 생각한다.
for _ in range(500):

    logits = router(X)

    loss = torch.nn.functional.cross_entropy(
        logits,
        semantic,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# --------------------------------------------------
# rotation
# --------------------------------------------------

def rotate(X, degree):

    t = math.radians(degree)

    R = torch.tensor([
        [math.cos(t), -math.sin(t)],
        [math.sin(t),  math.cos(t)],
    ])

    return X @ R.T


previous_v = None


print(
    "degree | learned | subspace | "
    "mix(.25) | mix(.50) | mix(.75)"
)

print("-" * 70)


for degree in range(0, 181, 15):

    Xr = rotate(X, degree)


    # ----------------------------------------------
    # 1. learned gate
    # ----------------------------------------------

    learned_logits = router(Xr)

    learned_pred = torch.argmax(
        learned_logits,
        dim=1,
    )

    learned_acc = (
        learned_pred == semantic
    ).float().mean()


    # ----------------------------------------------
    # 2. principal subspace
    # ----------------------------------------------

    centered = Xr - Xr.mean(dim=0)

    cov = (
        centered.T @ centered
        / (len(Xr) - 1)
    )

    _, eigvec = torch.linalg.eigh(cov)

    v = eigvec[:, -1]


    # orientation continuity
    if previous_v is None:

        if v[0] < 0:
            v = -v

    elif torch.dot(v, previous_v) < 0:

        v = -v

    previous_v = v.clone()


    # projection
    p = Xr @ v

    # A → positive,
    # B → negative라고 현재 orientation이 기억함
    structure_logits = torch.stack(
        [p, -p],
        dim=1,
    )

    subspace_pred = torch.argmax(
        structure_logits,
        dim=1,
    )

    subspace_acc = (
        subspace_pred == semantic
    ).float().mean()


    # ----------------------------------------------
    # scale normalize
    #
    # 두 score의 크기가 너무 다르면
    # beta 비교가 무의미하므로 표준화
    # ----------------------------------------------

    L = learned_logits
    S = structure_logits

    L = L / L.std()
    S = S / S.std()


    accs = []

    for beta in [0.25, 0.50, 0.75]:

        combined = (
            (1 - beta) * L
            + beta * S
        )

        pred = torch.argmax(
            combined,
            dim=1,
        )

        acc = (
            pred == semantic
        ).float().mean()

        accs.append(acc.item())


    print(
        f"{degree:6d} | "
        f"{learned_acc.item():7.2%} | "
        f"{subspace_acc.item():8.2%} | "
        f"{accs[0]:8.2%} | "
        f"{accs[1]:8.2%} | "
        f"{accs[2]:8.2%}"
    )