import torch

torch.manual_seed(0)

n = 4000

# --------------------------------------------------
# 1. semantic은 x축으로 구분
#
# A: x ≈ -1
# B: x ≈ +1
#
# 그런데 y variance를 아주 크게 만든다.
# → PCA는 y축을 principal direction으로 잡음
# --------------------------------------------------

A = torch.empty(n // 2, 2)
B = torch.empty(n // 2, 2)

A[:, 0] = torch.randn(n // 2) * 0.25 - 1.0
B[:, 0] = torch.randn(n // 2) * 0.25 + 1.0

A[:, 1] = torch.randn(n // 2) * 4.0
B[:, 1] = torch.randn(n // 2) * 4.0

X = torch.cat([A, B], dim=0)

semantic = torch.cat([
    torch.zeros(n // 2, dtype=torch.long),
    torch.ones(n // 2, dtype=torch.long),
])


# --------------------------------------------------
# 2. learned router
# --------------------------------------------------

router = torch.nn.Linear(2, 2)

optimizer = torch.optim.Adam(
    router.parameters(),
    lr=0.03,
)

for _ in range(500):

    logits = router(X)

    loss = torch.nn.functional.cross_entropy(
        logits,
        semantic,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


with torch.no_grad():

    learned_logits = router(X)

    learned_pred = torch.argmax(
        learned_logits,
        dim=1,
    )

    learned_acc = (
        learned_pred == semantic
    ).float().mean()


# --------------------------------------------------
# 3. PCA / subspace router
# --------------------------------------------------

centered = X - X.mean(dim=0)

cov = (
    centered.T @ centered
    / (len(X) - 1)
)

eigenvalues, eigenvectors = torch.linalg.eigh(
    cov
)

v = eigenvectors[:, -1]

print("covariance:")
print(cov)

print("\nprincipal direction:")
print(v)

print("\neigenvalues:")
print(eigenvalues)


# PCA projection만 가지고 expert 결정
score = X @ v

# 부호 orientation은 어느 쪽이든 가능하므로
# 둘 다 계산해서 더 좋은 쪽을 사용한다.
pred1 = (score >= 0).long()
pred2 = 1 - pred1

acc1 = (
    pred1 == semantic
).float().mean()

acc2 = (
    pred2 == semantic
).float().mean()

subspace_acc = max(
    acc1.item(),
    acc2.item(),
)


# --------------------------------------------------
# 4. learned + subspace mixed score
# --------------------------------------------------

# orientation 선택
if acc2 > acc1:
    score = -score

structure_logits = torch.stack(
    [
        -score,
        score,
    ],
    dim=1,
)

# scale normalize
L = learned_logits / learned_logits.std()
S = structure_logits / structure_logits.std()


print("\n--- Accuracy ---")

print(
    f"learned only : "
    f"{learned_acc.item():.2%}"
)

print(
    f"subspace only: "
    f"{subspace_acc:.2%}"
)


for beta in [
    0.0,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    1.0,
]:

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

    print(
        f"beta={beta:4.2f}: "
        f"{acc.item():.2%}"
    )