import torch
import math

torch.manual_seed(0)

n = 4000


# --------------------------------------------------
# 1. semantic A / B
# --------------------------------------------------

A = torch.randn(n // 2, 2) * 0.4
A[:, 0] += 2.0

B = torch.randn(n // 2, 2) * 0.4
B[:, 0] -= 2.0

X = torch.cat([A, B], dim=0)

semantic = torch.cat([
    torch.zeros(n // 2, dtype=torch.long),
    torch.ones(n // 2, dtype=torch.long),
])


# --------------------------------------------------
# 2. experts는 rotation-invariant한 feature를 사용
#
# r = ||x||
#
# A: y =  2r + 0.5
# B: y = -3r + 1
# --------------------------------------------------

r = torch.linalg.norm(
    X,
    dim=1,
    keepdim=True,
)

Y = torch.empty(n, 1)

mask_a = semantic == 0
mask_b = semantic == 1

Y[mask_a] = (
    2.0 * r[mask_a] + 0.5
)

Y[mask_b] = (
    -3.0 * r[mask_b] + 1.0
)


# --------------------------------------------------
# 3. tiny MoE
# --------------------------------------------------

router = torch.nn.Linear(2, 2)

expert0 = torch.nn.Linear(1, 1)
expert1 = torch.nn.Linear(1, 1)

optimizer = torch.optim.Adam(
    list(router.parameters())
    + list(expert0.parameters())
    + list(expert1.parameters()),
    lr=0.03,
)


# --------------------------------------------------
# 4. train
# --------------------------------------------------

for step in range(1001):

    probs = torch.softmax(
        router(X),
        dim=1,
    )

    y0 = expert0(r)
    y1 = expert1(r)

    pred = (
        probs[:, 0:1] * y0
        + probs[:, 1:2] * y1
    )

    loss = torch.mean(
        (pred - Y) ** 2
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# --------------------------------------------------
# 5. 어떤 expert가 A 전문인지 확인
# --------------------------------------------------

with torch.no_grad():

    train_route = torch.argmax(
        router(X),
        dim=1,
    )

    expert_for_a = int(
        torch.mode(
            train_route[mask_a]
        ).values
    )

    expert_for_b = 1 - expert_for_a


print("expert for A:", expert_for_a)
print("expert for B:", expert_for_b)

print()

print(
    "degree | standard_acc | standard_loss "
    "| subspace_acc | subspace_loss"
)

print("-" * 80)


# --------------------------------------------------
# 6. representation rotation
# --------------------------------------------------

previous_v = None

target_route = torch.where(
    semantic == 0,
    torch.tensor(expert_for_a),
    torch.tensor(expert_for_b),
)


for degree in [
    0,
    15,
    30,
    45,
    60,
    75,
    90,
]:

    theta = math.radians(degree)

    R = torch.tensor([
        [
            math.cos(theta),
            -math.sin(theta),
        ],
        [
            math.sin(theta),
            math.cos(theta),
        ],
    ])

    X_rot = X @ R.T

    # norm은 rotation해도 같음
    r_rot = torch.linalg.norm(
        X_rot,
        dim=1,
        keepdim=True,
    )

    y0 = expert0(r_rot)
    y1 = expert1(r_rot)


    # --------------------------------------------------
    # standard learned router
    # --------------------------------------------------

    route_standard = torch.argmax(
        router(X_rot),
        dim=1,
    )

    pred_standard = torch.where(
        (route_standard == 0).unsqueeze(1),
        y0,
        y1,
    )

    standard_acc = torch.mean(
        (
            route_standard
            == target_route
        ).float()
    )

    standard_loss = torch.mean(
        (
            pred_standard - Y
        ) ** 2
    )


    # --------------------------------------------------
    # structure-aware toy router
    #
    # PCA principal direction 사용
    # --------------------------------------------------

    centered = (
        X_rot
        - X_rot.mean(dim=0)
    )

    cov = (
        centered.T @ centered
        / (len(X_rot) - 1)
    )

    eigenvalues, eigenvectors = (
        torch.linalg.eigh(cov)
    )

    v = eigenvectors[:, -1]


    # PCA axis의 ± ambiguity 해결
    if previous_v is None:

        if v[0] < 0:
            v = -v

    else:

        if torch.dot(
            v,
            previous_v,
        ) < 0:
            v = -v

    previous_v = v.clone()


    score = X_rot @ v

    route_subspace = torch.where(
        score >= 0,
        torch.tensor(expert_for_a),
        torch.tensor(expert_for_b),
    )

    pred_subspace = torch.where(
        (route_subspace == 0).unsqueeze(1),
        y0,
        y1,
    )

    subspace_acc = torch.mean(
        (
            route_subspace
            == target_route
        ).float()
    )

    subspace_loss = torch.mean(
        (
            pred_subspace - Y
        ) ** 2
    )


    print(
        f"{degree:6d} | "
        f"{standard_acc.item():12.2%} | "
        f"{standard_loss.item():13.4f} | "
        f"{subspace_acc.item():12.2%} | "
        f"{subspace_loss.item():12.4f}"
    )