import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)


# ==================================================
# Same visual observation,
# but two equally valid pose labels
#
# 0 deg or 180 deg
# ==================================================

N = 1000

labels = np.random.choice(
    [0.0, np.pi],
    size=N,
)

y = torch.tensor(
    labels,
    dtype=torch.float32,
)


# ==================================================
# 1. Single pose regression
#
# 하나의 theta만 출력
# ==================================================

theta = torch.nn.Parameter(
    torch.tensor(1.0)
)

optimizer = torch.optim.Adam(
    [theta],
    lr=0.03,
)


for step in range(1000):

    loss = (
        (theta - y) ** 2
    ).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


print(
    "=== SINGLE REGRESSION ==="
)

print(
    "predicted angle:",
    theta.item(),
)

print(
    "degrees:",
    np.degrees(
        theta.item()
    ),
)


# ==================================================
# 2. Two-mode distribution
#
# 하나의 답 대신 두 개의 pose hypothesis를 학습
# ==================================================

centers = torch.nn.Parameter(
    torch.tensor([
        0.5,
        2.5,
    ])
)

optimizer = torch.optim.Adam(
    [centers],
    lr=0.03,
)

sigma = 0.15


for step in range(1500):

    # [N, 2]
    dist2 = (
        y[:, None]
        - centers[None, :]
    ) ** 2


    # Gaussian mixture likelihood
    log_prob = torch.logsumexp(
        -dist2
        / (2 * sigma ** 2),
        dim=1,
    )

    loss = (
        -log_prob.mean()
    )


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


print(
    "\n=== MULTIMODAL MODEL ==="
)

for i, center in enumerate(
    centers.detach()
):

    print(
        f"mode {i}:",
        f"{center.item():.3f} rad",
        f"= {np.degrees(center.item()):.1f} deg",
    )