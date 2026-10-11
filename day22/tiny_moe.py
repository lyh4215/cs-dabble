import torch

torch.manual_seed(0)

n = 4000

# --------------------------------------------------
# 1. 두 semantic group
# --------------------------------------------------

A = torch.randn(n // 2, 2) * 0.5
A[:, 0] += 2.0

B = torch.randn(n // 2, 2) * 0.5
B[:, 0] -= 2.0

X = torch.cat([A, B], dim=0)

semantic = torch.cat([
    torch.zeros(n // 2, dtype=torch.long),
    torch.ones(n // 2, dtype=torch.long),
])


# --------------------------------------------------
# 2. semantic마다 필요한 함수가 다름
#
# A: y = 2*x0 + 0.5*x1
# B: y = -1*x0 + 3*x1
# --------------------------------------------------

Y = torch.empty(n, 1)

mask_a = semantic == 0
mask_b = semantic == 1

Y[mask_a, 0] = (
    2.0 * X[mask_a, 0]
    + 0.5 * X[mask_a, 1]
)

Y[mask_b, 0] = (
    -1.0 * X[mask_b, 0]
    + 3.0 * X[mask_b, 1]
)


# --------------------------------------------------
# 3. Router + Experts
# --------------------------------------------------

router = torch.nn.Linear(2, 2)

expert0 = torch.nn.Linear(2, 1)
expert1 = torch.nn.Linear(2, 1)

params = (
    list(router.parameters())
    + list(expert0.parameters())
    + list(expert1.parameters())
)

optimizer = torch.optim.Adam(
    params,
    lr=0.03,
)


# --------------------------------------------------
# 4. end-to-end training
# --------------------------------------------------

for step in range(1001):

    logits = router(X)

    probs = torch.softmax(
        logits,
        dim=1,
    )

    y0 = expert0(X)
    y1 = expert1(X)

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

    if step % 200 == 0:

        with torch.no_grad():

            hard_route = torch.argmax(
                probs,
                dim=1,
            )

            a_route = hard_route[mask_a]
            b_route = hard_route[mask_b]

            print(
                f"step={step:4d}",
                f"loss={loss.item():.5f}",
            )

            print(
                " A route:",
                torch.bincount(
                    a_route,
                    minlength=2,
                ).tolist(),
            )

            print(
                " B route:",
                torch.bincount(
                    b_route,
                    minlength=2,
                ).tolist(),
            )


# --------------------------------------------------
# 5. 최종 expert weight
# --------------------------------------------------

print("\nExpert 0 weight:")
print(expert0.weight.detach().numpy())

print("Expert 1 weight:")
print(expert1.weight.detach().numpy())