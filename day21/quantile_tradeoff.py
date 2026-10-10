import numpy as np

rng = np.random.default_rng(0)

d = 16
n = 5000


# --------------------------------------------------
# 1. pretrained W0
# --------------------------------------------------

U, _ = np.linalg.qr(rng.normal(size=(d, d)))
V, _ = np.linalg.qr(rng.normal(size=(d, d)))

base_s = np.linspace(6.0, 0.5, d)

W0 = U @ np.diag(base_s) @ V.T


# --------------------------------------------------
# 2. old task
#
# top singular directions를 더 많이 사용하는 데이터
# --------------------------------------------------

old_scales = np.linspace(3.0, 0.2, d)

X_old = (
    rng.normal(size=(n, d))
    * old_scales
) @ V.T

Y_old = X_old @ W0.T


# --------------------------------------------------
# 3. new task가 원하는 update
#
# 여러 singular direction을 동시에 수정
# --------------------------------------------------

directions = [0, 4, 9, 15]

target_s = np.array([
    4.0,
    3.0,
    2.0,
    1.0,
])


delta_target = np.zeros((d, d))

for idx, strength in zip(
    directions,
    target_s,
):
    delta_target += (
        strength
        * np.outer(
            U[:, idx],
            V[:, idx],
        )
    )


W_target = W0 + delta_target


X_new = rng.normal(size=(n, d))
Y_new = X_new @ W_target.T


def mse(a, b):
    return np.mean((a - b) ** 2)


# --------------------------------------------------
# 4. quantile sweep
# --------------------------------------------------

qs = [
    0.0,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    1.0,
]


print(
    "q    threshold   clipped_singular_values"
    "              old_loss    new_loss"
)

print("-" * 90)


for q in qs:

    threshold = np.quantile(
        base_s,
        q,
    )

    clipped_s = np.minimum(
        target_s,
        threshold,
    )

    delta = np.zeros((d, d))

    for idx, strength in zip(
        directions,
        clipped_s,
    ):

        delta += (
            strength
            * np.outer(
                U[:, idx],
                V[:, idx],
            )
        )

    W = W0 + delta

    old_loss = mse(
        X_old @ W.T,
        Y_old,
    )

    new_loss = mse(
        X_new @ W.T,
        Y_new,
    )

    print(
        f"{q:3.2f}   "
        f"{threshold:8.3f}   "
        f"{np.round(clipped_s, 2)}   "
        f"{old_loss:10.5f}   "
        f"{new_loss:10.5f}"
    )