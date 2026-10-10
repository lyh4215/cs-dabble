import numpy as np

rng = np.random.default_rng(0)

d = 16
n = 5000


# --------------------------------------------------
# pretrained W0
# --------------------------------------------------

U, _ = np.linalg.qr(rng.normal(size=(d, d)))
V, _ = np.linalg.qr(rng.normal(size=(d, d)))

s = np.linspace(6.0, 0.5, d)

W0 = U @ np.diag(s) @ V.T


# --------------------------------------------------
# old-task data
#
# top singular direction v1을 많이 사용하도록 만든다.
# --------------------------------------------------

coeff = rng.normal(
    scale=np.linspace(3.0, 0.2, d),
    size=(n, d),
)

X_old = coeff @ V.T

Y_old = X_old @ W0.T


def mse(a, b):
    return np.mean((a - b) ** 2)


# --------------------------------------------------
# 같은 spectral norm을 가진 두 update
# --------------------------------------------------

alpha = 2.0

# 중요한 top direction 변경
delta_top = alpha * np.outer(U[:, 0], V[:, 0])

# 덜 사용되는 bottom direction 변경
delta_bottom = alpha * np.outer(U[:, -1], V[:, -1])


print("spectral norm top:")
print(np.linalg.norm(delta_top, ord=2))

print("spectral norm bottom:")
print(np.linalg.norm(delta_bottom, ord=2))


# --------------------------------------------------
# 기존 task 손실
# --------------------------------------------------

W_top = W0 + delta_top
W_bottom = W0 + delta_bottom

old_loss_top = mse(
    X_old @ W_top.T,
    Y_old,
)

old_loss_bottom = mse(
    X_old @ W_bottom.T,
    Y_old,
)


print("\nold loss after TOP-direction update:")
print(old_loss_top)

print("\nold loss after BOTTOM-direction update:")
print(old_loss_bottom)