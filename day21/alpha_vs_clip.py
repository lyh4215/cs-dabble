import numpy as np

rng = np.random.default_rng(0)

d = 16
n = 5000


# --------------------------------------------------
# pretrained W0
# --------------------------------------------------

U, _ = np.linalg.qr(rng.normal(size=(d, d)))
V, _ = np.linalg.qr(rng.normal(size=(d, d)))

base_s = np.linspace(6.0, 0.5, d)
W0 = U @ np.diag(base_s) @ V.T


# --------------------------------------------------
# target LoRA update
#
# singular values = [20, 4, 1]
# --------------------------------------------------

dirs = [0, 7, 15]
target_s = np.array([20.0, 4.0, 1.0])

delta_target = np.zeros((d, d))

for idx, strength in zip(dirs, target_s):
    delta_target += (
        strength
        * np.outer(U[:, idx], V[:, idx])
    )

W_target = W0 + delta_target


# --------------------------------------------------
# old task:
# 첫 번째 방향에 특히 민감하도록 구성
# --------------------------------------------------

old_scales = np.ones(d) * 0.05
old_scales[0] = 3.0

X_old = (
    rng.normal(size=(n, d))
    * old_scales
) @ V.T

Y_old = X_old @ W0.T


# --------------------------------------------------
# new task:
# 세 방향 모두 필요
# --------------------------------------------------

X_new = rng.normal(size=(n, d))
Y_new = X_new @ W_target.T


def mse(a, b):
    return np.mean((a - b) ** 2)


# --------------------------------------------------
# Method 1: alpha scaling
# top 20 -> 5가 되도록 전체 scaling
# --------------------------------------------------

cap = 5.0

alpha = cap / target_s.max()

scaled_s = target_s * alpha


# --------------------------------------------------
# Method 2: spectral clipping
# --------------------------------------------------

clipped_s = np.minimum(target_s, cap)


def make_delta(singular_values):
    delta = np.zeros((d, d))

    for idx, strength in zip(
        dirs,
        singular_values,
    ):
        delta += (
            strength
            * np.outer(
                U[:, idx],
                V[:, idx],
            )
        )

    return delta


delta_alpha = make_delta(scaled_s)
delta_clip = make_delta(clipped_s)


W_alpha = W0 + delta_alpha
W_clip = W0 + delta_clip


# --------------------------------------------------
# losses
# --------------------------------------------------

old_alpha = mse(
    X_old @ W_alpha.T,
    Y_old,
)

old_clip = mse(
    X_old @ W_clip.T,
    Y_old,
)

new_alpha = mse(
    X_new @ W_alpha.T,
    Y_new,
)

new_clip = mse(
    X_new @ W_clip.T,
    Y_new,
)


print("target singular values:")
print(target_s)

print("\nalpha scaling:")
print("alpha =", alpha)
print("singular values =", scaled_s)

print("\nspectral clipping:")
print("cap =", cap)
print("singular values =", clipped_s)


print("\n--- Loss ---")

print(
    "alpha scaling  ",
    f"old_loss={old_alpha:.5f}",
    f"new_loss={new_alpha:.5f}",
)

print(
    "spectral clip  ",
    f"old_loss={old_clip:.5f}",
    f"new_loss={new_clip:.5f}",
)