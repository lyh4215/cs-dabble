import numpy as np

rng = np.random.default_rng(0)

d = 16
n_old = 4000
n_new = 4000


# --------------------------------------------------
# 1. pretrained weight W0 만들기
# --------------------------------------------------

U, _ = np.linalg.qr(rng.normal(size=(d, d)))
V, _ = np.linalg.qr(rng.normal(size=(d, d)))

singular_values = np.linspace(6.0, 0.5, d)

W0 = U @ np.diag(singular_values) @ V.T


# --------------------------------------------------
# 2. old task
# --------------------------------------------------

X_old = rng.normal(size=(n_old, d))
Y_old = X_old @ W0.T


# --------------------------------------------------
# 3. new task
#
# 가장 작은 singular direction을
# 새 task에서 수정해야 한다고 가정
# --------------------------------------------------

u = U[:, -1]
v = V[:, -1]

target_strength = 2.0

delta_target = target_strength * np.outer(u, v)

W_new = W0 + delta_target


X_new = rng.normal(size=(n_new, d))
Y_new = X_new @ W_new.T


def mse(a, b):
    return np.mean((a - b) ** 2)


print("pretrained singular values:")
print(np.round(singular_values, 2))

print("\nalpha | adapter_norm | old_loss | new_loss")
print("-" * 48)


# --------------------------------------------------
# 4. adapter singular value를 점점 키워본다
# --------------------------------------------------

for alpha in [
    0.0,
    0.25,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
]:

    delta = alpha * np.outer(u, v)

    W = W0 + delta

    pred_old = X_old @ W.T
    pred_new = X_new @ W.T

    old_loss = mse(pred_old, Y_old)
    new_loss = mse(pred_new, Y_new)

    adapter_norm = np.linalg.norm(delta, ord=2)

    print(
        f"{alpha:5.2f} |"
        f"{adapter_norm:13.3f} |"
        f"{old_loss:9.5f} |"
        f"{new_loss:9.5f}"
    )