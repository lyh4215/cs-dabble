import numpy as np

rng = np.random.default_rng(0)

d = 12
rank = 3
n = 2000
lr = 0.05
steps = 200

# --------------------------------------------------
# pretrained W0
# --------------------------------------------------

U0, _ = np.linalg.qr(rng.normal(size=(d, d)))
V0, _ = np.linalg.qr(rng.normal(size=(d, d)))

base_s = np.linspace(5.0, 0.5, d)

W0 = U0 @ np.diag(base_s) @ V0.T


# --------------------------------------------------
# new task가 원하는 update
#
# 3개 방향을 사용
# --------------------------------------------------

dirs = [0, 5, 11]

target_s = np.array([
    8.0,
    3.0,
    1.0,
])

U = U0[:, dirs]
V = V0[:, dirs]

delta_target = (
    U
    @ np.diag(target_s)
    @ V.T
)

W_target = W0 + delta_target


# --------------------------------------------------
# datasets
# --------------------------------------------------

X_new = rng.normal(size=(n, d))
Y_new = X_new @ W_target.T

# old task는 첫 번째 방향에 특히 민감하게
old_scale = np.ones(d) * 0.1
old_scale[0] = 3.0

X_old = (
    rng.normal(size=(n, d))
    * old_scale
) @ V0.T

Y_old = X_old @ W0.T


def mse(a, b):
    return np.mean((a - b) ** 2)


def make_delta(s):
    return U @ np.diag(s) @ V.T


# --------------------------------------------------
# 세 방식의 singular parameters
# --------------------------------------------------

s_free = np.zeros(rank)
s_alpha = np.zeros(rank)
s_clip = np.zeros(rank)

cap = 3.0
alpha = 0.4


print(
    "step |"
    " free(old,new,s) |"
    " alpha(old,new,s) |"
    " clip(old,new,s)"
)

print("-" * 110)


for step in range(steps + 1):

    # --------------------------------------------------
    # evaluation
    # --------------------------------------------------

    if step % 20 == 0:

        results = []

        for s in [
            s_free,
            s_alpha,
            s_clip,
        ]:

            W = W0 + make_delta(s)

            old_loss = mse(
                X_old @ W.T,
                Y_old,
            )

            new_loss = mse(
                X_new @ W.T,
                Y_new,
            )

            results.append(
                (
                    old_loss,
                    new_loss,
                    s.copy(),
                )
            )

        print(
            f"{step:4d} | "
            f"{results[0][0]:6.3f},"
            f"{results[0][1]:6.3f},"
            f"{np.round(results[0][2], 2)} | "
            f"{results[1][0]:6.3f},"
            f"{results[1][1]:6.3f},"
            f"{np.round(results[1][2], 2)} | "
            f"{results[2][0]:6.3f},"
            f"{results[2][1]:6.3f},"
            f"{np.round(results[2][2], 2)}"
        )

    if step == steps:
        break


    # --------------------------------------------------
    # numerical gradient
    #
    # s만 학습해서 dynamics를 명확하게 본다.
    # --------------------------------------------------

    def grad_for(s):

        grad = np.zeros_like(s)
        eps = 1e-5

        for i in range(rank):

            s1 = s.copy()
            s2 = s.copy()

            s1[i] += eps
            s2[i] -= eps

            W1 = W0 + make_delta(s1)
            W2 = W0 + make_delta(s2)

            l1 = mse(
                X_new @ W1.T,
                Y_new,
            )

            l2 = mse(
                X_new @ W2.T,
                Y_new,
            )

            grad[i] = (
                l1 - l2
            ) / (2 * eps)

        return grad


    # --------------------------------------------------
    # 1. unconstrained
    # --------------------------------------------------

    s_free -= (
        lr
        * grad_for(s_free)
    )


    # --------------------------------------------------
    # 2. global alpha scaling
    #
    # update 전체를 약하게
    # --------------------------------------------------

    g = grad_for(s_alpha)

    s_alpha -= (
        lr
        * alpha
        * g
    )


    # --------------------------------------------------
    # 3. spectral clipping
    # --------------------------------------------------

    s_clip -= (
        lr
        * grad_for(s_clip)
    )

    s_clip = np.clip(
        s_clip,
        0,
        cap,
    )