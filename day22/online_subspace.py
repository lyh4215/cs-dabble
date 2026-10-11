import numpy as np

rng = np.random.default_rng(0)

n = 2000


def make_data(degree):

    A = rng.normal(
        loc=[2.0, 0.0],
        scale=0.5,
        size=(n // 2, 2),
    )

    B = rng.normal(
        loc=[-2.0, 0.0],
        scale=0.5,
        size=(n // 2, 2),
    )

    X = np.vstack([A, B])

    labels = np.array(
        [0] * (n // 2)
        + [1] * (n // 2)
    )

    theta = np.deg2rad(degree)

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])

    X = X @ R.T

    # streaming order 섞기
    idx = rng.permutation(n)

    return X[idx], labels[idx]


X1, y1 = make_data(0)
X2, y2 = make_data(60)

X = np.vstack([X1, X2])
labels = np.concatenate([y1, y2])


# --------------------------------------------------
# GHA / Oja principal direction
# --------------------------------------------------

w = np.array([1.0, 0.0])

lr = 0.01


def normalize(v):
    return v / np.linalg.norm(v)


correct_window = []

print(
    "step | direction            | recent routing acc"
)

print("-" * 60)


for i, (x, true_expert) in enumerate(
    zip(X, labels),
    start=1,
):

    # 현재 learned direction으로 routing
    score = np.dot(w, x)

    pred = int(score < 0)

    correct_window.append(
        pred == true_expert
    )

    if len(correct_window) > 200:
        correct_window.pop(0)


    # --------------------------------------------------
    # Oja/GHA update
    # --------------------------------------------------

    proj = np.dot(w, x)

    w += lr * proj * (
        x - proj * w
    )

    w = normalize(w)


    if i % 250 == 0:

        acc = np.mean(
            correct_window
        )

        print(
            f"{i:4d} | "
            f"[{w[0]: .3f}, {w[1]: .3f}] | "
            f"{acc:7.2%}"
        )