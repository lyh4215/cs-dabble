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
X2, y2 = make_data(90)

X = np.vstack([X1, X2])
labels = np.concatenate([y1, y2])

w_online = np.array([1.0, 0.0])
w_frozen = np.array([1.0, 0.0])

lr = 0.002

online_correct = []
frozen_correct = []
fixed_correct = []

print(
    "step | online direction      | "
    "fixed | frozen | online"
)
print("-" * 75)

for i, (x, true_expert) in enumerate(
    zip(X, labels),
    start=1,
):

    # fixed x-axis
    pred_fixed = int(x[0] < 0)

    # train 때 배운 subspace 그대로
    pred_frozen = int(
        np.dot(w_frozen, x) < 0
    )

    # 계속 업데이트되는 subspace
    pred_online = int(
        np.dot(w_online, x) < 0
    )

    fixed_correct.append(
        pred_fixed == true_expert
    )

    frozen_correct.append(
        pred_frozen == true_expert
    )

    online_correct.append(
        pred_online == true_expert
    )

    # 최근 50개만 보기
    for arr in [
        fixed_correct,
        frozen_correct,
        online_correct,
    ]:
        if len(arr) > 50:
            arr.pop(0)

    # online만 GHA update
    proj = np.dot(w_online, x)

    w_online += lr * proj * (
        x - proj * w_online
    )

    w_online /= np.linalg.norm(
        w_online
    )

    if i % 50 == 0:

        print(
            f"{i:4d} | "
            f"[{w_online[0]: .3f}, "
            f"{w_online[1]: .3f}] | "
            f"{np.mean(fixed_correct):6.1%} | "
            f"{np.mean(frozen_correct):6.1%} | "
            f"{np.mean(online_correct):6.1%}"
        )