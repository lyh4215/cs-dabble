import numpy as np

rng = np.random.default_rng(0)

n = 2000


# --------------------------------------------------
# 1. 두 semantic group
# --------------------------------------------------

A = rng.normal(
    loc=[2.0, 0.0],
    scale=0.5,
    size=(n, 2),
)

B = rng.normal(
    loc=[-2.0, 0.0],
    scale=0.5,
    size=(n, 2),
)

X = np.vstack([A, B])

true_expert = np.array(
    [0] * n + [1] * n
)


# --------------------------------------------------
# 2. 단순 linear router
#
# x축 부호만 보고 expert 결정
# --------------------------------------------------

def linear_router(X):
    return (X[:, 0] < 0).astype(int)


pred = linear_router(X)

base_acc = np.mean(
    pred == true_expert
)


# --------------------------------------------------
# 3. representation을 회전시킨다
# --------------------------------------------------

def rotate(X, degree):

    theta = np.deg2rad(degree)

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])

    return X @ R.T


print(
    "degree | routing accuracy"
)

print("-" * 30)


for degree in [
    0,
    15,
    30,
    45,
    60,
    75,
    90,
]:

    X_rot = rotate(
        X,
        degree,
    )

    pred = linear_router(
        X_rot
    )

    acc = np.mean(
        pred == true_expert
    )

    print(
        f"{degree:6d} | "
        f"{acc:7.2%}"
    )