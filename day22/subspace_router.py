import numpy as np

rng = np.random.default_rng(0)

n = 2000

# --------------------------------------------------
# semantic groups
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


def rotate(X, degree):
    theta = np.deg2rad(degree)

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])

    return X @ R.T


# --------------------------------------------------
# fixed router
# --------------------------------------------------

def fixed_router(X):
    return (X[:, 0] < 0).astype(int)


# --------------------------------------------------
# subspace-aware router
#
# 현재 데이터의 principal direction을 찾는다.
# --------------------------------------------------

def principal_direction(X):

    X_centered = X - X.mean(axis=0)

    cov = np.cov(
        X_centered,
        rowvar=False,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 가장 큰 eigenvalue의 eigenvector
    v = eigenvectors[:, np.argmax(eigenvalues)]

    return v


previous_v = None

print(
    "degree | fixed | subspace | principal direction"
)

print("-" * 65)


for degree in [
    0,
    15,
    30,
    45,
    60,
    75,
    90,
]:

    X_rot = rotate(X, degree)

    # -------------------------
    # fixed router
    # -------------------------

    pred_fixed = fixed_router(X_rot)

    fixed_acc = np.mean(
        pred_fixed == true_expert
    )


    # -------------------------
    # current principal axis
    # -------------------------

    v = principal_direction(X_rot)

    # eigenvector의 부호는 ± 둘 다 같은 축을 의미함.
    # 이전 방향과 연속되도록 sign 정렬.
    if previous_v is None:

        if v[0] < 0:
            v = -v

    else:

        if np.dot(v, previous_v) < 0:
            v = -v

    previous_v = v.copy()


    # -------------------------
    # learned axis로 projection
    # -------------------------

    score = X_rot @ v

    pred_subspace = (
        score < 0
    ).astype(int)

    subspace_acc = np.mean(
        pred_subspace
        == true_expert
    )


    print(
        f"{degree:6d} | "
        f"{fixed_acc:6.2%} | "
        f"{subspace_acc:8.2%} | "
        f"[{v[0]: .3f}, {v[1]: .3f}]"
    )