
import numpy as np

rng = np.random.default_rng(17)


# ==========================================
# 1. Rotation and Kabsch
# ==========================================

def rotation(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)

    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])


def kabsch(A, B):
    am = A.mean(axis=0)
    bm = B.mean(axis=0)

    U, _, Vt = np.linalg.svd(
        (A - am).T @ (B - bm)
    )

    D = np.eye(3)
    D[-1, -1] = np.linalg.det(U @ Vt)

    R = U @ D @ Vt
    t = bm - am @ R

    return R, t


# ==========================================
# 2. Nearest neighbor search
# ==========================================

def nearest(A, B):
    squared = (
        (A[:, None, :] - B[None, :, :]) ** 2
    ).sum(axis=2)

    return (
        squared.argmin(axis=1),
        np.sqrt(squared.min(axis=1)),
    )


# ==========================================
# 3. Rotation-invariant descriptors
#
# Each point is represented by distances
# to its 12 nearest neighbors.
# ==========================================

def descriptors(points, k=12):

    distances = np.sqrt(
        (
            (
                points[:, None, :]
                - points[None, :, :]
            ) ** 2
        ).sum(axis=2)
    )

    distances.sort(axis=1)

    return distances[:, 1:k + 1]


# ==========================================
# 4. Feature correspondence matching
#
# Use mutual nearest neighbors
# in feature space.
# ==========================================

def correspondences(A, B, k=12):

    fa = descriptors(A, k)
    fb = descriptors(B, k)

    squared = (
        (fa[:, None, :] - fb[None, :, :]) ** 2
    ).sum(axis=2)

    ab = squared.argmin(axis=1)
    ba = squared.argmin(axis=0)

    pairs = np.array([
        (i, j)
        for i, j in enumerate(ab)
        if ba[j] == i
    ], dtype=int)

    return pairs


# ==========================================
# 5. RANSAC
#
# Randomly select 3 correspondences.
# Estimate transform using Kabsch.
# Count consistent correspondences.
# Keep the transform with most inliers.
# ==========================================

def ransac(
    A,
    B,
    pairs,
    trials=3000,
    threshold=0.09,
):

    if len(pairs) < 3:
        raise ValueError(
            "Not enough correspondences"
        )

    aa = A[pairs[:, 0]]
    bb = B[pairs[:, 1]]

    best = np.zeros(
        len(pairs),
        dtype=bool,
    )

    best_R = np.eye(3)
    best_t = np.zeros(3)

    for _ in range(trials):

        ids = rng.choice(
            len(pairs),
            3,
            replace=False,
        )

        sample = aa[ids]

        if np.linalg.matrix_rank(
            sample - sample.mean(axis=0)
        ) < 2:
            continue

        R, t = kabsch(
            aa[ids],
            bb[ids],
        )

        errors = np.linalg.norm(
            aa @ R + t - bb,
            axis=1,
        )

        inliers = errors < threshold

        if inliers.sum() > best.sum():
            best = inliers
            best_R = R
            best_t = t

    if best.sum() >= 3:
        best_R, best_t = kabsch(
            aa[best],
            bb[best],
        )

    return best_R, best_t, int(best.sum())


# ==========================================
# 6. ICP refinement
# ==========================================

def icp(
    A,
    B,
    R=None,
    t=None,
    steps=40,
    trim=0.8,
):

    R = np.eye(3) if R is None else R.copy()
    t = np.zeros(3) if t is None else t.copy()

    for _ in range(steps):

        aligned = A @ R + t

        indices, distances = nearest(
            aligned,
            B,
        )

        count = int(trim * len(A))
        keep = np.argsort(distances)[:count]

        dR, dt = kabsch(
            aligned[keep],
            B[indices[keep]],
        )

        R = R @ dR
        t = t @ dR + dt

    return R, t


# ==========================================
# 7. Evaluation
# ==========================================

def report(label, R, t):

    angle = np.degrees(
        np.arccos(
            np.clip(
                (np.trace(R.T @ true_R) - 1) / 2,
                -1,
                1,
            )
        )
    )

    translation_error = np.linalg.norm(
        t - true_t
    )

    print(
        f"{label:27s} "
        f"rotation={angle:7.2f}° "
        f"translation={translation_error:.4f}"
    )


# ==========================================
# 8. Generate synthetic 3D object
# ==========================================

A = np.vstack([
    rng.normal(
        [0, 0, 0],
        [.35, .12, .08],
        (55, 3),
    ),
    rng.normal(
        [1.7, .3, .1],
        [.12, .32, .15],
        (40, 3),
    ),
    rng.normal(
        [.1, 1.2, .7],
        [.2, .12, .35],
        (35, 3),
    ),
])

true_R = rotation(38)
true_t = np.array([.65, -.25, .35])

B = (
    A @ true_R
    + true_t
    + rng.normal(0, .008, A.shape)
)

# Shuffle point order
B = B[rng.permutation(len(B))]


# ==========================================
# 9. Full scan comparison
# ==========================================

print(
    "=== Full scan, unknown correspondences ==="
)

R0, t0 = icp(A, B)
report("ICP from identity", R0, t0)

pairs = correspondences(A, B)

print(
    "Mutual feature matches:",
    len(pairs),
)

R1, t1, inliers = ransac(
    A,
    B,
    pairs,
)

print("RANSAC inliers:", inliers)
report("Features + RANSAC", R1, t1)

R2, t2 = icp(A, B, R1, t1)
report("RANSAC + ICP", R2, t2)


# ==========================================
# 10. Partial scan
# ==========================================

Bpartial = B[
    rng.choice(
        len(B),
        100,
        replace=False,
    )
]

print(
    "\n=== Partial scan: 100 of 130 points ==="
)

pairs = correspondences(
    A,
    Bpartial,
)

print(
    "Mutual feature matches:",
    len(pairs),
)

if len(pairs) >= 3:

    R3, t3, inliers = ransac(
        A,
        Bpartial,
        pairs,
        threshold=.12,
    )

    print("RANSAC inliers:", inliers)

    report(
        "Partial Features+RANSAC",
        R3,
        t3,
    )

    R4, t4 = icp(
        A,
        Bpartial,
        R3,
        t3,
    )

    report(
        "Partial RANSAC+ICP",
        R4,
        t4,
    )

print(
    "\nNo neural network was trained."
)
