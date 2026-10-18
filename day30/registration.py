
import numpy as np

rng = np.random.default_rng(17)


# ==========================================
# 1. 3D Rotation
# ==========================================

def rotation(degrees):
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)

    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])


# ==========================================
# 2. Kabsch Algorithm
#
# Given corresponding points A and B,
# find R and t minimizing:
#
# ||A @ R + t - B||²
# ==========================================

def kabsch(source, destination):
    a = source.mean(axis=0)
    b = destination.mean(axis=0)

    centered_a = source - a
    centered_b = destination - b

    H = centered_a.T @ centered_b

    U, _, Vt = np.linalg.svd(H)

    D = np.eye(3)
    D[-1, -1] = np.linalg.det(U @ Vt)

    R = U @ D @ Vt
    t = b - a @ R

    return R, t


# ==========================================
# 3. ICP
#
# 1. Transform points
# 2. Find nearest neighbors
# 3. Apply Kabsch
# 4. Repeat
# ==========================================

def icp(
    source,
    destination,
    steps=40,
    trim=1.0,
    init_R=None,
    init_t=None,
):
    R = (
        np.eye(3)
        if init_R is None
        else init_R.copy()
    )

    t = (
        np.zeros(3)
        if init_t is None
        else init_t.copy()
    )

    history = []

    for _ in range(steps):

        aligned = source @ R + t

        squared = (
            (
                aligned[:, None, :]
                - destination[None, :, :]
            ) ** 2
        ).sum(axis=2)

        closest = squared.argmin(axis=1)

        error = squared[
            np.arange(len(source)),
            closest,
        ]

        # Trimmed ICP:
        # discard high-error correspondences
        k = max(
            3,
            int(trim * len(source)),
        )

        keep = np.argsort(error)[:k]

        dR, dt = kabsch(
            aligned[keep],
            destination[closest[keep]],
        )

        R = R @ dR
        t = t @ dR + dt

        history.append(
            np.sqrt(error[keep].mean())
        )

    return R, t, history


# ==========================================
# 4. Evaluation
# ==========================================

def metrics(R, t, source, true_R, true_t):

    angle = np.rad2deg(
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

    aligned = source @ R + t
    ground_truth = source @ true_R + true_t

    rmse = np.sqrt(
        np.mean(
            np.sum(
                (aligned - ground_truth) ** 2,
                axis=1,
            )
        )
    )

    return angle, translation_error, rmse


# ==========================================
# 5. Create Asymmetric 3D Point Cloud
# ==========================================

source = np.vstack([
    rng.normal(
        [0, 0, 0],
        [0.35, 0.12, 0.08],
        (55, 3),
    ),
    rng.normal(
        [1.7, 0.3, 0.1],
        [0.12, 0.32, 0.15],
        (40, 3),
    ),
    rng.normal(
        [0.1, 1.2, 0.7],
        [0.2, 0.12, 0.35],
        (35, 3),
    ),
])

true_R = rotation(38)
true_t = np.array([0.65, -0.25, 0.35])

full_scan = source @ true_R + true_t

paired = full_scan + rng.normal(
    0,
    0.013,
    full_scan.shape,
)


# ==========================================
# 6. Experiment A: Known Correspondences
# ==========================================

R, t = kabsch(source, paired)

print("\n=== KNOWN CORRESPONDENCES ===")
print(
    "Kabsch:",
    metrics(R, t, source, true_R, true_t),
)


# ==========================================
# 7. Partial Scan
#
# 30 points are missing.
# Point order is shuffled.
# ==========================================

observed = paired[
    rng.choice(
        len(paired),
        100,
        replace=False,
    )
]

observed = observed[
    rng.permutation(len(observed))
]

outliers = rng.uniform(
    [-1.5, -2, -1],
    [3, 2, 2],
    (35, 3),
)

noisy_scan = np.vstack([
    observed,
    outliers,
])


# ==========================================
# 8. Experiments B, C, D
# ==========================================

experiments = [
    ("ICP / partial", observed, 1.0),
    ("ICP / outliers", noisy_scan, 1.0),
    ("Trimmed ICP", noisy_scan, 0.7),
]

for name, scan, trim in experiments:

    R, t, history = icp(
        source,
        scan,
        trim=trim,
    )

    angle, trans, rmse = metrics(
        R, t, source, true_R, true_t
    )

    print(f"\n=== {name} ===")
    print(f"Rotation error: {angle:.2f} deg")
    print(f"Translation error: {trans:.3f}")
    print(f"Alignment RMSE: {rmse:.3f}")
    print(
        f"Nearest-point loss: "
        f"{history[0]:.3f} -> {history[-1]:.3f}"
    )


# ==========================================
# 9. Experiment E: Coarse Initialization
# ==========================================

R, t, history = icp(
    source,
    observed,
    trim=0.8,
    init_R=rotation(25),
    init_t=np.array([0.4, -0.1, 0.2]),
)

angle, trans, rmse = metrics(
    R, t, source, true_R, true_t
)

print("\n=== ICP WITH INITIALIZATION ===")
print(f"Rotation error: {angle:.2f} deg")
print(f"Translation error: {trans:.3f}")
print(f"Alignment RMSE: {rmse:.3f}")
