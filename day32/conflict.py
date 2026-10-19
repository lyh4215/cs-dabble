
import numpy as np

rng = np.random.default_rng(32)

H, W = 48, 64
Y, X = np.mgrid[:H, :W]

mask = (
    (X >= 17) & (X < 47)
    & (Y >= 12) & (Y < 38)
)

# Ground truth depth
truth = np.where(
    mask,
    1.5 + .004 * Y,
    4.0 + .005 * X,
)

# Sparse sensor observations
observed = rng.random((H, W)) < .02

coords = np.argwhere(observed)
values = truth[observed]

print(
    "Observed:", observed.sum(),
    "/", H * W
)

# All query pixels
queries = np.stack([
    Y.ravel(),
    X.ravel(),
], axis=1)

# Distance to every observed depth point
dist2 = (
    (
        queries[:, None, :]
        - coords[None, :, :]
    ) ** 2
).sum(axis=2)

# 20 nearest sensor measurements
ids = np.argsort(dist2, axis=1)[:, :20]

d = dist2[
    np.arange(H * W)[:, None],
    ids,
]

v = values[ids]


# ==========================================
# Depth completion methods
# ==========================================

def complete(rgb, mode):
    rgb = rgb.reshape(-1, 3)

    observed_colors = rgb[
        coords[:, 0] * W + coords[:, 1]
    ]

    # Spatial similarity
    weights = np.exp(
        -d / (2 * 12 ** 2)
    )

    if mode in ("rgb", "gated"):
        color_dist2 = (
            (
                rgb[:, None, :]
                - observed_colors[ids]
            ) ** 2
        ).sum(axis=2)

        weights *= np.exp(
            -color_dist2 / (2 * .18 ** 2)
        )

    prediction = (
        (weights * v).sum(axis=1)
        / np.maximum(weights.sum(axis=1), 1e-30)
    )

    if mode == "gated":

        # Depth disagreement among nearby
        # sensor measurements
        near = v[:, :8]

        spread = np.sqrt(
            np.mean(
                (
                    near
                    - near.mean(
                        axis=1,
                        keepdims=True,
                    )
                ) ** 2,
                axis=1,
            )
        )

        # If neighboring depth measurements
        # disagree, favor the closest depth.
        #
        # Deliberately simple heuristic:
        # NOT masked joint attention.
        gate = np.clip(
            (spread - .15) / .8,
            0,
            1,
        )

        prediction = (
            (1 - gate) * prediction
            + gate * v[:, 0]
        )

    output = prediction.reshape(H, W)

    # Preserve measured depth
    output[observed] = truth[observed]

    return output


def rmse(prediction):
    return np.sqrt(
        np.mean(
            (prediction - truth) ** 2
        )
    )


# ==========================================
# Four RGB-depth relationships
# ==========================================

for scene in (
    "aligned",
    "texture",
    "camouflage",
    "both",
):
    rgb = np.zeros((H, W, 3))
    rgb[:] = [.15, .35, .75]

    # Real depth boundary matches color
    if scene in ("aligned", "texture"):
        rgb[mask] = [.85, .25, .15]

    # False color boundaries (stripes)
    if scene in ("texture", "both"):
        stripes = (X // 5) % 2 == 0
        rgb[stripes] = [.85, .25, .15]

    # Same RGB color for foreground/background
    if scene == "camouflage":
        rgb[:] = [.30, .40, .55]

    rgb += rng.normal(
        0, .01, rgb.shape
    )
    rgb = np.clip(rgb, 0, 1)

    print(f"\n=== {scene.upper()} ===")

    for mode in (
        "spatial",
        "rgb",
        "gated",
    ):
        result = complete(rgb, mode)

        print(
            f"{mode:10s}",
            f"RMSE={rmse(result):.4f} m"
        )
