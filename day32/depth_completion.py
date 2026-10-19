
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(32)

H, W = 48, 64
Y, X = np.mgrid[:H, :W]

# ==========================================
# 1. Synthetic scene
#
# Background: around 4 meters
# Foreground object: around 1.5 meters
# ==========================================

object_mask = (
    (X >= 17) & (X < 47)
    & (Y >= 12) & (Y < 38)
)

depth_gt = 4.0 + .005 * X
depth_gt = np.where(
    object_mask,
    1.5 + .004 * Y,
    depth_gt,
)

rgb = np.zeros((H, W, 3))
rgb[:] = [.15, .35, .75]
rgb[object_mask] = [.85, .25, .15]

# Add texture to the RGB image
rgb += rng.normal(0, .015, rgb.shape)
rgb = np.clip(rgb, 0, 1)


# ==========================================
# 2. Simulated sparse dToF measurements
#
# Only 2% of pixels are observed.
# ==========================================

observed = rng.random((H, W)) < .02

sparse = np.full((H, W), np.nan)
sparse[observed] = depth_gt[observed]

print("Observed:", observed.sum(), "/", H * W)


# ==========================================
# 3. Depth completion
#
# Each output pixel is estimated from
# sparse sensor measurements.
#
# spatial_weight:
#   nearby observations get higher weights
#
# color_weight:
#   observations with similar RGB get
#   higher weights
# ==========================================

def complete(
    sparse,
    rgb,
    spatial_sigma=12,
    color_sigma=None,
    k=20,
):
    coords = np.argwhere(np.isfinite(sparse))
    values = sparse[np.isfinite(sparse)]
    colors = rgb[coords[:, 0], coords[:, 1]]

    query = np.argwhere(
        np.ones((H, W), dtype=bool)
    )

    output = np.zeros((H, W))

    for y, x in query:
        dist2 = (
            (coords[:, 0] - y) ** 2
            + (coords[:, 1] - x) ** 2
        )

        ids = np.argsort(dist2)[:k]

        w = np.exp(
            -dist2[ids] / (2 * spatial_sigma ** 2)
        )

        if color_sigma is not None:
            color_dist2 = np.sum(
                (
                    colors[ids]
                    - rgb[y, x]
                ) ** 2,
                axis=1,
            )

            w *= np.exp(
                -color_dist2 / (2 * color_sigma ** 2)
            )

        output[y, x] = (
            np.sum(w * values[ids])
            / max(w.sum(), 1e-12)
        )

    # Preserve original sensor measurements
    valid = np.isfinite(sparse)
    output[valid] = sparse[valid]

    return output


# ==========================================
# 4. Compare three methods
# ==========================================

nearest = complete(
    sparse, rgb,
    spatial_sigma=1,
    k=1,
)

spatial = complete(
    sparse, rgb,
    spatial_sigma=12,
    k=20,
)

rgb_guided = complete(
    sparse, rgb,
    spatial_sigma=12,
    color_sigma=.18,
    k=20,
)


def rmse(pred, truth):
    return np.sqrt(
        np.mean((pred - truth) ** 2)
    )


print("\n=== DEPTH COMPLETION RMSE (meters) ===")

methods = {
    "Nearest depth": nearest,
    "Spatial interpolation": spatial,
    "RGB-guided interpolation": rgb_guided,
}

for name, prediction in methods.items():
    print(
        f"{name:27s}"
        f"RMSE={rmse(prediction, depth_gt):.4f}"
    )


# ==========================================
# 5. Noisy sensor measurements
# ==========================================

noisy_sparse = sparse.copy()
noisy_sparse[observed] += rng.normal(
    0, .15, observed.sum()
)

print("\n=== NOISY dToF TEST ===")

for name, sigma in [
    ("Spatial interpolation", None),
    ("RGB-guided interpolation", .18),
]:
    pred = complete(
        noisy_sparse,
        rgb,
        color_sigma=sigma,
    )

    print(
        f"{name:27s}"
        f"RMSE={rmse(pred, depth_gt):.4f}"
    )


# ==========================================
# 6. Visualization
# ==========================================

fig, axes = plt.subplots(2, 3, figsize=(12, 7))

images = [
    ("RGB", rgb, None),
    ("Ground Truth Depth", depth_gt, "viridis"),
    ("Sparse dToF", sparse, "viridis"),
    ("Nearest", nearest, "viridis"),
    ("Spatial", spatial, "viridis"),
    ("RGB-guided", rgb_guided, "viridis"),
]

for ax, (title, image, cmap) in zip(
    axes.flat, images
):
    ax.imshow(
        image,
        cmap=cmap,
        vmin=1 if cmap else None,
        vmax=4.5 if cmap else None,
    )
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig("day32_results.png", dpi=150)
plt.show()
