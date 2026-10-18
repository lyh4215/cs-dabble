
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Synthetic scene
# ==========================================

H, W = 96, 144
Y, X = np.mgrid[:H, :W]

# Static background texture
bg = np.stack([
    .16 + .08 * np.sin(X * .14) * np.cos(Y * .11),
    .24 + .06 * np.cos(X * .09 + Y * .07),
    .25 + .08 * np.sin(Y * .13),
], axis=-1)

bg = np.clip(bg, 0, 1)


# Moving geometric object
def sprite_mask(t):
    left = int(round(24 + 48 * t))

    return (
        (X >= left)
        & (X < left + 29)
        & (Y >= 42)
        & (Y < 82)
    )


# Animated light effect
# Not captured by the geometry motion vector
def glow(t):
    cx = 34 + 50 * t

    return .7 * np.exp(
        -(
            (X - cx) ** 2
            + (Y - 21) ** 2
        ) / (2 * 7 ** 2)
    )


def render(t):
    image = bg.copy()
    image[..., 0] += glow(t)
    image = np.clip(image, 0, 1)

    mask = sprite_mask(t)
    local_x = X - (24 + 48 * t)

    image[mask, 0] = .13
    image[mask, 1] = (
        .65 + .13 * np.sin(local_x[mask] * .4)
    )
    image[mask, 2] = .9

    return np.clip(image, 0, 1)


# ==========================================
# 2. Render two endpoints and ground truth
# ==========================================

I0 = render(0.0)
GT = render(0.5)
I1 = render(1.0)

m0 = sprite_mask(0.0)
mh = sprite_mask(0.5)
m1 = sprite_mask(1.0)


# ==========================================
# 3. Baseline A: Linear blending
#
# Simply average two images
# ==========================================

blend = (I0 + I1) / 2


# ==========================================
# 4. Forward warping
#
# Shift foreground pixels based on
# geometry motion vectors
# ==========================================

def translate_foreground(
    image,
    source_mask,
    pixels,
):
    result = np.full_like(
        image,
        np.nan,
    )

    yy, xx = np.where(source_mask)
    nx = xx + pixels

    valid = (nx >= 0) & (nx < W)

    result[
        yy[valid],
        nx[valid],
    ] = image[
        yy[valid],
        xx[valid],
    ]

    return result


# Object moves 48 pixels from t=0 to t=1.
# At t=0.5, displacement is 24 pixels.

fg0 = translate_foreground(
    I0, m0, +24
)

fg1 = translate_foreground(
    I1, m1, -24
)


# ==========================================
# 5. Baseline B: Engine motion vector only
#
# Move foreground from frame 0.
# Missing background becomes black.
# ==========================================

mv_only = I0.copy()

# Foreground leaves behind an unknown area
mv_only[m0] = 0

# Warped object at target time
mv_only[mh] = fg0[mh]


# ==========================================
# 6. Baseline C: Bilateral geometry
#
# Recover exposed background from
# either endpoint when possible.
# ==========================================

background = np.zeros_like(I0)

# Background visible in frame 0
background[~m0] = I0[~m0]

# Hidden in frame 0, visible in frame 1
background[m0 & ~m1] = I1[m0 & ~m1]

# Hidden in both endpoints:
# here we use frame 0 as fallback
background[m0 & m1] = I0[m0 & m1]

bilateral = background.copy()

# Combine two warped foreground observations
bilateral[mh] = (
    fg0[mh] + fg1[mh]
) / 2


# ==========================================
# 7. Synthetic appearance-aware correction
#
# Estimate animated light center from
# the image context.
#
# Educational upper-bound:
# we KNOW static background and light shape.
# This is NOT the paper's learned estimator.
# ==========================================

def glow_center(image):
    residual = np.maximum(
        image[..., 0] - bg[..., 0],
        0,
    )

    # Light occupies the upper part
    residual[42:, :] = 0

    return (
        (residual * X).sum()
        / residual.sum()
    )


c0 = glow_center(I0)
c1 = glow_center(I1)

center = (c0 + c1) / 2

appearance_correction = .7 * np.exp(
    -(
        (X - center) ** 2
        + (Y - 21) ** 2
    ) / (2 * 7 ** 2)
)

corrected = bilateral.copy()

# Replace the original t=0 light
# with the inferred midpoint light.
valid = ~mh

new_red = np.clip(
    background[..., 0]
    - glow(0)
    + appearance_correction,
    0,
    1,
)

corrected[..., 0][valid] = new_red[valid]


# ==========================================
# 8. Quantitative evaluation
# ==========================================

def psnr(image, ground_truth):
    mse = np.mean(
        (image - ground_truth) ** 2
    )

    return -10 * np.log10(
        max(mse, 1e-12)
    )


methods = {
    "Linear blend": blend,
    "Engine MV only": mv_only,
    "Bilateral geometry": bilateral,
    "Bilateral + image context": corrected,
}

print("\n=== FRAME GENERATION ===")

print(
    f"Glow center: "
    f"{c0:.2f} -> {c1:.2f}, "
    f"midpoint={center:.2f}"
)

for name, image in methods.items():
    mae = np.mean(
        np.abs(image - GT)
    )

    print(
        f"{name:27s} "
        f"PSNR={psnr(image, GT):6.2f} dB "
        f"MAE={mae:.4f}"
    )


# ==========================================
# 9. Visualization
# ==========================================

fig, axes = plt.subplots(
    2, 3,
    figsize=(13, 8),
)

images = [
    ("Rendered t=0", I0),
    ("Rendered t=0.5 (GT)", GT),
    ("Rendered t=1", I1),
    ("Linear blend", blend),
    ("Engine MV only", mv_only),
    ("Bilateral + image context", corrected),
]

for ax, (name, image) in zip(
    axes.flat,
    images,
):
    ax.imshow(image, vmin=0, vmax=1)
    ax.set_title(name)
    ax.axis("off")

fig.tight_layout()

fig.savefig(
    "day31_frame_comparison.png",
    dpi=145,
)

plt.show()

print(
    "\nSaved: day31_frame_comparison.png"
)
