
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 31
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(min(torch.get_num_threads(), 4))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
H, W = 32, 48

Y, X = torch.meshgrid(
    torch.arange(H),
    torch.arange(W),
    indexing="ij",
)
X = X.float()[None, None]
Y = Y.float()[None, None]


# ==========================================
# 1. Generate synthetic moving scenes
# ==========================================

def batch_scene(batch, seed, split):
    g = torch.Generator().manual_seed(seed)

    def uniform(lo, hi, shape):
        return lo + (hi - lo) * torch.rand(
            shape, generator=g
        )

    dx = torch.randint(
        4, 14, (batch, 1, 1, 1), generator=g
    ).float()

    if split == "test":
        dx = torch.randint(
            12, 19, (batch, 1, 1, 1), generator=g
        ).float()

    left0 = torch.randint(
        4, 13, (batch, 1, 1, 1), generator=g
    ).float()

    cy = torch.randint(
        12, 22, (batch, 1, 1, 1), generator=g
    ).float()

    phase = uniform(0, 6.28, (batch, 1, 1, 1))

    base = torch.cat([
        .25 + .12 * torch.sin(.22 * X + .15 * Y + phase),
        .35 + .10 * torch.cos(.17 * X - .13 * Y + phase),
        .40 + .10 * torch.sin(.11 * Y + .19 * X - phase),
    ], dim=1)

    color = uniform(.3, .85, (batch, 3, 1, 1))

    lightx = uniform(8, 37, (batch, 1, 1, 1))
    lightdx = uniform(-9, 9, (batch, 1, 1, 1))

    def render(t):
        foreground = (
            (X >= left0 + dx * t)
            & (X < left0 + dx * t + 11)
            & (Y >= cy)
            & (Y < cy + 13)
        )

        image = torch.where(
            foreground.expand(-1, 3, -1, -1),
            color.expand(-1, -1, H, W),
            base,
        )

        # Non-geometric light movement:
        # not described by object motion vector.
        glow = .25 * torch.exp(
            -(
                (X - lightx - lightdx * t) ** 2
                + (Y - 7) ** 2
            ) / (2 * 4.0 ** 2)
        )

        image += torch.cat([
            glow, glow * .25, glow * .05
        ], dim=1)

        return image.clamp(0, 1), foreground.float()

    i0, m0 = render(0)
    i1, m1 = render(1)
    midpoint, _ = render(.5)

    # Engine motion vector:
    # foreground displacement from t=0 to t=1.
    half = dx.view(-1).long() // 2

    warped0, warped1 = [], []
    wm0, wm1 = [], []

    for b in range(batch):
        shift = int(half[b])

        warped0.append(
            torch.roll(i0[b] * m0[b], shift, dims=-1)
        )
        warped1.append(
            torch.roll(i1[b] * m1[b], -shift, dims=-1)
        )
        wm0.append(torch.roll(m0[b], shift, dims=-1))
        wm1.append(torch.roll(m1[b], -shift, dims=-1))

    warped0 = torch.stack(warped0)
    warped1 = torch.stack(warped1)
    wm0 = torch.stack(wm0)
    wm1 = torch.stack(wm1)

    velocity = dx.expand(-1, 1, H, W) / W

    # Additional cues available only to MV-guided model
    hints = torch.cat([
        warped0,
        warped1,
        wm0,
        wm1,
        velocity,
    ], dim=1)

    images = torch.cat([i0, i1], dim=1)
    return images, hints, midpoint


# ==========================================
# 2. CNN frame interpolation model
# ==========================================

class CNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(48, 3, 3, padding=1),
        )

    def forward(self, x):
        baseline = .5 * (x[:, :3] + x[:, 3:6])
        return (baseline + self.net(x)).clamp(0, 1)


# ==========================================
# 3. PSNR
# ==========================================

def psnr(prediction, target):
    mse = float(((prediction - target) ** 2).mean())

    return -10 * np.log10(
        max(mse, 1e-12)
    )


# ==========================================
# 4. Training
# ==========================================

def train(name, use_mv, steps=100):

    torch.manual_seed(SEED)

    # Original images: 6 channels
    # Additional hints: 9 channels
    model = CNN(
        15 if use_mv else 6
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3
    )

    for step in range(steps + 1):

        images, hints, target = batch_scene(
            12, 1000 + step, "train"
        )

        if use_mv:
            inputs = torch.cat(
                [images, hints], dim=1
            )
        else:
            inputs = images

        inputs = inputs.to(DEVICE)
        target = target.to(DEVICE)

        prediction = model(inputs)
        loss = F.l1_loss(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 25 == 0:
            print(
                f"{name:15s} "
                f"step={step:3d} "
                f"L1={loss.item():.4f}"
            )

    return model.eval()


# ==========================================
# 5. Evaluate
# ==========================================

print("Device:", DEVICE)

image_model = train(
    "Image-only",
    use_mv=False,
)

guided_model = train(
    "Image + MV",
    use_mv=True,
)

print("\n=== HELD-OUT FRAME INTERPOLATION ===")

for split in ["train", "test"]:

    images, hints, target = batch_scene(
        32, 98765, split
    )

    target = target.to(DEVICE)

    with torch.no_grad():
        baseline = .5 * (
            images[:, :3] + images[:, 3:]
        )

        image_prediction = image_model(
            images.to(DEVICE)
        )

        mv_prediction = guided_model(
            torch.cat([images, hints], dim=1).to(DEVICE)
        )

    print(
        f"{split:5s} "
        f"blend={psnr(baseline, target):.2f} dB  "
        f"image-only={psnr(image_prediction, target):.2f} dB  "
        f"image+MV={psnr(mv_prediction, target):.2f} dB"
    )

print("\nDone.")
