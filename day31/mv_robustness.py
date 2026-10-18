
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(31)
torch.set_num_threads(min(torch.get_num_threads(), 4))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
H, W = 32, 48

yy, xx = torch.meshgrid(
    torch.arange(H),
    torch.arange(W),
    indexing="ij",
)
X, Y = xx.float()[None, None], yy.float()[None, None]


# ==========================================
# 1. Backward warping
# ==========================================

def warp(image, flow):
    """
    image: [B, 3, H, W]
    flow:  [B, 2, H, W]

    Flow는 출력 픽셀에서 원본 이미지의
    어느 좌표를 읽을지 나타냄.
    """
    b, _, h, w = image.shape

    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=image.device),
        torch.linspace(-1, 1, w, device=image.device),
        indexing="ij",
    )

    grid = torch.stack(
        (gx, gy), dim=-1
    )[None].expand(b, -1, -1, -1).clone()

    grid[..., 0] += 2 * flow[:, 0] / (w - 1)
    grid[..., 1] += 2 * flow[:, 1] / (h - 1)

    return F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


# ==========================================
# 2. Synthetic scene generation
# ==========================================

def make_batch(n, seed, split):
    rng = torch.Generator().manual_seed(seed)

    def rand(low, high, shape):
        return low + (high - low) * torch.rand(
            shape, generator=rng
        )

    shape = (n, 1, 1, 1)

    dx = (
        rand(4, 13, shape)
        if split == "train"
        else rand(13, 20, shape)
    )

    x0 = rand(3, 11, shape)
    cy = rand(10, 19, shape)
    phase = rand(0, 6.28, shape)
    light_x = rand(9, 35, shape)
    light_dx = rand(-11, 11, shape)
    color = rand(.25, .85, (n, 3, 1, 1))

    bg = torch.cat([
        .22 + .09 * torch.sin(.17 * X + .13 * Y + phase),
        .32 + .08 * torch.cos(.14 * X - .11 * Y + phase),
        .38 + .09 * torch.sin(.08 * Y + .15 * X - phase),
    ], dim=1)

    def render(t):
        x = x0 + t * dx

        mask = (
            (X >= x)
            & (X < x + 11)
            & (Y >= cy)
            & (Y < cy + 12)
        ).float()

        image = torch.where(
            mask.bool().expand(-1, 3, -1, -1),
            color.expand(-1, -1, H, W),
            bg,
        )

        glow = .30 * torch.exp(
            -(
                (X - light_x - t * light_dx) ** 2
                + (Y - 6.5) ** 2
            ) / (2 * 3.8 ** 2)
        )

        image = image + torch.cat([
            glow,
            .18 * glow,
            .04 * glow,
        ], dim=1)

        return image.clamp(0, 1), mask

    i0, m0 = render(0.0)
    i1, m1 = render(1.0)
    target, _ = render(0.5)

    halfway = torch.cat([
        dx.expand(-1, 1, H, W) / 2,
        torch.zeros(n, 1, H, W),
    ], dim=1)

    mask_mid_0 = warp(m0, -halfway)
    mask_mid_1 = warp(m1, +halfway)

    midmask = (
        (mask_mid_0 + mask_mid_1) * .5
    ).clamp(0, 1)

    guide0 = torch.cat([
        -dx.expand(-1, 1, H, W) / 2 * midmask,
        torch.zeros(n, 1, H, W),
    ], dim=1)

    guide1 = torch.cat([
        +dx.expand(-1, 1, H, W) / 2 * midmask,
        torch.zeros(n, 1, H, W),
    ], dim=1)

    w0 = warp(i0, guide0)
    w1 = warp(i1, guide1)

    features = torch.cat([
        i0,
        i1,
        guide0 / W,
        guide1 / W,
        w0,
        w1,
        midmask,
    ], dim=1)

    return features, target, i0, i1, guide0, guide1


# ==========================================
# 3. Neural networks
#
# rgb:
#   RGB residual 3 channels
#
# flow:
#   Bidirectional flow residual 4 channels
#
# flow_mask:
#   Flow residual 4 + blending mask 1
# ==========================================

class Net(nn.Module):
    def __init__(self, kind):
        super().__init__()

        self.kind = kind

        out = {
            "rgb": 3,
            "flow": 4,
            "flow_mask": 5,
        }[kind]

        self.net = nn.Sequential(
            nn.Conv2d(17, 24, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(24, 24, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(24, 24, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(24, out, 3, padding=1),
        )

        # Start from engine-guided warping.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, i0, i1, guide0, guide1):
        raw = self.net(features)

        if self.kind == "rgb":
            baseline = (
                warp(i0, guide0)
                + warp(i1, guide1)
            ) * .5

            return (baseline + raw).clamp(0, 1)

        # Predict residual flow, maximum about 2 px.
        residual = 2.0 * torch.tanh(raw[:, :4])

        f0 = guide0 + residual[:, :2]
        f1 = guide1 + residual[:, 2:4]

        w0 = warp(i0, f0)
        w1 = warp(i1, f1)

        if self.kind == "flow":
            alpha = .5
        else:
            alpha = torch.sigmoid(raw[:, 4:5])

        return (
            alpha * w0
            + (1 - alpha) * w1
        ).clamp(0, 1)


# ==========================================
# 4. Evaluation metric
# ==========================================

def score(pred, target):
    mse = F.mse_loss(pred, target).item()
    return -10 * math.log10(max(mse, 1e-12))


# ==========================================
# 5. Train on CLEAN motion vectors
# ==========================================

def train(kind, steps=80):
    torch.manual_seed(31)

    model = Net(kind).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=.001,
    )

    for step in range(steps + 1):
        features, target, i0, i1, g0, g1 = make_batch(
            8, 1000 + step, "train"
        )

        features, target, i0, i1, g0, g1 = [
            a.to(DEVICE)
            for a in (
                features, target, i0, i1, g0, g1
            )
        ]

        prediction = model(
            features, i0, i1, g0, g1
        )

        loss = F.l1_loss(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(
                f"{kind:10s} "
                f"step={step:3d} "
                f"L1={loss.item():.4f}"
            )

    return model.eval()


print("Device:", DEVICE)

models = {
    kind: train(kind)
    for kind in ("rgb", "flow", "flow_mask")
}


# ==========================================
# 6. Original clean-MV evaluation
# ==========================================

print("\n=== HELD-OUT PSNR (dB) ===")

for split in ("train", "test"):
    feat, gt, i0, i1, g0, g1 = make_batch(
        32, 98765, split
    )

    feat, gt, i0, i1, g0, g1 = [
        a.to(DEVICE)
        for a in (feat, gt, i0, i1, g0, g1)
    ]

    with torch.no_grad():
        blend = .5 * (i0 + i1)

        guided = .5 * (
            warp(i0, g0) + warp(i1, g1)
        )

        values = {
            "blend": score(blend, gt),
            "guided": score(guided, gt),
        }

        for name, net in models.items():
            values[name] = score(
                net(feat, i0, i1, g0, g1),
                gt,
            )

    print(
        split,
        " ".join(
            f"{k}={v:.2f}"
            for k, v in values.items()
        ),
    )


# ==========================================
# 7. Corrupt motion vector inputs
#
# clean:
#   Original engine-derived flow
#
# noise:
#   Random flow noise (~1.5 pixels)
#
# missing:
#   Remove ~50% of nonzero motion vectors
#
# bias:
#   Add +3 pixels of X displacement
# ==========================================

def corrupted_features(
    i0, i1, g0, g1,
    corruption,
    seed=42,
):
    gen = torch.Generator(
        device=i0.device
    ).manual_seed(seed)

    q0 = g0.clone()
    q1 = g1.clone()

    if corruption == "noise":
        for q in (q0, q1):
            moving = (
                q[:, :1].abs() > .01
            ).float()

            q += (
                1.5
                * torch.randn(
                    q.shape,
                    generator=gen,
                    device=q.device,
                )
                * moving
            )

    elif corruption == "missing":
        for q in (q0, q1):
            keep = (
                torch.rand(
                    q[:, :1].shape,
                    generator=gen,
                    device=q.device,
                ) > .5
            ).float()

            q *= keep

    elif corruption == "bias":
        for q in (q0, q1):
            moving = (
                q[:, :1].abs() > .01
            ).float()

            q[:, :1] += 3.0 * moving

    elif corruption != "clean":
        raise ValueError(corruption)

    # Rebuild guidance from corrupted flows.
    mask = (
        (q0[:, :1].abs() > .01)
        | (q1[:, :1].abs() > .01)
    ).float()

    w0 = warp(i0, q0)
    w1 = warp(i1, q1)

    feat = torch.cat([
        i0,
        i1,
        q0 / W,
        q1 / W,
        w0,
        w1,
        mask,
    ], dim=1)

    return feat, q0, q1


# ==========================================
# 8. Robustness evaluation
#
# No retraining on corrupted motion vectors.
# ==========================================

print("\n=== MOTION VECTOR ROBUSTNESS: HELD-OUT TEST ===")

print(
    f'{"condition":10s} '
    f'{"guided":>9s} '
    f'{"RGB":>9s} '
    f'{"flow":>9s} '
    f'{"flow+mask":>11s}'
)

_, gt, i0, i1, g0, g1 = make_batch(
    32, 98765, "test"
)

gt, i0, i1, g0, g1 = [
    z.to(DEVICE)
    for z in (gt, i0, i1, g0, g1)
]

for condition in (
    "clean",
    "noise",
    "missing",
    "bias",
):
    feat, c0, c1 = corrupted_features(
        i0, i1, g0, g1,
        condition,
    )

    with torch.no_grad():
        outputs = [
            .5 * (
                warp(i0, c0)
                + warp(i1, c1)
            ),

            models["rgb"](
                feat, i0, i1, c0, c1
            ),

            models["flow"](
                feat, i0, i1, c0, c1
            ),

            models["flow_mask"](
                feat, i0, i1, c0, c1
            ),
        ]

        scores = [
            score(output, gt)
            for output in outputs
        ]

    print(
        f"{condition:10s} "
        + " ".join(
            f"{s:9.2f}"
            for s in scores
        )
    )

print(
    "\nModels were trained only with CLEAN motion vectors."
)
