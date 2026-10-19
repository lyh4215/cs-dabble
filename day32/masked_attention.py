
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(32)
torch.set_num_threads(min(4, torch.get_num_threads()))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
H = W = 32

Y, X = torch.meshgrid(
    torch.arange(H), torch.arange(W),
    indexing="ij"
)
X = X.float()[None, None]
Y = Y.float()[None, None]


# ==========================================
# 1. Synthetic RGB + sparse depth
# ==========================================

def make_data(batch, seed, scenario="mixed"):
    g = torch.Generator().manual_seed(seed)

    def rand(a, b, shape):
        return a + (b - a) * torch.rand(
            shape, generator=g
        )

    shape = (batch, 1, 1, 1)

    left = rand(3, 12, shape)
    top = rand(3, 12, shape)
    width = rand(8, 17, shape)
    height = rand(8, 17, shape)

    obj = (
        (X >= left) & (X < left + width)
        & (Y >= top) & (Y < top + height)
    )

    bg_depth = rand(3.4, 5.2, shape)
    fg_depth = rand(1.0, 2.8, shape)

    gt = torch.where(
        obj,
        fg_depth + .006 * Y,
        bg_depth + .004 * X,
    )

    bg_color = rand(.15, .85, (batch, 3, 1, 1))
    fg_color = rand(.15, .85, (batch, 3, 1, 1))

    if scenario == "mixed":
        mode = seed % 4
    else:
        mode = {
            "aligned": 0,
            "texture": 1,
            "camouflage": 2,
            "both": 3,
        }[scenario]

    if mode in (2, 3):
        fg_color = bg_color.clone()

    rgb = torch.where(
        obj.expand(-1, 3, -1, -1),
        fg_color,
        bg_color,
    ).expand(-1, -1, H, W).clone()

    if mode in (1, 3):
        stripes = ((X // 4) % 2).bool()
        stripe_color = .12 + .75 * (1 - bg_color)

        rgb = torch.where(
            stripes.expand(-1, 3, -1, -1),
            stripe_color,
            rgb,
        )

    rgb += .02 * torch.randn(
        rgb.shape, generator=g
    )
    rgb = rgb.clamp(0, 1)

    # Sparse dToF: about 3% valid measurements
    valid = (
        torch.rand(gt.shape, generator=g) < .03
    ).float()

    valid[:, :, 0, 0] = 1.0

    sparse = gt * valid

    return rgb, sparse, valid, gt


# ==========================================
# 2. CNN baselines
# ==========================================

class CNN(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 3. Tiny Joint Attention
#
# RGB feature -> Queries
# Sparse depth -> Keys, Values
#
# Masked version ignores depth tokens
# without sensor observations.
# ==========================================

class JointAttention(nn.Module):
    def __init__(self, use_mask):
        super().__init__()

        self.use_mask = use_mask

        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.ReLU(),
        )

        self.q = nn.Linear(26, 32)
        self.k = nn.Linear(4, 32)
        self.v = nn.Linear(4, 32)

        self.decoder = nn.Sequential(
            nn.Conv2d(56, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )

        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, 8),
            torch.linspace(-1, 1, 8),
            indexing="ij",
        )

        pos = torch.stack(
            [gx, gy], dim=-1
        ).reshape(1, 64, 2)

        self.register_buffer("pos", pos)

    def forward(self, rgb, depth, valid):
        batch = rgb.shape[0]

        rgb_feature = self.rgb_encoder(rgb)

        # 32x32 -> 8x8 RGB tokens
        rgb_tokens = F.avg_pool2d(
            rgb_feature, 4
        )

        # Count-aware sparse depth pooling
        counts = F.avg_pool2d(valid, 4)

        depth_sum = F.avg_pool2d(depth, 4)

        depth_tokens = depth_sum / (
            counts + 1e-6
        )

        token_valid = (
            counts.reshape(batch, 64) > 0
        )

        position = self.pos.expand(
            batch, -1, -1
        )

        # Queries: RGB + position
        rgb_flat = rgb_tokens.flatten(
            2
        ).transpose(1, 2)

        query_input = torch.cat(
            [rgb_flat, position], dim=-1
        )

        # Keys / Values:
        # depth + observation density + position
        depth_input = torch.cat([
            depth_tokens.flatten(2).transpose(1, 2) / 5,
            counts.flatten(2).transpose(1, 2),
            position,
        ], dim=-1)

        Q = self.q(query_input)
        K = self.k(depth_input)
        V = self.v(depth_input)

        scores = (
            Q @ K.transpose(1, 2)
        ) / math.sqrt(32)

        # The ablation's critical difference
        if self.use_mask:
            scores = scores.masked_fill(
                ~token_valid[:, None, :],
                -1e4,
            )

        attention = torch.softmax(
            scores, dim=-1
        )

        context = attention @ V

        context = context.transpose(
            1, 2
        ).reshape(batch, 32, 8, 8)

        context = F.interpolate(
            context,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        fused = torch.cat(
            [rgb_feature, context], dim=1
        )

        return self.decoder(fused)


# ==========================================
# 4. Create models
# ==========================================

def create_model(kind):
    if kind == "rgb":
        return CNN(3)

    if kind == "concat":
        return CNN(5)

    if kind == "unmasked":
        return JointAttention(False)

    if kind == "masked":
        return JointAttention(True)

    raise ValueError(kind)


def predict(model, kind, rgb, depth, valid):
    if kind == "rgb":
        return model(rgb)

    if kind == "concat":
        return model(torch.cat([
            rgb, depth / 5, valid
        ], dim=1))

    return model(rgb, depth, valid)


# ==========================================
# 5. Training
# ==========================================

def train(kind, steps=160):
    torch.manual_seed(32)

    model = create_model(kind).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=.002
    )

    for step in range(steps + 1):
        rgb, depth, valid, gt = make_data(
            16, 1000 + step
        )

        rgb, depth, valid, gt = [
            x.to(DEVICE)
            for x in (rgb, depth, valid, gt)
        ]

        prediction = predict(
            model, kind, rgb, depth, valid
        )

        loss = F.l1_loss(prediction, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 40 == 0:
            print(
                f"{kind:8s} "
                f"step={step:3d} "
                f"MAE={loss.item():.4f}"
            )

    return model.eval()


print("Device:", DEVICE)

models = {}

for name in (
    "rgb",
    "concat",
    "unmasked",
    "masked",
):
    models[name] = train(name)


# ==========================================
# 6. Unseen test scenes
# ==========================================

print("\n=== UNSEEN DEPTH RMSE (meters) ===")

names = list(models)

print(
    f'{"Scene":12s}'
    + "".join(f"{name:>12s}" for name in names)
)

scenarios = (
    "aligned",
    "texture",
    "camouflage",
    "both",
)

for index, scenario in enumerate(scenarios):

    rgb, depth, valid, gt = make_data(
        48, 5551 + index, scenario
    )

    rgb, depth, valid, gt = [
        x.to(DEVICE)
        for x in (rgb, depth, valid, gt)
    ]

    results = []

    with torch.no_grad():
        for name, model in models.items():

            prediction = predict(
                model, name,
                rgb, depth, valid
            )

            error = F.mse_loss(
                prediction, gt
            ).sqrt().item()

            results.append(error)

    print(
        f"{scenario:12s}"
        + "".join(
            f"{value:12.3f}" for value in results
        )
    )

print("\nDone.")
