import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(31)
torch.set_num_threads(min(torch.get_num_threads(), 4))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
H, W = 32, 48
yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
X, Y = xx.float()[None, None], yy.float()[None, None]


def warp(image, flow):
    """flow: (B,2,H,W), target-to-source displacement in pixels."""
    b, _, h, w = image.shape
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=image.device),
        torch.linspace(-1, 1, w, device=image.device),
        indexing="ij")
    grid = torch.stack((gx, gy), -1)[None].expand(b, -1, -1, -1).clone()
    grid[..., 0] += 2 * flow[:, 0] / (w - 1)
    grid[..., 1] += 2 * flow[:, 1] / (h - 1)
    return F.grid_sample(image, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)


def make_batch(n, seed, split):
    rng = torch.Generator().manual_seed(seed)
    def rand(low, high, shape):
        return low + (high - low) * torch.rand(shape, generator=rng)
    shape = (n, 1, 1, 1)
    dx = rand(4, 13, shape) if split == "train" else rand(13, 20, shape)
    x0 = rand(3, 11, shape)
    cy = rand(10, 19, shape)
    phase = rand(0, 6.28, shape)
    light_x = rand(9, 35, shape)
    light_dx = rand(-11, 11, shape)
    color = rand(.25, .85, (n, 3, 1, 1))

    bg = torch.cat([
        .22 + .09 * torch.sin(.17*X + .13*Y + phase),
        .32 + .08 * torch.cos(.14*X - .11*Y + phase),
        .38 + .09 * torch.sin(.08*Y + .15*X - phase),
    ], dim=1)
    def render(t):
        x = x0 + t*dx
        mask = ((X >= x) & (X < x + 11) &
                (Y >= cy) & (Y < cy + 12)).float()
        image = torch.where(mask.bool().expand(-1,3,-1,-1),
                            color.expand(-1,-1,H,W), bg)
        glow = .30 * torch.exp(-((X-light_x-t*light_dx)**2 +
                                (Y-6.5)**2)/(2*3.8**2))
        image = image + torch.cat([glow, .18*glow, .04*glow], 1)
        return image.clamp(0,1), mask

    i0, m0 = render(0.0)
    i1, m1 = render(1.0)
    target, _ = render(.5)

    # Engine geometry guides: endpoint object masks and its velocity.
    # Midpoint mask is computed from the endpoint mask + known engine dx,
    # not from the ground-truth midpoint image.
    halfway = torch.cat([dx.expand(-1,1,H,W)/2,
                         torch.zeros(n,1,H,W)], dim=1)
    mask_mid_0 = warp(m0, -halfway)
    mask_mid_1 = warp(m1, +halfway)
    midmask = ((mask_mid_0 + mask_mid_1)*.5).clamp(0,1)
    guide0 = torch.cat([-dx.expand(-1,1,H,W)/2 * midmask,
                        torch.zeros(n,1,H,W)], 1)
    guide1 = torch.cat([+dx.expand(-1,1,H,W)/2 * midmask,
                        torch.zeros(n,1,H,W)], 1)
    w0, w1 = warp(i0, guide0), warp(i1, guide1)
    # Shared input for all models: same visual and motion information
    features = torch.cat([i0,i1,guide0/W,guide1/W,w0,w1,midmask],1)
    return features, target, i0, i1, guide0, guide1


class Net(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        out = {"rgb": 3, "flow": 4, "flow_mask": 5}[kind]
        self.net = nn.Sequential(
            nn.Conv2d(17,24,3,padding=1),nn.ReLU(),
            nn.Conv2d(24,24,3,padding=1),nn.ReLU(),
            nn.Conv2d(24,24,3,padding=1),nn.ReLU(),
            nn.Conv2d(24,out,3,padding=1))
        # Start with analytical engine warping; learn residual corrections.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, i0, i1, guide0, guide1):
        raw = self.net(features)
        if self.kind == "rgb":
            # RGB direct prediction, same motion-guided features
            baseline = (warp(i0,guide0) + warp(i1,guide1))*.5
            return (baseline + raw).clamp(0,1)
        residual = 2.0 * torch.tanh(raw[:,:4])
        f0 = guide0 + residual[:,:2]
        f1 = guide1 + residual[:,2:4]
        w0,w1 = warp(i0,f0),warp(i1,f1)
        if self.kind == "flow":
            alpha = .5
        else:
            alpha = torch.sigmoid(raw[:,4:5])
        return (alpha*w0 + (1-alpha)*w1).clamp(0,1)


def score(pred, target):
    mse = F.mse_loss(pred,target).item()
    return -10*math.log10(max(mse,1e-12))


def train(kind, steps=80):
    torch.manual_seed(31)
    model = Net(kind).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),lr=.001)
    for step in range(steps+1):
        features, target, i0,i1,g0,g1 = make_batch(8,1000+step,"train")
        features,target,i0,i1,g0,g1 = [a.to(DEVICE) for a in (features,target,i0,i1,g0,g1)]
        pred = model(features,i0,i1,g0,g1)
        loss = F.l1_loss(pred,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            print(f"{kind:10s} step={step:3d} L1={loss.item():.4f}")
    return model.eval()


print("Device:",DEVICE)
models = {kind:train(kind) for kind in ("rgb","flow","flow_mask")}

print("\n=== HELD-OUT PSNR (dB) ===")
for split in ("train","test"):
    feat,gt,i0,i1,g0,g1 = make_batch(32,98765,split)
    feat,gt,i0,i1,g0,g1 = [a.to(DEVICE) for a in (feat,gt,i0,i1,g0,g1)]
    with torch.no_grad():
        blend = .5*(i0+i1)
        guided = .5*(warp(i0,g0)+warp(i1,g1))
        values = {"blend":score(blend,gt),"guided":score(guided,gt)}
        for name,net in models.items():
            values[name] = score(net(feat,i0,i1,g0,g1),gt)
    print(split, " ".join(f"{k}={v:.2f}" for k,v in values.items()))
print("\nCaveat: synthetic known engine geometry; not the paper's model.")
