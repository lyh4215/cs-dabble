#!/usr/bin/env python3
"""Educational blockwise flow matching: shared model vs time-specialized experts.

Not the official NeurIPS BFM architecture (no semantic feature guidance or FRN).
Uses eight 2D Gaussian clusters, straight conditional flow matching, Euler ODE.
Requires: torch, matplotlib.
"""
import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn


def target_samples(n, device, generator):
    a = torch.arange(8, device=device, dtype=torch.float32) * (math.pi / 4)
    centers = 3 * torch.stack((a.cos(), a.sin()), dim=-1)
    indices = torch.randint(8, (n,), generator=generator, device=device)
    noise = torch.randn(n, 2, generator=generator, device=device)
    return centers[indices] + noise * 0.20


class VelocityMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t):
        return self.net(torch.cat((x, t), dim=-1))


class PiecewiseVelocity(nn.Module):
    def __init__(self, num_segments, hidden):
        super().__init__()
        self.experts = nn.ModuleList(
            [VelocityMLP(hidden) for _ in range(num_segments)]
        )
        self.num_segments = num_segments

    def forward(self, x, t):
        # For a batch with arbitrary t, route each item to the expert for its interval.
        ids = (t[:, 0] * self.num_segments).long().clamp(0, self.num_segments - 1)
        output = torch.zeros_like(x)
        # where avoids in-place assignment on gradient-bearing tensors.
        for i, expert in enumerate(self.experts):
            positions = torch.where(ids == i)[0]
            if positions.numel() > 0:
                output = output.index_add(
                    0, positions, expert(x.index_select(0, positions),
                                         t.index_select(0, positions))
                )
        return output

    def forward_segment(self, x, t, segment_id):
        # Fast route during sampling: the entire batch has the same t.
        return self.experts[segment_id](x, t)


def draw_train_batch(batch, device, generator):
    x0 = torch.randn(batch, 2, generator=generator, device=device)
    x1 = target_samples(batch, device, generator)
    t = torch.rand(batch, 1, generator=generator, device=device)
    xt = (1 - t) * x0 + t * x1
    return xt, t, x1 - x0


def train_model(model, batches, device, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    losses = []
    begin = time.perf_counter()
    for i, (xt, t, v_target) in enumerate(batches):
        loss = (model(xt, t) - v_target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if (i + 1) % max(1, len(batches) // 4) == 0:
            print(f"{model.__class__.__name__} step={i+1}/{len(batches)} "
                  f"loss={losses[-1]:.4f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return losses, time.perf_counter() - begin


@torch.inference_mode()
def generate(model, initial, nfe, device):
    model.eval()
    x = initial.clone()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for k in range(nfe):
        # Midpoint time evaluation and Euler integration.
        time_value = (k + 0.5) / nfe
        t = torch.full((len(x), 1), time_value, device=device)
        if isinstance(model, PiecewiseVelocity):
            expert_id = min(int(time_value * model.num_segments), model.num_segments - 1)
            velocity = model.forward_segment(x, t, expert_id)
        else:
            velocity = model(x, t)
        x = x + velocity / nfe
    if device.type == "cuda":
        torch.cuda.synchronize()
    return x, (time.perf_counter() - start) * 1000


@torch.inference_mode()
def sliced_wasserstein(a, b, projections=128):
    assert len(a) == len(b)
    # Fixed projection vectors for apples-to-apples comparisons.
    gen = torch.Generator(device=a.device).manual_seed(901)
    directions = torch.randn(2, projections, generator=gen, device=a.device)
    directions = directions / directions.norm(dim=0, keepdim=True)
    aa = (a @ directions).sort(dim=0).values
    bb = (b @ directions).sort(dim=0).values
    return float((aa - bb).abs().mean())

def plot_samples(reference, large, small, block, save_to):
    fig, axes = plt.subplots(
        1, 4, figsize=(16, 4), sharex=True, sharey=True
    )

    for ax, points, title in zip(
        axes,
        (reference, large, small, block),
        (
            "Target: 8 Gaussians",
            "Large Single (96)",
            "Small Single (48)",
            "Blockwise (48 x 3)",
        ),
    ):
        z = points.detach().cpu().numpy()
        ax.scatter(z[:, 0], z[:, 1], s=3, alpha=0.4)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

    fig.tight_layout()
    fig.savefig(save_to, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--samples", type=int, default=2048)
    p.add_argument("--nfe", type=int, default=30)
    p.add_argument("--segments", type=int, default=3)
    p.add_argument("--mono-hidden", type=int, default=96)
    p.add_argument("--expert-hidden", type=int, default=48)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("bfm_results"))
    args = p.parse_args()
    if min(args.steps, args.batch, args.samples, args.nfe, args.segments,
           args.mono_hidden, args.expert_hidden) < 1:
        p.error("Counts and model widths must be >= 1")

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator(device=device).manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Both models use the exact same stochastic training batches, in the same order.
    batches = [draw_train_batch(args.batch, device, rng) for _ in range(args.steps)]
    large = VelocityMLP(96).to(device)
    small = VelocityMLP(48).to(device)
    block = PiecewiseVelocity(3, 48).to(device)
    large_losses, large_train = train_model(
        large, batches, device, args.lr
    )

    small_losses, small_train = train_model(
        small, batches, device, args.lr
    )

    block_losses, block_train = train_model(
        block, batches, device, args.lr
    )

    reference = target_samples(args.samples, device, rng)
    initial = torch.randn(args.samples, 2, generator=rng, device=device)
    large_samples, large_ms = generate(
        large, initial, args.nfe, device
    )

    small_samples, small_ms = generate(
        small, initial, args.nfe, device
    )

    block_samples, block_ms = generate(
        block, initial, args.nfe, device
    )

    for name, samples, elapsed in [
        ("large", large_samples, large_ms),
        ("small", small_samples, small_ms),
        ("blockwise", block_samples, block_ms),
    ]:
        swd = sliced_wasserstein(samples, reference)

        print(
            f"{name:10s} "
            f"SWD={swd:.4f} "
            f"latency={elapsed:.2f}ms"
        )
    picture = args.output_dir / "samples.png"
    plot_samples(
        reference,
        large_samples,
        small_samples,
        block_samples,
        picture,
    )

    metrics = {
        "setting": vars(args) | {
            "output_dir": str(args.output_dir),
            "device": str(device),
        },
        "large_single": {
            "params_total": sum(p.numel() for p in large.parameters()),
            "params_active": sum(p.numel() for p in large.parameters()),
            "swd": sliced_wasserstein(large_samples, reference),
            "sample_ms": large_ms,
            "train_s": large_train,
            "recent_train_loss": sum(large_losses[-20:]) / min(20, len(large_losses)),
        },
        "small_single": {
            "params_total": sum(p.numel() for p in small.parameters()),
            "params_active": sum(p.numel() for p in small.parameters()),
            "swd": sliced_wasserstein(small_samples, reference),
            "sample_ms": small_ms,
            "train_s": small_train,
            "recent_train_loss": sum(small_losses[-20:]) / min(20, len(small_losses)),
        },
        "blockwise": {
            "params_total": sum(p.numel() for p in block.parameters()),
            "params_active": sum(p.numel() for p in block.experts[0].parameters()),
            "swd": sliced_wasserstein(block_samples, reference),
            "sample_ms": block_ms,
            "train_s": block_train,
            "recent_train_loss": sum(block_losses[-20:]) / min(20, len(block_losses)),
        },
        "warning": (
            "Toy educational time-segmentation only; "
            "not official BFM (no SFG/FRN)."
        ),
    }
    out = args.output_dir / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(json.dumps(metrics, indent=2, default=str))
    print(f"saved image: {picture}")
    print(f"saved metrics: {out}")


if __name__ == "__main__":
    main()
