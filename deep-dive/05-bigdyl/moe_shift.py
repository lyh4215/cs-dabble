"""BigDyL Deep Dive 05: router ablations under covariate shift.

Educational experiment inspired by structure-aware MoE routing, NOT a STAR reproduction.
Python >=3.10; pip install torch numpy
"""
import argparse
import csv
import math
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

D = 12
DOMAINS = 3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def data(n, seed, shift=0., noise=0.):
    """Three latent tasks. Labels invariant to nuisance/domain-identifier shift.

    Coordinates [0:3]: domain indicators; [3:6]: domain-specific predictive signal.
    Coordinates [6:12]: correlated nuisance with high variance.
    """
    rng = np.random.default_rng(seed)
    d = rng.integers(0, DOMAINS, size=n)
    x = rng.normal(0, 0.65, (n, D)).astype('float32')
    x[np.arange(n), d] += 2.5
    z = rng.normal(size=(n, DOMAINS)).astype('float32')
    x[:, 3:6] = z + rng.normal(0, 0.12, (n, DOMAINS))
    # Correlated high-variance nuisance dominates top principal directions.
    u = rng.normal(size=(n, 2)).astype('float32')
    x[:, 6] = 4.0 * u[:, 0]
    x[:, 7] = 3.5 * u[:, 0] + 0.4 * u[:, 1]
    x[:, 8] = 3.7 * u[:, 1]
    x[:, 9] = 2.0 * u[:, 1] + 0.3 * u[:, 0]
    # Fixed covariate shift: make domain indicators less reliable.
    if shift:
        x[:, 0:3] = (1-shift) * x[:, 0:3] + shift * x[:, [1, 2, 0]]
        x[:, 6:10] *= (1. + 0.30 * shift)
    if noise:
        x += rng.normal(0, noise, x.shape).astype('float32')
    targets = (z[np.arange(n), d] > 0).astype('int64')
    return (torch.from_numpy(x), torch.from_numpy(targets), torch.from_numpy(d))


def pca_basis(x, dim):
    centered = x - x.mean(0, keepdim=True)
    # High-variance directions on purpose: a falsifiable inductive bias.
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return vh[:dim].T.contiguous(), x.mean(0)


class Router(nn.Module):
    def __init__(self, kind, basis=None, center=None, hidden=32):
        super().__init__()
        self.kind = kind
        self.linear = nn.Linear(D, DOMAINS)
        if kind == 'mlp':
            self.extra = nn.Sequential(nn.Linear(D, hidden), nn.Tanh(), nn.Linear(hidden, DOMAINS))
        elif kind == 'subspace':
            assert basis is not None and center is not None
            self.register_buffer('basis', basis)
            self.register_buffer('center', center)
            self.extra = nn.Sequential(nn.Linear(basis.shape[1], hidden), nn.Tanh(), nn.Linear(hidden, DOMAINS))
        else:
            self.extra = None

    def forward(self, x):
        logits = self.linear(x)
        if self.kind == 'mlp':
            logits = logits + self.extra(x)
        elif self.kind == 'subspace':
            projected = (x - self.center) @ self.basis
            logits = logits + self.extra(projected.square())
        return logits


class MoE(nn.Module):
    def __init__(self, router):
        super().__init__()
        self.router = router
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, 2))
            for _ in range(DOMAINS)
        ])

    def forward(self, x, temperature=1.0):
        scores = self.router(x)
        probs = F.softmax(scores / temperature, dim=-1)
        idx = scores.argmax(dim=-1)
        hard = F.one_hot(idx, DOMAINS).to(probs.dtype)
        # Straight-through estimator: hard top-1 forward, soft backward.
        weights = hard - probs.detach() + probs
        expert_logits = torch.stack([expert(x) for expert in self.experts], dim=1)
        logits = (expert_logits * weights.unsqueeze(-1)).sum(dim=1)
        return logits, probs, idx


def train(model, ds, epochs, batch_size, lr, balance):
    x, y, _ = ds
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(x)
    for epoch in range(epochs):
        for ids in torch.randperm(n).split(batch_size):
            logits, probs, idx = model(x[ids])
            ce = F.cross_entropy(logits, y[ids])
            # Penalize unbalanced *average soft probability*, not true hard counts.
            avg = probs.mean(0)
            lb = ((avg - 1. / DOMAINS)**2).sum()
            loss = ce + balance * lb
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate(model, ds, jitter=0.15):
    x, y, d = ds
    model.eval()
    logits, probs, idx = model(x)
    clean = (logits.argmax(-1) == y).float().mean().item()
    # Stability: fraction maintaining expert on small isotropic perturbation.
    idx2 = model(x + torch.randn_like(x) * jitter)[2]
    stability = (idx == idx2).float().mean().item()
    utilization = torch.bincount(idx, minlength=DOMAINS).float() / len(idx)
    cond = torch.zeros((DOMAINS, DOMAINS))
    for domain in range(DOMAINS):
        cond[domain] = torch.bincount(idx[d == domain], minlength=DOMAINS).float()
        cond[domain] /= cond[domain].sum().clamp_min(1)
    # NMI(domain, expert) / H(domain), 0 = no domain specialization.
    joint = torch.zeros((DOMAINS, DOMAINS))
    for i in range(DOMAINS):
        for j in range(DOMAINS):
            joint[i, j] = ((d == i) & (idx == j)).float().mean()
    pdom = joint.sum(1, keepdim=True)
    pexp = joint.sum(0, keepdim=True)
    valid = joint > 0
    mi = (joint[valid] * torch.log(joint[valid] / (pdom @ pexp)[valid])).sum()
    hd = -(pdom[pdom > 0] * pdom[pdom > 0].log()).sum()
    normalized_mi = (mi / hd).item() if hd > 0 else 0.
    return {'acc': clean, 'stability': stability, 'min_util': utilization.min().item(),
            'domain_mi': normalized_mi, 'utilization': utilization.tolist(),
            'domain_routing': cond.tolist()}


def run(args):
    records = []
    for seed in range(args.seeds):
        set_seed(args.seed + seed)
        tr = data(args.train_n, args.seed + seed)
        clean = data(args.test_n, 10000 + args.seed + seed)
        shifted = data(args.test_n, 10000 + args.seed + seed, shift=args.shift)
        basis, center = pca_basis(tr[0], args.pca_dim)
        for name in ('linear', 'mlp', 'subspace'):
            # Reset initialization for reproducibility, independent of run order.
            torch.manual_seed(args.seed + seed)
            model = MoE(Router(name, basis, center, hidden=args.hidden))
            train(model, tr, args.epochs, args.batch, args.lr, args.balance)
            for split, ds in [('iid', clean), ('shifted', shifted)]:
                result = evaluate(model, ds, args.jitter)
                row = {'seed': args.seed + seed, 'router': name, 'split': split,
                       **{k: v for k, v in result.items() if k not in ('utilization', 'domain_routing')}}
                records.append(row)
                print(f"seed={seed} {name:8} {split:7} acc={row['acc']:.3f} "
                      f"stable={row['stability']:.3f} min_util={row['min_util']:.3f} "
                      f"domain_mi={row['domain_mi']:.3f}  utilization={result['utilization']}")
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    print(f'\nSaved {args.out}')
    for split in ('iid','shifted'):
        for name in ('linear','mlp','subspace'):
            group = [r for r in records if r['router']==name and r['split']==split]
            print(f"{split:7} {name:8} " + ' '.join(
                f"{m}={np.mean([r[m] for r in group]):.3f}±{np.std([r[m] for r in group]):.3f}"
                for m in ('acc','stability','domain_mi')))


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=35)
    ap.add_argument('--train-n', type=int, default=2400)
    ap.add_argument('--test-n', type=int, default=1200)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.002)
    ap.add_argument('--balance', type=float, default=0.1)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--pca-dim', type=int, default=4)
    ap.add_argument('--shift', type=float, default=0.7)
    ap.add_argument('--jitter', type=float, default=0.15)
    ap.add_argument('--out', type=Path, default=Path('moe_shift_results.csv'))
    args = ap.parse_args()
    if args.pca_dim < 1 or args.pca_dim > D: ap.error('pca-dim must be 1..12')
    if not 0 <= args.shift <= 1: ap.error('shift must be 0..1')
    run(args)

if __name__ == '__main__':
    cli()
