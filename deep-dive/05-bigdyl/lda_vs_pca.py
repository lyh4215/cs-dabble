"""BigDyL Deep Dive 05 — supervised vs unsupervised projection for MoE routing.
Run beside moe_shift.py. Python 3.11. pip install torch numpy
NOT a faithful reproduction of STAR or a deployment-ready supervised router.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from moe_shift import D, DOMAINS, MoE, data, evaluate, pca_basis, set_seed, train


def lda_basis(x: torch.Tensor, labels: torch.Tensor, n_components: int, ridge: float = 0.05):
    """Regularized Fisher LDA, orthonormal directions, training set only.

    Uses covariance whitening for generalized eigenproblem S_B v=lambda S_W v.
    Orthonormalizing retained directions is convenient for projection comparisons,
    but differs from retaining canonical LDA scaling.
    """
    x = x.to(torch.float64)
    center = x.mean(0)
    classes = torch.unique(labels)
    assert 1 <= n_components <= len(classes) - 1
    sw = torch.zeros((D, D), dtype=torch.float64)
    sb = torch.zeros_like(sw)
    for cls in classes:
        group = x[labels == cls]
        mean = group.mean(0)
        delta = group - mean
        sw += delta.T @ delta
        diff = mean - center
        sb += len(group) * torch.outer(diff, diff)
    sw /= len(x)
    sb /= len(x)
    # Ridge makes LDA stable even if nuisance features are correlated.
    sw = sw + ridge * torch.eye(D, dtype=torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(sw)
    whiten = (eigenvectors * eigenvalues.clamp_min(1e-10).rsqrt()) @ eigenvectors.T
    mat = whiten @ sb @ whiten
    _, directions = torch.linalg.eigh(mat)
    raw = whiten @ directions[:, -n_components:].flip(1)
    # Choose an orthonormal basis of the same discriminative subspace.
    q, _ = torch.linalg.qr(raw, mode='reduced')
    return q.to(torch.float32), center.to(torch.float32)


class ProjectedRouter(nn.Module):
    """Identical trainable architecture across PCA and LDA variants.

    For LDA-1 we zero-pad its second projection coordinate: parameter count
    matches 2D variants, but effective information dimension remains one.
    """
    def __init__(self, basis, center, hidden=32):
        super().__init__()
        self.linear = nn.Linear(D, DOMAINS)
        self.register_buffer('basis', basis)
        self.register_buffer('center', center)
        self.extra = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, DOMAINS))

    def forward(self, x):
        proj = (x - self.center) @ self.basis
        if proj.shape[-1] == 1:
            proj = torch.cat((proj, torch.zeros_like(proj)), dim=-1)
        features = torch.cat((proj, proj.square()), dim=-1)
        return self.linear(x) + self.extra(features)


def shifted_dataset(base_ds, kind, shift):
    x, y, d = base_ds
    x = x.clone()
    if kind in ('domain', 'both'):
        original = x[:, :3].clone()
        x[:, :3] = (1 - shift) * original + shift * original[:, [1, 2, 0]]
    if kind in ('nuisance', 'both'):
        x[:, 6:10] *= 1 + 0.3 * shift
    return x, y, d


def run(args):
    records = []
    model_names = ['linear', 'pca1', 'pca2', 'label_lda1', 'domain_lda2']
    for seed in range(args.seed, args.seed + args.seeds):
        set_seed(seed)
        tr = data(args.train_n, seed)
        base = data(args.test_n, 10000 + seed)
        splits = {name: shifted_dataset(base, name, args.shift)
                  for name in ('iid', 'domain', 'nuisance', 'both')}
        pb1, pc = pca_basis(tr[0], 1)
        pb2, _ = pca_basis(tr[0], 2)
        lb, lc = lda_basis(tr[0], tr[1], 1, args.ridge)
        db, dc = lda_basis(tr[0], tr[2], 2, args.ridge)
        variants = {
            'pca1': (pb1, pc), 'pca2': (pb2, pc),
            'label_lda1': (lb, lc), 'domain_lda2': (db, dc),
        }
        for name in model_names:
            torch.manual_seed(seed)
            if name == 'linear':
                from moe_shift import Router
                router = Router('linear')
            else:
                router = ProjectedRouter(*variants[name], hidden=args.hidden)
            model = MoE(router)
            train(model, tr, args.epochs, args.batch, args.lr, args.balance)
            for split, ds in splits.items():
                torch.manual_seed(30000 + seed)
                m = evaluate(model, ds, jitter=args.jitter)
                row = {'seed': seed, 'router': name, 'split': split,
                       'acc': m['acc'], 'domain_mi': m['domain_mi'],
                       'stability': m['stability'], 'min_util': m['min_util']}
                records.append(row)
                print(f'{seed} {name:12} {split:8} acc={m["acc"]:.3f} '
                      f'MI={m["domain_mi"]:.3f} stable={m["stability"]:.3f}')
        print(f'seed {seed} basis squared loadings:')
        for name, (basis, _) in variants.items():
            energies = [basis[s].square().sum().item() / basis.square().sum().item()
                        for s in (slice(0, 3), slice(3, 6), slice(6, 12))]
            print(f'  {name:12} domain={energies[0]:.3f} task={energies[1]:.3f} nuisance={energies[2]:.3f}')
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print('\n=== MEAN ACROSS SEEDS ===')
    for name in model_names:
        for split in splits:
            subset = [r for r in records if r['router'] == name and r['split'] == split]
            print(f'{name:12} {split:8} acc={np.mean([r["acc"] for r in subset]):.3f} '
                  f'MI={np.mean([r["domain_mi"] for r in subset]):.3f}')
    print('Saved:', args.out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=35)
    p.add_argument('--train-n', type=int, default=2400)
    p.add_argument('--test-n', type=int, default=1200)
    p.add_argument('--shift', type=float, default=0.7)
    p.add_argument('--ridge', type=float, default=0.05)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.002)
    p.add_argument('--balance', type=float, default=0.1)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--jitter', type=float, default=0.15)
    p.add_argument('--out', type=Path, default=Path('lda_vs_pca_results.csv'))
    run(p.parse_args())
