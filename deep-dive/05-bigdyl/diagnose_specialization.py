"""BigDyL Deep Dive 05: PCA-feature attribution and expert specialization.
Run next to moe_shift.py. Python 3.11; requires torch, numpy.
"""
import argparse
import csv
import numpy as np
import torch
from moe_shift import (D, DOMAINS, MoE, Router, data, evaluate,
                       pca_basis, set_seed, train)


def coordinate_basis(indices):
    basis = torch.eye(D)[:, indices].clone()
    return basis


def basis_energy(basis, idx):
    # Each PCA dimension has unit norm; quantity is its fraction of squared
    # loading mass in a coordinate group, across all selected dimensions.
    return basis[idx].square().sum().item() / basis.square().sum().item()


def variants(training_x, dims):
    center = training_x.mean(0)
    yield 'linear', 'linear', None, None
    yield 'mlp', 'mlp', None, None
    for k in dims:
        basis, pca_center = pca_basis(training_x, k)
        yield f'pca{k}', 'subspace', basis, pca_center
    # Diagnostic oracle: features 3,4,5 were deliberately constructed as
    # label-relevant. Not an unsupervised or fair deployable competitor.
    yield 'oracle_task3', 'subspace', coordinate_basis([3, 4, 5]), center


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=35)
    ap.add_argument('--train-n', type=int, default=2400)
    ap.add_argument('--test-n', type=int, default=1200)
    ap.add_argument('--shift', type=float, default=0.7)
    ap.add_argument('--dims', type=int, nargs='+', default=[2, 4, 8])
    ap.add_argument('--out', default='specialization_results.csv')
    args = ap.parse_args()
    rows = []
    for seed in range(args.seed, args.seed + args.seeds):
        set_seed(seed)
        tr = data(args.train_n, seed)
        test = data(args.test_n, 10000 + seed)
        shifted = data(args.test_n, 10000 + seed, shift=args.shift)
        for name, kind, basis, center in variants(tr[0], args.dims):
            torch.manual_seed(seed)
            model = MoE(Router(kind, basis, center))
            train(model, tr, args.epochs, 128, .002, .1)
            for split, ds in [('iid', test), ('shifted', shifted)]:
                torch.manual_seed(30000 + seed)
                metrics = evaluate(model, ds)
                row = dict(seed=seed, model=name, split=split, acc=metrics['acc'],
                           stability=metrics['stability'], min_util=metrics['min_util'],
                           domain_mi=metrics['domain_mi'],
                           pca_task_energy=basis_energy(basis, slice(3, 6)) if basis is not None else '',
                           pca_nuisance_energy=basis_energy(basis, slice(6, 12)) if basis is not None else '')
                rows.append(row)
                print(f'{seed} {name:12} {split:7} acc={row["acc"]:.3f} '
                      f'DomainMI={row["domain_mi"]:.3f} '
                      f'task-energy={str(row["pca_task_energy"])[:5]} '
                      f'nuisance-energy={str(row["pca_nuisance_energy"])[:5]}')
                if split == 'iid' and seed == args.seed:
                    print('  P(expert | domain):')
                    for i, probs in enumerate(metrics['domain_routing']):
                        print(f'    domain {i}: ' + ' '.join(f'{p:.2f}' for p in probs))
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('\nSummary (mean across seeds)')
    for name in dict.fromkeys(r['model'] for r in rows):
        for split in ('iid', 'shifted'):
            g = [r for r in rows if r['model'] == name and r['split'] == split]
            print(f'{name:12} {split:7} acc={np.mean([r["acc"] for r in g]):.3f} '
                  f'MI={np.mean([r["domain_mi"] for r in g]):.3f} '
                  f'stability={np.mean([r["stability"] for r in g]):.3f}')
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
