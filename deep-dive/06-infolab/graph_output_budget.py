"""InfoLab-inspired variable-output GPU memory budgeting experiment.

Not a reproduction of INFINEL/GFlux. It compares output allocation strategies
for a synthetic graph expansion where row i emits degree[i] integers.
Requires: pip install torch numpy. CUDA optional.
"""
import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import torch


def make_degrees(n: int, total: int, pattern: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if pattern == 'uniform':
        weights = np.ones(n)
    elif pattern == 'skew':
        # 2% of vertices get 85% of the total expansion work.
        weights = np.full(n, 0.15 / n)
        heavy = rng.choice(n, max(1, n // 50), replace=False)
        weights[heavy] += 0.85 / len(heavy)
    elif pattern == 'powerlaw':
        weights = rng.zipf(1.6, n).astype(float)
        weights = np.minimum(weights, 10000)
    else:
        raise ValueError(pattern)
    return rng.multinomial(total, weights / weights.sum()).astype(np.int64)


def sync(device: str) -> None:
    if device.startswith('cuda'):
        torch.cuda.synchronize()


def count_checksum(items: torch.Tensor) -> tuple[int, int]:
    # Exact integer checksum for indices 0 <= index < n.
    return int(items.numel()), int(items.sum(dtype=torch.int64).item())


def expand_all(degrees: torch.Tensor, budget: int):
    total = int(degrees.sum().item())
    if total > budget:
        return {'status': 'OOM-budget', 'chunks': 0, 'peak': total,
                'utilization': float('nan'), 'checksum': None}
    out = torch.repeat_interleave(torch.arange(len(degrees), device=degrees.device), degrees)
    num, checksum = count_checksum(out)
    return {'status': 'ok', 'chunks': 1, 'peak': num,
            'utilization': num / budget, 'checksum': checksum}


def expand_fixed(degrees: torch.Tensor, budget: int, rows_per_batch: int):
    chunks, peak, util_sum, checksum = 0, 0, 0.0, 0
    for start in range(0, len(degrees), rows_per_batch):
        sub = degrees[start:start + rows_per_batch]
        required = int(sub.sum().item())
        peak = max(peak, required)
        if required > budget:
            return {'status': 'OOM-budget', 'chunks': chunks,
                    'peak': peak, 'utilization': float('nan'), 'checksum': None}
        out = torch.repeat_interleave(
            torch.arange(start, start + len(sub), device=degrees.device), sub)
        _, s = count_checksum(out)
        checksum += s
        util_sum += len(out) / budget
        chunks += 1
    return {'status': 'ok', 'chunks': chunks, 'peak': peak,
            'utilization': util_sum / chunks, 'checksum': checksum}


def expand_output_budget(degrees: torch.Tensor, budget: int):
    """Chunk by OUTPUT positions, even when one row emits > budget items."""
    prefix = torch.cat([torch.zeros(1, dtype=torch.long, device=degrees.device),
                        degrees.cumsum(0)])
    total = int(prefix[-1].item())
    chunks, checksum, peak = 0, 0, 0
    for start in range(0, total, budget):
        end = min(total, start + budget)
        # Searchsorted maps output offsets into source rows; repeated prefix
        # entries are zero-degree rows and correctly skipped with right=True.
        offsets = torch.arange(start, end, device=degrees.device, dtype=torch.long)
        rows = torch.searchsorted(prefix[1:].contiguous(), offsets, right=True)
        _, s = count_checksum(rows)
        checksum += s
        peak = max(peak, len(rows))
        chunks += 1
    return {'status': 'ok', 'chunks': chunks, 'peak': peak,
            'utilization': total / (chunks * budget), 'checksum': checksum}


def measure(fn, device: str, repeats: int):
    durations = []
    result = None
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        result = fn()
        sync(device)
        durations.append((time.perf_counter() - t0) * 1000)
    return result, statistics.median(durations)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--rows', type=int, default=4000)
    ap.add_argument('--outputs', type=int, default=80000)
    ap.add_argument('--budget', type=int, default=8192)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--csv', default='infolab_output_results.csv')
    args = ap.parse_args()
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA device unavailable; try --device cpu')
    if args.budget <= 0 or args.outputs <= 0 or args.rows <= 0:
        raise ValueError('rows, outputs and budget must be positive')
    print(f'device={device}, rows={args.rows}, outputs={args.outputs}, '
          f'output_budget={args.budget} int64 elements (~{8*args.budget/1024:.1f} KiB)')
    records = []
    for pattern in ('uniform', 'skew', 'powerlaw'):
        deg_np = make_degrees(args.rows, args.outputs, pattern, args.seed)
        deg = torch.as_tensor(deg_np, device=device)
        oracle = int(np.dot(np.arange(args.rows, dtype=np.int64), deg_np))
        max_degree = int(deg_np.max())
        print(f'\n{pattern}: max_degree={max_degree}, oracle_checksum={oracle}')
        methods = [('all-at-once', lambda: expand_all(deg, args.budget))]
        for b in (16, 64, 256):
            methods.append((f'fixed-rows-{b}', lambda b=b: expand_fixed(deg, args.budget, b)))
        methods.append(('output-budgeted', lambda: expand_output_budget(deg, args.budget)))
        for name, fn in methods:
            r, ms = measure(fn, device, args.repeats)
            ok = r['status'] == 'ok' and r['checksum'] == oracle
            if r['status'] == 'ok' and not ok:
                raise AssertionError(f'{pattern}/{name}: checksum mismatch')
            print(f"{name:18} status={r['status']:10} chunks={r['chunks']:4} "
                  f"peak={r['peak']:7} avg_util={r['utilization']:.3f} "
                  f"median_ms={ms:.2f}")
            records.append({'device': device, 'pattern': pattern, 'method': name,
                            'rows': args.rows, 'outputs': args.outputs,
                            'budget': args.budget, 'max_degree': max_degree,
                            'status': r['status'], 'chunks': r['chunks'],
                            'peak': r['peak'], 'utilization': r['utilization'],
                            'median_ms': ms, 'correct': ok})
    with open(args.csv, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=records[0].keys())
        wr.writeheader()
        wr.writerows(records)
    print(f'\nSaved {Path(args.csv).resolve()}')


if __name__ == '__main__':
    main()
