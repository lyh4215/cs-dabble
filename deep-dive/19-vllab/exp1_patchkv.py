#!/usr/bin/env python3
"""PatchKV-inspired, CPU NumPy one-block activation matching experiment.

This is NOT the official NeurIPS 2026 implementation. Instead it preserves the
paper's block-level target and closed-form ridge solution in a tiny attention model.
Reference queries are sampled from the context, never from evaluation queries.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Context:
    keys: np.ndarray
    values: np.ndarray


@dataclass
class Block:
    a_q: np.ndarray
    a_z: np.ndarray
    bias: np.ndarray
    w_out: np.ndarray


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def forward(q: np.ndarray, ctx: Context, block: Block, scale: float, patch: np.ndarray | None = None):
    d = q.shape[1]
    z = softmax((q @ ctx.keys.T) * (scale / math.sqrt(d))) @ ctx.values
    h = np.tanh(q @ block.a_q + z @ block.a_z + block.bias)
    # Teacher and student share the model; only cache entries and W patch differ.
    y = z + h @ (block.w_out if patch is None else block.w_out + patch)
    return y, h, z


def make_block(rng: np.random.Generator, d: int, dff: int) -> Block:
    return Block(
        a_q=rng.normal(size=(d, dff)) / math.sqrt(d),
        a_z=rng.normal(size=(d, dff)) / math.sqrt(d),
        bias=rng.normal(size=dff) * .07,
        w_out=rng.normal(size=(dff, d)) / math.sqrt(dff),
    )


def make_context(rng: np.random.Generator, d: int, n: int, n_group: int = 8) -> Context:
    group = rng.integers(0, n_group, size=n)
    centers = rng.normal(size=(n_group, d)) * 1.9
    keys = centers[group] + rng.normal(size=(n, d)) * .85
    base_map = rng.normal(size=(d, d)) / math.sqrt(d)
    # Related keys carry related but context-dependent values.
    group_shift = rng.normal(size=(n_group, d)) * .65
    values = np.tanh(keys @ base_map) + group_shift[group] + rng.normal(size=(n, d)) * .10
    return Context(keys=keys, values=values)


def make_queries(rng: np.random.Generator, ctx: Context, n: int, kind: str) -> np.ndarray:
    if kind == 'in':
        ix = rng.integers(0, len(ctx.keys), size=n)
        return ctx.keys[ix] + rng.normal(size=(n, ctx.keys.shape[1])) * .75
    if kind == 'shift':
        # New query distribution: mixtures of two context keys, not shown at fit.
        a = ctx.keys[rng.integers(0, len(ctx.keys), size=n)]
        b = ctx.keys[rng.integers(0, len(ctx.keys), size=n)]
        return .50 * a + .50 * b + rng.normal(size=a.shape) * 1.8
    raise ValueError(kind)


def compressed(ctx: Context, ratio: float, rng: np.random.Generator) -> Context:
    n = max(1, int(round(len(ctx.keys) * ratio)))
    idx = np.sort(rng.choice(len(ctx.keys), size=n, replace=False))
    return Context(ctx.keys[idx], ctx.values[idx])


def ridge_stats(h: np.ndarray, targets: np.ndarray, lam0: float, chunk: int) -> tuple[np.ndarray, dict]:
    """Solve min ||H @ patch - targets||^2 + lambda ||patch||^2.

    Uses the same normal-equation sufficient statistics as the paper. The
    orientation here is H:(N,dff), patch:(dff,d) unlike paper DeltaW:(d,dff).
    """
    dff, dout = h.shape[1], targets.shape[1]
    sh = np.zeros((dff, dff), dtype=np.float64)
    st = np.zeros((dff, dout), dtype=np.float64)
    for start in range(0, len(h), chunk):
        hb = h[start:start + chunk]
        tb = targets[start:start + chunk]
        sh += hb.T @ hb
        st += hb.T @ tb
    trace = max(float(np.trace(sh)), 1e-12)
    lam = lam0 * float(np.sum(sh * sh)) / trace
    patch = np.linalg.solve(sh + lam * np.eye(dff), st)
    return patch, {'lambda_effective': lam, 'patch_fro': float(np.linalg.norm(patch))}


def mse(a, b) -> float:
    return float(np.mean((a - b) ** 2))


def evaluate(q, full, small, block, scale, patch):
    truth = forward(q, full, block, scale)[0]
    baseline = forward(q, small, block, scale)[0]
    patched = forward(q, small, block, scale, patch=patch)[0]
    base = mse(truth, baseline)
    after = mse(truth, patched)
    recovery = (base - after) / base if base > 1e-12 else 0.0
    return {'baseline_mse': base, 'patched_mse': after, 'recovery': recovery}


def run(args) -> dict:
    all_rows = []
    check_rows = []
    for s in range(args.seeds):
        rng = np.random.default_rng(args.seed + s)
        block = make_block(rng, args.d, args.dff)
        ctx = make_context(rng, args.d, args.context)
        other = make_context(rng, args.d, args.context)
        qref = make_queries(rng, ctx, args.refs, 'in')
        qin = make_queries(rng, ctx, args.tests, 'in')
        qshift = make_queries(rng, ctx, args.tests, 'shift')
        qother = make_queries(rng, other, args.tests, 'in')
        teacher_ref, hfull, zfull = forward(qref, ctx, block, args.scale)
        for ratio in args.ratios:
            small = compressed(ctx, ratio, rng)
            _, hsmall, zsmall = forward(qref, small, block, args.scale)
            # target = W (h - h') + (z-z') for row-major matrix notation
            target = (hfull - hsmall) @ block.w_out + (zfull - zsmall)
            patch, info = ridge_stats(hsmall, target, args.lam, args.chunk)
            # Chunked sufficient stats must agree with one-shot construction.
            direct, _ = ridge_stats(hsmall, target, args.lam, len(qref))
            consistency = float(np.max(np.abs(patch - direct)))
            assert consistency < 1e-8, f'Chunked ridge mismatch: {consistency}'
            for split, queries in [('reference', qref), ('heldout', qin), ('shifted', qshift)]:
                row = evaluate(queries, ctx, small, block, args.scale, patch)
                row.update(seed=args.seed + s, ratio=ratio, retained=len(small.keys), split=split, **info)
                all_rows.append(row)
            # Negative control: use context A's patch while serving unrelated B.
            other_small = compressed(other, ratio, rng)
            row = evaluate(qother, other, other_small, block, args.scale, patch)
            row.update(seed=args.seed + s, ratio=ratio, retained=len(other_small.keys), split='cross_context', **info)
            all_rows.append(row)
            check_rows.append({'seed': args.seed + s, 'ratio': ratio, 'chunk_vs_full_max_abs': consistency})
    by_split = {}
    for split in ['reference', 'heldout', 'shifted', 'cross_context']:
        by_split[split] = []
        for ratio in args.ratios:
            rows = [r for r in all_rows if r['split'] == split and r['ratio'] == ratio]
            by_split[split].append({
                'ratio': ratio,
                'baseline_mse': float(np.mean([r['baseline_mse'] for r in rows])),
                'patched_mse': float(np.mean([r['patched_mse'] for r in rows])),
                'recovery': float(np.mean([r['recovery'] for r in rows])),
            })
    report = {'config': vars(args), 'summary': by_split, 'rows': all_rows, 'ridge_checks': check_rows}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('=== PATCHKV-INSPIRED ONE-BLOCK EXPERIMENT ===')
    print(f'contexts={args.seeds} n_context_tokens={args.context} reference_queries={args.refs} hidden={args.dff} feature_dim={args.d}')
    print('ratio  retained  split           baseline_MSE  patched_MSE  recovery')
    for ratio in args.ratios:
        retained = max(1, int(round(args.context * ratio)))
        for split in by_split:
            d = next(x for x in by_split[split] if x['ratio'] == ratio)
            print(f'{ratio:5.2f}  {retained:8}  {split:14s}  {d["baseline_mse"]:12.6f}  {d["patched_mse"]:11.6f}  {d["recovery"]:+8.1%}')
    print(f'chunked sufficient-statistics check: PASS ({len(check_rows)} cases)')
    print(f'Report: {out / "report.json"}')
    return report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='results/exp1_patchkv')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--context', type=int, default=320)
    p.add_argument('--d', type=int, default=12)
    p.add_argument('--dff', type=int, default=32)
    p.add_argument('--refs', type=int, default=192)
    p.add_argument('--tests', type=int, default=256)
    p.add_argument('--scale', type=float, default=1.8)
    p.add_argument('--lam', type=float, default=0.001)
    p.add_argument('--chunk', type=int, default=48)
    p.add_argument('--ratios', type=float, nargs='+', default=[.05,.10,.20,.40])
    args = p.parse_args()
    if args.seeds < 1 or args.refs < 1 or args.chunk < 1 or not all(0 < r <= 1 for r in args.ratios) or args.lam <= 0:
        p.error('Invalid arguments: seeds/refs/chunk>=1, 0<ratio<=1, lam>0')
    return args


if __name__ == '__main__':
    run(parse_args())
