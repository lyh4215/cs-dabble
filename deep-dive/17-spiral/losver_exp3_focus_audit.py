#!/usr/bin/env python3
"""CS Dabble 2.0 / Deep Dive 17 / LOSVER -- Exp 3.

EXPLORATORY, LIGHTWEIGHT downstream vulnerability detection study:
  - Loads the *trained* Exp 2 localizer and freezes its CodeBERT encoder.
  - Computes function representations using all sampled token windows, not only head.
  - Uses identical embeddings, split, classifier type, hyperparameters across
    UNIFORM, RANDOM, LONGEST, LEXICAL, LOCALIZER line-focus signals.
  - Trains a NEW logistic regression detector for each signal (NO additional GPU fine-tuning).
  - Evaluates vulnerability vs fixed counterpart using pair ranking and AUROC/AUPRC.
  - Excludes exact vulnerability samples used to train the Exp 2 localizer from
    detector training, and avoids overlap across detector splits.
  - Does NOT use patch-derived labels for choosing focus at validation/test time.

This is an independent controlled study, NOT a reproduction of the authors'
full UniXcoder/LOSVER detector. Patch-localization != vulnerability localization.

Requires: Exp 1 + Exp 2 scripts, original train/valid/test paired JSONL,
          results/exp2_localizer/checkpoint.pt, GPU recommended.
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from losver_exp1_signal_audit import rank_lexical, rank_longest
from losver_exp2_localizer import LineLocalizer, make_windows, read_pairs

METHODS = ('uniform', 'random', 'longest', 'lexical', 'localizer')


def _hash(code):
    return hashlib.sha256('\n'.join(code.splitlines()).encode('utf-8')).hexdigest()


def load_pairs(path):
    pairs = []
    with open(path, encoding='utf-8') as f:
        pending = None
        for row in f:
            if not row.strip():
                continue
            x = json.loads(row)
            if pending is None:
                pending = x
                continue
            a, b = pending, x
            pending = None
            if {a.get('target'), b.get('target')} != {0, 1} or not a.get('func') or not b.get('func'):
                continue
            vuln = a if a['target'] == 1 else b
            fixed = b if a['target'] == 1 else a
            if vuln['func'] == fixed['func']:
                continue
            pairs.append({'vuln': vuln['func'], 'fixed': fixed['func'],
                          'hashes': {_hash(vuln['func']), _hash(fixed['func'])},
                          'id': str(vuln.get('idx', len(pairs)))})
    return pairs


def sample_excluding(rows, limit, excluded, seed):
    filtered = [p for p in rows if p['hashes'].isdisjoint(excluded)]
    random.Random(seed).shuffle(filtered)
    chosen, hashes = [], set()
    for p in filtered:
        if not p['hashes'].isdisjoint(hashes):
            continue
        chosen.append(p)
        hashes.update(p['hashes'])
        if limit > 0 and len(chosen) >= limit:
            break
    return chosen, hashes, len(filtered)


def choose_windows(windows, max_windows):
    if not max_windows or len(windows) <= max_windows:
        return windows
    indexes = np.linspace(0, len(windows) - 1, max_windows, dtype=int)
    return [windows[i] for i in sorted(set(indexes.tolist()))]


@torch.inference_mode()
def extract_features(model, tokenizer, code, batch_size, max_tokens, overlap, max_windows, device):
    lines = code.splitlines()
    pair = {'id': 'current', 'lines': lines, 'changed': set()}
    windows = choose_windows(make_windows(pair, tokenizer, max_tokens, overlap), max_windows)
    H = model.encoder.config.hidden_size
    summed = np.zeros((len(lines), H), np.float32)
    scores = np.zeros((len(lines),), np.float64)
    counts = np.zeros((len(lines),), np.int32)
    for beg in range(0, len(windows), batch_size):
        batch = windows[beg:beg + batch_size]
        max_len = max(len(w['ids']) for w in batch)
        ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
        masks = torch.zeros_like(ids)
        for j, w in enumerate(batch):
            n = len(w['ids'])
            ids[j, :n] = torch.tensor(w['ids'], dtype=torch.long, device=device)
            masks[j, :n] = 1
        # Autocast during inference reduces GPU memory. CPU path remains fp32.
        with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda'), dtype=torch.float16):
            states = model.encoder(input_ids=ids, attention_mask=masks).last_hidden_state
            for j, w in enumerate(batch):
                line_ids = torch.tensor(w['line_ids'], dtype=torch.long, device=device)
                valid = line_ids >= 0
                count = len(w['original_lines'])
                ss = torch.zeros((count, H), dtype=states.dtype, device=device)
                cc = torch.zeros((count, 1), dtype=states.dtype, device=device)
                ix = line_ids.clamp_min(0).unsqueeze(-1)
                ss.scatter_add_(0, ix.expand(-1, H), states[j, :len(line_ids)] * valid.unsqueeze(-1))
                cc.scatter_add_(0, ix, valid.unsqueeze(-1).to(states.dtype))
                means = ss / cc.clamp_min(1)
                cls = states[j, 0:1].expand(count, -1)
                logits = model.classifier(torch.cat((means, cls), dim=-1)).squeeze(-1)
                reps = means.float().cpu().numpy()
                logits_np = logits.float().cpu().numpy()
                mask_np = (cc.squeeze(-1) > 0).cpu().numpy()
                for local, original in enumerate(w['original_lines']):
                    if mask_np[local]:
                        summed[original] += reps[local]
                        scores[original] += float(logits_np[local])
                        counts[original] += 1
    available = np.flatnonzero(counts > 0)
    if len(available) == 0:
        return None
    reps = summed[available] / counts[available, None]
    logits = scores[available] / counts[available]
    return lines, available, reps, logits


def build_representation(extracted, code_id, topk, boost, seed):
    lines, available, reps, logits = extracted
    n = len(available)
    index = {int(a): i for i, a in enumerate(available)}
    random_order = list(range(n))
    rng = random.Random(int.from_bytes(hashlib.sha256(f'{seed}:{code_id}'.encode()).digest()[:8], 'big'))
    rng.shuffle(random_order)
    candidate = {
        'uniform': [],
        'random': random_order,
        'longest': [index[i] for i in rank_longest(lines) if i in index],
        'lexical': [index[i] for i in rank_lexical(lines) if i in index],
        'localizer': np.argsort(-logits, kind='stable').tolist(),
    }
    vectors = {}
    for method in METHODS:
        weights = np.ones(n, dtype=np.float32)
        for idx in candidate[method][:topk]:
            weights[idx] = boost
        vectors[method] = np.average(reps, axis=0, weights=weights).astype(np.float32)
    return vectors, n / max(1, len(lines))


def extract_split(name, pairs, model, tokenizer, args, device):
    X = {method: [] for method in METHODS}
    ys = []
    coverage = []
    for i, pair in enumerate(pairs):
        item = {}
        for side, label in (('vuln', 1), ('fixed', 0)):
            code = pair[side]
            z = extract_features(model, tokenizer, code, args.batch_size,
                                 args.max_tokens, args.overlap, args.max_windows, device)
            if z is None:
                break
            vals, covered = build_representation(z, f"{pair['id']}:{side}", args.topk, args.boost, args.seed)
            item[side] = (vals, covered)
        if len(item) == 2:
            for side, label in (('vuln', 1), ('fixed', 0)):
                vectors, covered = item[side]
                for m in METHODS:
                    X[m].append(vectors[m])
                ys.append(label)
                coverage.append(covered)
        if (i + 1) % 20 == 0 or i + 1 == len(pairs):
            print(f'  {name}: {i + 1}/{len(pairs)} pairs encoded', flush=True)
    X = {m: np.stack(v) for m, v in X.items()}
    print(f'  {name}: used={len(ys)//2} pairs, mean line coverage={np.mean(coverage):.3f}')
    return X, np.asarray(ys, dtype=int)


def fit_detector(X, y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    clf.fit(X, y)
    return clf


def metrics(model, X, y):
    p = model.predict_proba(X)[:, 1]
    pred = (p >= .5).astype(int)
    if len(y) % 2 != 0 or not np.all(y[0::2] == 1) or not np.all(y[1::2] == 0):
        raise ValueError('Expected alternating vulnerable/fixed pairs')
    pair_hit = (p[::2] > p[1::2]).astype(np.float32)
    result = {'n_pairs': len(pair_hit), 'accuracy': float(accuracy_score(y, pred)),
              'f1': float(f1_score(y, pred, zero_division=0)),
              'auroc': float(roc_auc_score(y, p)),
              'auprc': float(average_precision_score(y, p)),
              'pair_accuracy': float(pair_hit.mean())}
    return result, pair_hit


def paired_ci(a, b, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    diff = a - b
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    return np.quantile(diff[idx].mean(axis=1), [.025, .975]).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True, type=Path)
    p.add_argument('--valid', required=True, type=Path)
    p.add_argument('--test', required=True, type=Path)
    p.add_argument('--checkpoint', default='results/exp2_localizer/checkpoint.pt', type=Path)
    p.add_argument('--backbone', default='microsoft/codebert-base')
    p.add_argument('--train-pairs', type=int, default=220)
    p.add_argument('--valid-pairs', type=int, default=80)
    p.add_argument('--test-pairs', type=int, default=180)
    p.add_argument('--localizer-train-pairs', type=int, default=900,
                   help='Must match Exp 2 training sample size (900 by default)')
    p.add_argument('--seed', type=int, default=42,
                   help='Must match Exp 2 sampling seed (42 by default)')
    p.add_argument('--max-tokens', type=int, default=384)
    p.add_argument('--overlap', type=int, default=96)
    p.add_argument('--max-windows', type=int, default=8, help='0=all windows (slower)')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--topk', type=int, default=5)
    p.add_argument('--boost', type=float, default=8.0)
    p.add_argument('--out', type=Path, default=Path('results/exp3_focus_audit'))
    args = p.parse_args()
    if not args.checkpoint.exists():
        p.error(f'checkpoint not found: {args.checkpoint}')
    if args.max_tokens < 16 or args.overlap < 0 or args.overlap >= args.max_tokens-2:
        p.error('Invalid max-tokens/overlap')
    if args.topk < 1 or args.boost < 1:
        p.error('--topk >= 1 and --boost >= 1 required')
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    if device.type != 'cuda':
        print('WARNING: GPU recommended for feature extraction')

    print('Loading paired data and reproducing Exp 2 excluded hashes...')
    localizer_train, _ = read_pairs(args.train, args.localizer_train_pairs, args.seed + 2)
    excluded = {p['hash'] for p in localizer_train}
    # Choose test and validation first; do not let detector training contain their exact code.
    test, test_h, test_eligible = sample_excluding(load_pairs(args.test), args.test_pairs, set(), args.seed + 7)
    valid, valid_h, valid_eligible = sample_excluding(load_pairs(args.valid), args.valid_pairs, test_h, args.seed + 6)
    train, train_h, train_eligible = sample_excluding(load_pairs(args.train), args.train_pairs,
                                                     test_h | valid_h | excluded, args.seed + 5)
    print('pairs (train/valid/test):', len(train), len(valid), len(test))
    print('eligible after exclusions:', train_eligible, valid_eligible, test_eligible)
    if min(len(train), len(valid), len(test)) < 5:
        raise RuntimeError('Insufficient pairs after split/dedup filtering')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    model = LineLocalizer(args.backbone)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    arrays = {}
    for split, pairs in [('train', train), ('valid', valid), ('test', test)]:
        print('\nExtracting frozen CodeBERT features:', split, flush=True)
        X, y = extract_split(split, pairs, model, tokenizer, args, device)
        arrays[split] = (X, y)
    # Unlike classifier fine-tuning, features are only calculated once.
    print('\n=== EXP 3: VULNERABILITY DETECTION (paired vulnerable/fixed code) ===')
    print('focus        valid AUPRC   test AUPRC   test AUROC   test PairAcc   test F1')
    results, pair_hits = {}, {}
    for m in METHODS:
        clf = fit_detector(arrays['train'][0][m], arrays['train'][1])
        vm, _ = metrics(clf, arrays['valid'][0][m], arrays['valid'][1])
        tm, hits = metrics(clf, arrays['test'][0][m], arrays['test'][1])
        results[m] = {'valid': vm, 'test': tm}
        pair_hits[m] = hits
        print(f'{m:<12} {vm["auprc"]:10.3f}   {tm["auprc"]:10.3f}   '
              f'{tm["auroc"]:10.3f}   {tm["pair_accuracy"]:12.3f}   {tm["f1"]:7.3f}')
    delta = pair_hits['localizer'].mean() - pair_hits['uniform'].mean()
    lo, hi = paired_ci(pair_hits['localizer'], pair_hits['uniform'])
    print(f'Localizer - Uniform PairAcc: {delta:+.4f} (95% paired bootstrap [{lo:+.4f}, {hi:+.4f}])')
    result = {
        'note': 'Frozen Exp2 localizer embeddings + independently trained logistic regressions; '
                'not full fine-tuned LOSVER. Oracle patch lines intentionally excluded.',
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'counts': {'train': len(arrays['train'][1]) // 2,
                   'valid': len(arrays['valid'][1]) // 2, 'test': len(arrays['test'][1]) // 2},
        'results': results,
        'localizer_vs_uniform_pairacc': {'delta': float(delta), 'ci_95': [lo, hi]},
    }
    (args.out / 'report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print('Results saved to', args.out / 'report.json')
    print('Interpretation: paired vulnerability classification, NOT patch-line hit rate.')


if __name__ == '__main__':
    main()
