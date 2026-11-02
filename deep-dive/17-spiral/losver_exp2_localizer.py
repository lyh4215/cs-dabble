#!/usr/bin/env python3
"""CS Dabble / Deep Dive 17 / LOSVER - Exp 2.

Train an INDEPENDENT CodeBERT patch-line localizer on PrimeVul's paired data.
This is not the authors' exact UniXcoder/attention architecture.

- Train only on official train pairs; checkpoint selected on validation pairs.
- Test pairs evaluated once, after checkpoint selection.
- Separate long functions into overlapping token windows (no 512-token-head bias).
- Compare Top-{1,3,5} PATCH-LINE hit rate against Exp 1 controls.
- Patch-line annotations are noisy proxies, NOT verified vulnerable lines.
- Insert-only pairs have no existing before-line label and are excluded.

Requires: pip install 'transformers>=4.45,<5' torch numpy scikit-learn
Run `python losver_exp2_localizer.py --help` for arguments.
"""
import argparse
import bisect
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from losver_exp1_signal_audit import changed_before_lines, rank_lexical, rank_longest


def read_pairs(path, limit, seed):
    """Reservoir sample eligible VULNERABLE->PATCHED pairs, tracking exclusions."""
    rng = random.Random(seed)
    stats = Counter()
    reservoir = []
    with open(path, encoding='utf-8') as f:
        pending = None
        for raw in f:
            if not raw.strip():
                continue
            ex = json.loads(raw)
            if pending is None:
                pending = ex
                continue
            a, b = pending, ex
            pending = None
            stats['candidate_pairs'] += 1
            if {a.get('target'), b.get('target')} != {0, 1} or not a.get('func') or not b.get('func'):
                stats['invalid_pairs'] += 1
                continue
            vuln = a if a['target'] == 1 else b
            fixed = b if a['target'] == 1 else a
            lines = vuln['func'].splitlines()
            changed, inserted = changed_before_lines(vuln['func'], fixed['func'])
            if not lines:
                stats['empty'] += 1
                continue
            if not changed:
                stats['insert_only_or_unusable'] += 1
                if inserted:
                    stats['insert_only'] += 1
                continue
            stats['eligible'] += 1
            item = {
                'id': f"{stats['candidate_pairs']}:{vuln.get('idx', 'unknown')}",
                'lines': lines,
                'changed': changed,
                'hash': hashlib.sha256(('\n'.join(lines)).encode('utf-8')).hexdigest(),
            }
            if limit <= 0 or len(reservoir) < limit:
                reservoir.append(item)
            else:
                ix = rng.randrange(stats['eligible'])
                if ix < limit:
                    reservoir[ix] = item
    if pending is not None:
        stats['dangling_row'] += 1
    return reservoir, stats


def filter_overlap(train, valid, test):
    """Prevent exact vulnerable-function duplication across splits."""
    test_h = {p['hash'] for p in test}
    valid = [p for p in valid if p['hash'] not in test_h]
    valid_h = {p['hash'] for p in valid}
    train = [p for p in train if p['hash'] not in test_h and p['hash'] not in valid_h]
    def unique(seq):
        seen, result = set(), []
        for p in seq:
            if p['hash'] not in seen:
                seen.add(p['hash'])
                result.append(p)
        return result
    return unique(train), unique(valid), test  # Preserve all TEST examples for Exp 1 comparability.


def make_windows(pair, tokenizer, max_tokens, overlap):
    """Tokenize whole function (fast tokenizer offsets) and slide over all tokens."""
    code = '\n'.join(pair['lines'])
    result = tokenizer(code, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    ids = result['input_ids']
    offsets = result['offset_mapping']
    if not ids:
        return []
    if not (0 <= overlap < max_tokens - 2):
        raise ValueError('--overlap must be < --max-tokens - 2')
    starts = [0] + [m.end() for m in re.finditer('\n', code)]
    token_lines = [min(len(pair['lines']) - 1, bisect.bisect_right(starts, start) - 1) for start, end in offsets]
    budget = max_tokens - 2
    step = budget - overlap
    windows = []
    for beg in range(0, len(ids), step):
        end = min(beg + budget, len(ids))
        chunk_lines = token_lines[beg:end]
        uniq_lines = sorted(set(chunk_lines))
        if not uniq_lines:
            continue
        to_local = {line: i for i, line in enumerate(uniq_lines)}
        windows.append({
            'pid': pair['id'],
            'ids': [tokenizer.cls_token_id] + ids[beg:end] + [tokenizer.sep_token_id],
            'line_ids': [-1] + [to_local[i] for i in chunk_lines] + [-1],
            'original_lines': uniq_lines,
            'labels': [int(line in pair['changed']) for line in uniq_lines],
        })
        if end == len(ids):
            break
    return windows


class Windows(Dataset):
    def __init__(self, xs):
        self.items = xs

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(samples, pad_id):
    B = len(samples)
    L = max(len(x['ids']) for x in samples)
    M = max(len(x['original_lines']) for x in samples)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    masks = torch.zeros((B, L), dtype=torch.long)
    token_lines = torch.full((B, L), -1, dtype=torch.long)
    labels = torch.full((B, M), -1, dtype=torch.float32)
    for i, s in enumerate(samples):
        n, m = len(s['ids']), len(s['original_lines'])
        ids[i, :n] = torch.tensor(s['ids'], dtype=torch.long)
        masks[i, :n] = 1
        token_lines[i, :n] = torch.tensor(s['line_ids'], dtype=torch.long)
        labels[i, :m] = torch.tensor(s['labels'], dtype=torch.float32)
    return ids, masks, token_lines, labels, samples


class LineLocalizer(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(backbone)
        h = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(nn.Linear(2 * h, h // 2), nn.GELU(),
                                        nn.Dropout(.1), nn.Linear(h // 2, 1))

    def forward(self, ids, masks, token_lines, max_lines):
        states = self.encoder(input_ids=ids, attention_mask=masks).last_hidden_state
        B, L, H = states.shape
        sums = torch.zeros((B, max_lines, H), dtype=states.dtype, device=states.device)
        counts = torch.zeros((B, max_lines, 1), dtype=states.dtype, device=states.device)
        good = (token_lines >= 0) & masks.bool()
        indices = token_lines.clamp_min(0).unsqueeze(-1)
        sums.scatter_add_(1, indices.expand(-1, -1, H), states * good.unsqueeze(-1))
        counts.scatter_add_(1, indices, good.unsqueeze(-1).to(states.dtype))
        means = sums / counts.clamp_min(1)
        cls = states[:, :1].expand(-1, max_lines, -1)
        logits = self.classifier(torch.cat([means, cls], dim=-1)).squeeze(-1)
        return logits, counts.squeeze(-1) > 0


def model_scores(model, records, batch_size, device, pad_id):
    """Average logit across overlapping windows; -inf for unrepresented blank lines."""
    loader = DataLoader(Windows(records), batch_size=batch_size, shuffle=False,
                        collate_fn=lambda x: collate(x, pad_id))
    score_sum = defaultdict(float)
    score_count = Counter()
    model.eval()
    with torch.no_grad():
        for ids, masks, line_ids, labels, samples in loader:
            logits, valid = model(ids.to(device), masks.to(device), line_ids.to(device), labels.shape[1])
            logits = logits.float().cpu().numpy()
            valid = valid.cpu().numpy()
            for b, s in enumerate(samples):
                for local_idx, original_idx in enumerate(s['original_lines']):
                    if valid[b, local_idx]:
                        key = (s['pid'], original_idx)
                        score_sum[key] += float(logits[b, local_idx])
                        score_count[key] += 1
    return {key: score_sum[key] / count for key, count in score_count.items()}


def analytical_random_hit(n, m, k):
    k = min(k, n)
    if k > n - m:
        return 1.
    miss = 1.
    for i in range(k):
        miss *= (n-m-i)/(n-i)
    return 1. - miss


def paired_bootstrap_diff(a, b, seed=42, n=2000):
    rng = np.random.default_rng(seed)
    delta = np.asarray(a) - np.asarray(b)
    if len(delta) == 0:
        return (float('nan'), float('nan'))
    means = np.empty(n)
    for i in range(n):
        means[i] = delta[rng.integers(0, len(delta), len(delta))].mean()
    return tuple(np.percentile(means, [2.5, 97.5]).tolist())


def evaluate(pairs, scores, label):
    k_values = [1, 3, 5]
    hits = {key: {k: [] for k in k_values} for key in ('model', 'random', 'longest', 'lexical')}
    covered = []
    for p in pairs:
        L = len(p['lines'])
        changed = p['changed']
        ranked = sorted(range(L), key=lambda i: (-scores.get((p['id'], i), -1e20), i))
        longest = rank_longest(p['lines'])
        lexical = rank_lexical(p['lines'])
        covered.append(sum((p['id'], ix) in scores for ix in changed) / len(changed))
        for k in k_values:
            for name, order in [('model', ranked), ('longest', longest), ('lexical', lexical)]:
                hits[name][k].append(float(bool(set(order[:k]) & changed)))
            hits['random'][k].append(analytical_random_hit(L, len(changed), k))
    print(f'\n=== {label} (n={len(pairs)}) PATCH-LINE TOP-K HIT ===')
    print(' k    random  longest  lexical    model')
    for k in k_values:
        print(f' {k:<4} ' + '  '.join(f'{np.mean(hits[name][k]):.3f}' for name in ('random', 'longest', 'lexical', 'model')))
    ci = paired_bootstrap_diff(hits['model'][5], hits['lexical'][5])
    print('model - lexical Top-5 difference (95% paired bootstrap CI):',
          f'{np.mean(hits["model"][5])-np.mean(hits["lexical"][5]):+.4f}',
          f'[{ci[0]:+.4f}, {ci[1]:+.4f}]')
    print('fraction of patch lines represented by tokenization:', f'{np.mean(covered):.3f}')
    return {'n': len(pairs), 'topk': {name: {str(k): float(np.mean(vals)) for k, vals in kv.items()} for name, kv in hits.items()},
            'top5_delta_ci': list(ci), 'patch_line_token_coverage': float(np.mean(covered))}


def select_train_windows(records, limit, seed):
    if limit <= 0 or limit >= len(records):
        return records
    rng = random.Random(seed)
    positive = [r for r in records if any(r['labels'])]
    negative = [r for r in records if not any(r['labels'])]
    rng.shuffle(positive)
    rng.shuffle(negative)
    n_pos = min(len(positive), (limit + 1) // 2)
    n_neg = min(len(negative), limit - n_pos)
    if n_pos + n_neg < limit:
        return (positive[:limit - n_neg] + negative[:n_neg])[:limit]
    xs = positive[:n_pos] + negative[:n_neg]
    rng.shuffle(xs)
    return xs


def build_records(pairs, tokenizer, max_tokens, overlap):
    records = []
    for pair in pairs:
        records.extend(make_windows(pair, tokenizer, max_tokens, overlap))
    return records


def train_epoch(model, loader, optim, device, pos_weight):
    model.train()
    total, batches = 0., 0
    for ids, masks, token_lines, labels, _ in loader:
        ids, masks = ids.to(device), masks.to(device)
        token_lines, labels = token_lines.to(device), labels.to(device)
        optim.zero_grad(set_to_none=True)
        logits, available = model(ids, masks, token_lines, labels.shape[1])
        valid = (labels >= 0) & available
        if not valid.any():
            continue
        loss = F.binary_cross_entropy_with_logits(logits[valid], labels[valid], pos_weight=pos_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total += float(loss.item())
        batches += 1
    return total / max(batches, 1)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--train', type=Path, required=True)
    a.add_argument('--valid', type=Path, required=True)
    a.add_argument('--test', type=Path, required=True)
    a.add_argument('--train-pairs', type=int, default=900, help='Reservoir sample eligible training pairs, 0=all')
    a.add_argument('--valid-pairs', type=int, default=170, help='0=all')
    a.add_argument('--train-windows', type=int, default=2600, help='Subsample windows, balancing positive-containing and negative windows; 0=all')
    a.add_argument('--max-tokens', type=int, default=384)
    a.add_argument('--overlap', type=int, default=96)
    a.add_argument('--backbone', default='microsoft/codebert-base')
    a.add_argument('--epochs', type=int, default=2)
    a.add_argument('--batch-size', type=int, default=8)
    a.add_argument('--lr', type=float, default=2e-5)
    a.add_argument('--seed', type=int, default=42)
    a.add_argument('--out', type=Path, default=Path('results/exp2_localizer'))
    args = a.parse_args()
    if args.max_tokens < 16 or not 0 <= args.overlap < args.max_tokens - 2:
        a.error('require --max-tokens >= 16 and 0 <= --overlap < --max-tokens - 2')
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device, '(GPU recommended for training)')

    # Test inputs are loaded for duplication filtering, not for hyperparameter selection.
    test, test_stats = read_pairs(args.test, 0, args.seed)
    valid, valid_stats = read_pairs(args.valid, args.valid_pairs, args.seed + 1)
    train, train_stats = read_pairs(args.train, args.train_pairs, args.seed + 2)
    before = [len(train), len(valid), len(test)]
    train, valid, test = filter_overlap(train, valid, test)
    print('PAIR COUNTS (train/valid/test), sampled:', before, 'after exact dedup:', [len(train), len(valid), len(test)])
    for name, stats in [('train', train_stats), ('valid', valid_stats), ('test', test_stats)]:
        print(name, dict(stats))
    if not train or not valid or not test:
        raise SystemExit('No eligible pairs in some split. Check data files.')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    if not tokenizer.is_fast:
        raise SystemExit('A fast tokenizer is required for offset_mapping.')
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None or tokenizer.pad_token_id is None:
        raise SystemExit('Tokenizer must have cls/sep/pad token IDs.')
    train_records = select_train_windows(build_records(train, tokenizer, args.max_tokens, args.overlap), args.train_windows, args.seed)
    valid_records = build_records(valid, tokenizer, args.max_tokens, args.overlap)
    test_records = build_records(test, tokenizer, args.max_tokens, args.overlap)
    print('windows (train/valid/test):', len(train_records), len(valid_records), len(test_records))
    positives = sum(sum(x['labels']) for x in train_records)
    negatives = sum(len(x['labels'])-sum(x['labels']) for x in train_records)
    if positives == 0:
        raise SystemExit('Training windows contain no patch-line positives.')
    pos_weight = torch.tensor(min(20., negatives / positives), device=device)
    print('training patch line positives:', positives, 'negatives:', negatives, 'pos_weight:', float(pos_weight))

    train_loader = DataLoader(Windows(train_records), batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda x: collate(x, tokenizer.pad_token_id))
    model = LineLocalizer(args.backbone).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    best = -float('inf')
    best_epoch = 0
    ckpt = args.out / 'checkpoint.pt'
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optim, device, pos_weight)
        scores = model_scores(model, valid_records, args.batch_size, device, tokenizer.pad_token_id)
        result = evaluate(valid, scores, f'VALID epoch={epoch}')
        score = result['topk']['model']['5']
        print(f'epoch={epoch} train_loss={loss:.5f} valid_top5={score:.4f}')
        if score > best:
            best, best_epoch = score, epoch
            torch.save(model.state_dict(), ckpt)
    print('selected epoch on VALID:', best_epoch)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    scores = model_scores(model, test_records, args.batch_size, device, tokenizer.pad_token_id)
    result = evaluate(test, scores, 'TEST (only after VALID selection)')
    with (args.out / 'test_scores.jsonl').open('w', encoding='utf-8') as f:
        for p in test:
            values = [scores.get((p['id'], i)) for i in range(len(p['lines']))]
            f.write(json.dumps({'id': p['id'], 'line_count': len(p['lines']), 'patch_lines_0based': sorted(p['changed']),
                                'line_scores': values}, ensure_ascii=False) + '\n')
    report = {'args': {key: str(val) if isinstance(val, Path) else val for key, val in vars(args).items()},
              'pair_stats': {'train': dict(train_stats), 'valid': dict(valid_stats), 'test': dict(test_stats)},
              'selected_epoch': best_epoch, 'test': result}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('results saved to', args.out)
    print('IMPORTANT: These scores measure PATCH localization, not vulnerability detection.')


if __name__ == '__main__':
    main()
