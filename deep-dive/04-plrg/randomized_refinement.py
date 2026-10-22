"""PLRG Experiment 4: seeded, heterogeneous context-refinement benchmarks.

Place beside engine.py and policy_engine.py; Python 3.11+.
Usage: python randomized_refinement.py --trials 12 --seed 42

Synthetic DAGs, finite value sets, and a ground-truth oracle. A comparison of
policy-selection heuristics, NOT a general static-analysis precision result.
"""
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path

from engine import BranchPositive, Call, Const, Div, Function, Return
from policy_engine import Analyzer


@dataclass(frozen=True)
class Evaluation:
    settings: tuple[int, ...]
    fp: int
    tp: int
    contexts: int
    steps: int
    runs: int
    histories: int


def make_case(rng: random.Random, case_id: int):
    """Generate 3 independent pipelines; depth 1/2/3, counts 2..5.

    Pipeline length L has L refinement knobs (one per layer) whose exact
    required call-string depths are 1..L in call order. All L knobs must be
    enabled to distinguish one safe and one unsafe call to that pipeline.
    """
    depths = [1, 2, 3]
    rng.shuffle(depths)
    # Shuffle helps eliminate dependence on chain enumeration order.
    counts = [rng.randint(2, 5) for _ in depths]
    program: dict[str, Function] = {}
    instructions = []
    knobs: list[tuple[str, int]] = []
    safe: set[str] = set()
    bad: set[str] = set()
    for chain, (depth, count) in enumerate(zip(depths, counts)):
        names = [f'c{chain}_layer{j}' for j in range(depth)]
        # Layer0 is first entered from main; final layer has branch.
        for j, name in enumerate(names):
            knobs.append((name, j + 1))
            if j == depth - 1:
                program[name] = Function('x', {
                    'entry': ([], BranchPositive('x', 'positive', 'zero')),
                    'positive': ([], Return('x')),
                    'zero': ([Const('z', 0)], Return('z')),
                })
            else:
                program[name] = Function('x', {
                    'entry': ([Call('r', names[j + 1], 'x', f'{name}:next')], Return('r')),
                })
        # Exactly one positive (safe) and others negative, but random order.
        positives = rng.sample(range(count), k=1)[0]
        negative = next(i for i in range(count) if i != positives)
        rets = []
        for i in range(count):
            var, ret = f'c{chain}_x{i}', f'c{chain}_ret{i}'
            value = rng.randint(1, 99) if i == positives else -rng.randint(1, 99)
            instructions.extend((Const(var, value), Call(ret, names[0], var, f'main:c{chain}_call{i}')))
            rets.append(ret)
        safe_site, bad_site = f'case{case_id}:c{chain}:safe', f'case{case_id}:c{chain}:bug'
        instructions.extend((Div(f'safe{chain}', 100, rets[positives], safe_site),
                             Div(f'bad{chain}', 100, rets[negative], bad_site)))
        safe.add(safe_site)
        bad.add(bad_site)
    instructions.append(Const('done', 1))
    program['main'] = Function(None, {'entry': (instructions, Return('done'))})
    return program, knobs, frozenset(safe), frozenset(bad), depths, counts


def run_case(case_id: int, rng: random.Random, budget_fractions=(0.33, 0.67, 1.0)):
    program, knobs, safe, bugs, depths, counts = make_case(rng, case_id)
    cache: dict[tuple[int, ...], Evaluation] = {}

    def evaluate(settings):
        settings = tuple(settings)
        if settings not in cache:
            policy = {fn: k for (fn, k), active in zip(knobs, settings) if active}
            result = Analyzer(policy, program).run()
            if not bugs <= result.warnings:
                raise AssertionError(f'case {case_id} missed actual division by zero')
            unexpected = result.warnings - safe - bugs
            if unexpected:
                raise AssertionError(f'unexpected warning sites: {unexpected}')
            cache[settings] = Evaluation(settings, len(result.warnings & safe),
                                         len(result.warnings & bugs), result.contexts,
                                         result.transfer_count, result.reanalyses,
                                         result.provenance_count)
        return cache[settings]

    all_results = [evaluate(s) for s in product((0, 1), repeat=len(knobs))]
    base = evaluate((0,) * len(knobs))
    full = evaluate((1,) * len(knobs))
    assert base.fp == len(safe), (case_id, 'unexpected baseline', base.fp)
    assert full.fp == 0, (case_id, 'full refinement failed', full.fp)

    def exact(budget):
        return min((r for r in all_results if r.contexts <= budget),
                   key=lambda r: (r.fp, r.contexts, r.steps))

    def refine(budget, max_group_size):
        current = base
        while True:
            remaining = [i for i, bit in enumerate(current.settings) if not bit]
            candidates = []
            for group_size in range(1, min(max_group_size, len(remaining)) + 1):
                for chosen in combinations(remaining, group_size):
                    settings = list(current.settings)
                    for idx in chosen:
                        settings[idx] = 1
                    candidate = evaluate(settings)
                    if candidate.contexts > budget or candidate.fp >= current.fp:
                        continue
                    gain = current.fp - candidate.fp
                    extra = max(candidate.contexts - current.contexts, 1)
                    score = (gain / extra, gain, -candidate.contexts)
                    candidates.append((score, candidate))
            if not candidates:
                return current
            current = max(candidates, key=lambda p: p[0])[1]

    span = full.contexts - base.contexts
    rows = []
    for fraction in budget_fractions:
        budget = base.contexts + round(span * fraction)
        choices = {'greedy-1': refine(budget, 1), 'lookahead-2': refine(budget, 2),
                   'lookahead-3': refine(budget, 3), 'exact': exact(budget)}
        for method, result in choices.items():
            rows.append({'case': case_id, 'depths': '-'.join(map(str, depths)),
                         'calls': '-'.join(map(str, counts)), 'fraction': fraction,
                         'budget': budget, 'method': method, 'fp': result.fp,
                         'tp': result.tp, 'contexts': result.contexts,
                         'steps': result.steps, 'runs': result.runs,
                         'histories': result.histories,
                         'selected': ','.join(name for (name, _), on in zip(knobs, result.settings) if on)})
    return rows, len(cache)


def report(rows):
    methods = ('greedy-1', 'lookahead-2', 'lookahead-3', 'exact')
    fractions = sorted({row['fraction'] for row in rows})
    print('Budget as fraction of baseline → fully refined context span')
    print('FP averages over generated cases; lower is better.')
    print(f'{"budget%":>8} ' + ' '.join(f'{m:>13}' for m in methods))
    for f in fractions:
        subset = [r for r in rows if r['fraction'] == f]
        avgs = [sum(r['fp'] for r in subset if r['method'] == m) / sum(r['method'] == m for r in subset) for m in methods]
        print(f'{f * 100:>7.0f}% ' + ' '.join(f'{v:>13.3f}' for v in avgs))
    print('\nOptimality gap count (number of trials where heuristic has more FP than exact):')
    for f in fractions:
        cases = sorted({r['case'] for r in rows if r['fraction'] == f})
        exact_fp = {r['case']: r['fp'] for r in rows if r['fraction'] == f and r['method'] == 'exact'}
        counts = {m: 0 for m in methods[:-1]}
        for r in rows:
            if r['fraction'] == f and r['method'] in counts and r['fp'] > exact_fp[r['case']]:
                counts[r['method']] += 1
        print(f'  {f:.0%} budget: ' + ', '.join(f'{m}={v}/{len(cases)}' for m, v in counts.items()))
    print('\nNOTE: Known FP ground truth guides the heuristic; this is an oracle-assisted policy-selection study.')
    print('NOTE: IR steps and contexts are proxies; convergence warnings can accumulate transiently.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='randomized_refinement_results.csv')
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be at least one')
    rng = random.Random(args.seed)
    rows = []
    policies = 0
    for case_id in range(args.trials):
        case_rows, count = run_case(case_id, rng)
        rows.extend(case_rows)
        policies += count
    with Path(args.output).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Cases={args.trials}, seed={args.seed}, policy evaluations={policies}; CSV={args.output}\n')
    report(rows)


if __name__ == '__main__':
    main()
