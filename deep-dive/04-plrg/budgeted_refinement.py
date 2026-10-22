"""PLRG Experiment 3: budgeted, dependency-aware call-string refinement.

Run beside your existing engine.py and policy_engine.py:
    python budgeted_refinement.py

Toy limitations: acyclic call graph/CFG, finite integer sets, one integer
parameter per function, monotonically accumulated warnings. This benchmark
checks warnings at convergence and requires a known execution oracle.
"""
from dataclasses import dataclass
from itertools import combinations, product

from engine import Function, Const, Call, Div, Return, BranchPositive
from policy_engine import Analyzer, Program


@dataclass(frozen=True)
class Evaluation:
    settings: tuple[int, ...]
    policy: dict[str, int]
    contexts: int
    steps: int
    runs: int
    histories: int
    false_positives: int
    true_positives: int
    warnings: frozenset[str]


def make_program(call_counts=(2, 4, 7)):
    """Independent two-layer call chains with both real and false warnings.

    One positive call and (n-1) negative calls per chain. main divides by
    both first (safe) and second (actual zero) returns. Distinct callsites
    are used for every main -> wrapper invocation.
    """
    p: Program = {}
    instructions = []
    expected_false = set()
    expected_true = set()
    knobs: list[tuple[str, int]] = []

    for chain, n in enumerate(call_counts):
        assert n >= 2
        transform = f'transform_{chain}'
        wrapper = f'wrapper_{chain}'
        p[transform] = Function('x', {
            'entry': ([], BranchPositive('x', 'positive', 'zero')),
            'positive': ([], Return('x')),
            'zero': ([Const('z', 0)], Return('z')),
        })
        p[wrapper] = Function('x', {
            'entry': ([Call('r', transform, 'x', f'{wrapper}:to_transform')], Return('r')),
        })
        # The optimization decision is binary for each function:
        # wrapper: 0 -> 1; transform: 0 -> 2.
        knobs.extend([(wrapper, 1), (transform, 2)])
        for i in range(n):
            var = f'x_{chain}_{i}'
            ret = f'r_{chain}_{i}'
            instructions.extend([
                Const(var, 10 if i == 0 else -10),
                Call(ret, wrapper, var, f'main:chain_{chain}_call_{i}'),
            ])
        good = f'chain_{chain}:safe_division'
        bad = f'chain_{chain}:actual_zero_division'
        expected_false.add(good)
        expected_true.add(bad)
        instructions.extend([
            Div(f'good_{chain}', 100, f'r_{chain}_0', good),
            Div(f'bad_{chain}', 100, f'r_{chain}_1', bad),
        ])

    instructions.append(Const('done', 1))
    p['main'] = Function(None, {'entry': (instructions, Return('done'))})
    return p, knobs, frozenset(expected_false), frozenset(expected_true)


def search(call_counts=(2, 4, 7), budgets=(10, 16, 22, 30)):
    program, knobs, false_sites, true_sites = make_program(call_counts)
    cache = {}

    def evaluate(settings: tuple[int, ...]):
        settings = tuple(settings)
        if settings not in cache:
            policy = {name: depth for (name, depth), on in zip(knobs, settings) if on}
            r = Analyzer(policy, program).run()
            cache[settings] = Evaluation(
                settings, policy, r.contexts, r.transfer_count,
                r.reanalyses, r.provenance_count,
                len(r.warnings & false_sites), len(r.warnings & true_sites),
                r.warnings,
            )
        return cache[settings]

    all_settings = tuple(product((0, 1), repeat=len(knobs)))
    all_results = [evaluate(s) for s in all_settings]
    assert all(r.true_positives == len(true_sites) for r in all_results), (
        'An actual bug was missed. Inspect abstract semantics before using precision metrics.'
    )

    def quality(r):
        return (r.false_positives, r.contexts, r.steps)

    def feasible(r, budget):
        return r.contexts <= budget

    def refine(budget, max_group_size):
        # 먼저 k=0으로 계산
        current = evaluate((0,) * len(knobs))

        # 그거도 불가능하면 return None
        if not feasible(current, budget):
            return None

        # 탐색: best 조합에 모든 comb + 1 해보면서 탐색
        while True:
            remaining = [i for i, bit in enumerate(current.settings) if bit == 0]
            candidates = []
            for size in range(1, max_group_size + 1):
                for chosen in combinations(remaining, size):
                    candidate = list(current.settings)
                    for i in chosen:
                        candidate[i] = 1
                    r = evaluate(candidate)
                    if feasible(r, budget) and r.false_positives < current.false_positives:
                        gain = current.false_positives - r.false_positives
                        extra = max(r.contexts - current.contexts, 1)
                        candidates.append((gain / extra, gain, -r.contexts, r))
            if not candidates:
                return current
            # Deterministic decision; best local gain per extra context.
            current = max(candidates, key=lambda x: (x[0], x[1], x[2]))[3]

    base = evaluate((0,) * len(knobs))
    full = evaluate((1,) * len(knobs))
    print('=== PLRG Experiment 3: budgeted context refinement ===')
    print(f'Call counts per chain: {call_counts}; policies evaluated: {len(all_results)}')
    print(f'Known safe divisions: {len(false_sites)}; known actual zero divisions: {len(true_sites)}')
    print(f'Baseline: FP={base.false_positives}, TP={base.true_positives}, '
          f'contexts={base.contexts}, steps={base.steps}, histories={base.histories}')
    print(f'All refined: FP={full.false_positives}, TP={full.true_positives}, '
          f'contexts={full.contexts}, steps={full.steps}, histories={full.histories}')
    print()
    print(f'{"budget":>6} {"method":>14} {"FP":>3} {"TP":>3} '
          f'{"ctx":>5} {"steps":>6} {"runs":>5} {"hist":>5}  selected refinements')

    rows = []
    for budget in budgets:
        feasible_results = [r for r in all_results if feasible(r, budget)]
        exact = min(feasible_results, key=quality) if feasible_results else None
        methods = [('greedy-1', refine(budget, 1)),
                   ('lookahead-2', refine(budget, 2)),
                   ('exact', exact)]
        for name, result in methods:
            if result is None:
                print(f'{budget:>6} {name:>14} -- infeasible')
                continue
            selected = ','.join(f'{n}:{d}' for n, d in result.policy.items()) or '(none)'
            print(f'{budget:>6} {name:>14} {result.false_positives:>3} '
                  f'{result.true_positives:>3} {result.contexts:>5} '
                  f'{result.steps:>6} {result.runs:>5} '
                  f'{result.histories:>5}  {selected}')
            rows.append((budget, name, result))
        print()
    return rows, cache


if __name__ == '__main__':
    search()
