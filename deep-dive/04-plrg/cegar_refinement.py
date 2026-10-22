"""PLRG Experiment 5: CEGAR-inspired refinement with a concrete feasibility check.

Python 3.11+. Run beside engine.py, policy_engine.py, randomized_refinement.py.

IMPORTANT: Programs generated here have fully specified, deterministic inputs.
Concrete execution therefore decides feasibility FOR THIS TOY TEST SET only. This is
not a general symbolic-execution/SMT CEGAR implementation; no oracle safe/bug
labels are used to CHOOSE refinements. Labels are used only for final evaluation.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

from engine import BranchPositive, Call, Const, Div, Jump, Return
from policy_engine import Analyzer
from randomized_refinement import make_case


@dataclass(frozen=True)
class RunInfo:
    policy: dict[str, int]
    warnings: frozenset[str]
    contexts: int
    steps: int
    runs: int
    histories: int


def inspect_policy(program, policy: dict[str, int]) -> RunInfo:
    result = Analyzer(policy, program).run()
    return RunInfo(dict(policy), result.warnings, result.contexts,
                   result.transfer_count, result.reanalyses,
                   result.provenance_count)


def concrete_execute(program):
    """Execute deterministic acyclic IR; return {division_site: denominator}.

    NOT an arbitrary-path verifier: only checks the unique concrete execution
    of the supplied fully specified program. A zero denominator is recorded;
    execution continues with a dummy result so all sites can be checked.
    """
    denominators: dict[str, int] = {}

    def execute(fn_name: str, arg: int | None, active: frozenset[str]):
        if fn_name in active:
            raise ValueError('Concrete executor does not support recursion')
        fn = program[fn_name]
        env = {fn.param: arg} if fn.param is not None else {}
        block = 'entry'
        steps = 0
        while True:
            steps += 1
            if steps > 10000:
                raise ValueError('Concrete execution exceeded 10000 blocks')
            instructions, terminal = fn.blocks[block]
            for instr in instructions:
                if isinstance(instr, Const):
                    env[instr.dst] = instr.value
                elif isinstance(instr, Call):
                    env[instr.dst] = execute(instr.callee,
                                             env[instr.arg] if instr.arg is not None else None,
                                             active | {fn_name})
                elif isinstance(instr, Div):
                    denominator = env[instr.denominator]
                    denominators[instr.site] = denominator
                    env[instr.dst] = instr.numerator // denominator if denominator else 0
                else:
                    raise TypeError(f'Unsupported IR: {instr!r}')
            if isinstance(terminal, Return):
                return env[terminal.var]
            if isinstance(terminal, BranchPositive):
                block = terminal.yes if env[terminal.var] > 0 else terminal.no
            elif isinstance(terminal, Jump):
                block = terminal.target
            else:
                raise TypeError(terminal)

    execute('main', None, frozenset())
    return denominators


def dependency_knobs(program, warning_site: str, knobs: list[tuple[str, int]]):
    """Find function chain feeding the denominator of a warning in main.

    This is intentionally a limited use-def walk: main has Const/Call/Div, each
    pipeline function has exactly one nested Call. It uses no ground-truth label.
    """
    main_instructions = program['main'].blocks['entry'][0]
    warning = next((ins for ins in main_instructions
                    if isinstance(ins, Div) and ins.site == warning_site), None)
    if warning is None:
        raise ValueError(f'No Div for {warning_site!r}')
    defining_call = next((ins for ins in main_instructions
                          if isinstance(ins, Call) and ins.dst == warning.denominator), None)
    if defining_call is None:
        raise ValueError('Expected denominator defined by a Call')
    chain = []
    current = defining_call.callee
    while True:
        chain.append(current)
        nested = [ins for instrs, _ in program[current].blocks.values()
                  for ins in instrs if isinstance(ins, Call)]
        if not nested:
            break
        if len(nested) != 1:
            raise ValueError('Expected linear synthetic call chain')
        current = nested[0].callee
    available = dict(knobs)
    return [(fn, available[fn]) for fn in chain]


def refine(program, knobs: list[tuple[str, int]], budget: int, verbose=False):
    """Check reported warnings, then refine dependencies of infeasible ones.

    At no point does this procedure inspect the fixture's safe/bug labels.
    """
    policy: dict[str, int] = {}
    checked = concrete_execute(program)
    audits = []
    current = inspect_policy(program, policy)
    ignored_feasible = set()

    while True:
        spurious = [site for site in sorted(current.warnings)
                    if site in checked and checked[site] != 0]
        feasible = [site for site in sorted(current.warnings)
                    if site in checked and checked[site] == 0]
        ignored_feasible.update(feasible)
        if verbose:
            print(f'  ctx={current.contexts:<3} warnings={len(current.warnings)} '
                  f'feasible={len(feasible)} infeasible={len(spurious)} policy={policy}')
        if not spurious:
            return current, audits, ignored_feasible, []

        made_progress = False
        for site in spurious:
            chain = dependency_knobs(program, site, knobs)
            # Refinement may require several related call-string precisions.
            # Turn on one stage at a time, using analysis reports to measure cost.
            for fn, depth in chain:
                if policy.get(fn, 0) >= depth:
                    continue
                proposal = dict(policy)
                proposal[fn] = depth
                trial = inspect_policy(program, proposal)
                if trial.contexts > budget:
                    audits.append(('budget-blocked', site, fn, depth, trial.contexts))
                    if verbose:
                        print(f'    blocked: {fn}={depth} -> ctx {trial.contexts} > {budget}')
                    continue
                policy = proposal
                current = trial
                audits.append(('refined', site, fn, depth, current.contexts))
                made_progress = True
                if site not in current.warnings:
                    break
            if made_progress:
                break
        if not made_progress:
            unresolved = [site for site in sorted(current.warnings)
                          if site in checked and checked[site] != 0]
            return current, audits, ignored_feasible, unresolved


def run_trials(trials: int, seed: int):
    rng = random.Random(seed)
    print('case  ctx_budget baseline(FP/TP) final(FP/TP)  final_ctx  uniform_ctx  refinements')
    for case_id in range(trials):
        program, knobs, safe, bugs, depths, counts = make_case(rng, case_id)
        base = inspect_policy(program, {})
        full = inspect_policy(program, dict(knobs))
        # Context count budget must admit full precision for this test.
        budget = full.contexts
        result, audits, feasible, unresolved = refine(program, knobs, budget)
        # Fixture labels are used ONLY here, after policy selection.
        base_fp = len(base.warnings & safe)
        base_tp = len(base.warnings & bugs)
        result_fp = len(result.warnings & safe)
        result_tp = len(result.warnings & bugs)
        if result_tp != len(bugs) or result_fp != 0 or unresolved:
            raise AssertionError((case_id, result_fp, result_tp, unresolved))
        if set(feasible) != set(bugs):
            raise AssertionError((case_id, 'Feasibility mismatch', feasible, bugs))
        selected = ','.join(f'{name}:{depth}' for name, depth in result.policy.items())
        print(f'{case_id:>4} {budget:>10} '
              f'{base_fp}/{base_tp:>2} {result_fp}/{result_tp:>2} '
              f'{result.contexts:>10} {full.contexts:>12} {len(audits):>12}  {selected}')
    print('\nAll fully specified synthetic cases: feasible zero-divisions retained, '
          'infeasible warnings removed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be positive')
    if args.trace:
        rng = random.Random(args.seed)
        program, knobs, safe, bugs, *_ = make_case(rng, 0)
        budget = inspect_policy(program, dict(knobs)).contexts
        result, actions, feasible, unresolved = refine(program, knobs, budget, verbose=True)
        print('\nRefinement actions:', actions)
        print('Verified feasible warnings:', sorted(feasible))
        print('Unresolved:', unresolved)
        print('Final summary:', result)
    else:
        run_trials(args.trials, args.seed)


if __name__ == '__main__':
    main()
