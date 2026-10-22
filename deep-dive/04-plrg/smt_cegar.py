"""PLRG Experiment 6: SMT-checked warnings + call-string refinement.

Python 3.11+. Install: python -m pip install z3-solver
Place alongside engine.py, policy_engine.py, randomized_refinement.py,
cegar_refinement.py. Run: python smt_cegar.py --trace

Scope: finite, acyclic toy IR with Const, Call, Div, Return,
BranchPositive, Jump. Enumerates all feasible CFG paths up to a path cap.
SMT classifies a warning as feasible if there exists a path reaching the
warning with denominator zero. It does NOT rely on supplied FP labels to
choose refinements. The test fixtures happen to use concrete constants;
the checker retains path constraints symbolically rather than replaying a
single execution. This is a CEGAR-inspired demonstration, not general CEGAR.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

try:
    from z3 import And, BoolVal, If, IntVal, Solver, sat, unsat, simplify
except ImportError as exc:
    raise SystemExit("Missing Z3: run `python -m pip install z3-solver`") from exc

from engine import BranchPositive, Call, Const, Div, Jump, Return
from policy_engine import Analyzer
from randomized_refinement import make_case
from cegar_refinement import dependency_knobs


@dataclass(frozen=True)
class DivisionPath:
    site: str
    condition: object   # Z3 BoolRef
    denominator: object # Z3 ArithRef
    calls: tuple[str, ...]


@dataclass(frozen=True)
class State:
    env: dict
    conditions: tuple
    callsites: tuple[str, ...]
    divisions: tuple[DivisionPath, ...]


class SymbolicChecker:
    """Bounded symbolic interpreter. A call returns possible (value, state) pairs."""

    def __init__(self, program, max_paths=10000):
        self.program = program
        self.max_paths = max_paths
        self.visits = 0
        self.solver_calls = 0
        self.division_paths: list[DivisionPath] = []

    def possible(self, conditions) -> bool:
        self.solver_calls += 1
        solver = Solver()
        solver.add(*conditions)
        result = solver.check()
        if result == sat:
            return True
        if result == unsat:
            return False
        raise RuntimeError(f'Solver returned {result}; cannot conclude feasibility')

    def execute(self, fn_name, argument=None, conditions=(), callsites=(), active=()):
        if fn_name in active:
            raise ValueError('Recursive call graph unsupported')
        fn = self.program[fn_name]
        initial = {fn.param: argument} if fn.param is not None else {}
        pending = [('entry', State(initial, tuple(conditions), tuple(callsites), ()))]
        outcomes = []
        while pending:
            self.visits += 1
            if self.visits > self.max_paths:
                raise RuntimeError('Symbolic path cap exceeded; feasibility unknown')
            block, state = pending.pop()
            instructions, terminal = fn.blocks[block]
            states = [state]
            for instr in instructions:
                following = []
                for s in states:
                    if isinstance(instr, Const):
                        env = dict(s.env)
                        env[instr.dst] = IntVal(instr.value)
                        following.append(State(env, s.conditions, s.callsites, s.divisions))
                    elif isinstance(instr, Call):
                        arg = s.env[instr.arg] if instr.arg is not None else None
                        # Split by callee's actual symbolic execution outcomes.
                        for val, result_state in self.execute(
                            instr.callee, arg, s.conditions,
                            s.callsites + (instr.site,), active + (fn_name,)
                        ):
                            env = dict(s.env)
                            env[instr.dst] = val
                            following.append(State(env, result_state.conditions,
                                                   s.callsites, s.divisions + result_state.divisions))
                    elif isinstance(instr, Div):
                        d = s.env[instr.denominator]
                        self.division_paths.append(DivisionPath(
                            instr.site, And(*s.conditions) if s.conditions else BoolVal(True),
                            d, s.callsites))
                        # Continue with dummy value if denominator is zero.
                        # Branching on the division result is outside scope.
                        env = dict(s.env)
                        env[instr.dst] = If(d == 0, IntVal(0), IntVal(instr.numerator) / d)
                        following.append(State(env, s.conditions, s.callsites, s.divisions))
                    else:
                        raise TypeError(f'Unsupported instruction: {instr!r}')
                states = following
            for s in states:
                if isinstance(terminal, Return):
                    outcomes.append((s.env[terminal.var], s))
                elif isinstance(terminal, BranchPositive):
                    condition = s.env[terminal.var] > 0
                    for target, c in ((terminal.yes, condition),
                                      (terminal.no, ~condition)):
                        conds = s.conditions + (c,)
                        if self.possible(conds):
                            pending.append((target, State(dict(s.env), conds,
                                                          s.callsites, s.divisions)))
                elif isinstance(terminal, Jump):
                    pending.append((terminal.target, s))
                else:
                    raise TypeError(f'Unsupported terminal: {terminal!r}')
        return outcomes

    def check_warnings(self, warning_sites: set[str]):
        """SAT means genuine feasible path; UNSAT means spurious warning."""
        if not self.division_paths:
            self.execute('main')
        by_site: dict[str, list[DivisionPath]] = {}
        for path in self.division_paths:
            by_site.setdefault(path.site, []).append(path)
        decisions = {}
        witnesses = {}
        for site in sorted(warning_sites):
            paths = by_site.get(site, [])
            if not paths:
                # Do not silently classify a site our verifier failed to reach.
                decisions[site] = 'UNKNOWN'
                continue
            found = False
            for path in paths:
                self.solver_calls += 1
                solver = Solver()
                solver.add(path.condition, path.denominator == 0)
                answer = solver.check()
                if answer == sat:
                    found = True
                    witnesses[site] = (path.calls, str(solver.model()))
                    break
                if answer != unsat:
                    decisions[site] = 'UNKNOWN'
                    break
            if site not in decisions:
                decisions[site] = 'SAT' if found else 'UNSAT'
        return decisions, witnesses


def run_refinement(program, knobs, budget, trace=False):
    """Refine only for UNSAT warnings; no fixture safe/bug labels used."""
    policy: dict[str, int] = {}
    base = Analyzer(policy, program).run()
    checker = SymbolicChecker(program)
    decisions, witnesses = checker.check_warnings(set(base.warnings))
    if trace:
        print('SMT decisions:', decisions)
        print('SAT witnesses:', witnesses)
        print('SMT queries:', checker.solver_calls, 'symbolic visits:', checker.visits)
    audited = []
    current = base
    while True:
        spurious = [s for s in sorted(current.warnings) if decisions.get(s) == 'UNSAT']
        if not spurious:
            break
        changed = False
        for site in spurious:
            for fn, k in dependency_knobs(program, site, knobs):
                if policy.get(fn, 0) >= k:
                    continue
                proposed = {**policy, fn: k}
                next_result = Analyzer(proposed, program).run()
                if next_result.contexts > budget:
                    audited.append(('budget', site, fn, k, next_result.contexts))
                    continue
                policy = proposed
                current = next_result
                audited.append(('refined', site, fn, k, current.contexts))
                changed = True
                if site not in current.warnings:
                    break
            if changed:
                break
        if not changed:
            break
    return base, current, decisions, witnesses, audited, checker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    rng = random.Random(args.seed)
    print('case  initial_warnings  final_warnings  initial_ctx  final_ctx  sat  unsat  smt_queries')
    for case_id in range(args.trials):
        program, knobs, safe, bugs, *_ = make_case(rng, case_id)
        budget = Analyzer(dict(knobs), program).run().contexts
        base, final, decisions, witnesses, audited, checker = run_refinement(
            program, knobs, budget, trace=args.trace and case_id == 0)
        # Fixture labels used here ONLY for reporting/tests, never selection.
        assert set(bugs) <= final.warnings, 'True zero-division lost'
        assert not (set(safe) & final.warnings), 'Spurious warning remained'
        assert all(decisions.get(s) == 'UNSAT' for s in safe)
        assert all(decisions.get(s) == 'SAT' for s in bugs)
        print(f'{case_id:>4} {len(base.warnings):>17} {len(final.warnings):>15} '
              f'{base.contexts:>12} {final.contexts:>10} '
              f'{sum(v == "SAT" for v in decisions.values()):>4} '
              f'{sum(v == "UNSAT" for v in decisions.values()):>6} '
              f'{checker.solver_calls:>11}')
        if args.trace and case_id == 0:
            print('Refinement actions:', *audited, sep='\n  ')
            print('Final policy:', {a[2]: a[3] for a in audited if a[0] == 'refined'})
    print('NOTE: These synthetic programs have constant inputs; symbolic branching'
          ' is encoded in Z3, but broader input domains are future work.')


if __name__ == '__main__':
    main()
