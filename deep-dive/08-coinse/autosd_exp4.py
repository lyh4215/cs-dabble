"""COINSE Deep Dive 08, Exp4: automatic hypothesis generation and debugger-based falsification.
Python 3.10+, standard library only. Run: python autosd_exp4.py

This is an AutoSD-inspired teaching experiment, NOT a reproduction of AutoSD.
"""
import ast
import bdb
import copy
import random
import textwrap
from collections import Counter
from dataclasses import dataclass

SOURCE = textwrap.dedent('''\
def resolve(user, org, default):
    if user:
        return user
    if org:
        return org
    return default
''')

# Contract/oracle is external to SUT: None alone means 'not specified'.
def oracle(user, org, default):
    if user is not None:
        return user
    if org is not None:
        return org
    return default

@dataclass(frozen=True)
class Hypothesis:
    line: int
    variable: str
    kind: str

    def name(self):
        return f"line{self.line}:{self.variable}:{self.kind}"


def candidates(source: str):
    """Extract candidate repairs from AST; not hardcoded to specific variable names."""
    tree = ast.parse(source)
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name):
            result.append(Hypothesis(node.lineno, node.test.id, "replace_truthiness_with_is_not_none"))
    return result


def mutate(source, hyp: Hypothesis):
    tree = ast.parse(source)
    count = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.lineno == hyp.line and node.test.id == hyp.variable):
            node.test = ast.Compare(left=ast.Name(id=hyp.variable, ctx=ast.Load()),
                                    ops=[ast.IsNot()], comparators=[ast.Constant(None)])
            count += 1
    if count != 1:
        raise ValueError(f"non-unique candidate {hyp}")
    ast.fix_missing_locations(tree)
    return compile(tree, f"<patch:{hyp.name()}>", "exec")


def load(code, filename):
    namespace = {}
    exec(compile(code, filename, 'exec') if isinstance(code, str) else code, namespace)
    return namespace['resolve']


class LineTracer(bdb.Bdb):
    def __init__(self):
        super().__init__()
        self.events = []

    def user_line(self, frame):
        if frame.f_code.co_name == 'resolve':
            self.events.append((frame.f_lineno, dict(frame.f_locals)))
        self.set_step()  # continue stepping, not set_continue()


def trace_call(fn, args):
    tracer = LineTracer()
    got = tracer.runcall(fn, *args)
    return got, tracer.events


def has_evidence(hyp, events):
    """Executed conditional with value 0/False/''; evidence for truthiness/None mismatch.
    This is candidate evidence, not a proof of causality.
    """
    return any(line == hyp.line and env.get(hyp.variable, object()) in (0, False, '')
               for line, env in events)


def main():
    original = load(SOURCE, '<buggy>')
    hyps = candidates(SOURCE)
    print('Hypotheses derived from AST:', [h.name() for h in hyps])
    print('Initial failing tests:')
    # Both branches buggy. First two failures identical externally, but different sites.
    failures = [(0, 20, 30), (None, 0, 30)]
    for args in failures:
        actual, events = trace_call(original, args)
        print(' input', args, 'expected', oracle(*args), 'actual', actual,
              'trace', [(n, {k: v for k, v in d.items() if k in ('user','org')}) for n,d in events])
        observed = [h.name() for h in hyps if has_evidence(h, events)]
        print('  candidate evidence:', observed)

    print('\nSingle-patch counterexample search:')
    # Test cases are deterministic, exhaustive over small values.
    values = [None, 0, False, '', 7, 20]
    tests = [(u, o, 30) for u in values for o in values]
    baseline_failed = sum(original(*args) != oracle(*args) for args in tests)
    print('baseline failures', baseline_failed, '/', len(tests))
    for h in hyps:
        patched = load(mutate(SOURCE, h), '<patched>')
        failing = [args for args in tests if patched(*args) != oracle(*args)]
        print(f' {h.name()}: {len(failing)}/{len(tests)} failures; example={failing[0] if failing else None}')

    # Apply all independent repairs only after observing that no single repair suffices.
    tree = ast.parse(SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name):
            n = node.test.id
            node.test = ast.Compare(left=ast.Name(id=n, ctx=ast.Load()),
                                    ops=[ast.IsNot()], comparators=[ast.Constant(None)])
    ast.fix_missing_locations(tree)
    fixed = load(compile(tree, '<both-patched>', 'exec'), '<both-patched>')
    assert all(fixed(*a) == oracle(*a) for a in tests)
    print(' both patches: 0/', len(tests), 'failures')

    print('\nImportant limitation: finding a falsy value at an if-statement')
    print('is only correlation; the oracle and counterexamples establish correctness')
    print('for this finite domain, not for arbitrary real-world programs.')

if __name__ == '__main__':
    main()
