"""COINSE Deep Dive 08, Exp 3: causal debugging with real execution tracing.
Run: python autosd_exp3.py
Requires: Python 3.10+, standard library only.
"""
from __future__ import annotations

import bdb
import importlib.util
from pathlib import Path
import tempfile
import textwrap

# The same observed failing output can result from either bug.
# Variant A: user setting checked by truthiness (user=0 ignored).
# Variant B: organization setting checked by truthiness (org=0 ignored).
SOURCES = {
    "user_bug": '''
        def choose(user, org, default):
            if user:                    # BUG_USER
                return user
            if org is not None:        # CORRECT_ORG
                return org
            return default
    ''',
    "org_bug": '''
        def choose(user, org, default):
            if user is not None:       # CORRECT_USER
                return user
            if org:                    # BUG_ORG
                return org
            return default
    ''',
}

# All cases use the same observed symptom: returned 30, expected 0.
FAILING = {
    "user_bug": (0, None, 30),
    "org_bug": (None, 0, 30),
}

# Test suite is separate from the diagnostic probe; it checks both falsy values.
REGRESSION = [
    (0, 17, 30), (False, 17, 30), ("", 17, 30),
    (None, 0, 30), (None, False, 30), (None, "", 30),
    (None, None, 30), (12, 17, 30), (None, 17, 30),
]


def specification(user, org, default):
    if user is not None:
        return user
    if org is not None:
        return org
    return default


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("debug_target", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TraceDebugger(bdb.Bdb):
    """Collect actual executed lines and local variables in choose()."""
    def __init__(self, target: Path):
        super().__init__()
        self.target = str(target.resolve())
        self.events = []

    def user_line(self, frame):
        if (frame.f_code.co_name == "choose" and
                frame.f_code.co_filename == self.target):
            self.events.append((frame.f_lineno, dict(frame.f_locals)))


def trace(path: Path, case):
    debugger = TraceDebugger(path)
    result = debugger.runcall(load_module(path).choose, *case)
    return result, debugger.events


def patch(source: str, hypothesis: str):
    """The candidates are predefined: this is NOT LLM hypothesis generation."""
    if hypothesis == "user_truthiness":
        old, new = "if user: ", "if user is not None: "
    elif hypothesis == "org_truthiness":
        old, new = "if org: ", "if org is not None: "
    else:
        raise ValueError(hypothesis)
    # Match the condition but preserve the explanatory comment on its line.
    lines = source.splitlines()
    for i, line in enumerate(lines):
        if old in line:
            lines[i] = line.replace(old, new, 1)
            return "\n".join(lines) + "\n"
    return source  # Wrong-location patch: no change.


def validate(path: Path, test_cases):
    fn = load_module(path).choose
    failures = []
    for args in test_cases:
        actual = fn(*args)
        expected = specification(*args)
        if type(actual) is not type(expected) or actual != expected:
            failures.append((args, expected, actual))
    return failures


def diagnose_from_trace(source: str, events, failing_case):
    """Infer the first incorrectly skipped branch from executed-line evidence.

    This is a deliberately restricted inference rule for the two-fixture family.
    """
    lines = source.splitlines()
    visited = {line for line, _ in events}
    for line_no, env in events:
        statement = lines[line_no - 1].strip()
        if statement.startswith("if user:") and env.get("user") is not None:
            # The next line, 'return user', is skipped for falsy user.
            if line_no + 1 not in visited:
                return "user_truthiness"
        if statement.startswith("if org:") and env.get("org") is not None:
            if line_no + 1 not in visited:
                return "org_truthiness"
    return None


def main():
    with tempfile.TemporaryDirectory(prefix="coinse_exp3_") as tmp:
        root = Path(tmp)
        success_symptom = success_trace = 0
        for variant, template in SOURCES.items():
            # Source 저장
            source = textwrap.dedent(template).lstrip("\n")
            source_path = root / f"{variant}.py"
            source_path.write_text(source, encoding="utf-8")
            
            failing_case = FAILING[variant]
            observed, events = trace(source_path, failing_case)
            expected = specification(*failing_case)
            assert observed == 30 and expected == 0

            # BASELINE: only knows the failure output. Always assumes user branch.
            symptom_guess = "user_truthiness"
            # TRACE: inspect actual executed branch and local environment.
            trace_guess = diagnose_from_trace(source, events, failing_case)
            print(f"\n[{variant}] input={failing_case} expected={expected!r} got={observed!r}")
            print("  Executed lines:")
            for ln, env in events:
                print(f"    L{ln}: {source.splitlines()[ln - 1].strip()}  locals={env}")
            print(f"  symptom-only guess: {symptom_guess}")
            print(f"  trace-based guess:  {trace_guess}")

            for name, hypothesis in [("symptom", symptom_guess), ("trace", trace_guess)]:
                if hypothesis is None:
                    raise AssertionError("Could not find a hypothesis")
                patched = patch(source, hypothesis)
                candidate_path = root / f"{variant}_{name}.py"
                candidate_path.write_text(patched, encoding="utf-8")
                fails = validate(candidate_path, REGRESSION)
                passed = len(REGRESSION) - len(fails)
                print(f"  {name:7s} patch: {passed}/{len(REGRESSION)} regression tests pass")
                if fails:
                    print(f"    first failure: {fails[0]}")
                if name == "symptom" and not fails:
                    success_symptom += 1
                if name == "trace" and not fails:
                    success_trace += 1
        print(f"\nSUMMARY: symptom-only repaired {success_symptom}/2; "
              f"trace-guided repaired {success_trace}/2")
        assert (success_symptom, success_trace) == (1, 2)


if __name__ == "__main__":
    main()
