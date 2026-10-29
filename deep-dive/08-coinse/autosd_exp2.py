"""COINSE AutoSD-inspired experiment 2: debugger evidence + patch validation.
Run: python autosd_exp2.py
Python stdlib only. Creates a small, multi-file package and a test suite.
"""
from __future__ import annotations

import ast
import bdb
import importlib.util
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

RESOLVER = '''\
def pick_timeout(user, org, default):
    """An explicit setting (including 0) overrides lower-priority settings.

    Only None means that the option was not configured.
    """
    if user:
        return user
    if org is not None:
        return org
    return default
'''
API = '''\
from .resolver import pick_timeout

def build_request(user_timeout, org_timeout, default_timeout=30):
    selected = pick_timeout(user_timeout, org_timeout, default_timeout)
    return {"timeout_seconds": selected}
'''
TESTS = '''\
import unittest
from settings_service.api import build_request

class SettingsTests(unittest.TestCase):
    def test_regular_user(self):
        self.assertEqual(build_request(15, 20)["timeout_seconds"], 15)

    def test_org_fallback(self):
        self.assertEqual(build_request(None, 20)["timeout_seconds"], 20)

    def test_default_fallback(self):
        self.assertEqual(build_request(None, None)["timeout_seconds"], 30)

    def test_bug_reproduction(self):
        self.assertEqual(build_request(0, 20)["timeout_seconds"], 0)
'''
HIDDEN = '''\
import unittest
from settings_service.api import build_request

class AdditionalTests(unittest.TestCase):
    def test_empty_string_explicit(self):
        self.assertEqual(build_request("", 20)["timeout_seconds"], "")

    def test_false_explicit(self):
        self.assertIs(build_request(False, 20)["timeout_seconds"], False)

    def test_user_zero_org_none(self):
        self.assertEqual(build_request(0, None)["timeout_seconds"], 0)
'''

PATCHES = {
    "P1_truthy_or_zero": "if user or user == 0:",
    "P2_not_none": "if user is not None:",
    "P3_not_false": "if user is not False:",
    "P4_not_empty": 'if user != "":',
}

class BreakpointRecorder(bdb.Bdb):
    def __init__(self):
        super().__init__()
        self.observations = []
    def user_line(self, frame):
        if frame.f_code.co_name == "pick_timeout":
            keys = ("user", "org", "default")
            self.observations.append({"line": frame.f_lineno, **{
                k: repr(frame.f_locals.get(k, "<missing>")) for k in keys
            }})
        self.set_continue()


def actual_debugger_probe(filename: Path):
    """Use bdb with a real breakpoint in the buggy program."""
    spec = importlib.util.spec_from_file_location("probe_resolver", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lines = filename.read_text().splitlines()
    line = next(i for i, s in enumerate(lines, 1) if s.strip() == "if user:")
    debugger = BreakpointRecorder()
    err = debugger.set_break(str(filename.resolve()), line)
    if err:
        raise RuntimeError(err)
    result = debugger.runcall(module.pick_timeout, 0, 20, 30)
    return line, debugger.observations, result


def run_tests(root: Path, suite: str):
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", suite, "-v"]
    proc = subprocess.run(cmd, cwd=root, text=True, capture_output=True)
    return proc.returncode == 0, proc.stdout + proc.stderr


def produce_patches(src: str):
    """Tiny AST-guided template synthesis. NOT LLM generation."""
    tree = ast.parse(src)
    first_if = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and
                    isinstance(n.test, ast.Name) and n.test.id == 'user')
    lines = src.splitlines(keepends=True)
    original = lines[first_if.lineno - 1]
    assert original.strip() == 'if user:'
    for name, replacement in PATCHES.items():
        altered = lines.copy()
        altered[first_if.lineno - 1] = original[:len(original)-len(original.lstrip())] + replacement + '\n'
        yield name, ''.join(altered)


def main():
    with tempfile.TemporaryDirectory(prefix="autosd_exp2_") as tmp:
        root = Path(tmp)
        pkg = root / "settings_service"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        resolver = pkg / "resolver.py"
        resolver.write_text(RESOLVER)
        (pkg / "api.py").write_text(API)
        for dirname, content in (("visible", TESTS), ("hidden", HIDDEN)):
            folder = root / dirname
            folder.mkdir()
            (folder / "test_settings.py").write_text(content)

        print("=== 1. Failing test, original implementation ===")
        passed, result = run_tests(root, "visible")
        print(f"Visible tests pass? {passed}")
        print('\n'.join(s for s in result.splitlines() if 'FAIL:' in s or 'Ran ' in s or 'FAILED (' in s))

        print("\n=== 2. Real bdb breakpoint observation ===")
        line, observations, observed = actual_debugger_probe(resolver)
        print(f"Breakpoint: resolver.py:{line}  ->  {observations}")
        print(f"Actual return for user=0, org=20: {observed!r} (expected 0)")
        assert observations and observations[0]['user'] == '0' and observed == 20

        print("\n=== 3. Hypothesis / prediction / conclusion ===")
        print("Hypothesis: `if user:` wrongly treats explicit 0 as absent")
        print("Prediction: debugger shows user=0 on that condition, and org=20 is returned")
        print("Observation matches prediction: SUPPORTED")
        print("Alternative hypothesis: org override itself is wrong -> not established")

        print("\n=== 4. Template-based patch candidates (visible suite) ===")
        survivors = []
        for name, patched in produce_patches(RESOLVER):
            resolver.write_text(patched)
            ok, _ = run_tests(root, 'visible')
            print(f"{name:24s} visible={'PASS' if ok else 'FAIL'}")
            if ok:
                survivors.append((name, patched))
        resolver.write_text(RESOLVER)

        print("\n=== 5. Additional / held-out tests ===")
        valid = []
        for name, patched in survivors:
            resolver.write_text(patched)
            ok, _ = run_tests(root, 'hidden')
            print(f"{name:24s} additional={'PASS' if ok else 'FAIL'}")
            if ok:
                valid.append(name)
        resolver.write_text(RESOLVER)
        print(f"Survivors after extra tests: {valid}")
        assert valid == ['P2_not_none'], valid
        print("\nConclusion: evidence-guided patch + extra tests selects `if user is not None:`")
        print("Limitation: template search space is manually specified; not the AutoSD LLM agent.")

if __name__ == '__main__':
    main()
