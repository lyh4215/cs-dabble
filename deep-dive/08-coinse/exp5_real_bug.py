
#!/usr/bin/env python3
"""COINSE 08 Exp5: reproduce a real BugsInPy/Black issue without spoiling the patch.
Run in a disposable workspace. Requires git. Historic dependencies require Python 3.8.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
import json

REPO = "https://github.com/psf/black.git"
BUGGY = "26c9465a22c732ab1e17b0dec578fa3432e9b558"
TEST = "tests.test_black.BlackTestCase.test_works_in_mono_process_only_environment"


def run(
    args: list[str],
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(args), flush=True)

    p = subprocess.run(
        args,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print(p.stdout[-10000:])

    if check and p.returncode:
        raise SystemExit(f"Command failed: exit {p.returncode}")

    return p


def setup(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    repo = root / "black-bug1"

    if not (repo / ".git").exists():
        run(["git", "clone", REPO, str(repo)])

    run(["git", "checkout", "--detach", BUGGY], cwd=repo)

    (repo / "tests" / "__init__.py").touch(exist_ok=True)

    print("\nBuggy source ready:", repo)
    print("WARNING: do not inspect the fixing commit before solving.\n")


def repro(root: Path, python: str):
    repo = root / "black-bug1"

    if not (repo / ".git").exists():
        raise SystemExit("Run setup first")

    start = time.perf_counter()

    p = run(
        [python, "-m", "unittest", "-q", TEST],
        cwd=repo,
        check=False,
    )

    record = {
        "test": TEST,
        "returncode": p.returncode,
        "elapsed_s": round(time.perf_counter() - start, 4),
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    (root / "repro.json").write_text(json.dumps(record, indent=2))
    (root / "repro.log").write_text(p.stdout)

    print("Recorded:", root / "repro.json", root / "repro.log")

    if p.returncode == 0:
        print(
            "PASS: verify you are on buggy version "
            "and actually exercising test"
        )
    else:
        print(
            "FAIL: distinguish intended assertion failure "
            "from missing dependencies/import errors"
        )


def status(root: Path):
    repo = root / "black-bug1"

    run(["git", "status", "--short"], cwd=repo)
    run(["git", "diff", "--stat"], cwd=repo)


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument(
        "action",
        choices=["setup", "repro", "status"],
    )

    p.add_argument(
        "--root",
        type=Path,
        default=Path("coinse_exp5").resolve(),
    )

    p.add_argument(
        "--python",
        default=sys.executable,
    )

    a = p.parse_args()

    {
        "setup": lambda: setup(a.root),
        "repro": lambda: repro(a.root, a.python),
        "status": lambda: status(a.root),
    }[a.action]()
