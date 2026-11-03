#!/usr/bin/env python3
"""Deep Dive 18 / Optimus-inspired first experiment.

Observe real Linux syscalls and compare static vs naive-adaptive allowlists.
No seccomp policy is installed; this program NEVER changes system policies.

Requires Linux, gcc, strace. Usage:
    python exp1_syscall_surface.py --out results/exp1
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

SYSCALL_LINE = re.compile(
    r"^\s*(?:(?:\[pid\s+\d+\]|\d+)\s+)?([a-zA-Z_][a-zA-Z_0-9]*)\("
)
MODES = ("startup", "steady", "rare", "poison")


def parse_trace(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            match = SYSCALL_LINE.match(raw)
            if match:
                counts[match.group(1)] += 1
    return counts


def execute(command: list[str]) -> None:
    try:
        p = subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or str(e)).strip()
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{msg}") from e
    if p.stderr.strip():
        print(p.stderr.strip(), file=sys.stderr)


def names(counter: Counter[str]) -> set[str]:
    return set(counter)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/exp1", help="Output directory")
    args = parser.parse_args()
    if sys.platform != "linux":
        sys.exit("This experiment requires Linux (Colab/Codespaces/Ubuntu).");
    for program in ("gcc", "strace"):
        if not shutil.which(program):
            sys.exit(f"Missing {program}. On Ubuntu: sudo apt-get update && sudo apt-get install -y strace gcc")

    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("workload.c")
    if not source.is_file():
        sys.exit("workload.c must be next to this script")
    binary = output / "workload"
    execute(["gcc", "-O2", "-Wall", "-Wextra", str(source), "-o", str(binary)])
    traces = {}
    for mode in MODES:
        path = output / f"{mode}.strace"
        try:
            execute(["strace", "-qq", "-f", "-o", str(path), str(binary), mode])
        except RuntimeError as e:
            sys.exit(str(e) + "\nIf ptrace is restricted by the container, use an unrestricted local Linux VM. Do not modify host security settings.")
        traces[mode] = parse_trace(path)

    observed = {mode: names(traces[mode]) for mode in MODES}
    # Cold-start and steady traces form an allowlist: note this is NOT a secured policy.
    base = observed["startup"] | observed["steady"]
    rare_missing = observed["rare"] - base
    # Incorrect policy: blindly trusting all new observations as needed behavior.
    naive_updated = base | observed["poison"]
    poison_added = naive_updated - base
    # A manually verified policy would allow the known rare path but reject untrusted updates.
    verified_example = base | rare_missing

    print("=== REAL SYSCALL INVENTORY ===")
    for mode in MODES:
        print(f"{mode:8}: {len(observed[mode]):3d} distinct syscalls | " + ", ".join(sorted(observed[mode])))
    print("\n=== INFERRED PROFILES ===")
    print(f"startup + steady (observed only) : {len(base)} allowed syscalls")
    print(f"unseen legitimate rare path     : {len(rare_missing)} missing: {sorted(rare_missing)}")
    print(f"blindly learning poison trace   : {len(poison_added)} newly allowed: {sorted(poison_added)}")
    print(f"manually verified example       : {len(verified_example)} allowed syscalls")
    print("\nNOTE: Actual syscall enforcement is intentionally deferred to Exp 2.")
    print("The traces include dynamic loader/runtime calls and differ by Linux environment.")
    report = {
        "experiment": "Deep Dive 18 Optimus-inspired syscall profiling",
        "profiles_are_enforced": False,
        "platform": sys.platform,
        "traces": {mode: dict(sorted(traces[mode].items())) for mode in MODES},
        "profile": {
            "startup_plus_steady": sorted(base),
            "unseen_legitimate_calls": sorted(rare_missing),
            "untrusted_extra_calls": sorted(poison_added),
            "verified_example": sorted(verified_example),
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {output / 'report.json'}")


if __name__ == "__main__":
    main()
