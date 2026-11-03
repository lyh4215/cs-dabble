#!/usr/bin/env python3
"""Exp2: ENFORCE syscall allowlists from Exp1 using actual Linux seccomp-BPF.

Run in the same Codespaces directory as exp1, with the Exp1 report on disk:
    python exp2_seccomp_enforce.py --report results/exp1/report.json --out results/exp2

Requirements: Linux x86-64, gcc; no sudo or libseccomp required.
Never changes host settings, only restricts short-lived experimental subprocesses.
"""
from __future__ import annotations
import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

NAME = re.compile(r"^[a-z][a-z0-9_]*$")
# profile, probe, expected allowed at the syscall level
CASES = [
    ("observed", "steady", True),
    ("observed", "rare", False),
    ("observed", "poison", False),
    ("verified", "rare", True),
    ("verified", "poison", False),
    ("poisoned", "poison", True),
    ("poisoned", "rare", False),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", default="results/exp1/report.json")
    p.add_argument("--out", default="results/exp2")
    args = p.parse_args()
    if sys.platform != "linux" or platform.machine() not in {"x86_64", "AMD64"}:
        p.error("Only x86-64 Linux is supported in this educational BPF example")
    if not shutil.which("gcc"):
        p.error("gcc missing: sudo apt-get install -y gcc")
    src = Path(__file__).with_name("workload_seccomp.c")
    if not src.is_file():
        p.error(f"Missing {src.name}; place both files in same directory")
    report_path = Path(args.report)
    if not report_path.is_file():
        p.error(f"Exp1 report not found at {report_path}; run Exp1 first")
    data = json.loads(report_path.read_text(encoding="utf-8"))
    if data.get("profiles_are_enforced") is not False:
        p.error("The provided report is not the expected Exp1 format")
    base = set(data["profile"]["startup_plus_steady"])
    verified = set(data["profile"]["verified_example"])
    untrusted = set(data["profile"]["untrusted_extra_calls"])
    profiles = {"observed": base, "verified": verified, "poisoned": base | untrusted}
    for names in profiles.values():
        if not names or any(not NAME.fullmatch(x) for x in names):
            p.error("Invalid or empty syscall names from report")
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    executables = {}
    for profile, names in profiles.items():
        directory = output / profile
        directory.mkdir(exist_ok=True)
        header = directory / "allowed_syscalls.h"
        header.write_text("#include <sys/syscall.h>\nstatic const int allowed_syscalls[] = {\n" + "".join(f"    SYS_{name},\n" for name in sorted(names)) + "};\n", encoding="utf-8")
        binary = directory / "workload_seccomp"
        compile_args = ["gcc", "-O2", "-Wall", "-Wextra", "-std=c11", "-I", str(directory), str(src), "-o", str(binary)]
        result = subprocess.run(compile_args, capture_output=True, text=True)
        if result.returncode:
            p.error(f"Could not compile {profile} policy; check kernel syscall names:\n{result.stderr}")
        executables[profile] = binary

    rows = []
    print("=== ACTUAL SECCOMP-BPF ENFORCEMENT (child processes only) ===")
    print("Profile    Mode      Expected   Actual     Verdict")
    for profile, mode, expected in CASES:
        proc = subprocess.run([str(executables[profile]), mode], capture_output=True, text=True)
        if proc.returncode == 90:
            print(proc.stderr)
            p.error("Kernel refused seccomp installation; Codespaces security settings may prevent nested seccomp. Do not loosen host security settings")
        allowed = proc.returncode == 0
        # Avoid interpreting signals/segfaults as an expected block.
        verdict = "OK" if (allowed == expected and proc.returncode in {0, 23}) else "MISMATCH"
        print(f"{profile:<10} {mode:<9} {('PASS' if expected else 'BLOCK'):<10} {('PASS' if allowed else 'BLOCK'):<10} {verdict}")
        print("  " + " | ".join(x for x in proc.stdout.strip().splitlines() if x.startswith("CALL") or x.startswith("RESULT")))
        if proc.stderr:
            print("  stderr:", proc.stderr.strip())
        rows.append({"profile": profile, "mode": mode, "expected_pass": expected, "actual_pass": allowed, "returncode": proc.returncode, "verdict": verdict, "stdout": proc.stdout, "stderr": proc.stderr})
    summary = {"experiment": "Deep Dive 18 Optimus-inspired Exp 2", "kernel_enforcement": "per-process seccomp-BPF (x86_64)", "profiles": {k: sorted(v) for k, v in profiles.items()}, "cases": rows, "all_matched": all(x["verdict"] == "OK" for x in rows)}
    (output / "report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nall_matched={summary['all_matched']} | Report: {output / 'report.json'}")
    if not summary["all_matched"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
