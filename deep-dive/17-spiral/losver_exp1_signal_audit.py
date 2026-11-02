#!/usr/bin/env python3
"""Deep Dive 17 / LOSVER, Experiment 1: audit patch-derived line signals.

Input: official PrimeVul paired JSONL (vulnerable and patched functions in consecutive rows).
Output: data quality report and simple line-selection control baselines.
No model or third-party libraries required.
"""
import argparse
import difflib
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

# A deliberately weak, transparent lexical control, NOT a vulnerability oracle.
SUSPICIOUS = re.compile(
    r"\b(?:strcpy|strcat|sprintf|gets|memcpy|malloc|free|realloc|"
    r"scanf|read|write)\s*\(|\b(?:sizeof|NULL)\b|"
    r"\b(?:if|while|for)\s*\(|(?:->|\[[^\]]+\])"
)


def changed_before_lines(before: str, after: str):
    """0-indexed lines deleted/replaced in BEFORE, excluding insert-only hunks."""
    a, b = before.splitlines(), after.splitlines()
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    line_ids = set()
    inserted = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("delete", "replace"):
            line_ids.update(range(i1, i2))
        if tag == "insert":
            inserted += j2 - j1
    return line_ids, inserted


def rank_longest(lines):
    return sorted(range(len(lines)), key=lambda i: (-len(lines[i].strip()), i))


def rank_lexical(lines):
    return sorted(
        range(len(lines)),
        key=lambda i: (-len(SUSPICIOUS.findall(lines[i])), -len(lines[i].strip()), i),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path, help="PrimeVul's primevul_test_paired.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--examples", type=int, default=3)
    args = p.parse_args()

    with args.data.open(encoding="utf-8") as f:
        samples = [json.loads(x) for x in f if x.strip()]
    if len(samples) % 2:
        print("WARNING: odd number of rows; ignoring the last unmatched row")
    totals = Counter()
    valid = []
    example_cases = []
    for ix in range(0, len(samples) - 1, 2):
        a, b = samples[ix], samples[ix + 1]
        labels = (a.get("target"), b.get("target"))
        if set(labels) != {0, 1} or "func" not in a or "func" not in b:
            totals["invalid_pairs"] += 1
            continue
        vulnerable = a if a["target"] == 1 else b
        patched = b if a["target"] == 1 else a
        before, after = vulnerable["func"], patched["func"]
        lines = before.splitlines()
        if not lines:
            totals["empty_functions"] += 1
            continue
        changed, inserted = changed_before_lines(before, after)
        totals["valid_pairs"] += 1
        if before == after:
            totals["identical_pairs"] += 1
        if not changed:
            totals["no_removed_or_replaced_lines"] += 1
            if inserted:
                totals["insert_only_patches"] += 1
            continue
        totals["eligible_pairs"] += 1
        totals["total_eligible_lines"] += len(lines)
        totals["total_changed_lines"] += len(changed)
        valid.append((lines, changed))
        if len(example_cases) < args.examples:
            example_cases.append((vulnerable.get("idx", ix), len(lines), sorted(changed), [lines[j].strip()[:100] for j in sorted(changed)[:3]]))

    print("=== DATA INTEGRITY ===")
    print("rows:", len(samples), "adjacent candidate pairs:", len(samples) // 2)
    for k, v in sorted(totals.items()):
        print(f"{k}: {v}")
    if not valid:
        raise SystemExit("No eligible pairs: check file format and pairing assumptions.")
    print("mean changed-line fraction:", round(statistics.mean(len(c)/len(l) for l, c in valid), 4))
    print("NOTE: changed lines are PATCH locations, not verified vulnerable lines.")

    ks = (1, 3, 5)
    results = {k: {name: [] for name in ("random", "longest", "lexical", "oracle")} for k in ks}
    rnd = random.Random(args.seed)
    for lines, changed in valid:
        longest = rank_longest(lines)
        lexical = rank_lexical(lines)
        oracle = list(sorted(changed)) + [i for i in range(len(lines)) if i not in changed]
        for k in ks:
            k_eff = min(k, len(lines))
            results[k]["longest"].append(int(bool(set(longest[:k_eff]) & changed)))
            results[k]["lexical"].append(int(bool(set(lexical[:k_eff]) & changed)))
            results[k]["oracle"].append(int(bool(set(oracle[:k_eff]) & changed)))
            # Repeated random draw gives stable Monte Carlo reference.
            random_hits = sum(bool(set(rnd.sample(range(len(lines)), k_eff)) & changed) for _ in range(args.repeats))
            results[k]["random"].append(random_hits / args.repeats)

    print("\n=== TOP-K PATCH-LINE HIT RATE (eligible pairs only) ===")
    print(" k    random  longest  lexical  oracle")
    for k in ks:
        print(f" {k:<4} " + "  ".join(f"{statistics.mean(results[k][name]):.3f}" for name in ("random", "longest", "lexical", "oracle")))
    print("\n=== EXAMPLES (PATCH LINE NUMBERS, 1-BASED) ===")
    for idx, n, changed, snippets in example_cases:
        print(f"idx={idx}, lines={n}, changed={[j+1 for j in changed]}")
        for s in snippets:
            print("   ", s)
    print("\nINTERPRETATION: This is not vulnerability detection accuracy.")
    print("Next: compare an independently trained LOSVER line-localizer with these matched controls.")


if __name__ == "__main__":
    main()
