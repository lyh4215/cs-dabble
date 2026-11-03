# Deep Dive 18 — Exp 3: Association Rule Mining on real syscall traces

Run from `deep-dive/18-cysec` after Exp 1 + Exp 2. Requires `strace` and normal Linux core utilities (already installed on most Codespaces).

```
python exp3_association.py --exp1 results/exp1/report.json --out results/exp3
```

Run a support sensitivity experiment without contaminating the original output:

```
python exp3_association.py --exp1 results/exp1/report.json --out results/exp3_s20 --support 0.20
```

Focus on:
1. Legitimate rare `getrusage` accepted/rejected? Why? 
2. Poison extra `uname` and `getppid` accepted/rejected?
3. Held-out recall for new real Linux CLI syscall occurrences.
4. Would blindly increasing the allowlist or lowering support harm security?

## Scope / caveats

- Offline training corpus consists of distinct **real Linux CLI command traces** automatically collected via `strace`; NOT the original paper's 71 Docker images.
- Training and holdout commands are distinct. Pairwise co-occurrence rules use support >= 0.40 and confidence >= 0.80 by default, inspired by Optimus. No FP-Growth implementation because only 2-item pairwise rules are mined.
- This is a **rule audit**. It DOES NOT install seccomp and does not alter host policy. The `Exp1` probes were not used as the training transactions.
- Held-out metric tests whether new syscall names *observed* in held-out commands would be validated by association rules; it does NOT measure attack prevention or real container availability.
- It cannot infer a rare legitimate syscall that never appears in the offline corpus. It may also accept an unwanted syscall that is correlated with benign operations. A correlation rule is not a proof of safety.
