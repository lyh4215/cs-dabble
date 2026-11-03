#!/usr/bin/env python3
"""Deep Dive 18 Exp3 — Optimus-inspired OFFLINE association-rule audit.

Unlike Exp1, derive support/confidence from independent real Linux commands,
not from the four evaluation modes. Rule matching does NOT install a seccomp
policy. Results are not a reproduction of the paper's 71-container evaluation.

Usage:
 python exp3_association.py --exp1 results/exp1/report.json --out results/exp3
 python exp3_association.py --exp1 results/exp1/report.json --out results/exp3_s20 --support .2
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Optional

SYSCALL_LINE = re.compile(r'^\s*(?:(?:\[pid\s+\d+\]|\d+)\s+)?([A-Za-z_][A-Za-z_0-9]*)\(')


@dataclass(frozen=True)
class Recipe:
    name: str
    args: tuple[str, ...]
    test: bool = False


def recipes(root: Path) -> list[Recipe]:
    input_file = str(root / 'input.txt')
    sorted_file = str(root / 'sorted.txt')
    csv_file = str(root / 'sample.csv')
    cases = [
        ('cat', ('cat', input_file)),
        ('head', ('head', '-n', '3', input_file)),
        ('tail', ('tail', '-n', '5', input_file)),
        ('wc-lines', ('wc', '-l', input_file)),
        ('wc-bytes', ('wc', '-c', input_file)),
        ('sort', ('sort', input_file)),
        ('uniq', ('uniq', input_file)),
        ('grep', ('grep', '-n', 'alpha', input_file)),
        ('sed', ('sed', '-n', '2,6p', input_file)),
        ('cut', ('cut', '-d', ',', '-f', '2', csv_file)),
        ('tr', ('tr', '[:lower:]', '[:upper:]')),
        ('sha256sum', ('sha256sum', input_file)),
        ('md5sum', ('md5sum', input_file)),
        ('cksum', ('cksum', input_file)),
        ('stat', ('stat', input_file)),
        ('ls', ('ls', '-la', str(root))),
        ('find', ('find', str(root), '-maxdepth', '1', '-type', 'f')),
        ('du', ('du', '-s', str(root))),
        ('dirname', ('dirname', input_file)),
        ('basename', ('basename', input_file)),
        ('date', ('date', '+%Y-%m-%d')),
        ('id', ('id',)),
        ('true', ('true',)),
        ('env', ('env',)),
        ('printf', ('printf', 'hello-%s\\n', 'world')),
        ('readlink', ('readlink', '-f', input_file)),
        ('realpath', ('realpath', input_file)),
        ('od', ('od', '-An', '-tx1', input_file)),
        ('fold', ('fold', '-w', '13', input_file)),
        ('paste', ('paste', '-d,', input_file, input_file)),
        ('comm', ('comm', sorted_file, sorted_file)),
        ('diff', ('diff', input_file, input_file)),
        ('cmp', ('cmp', input_file, input_file)),
        ('sleep', ('sleep', '0')),
        ('uname', ('uname', '-s')),
        ('whoami', ('whoami',)),
        ('groups', ('groups',)),
        ('df', ('df', '-P', str(root))),
        ('test', ('test', '-f', input_file)),
        ('expr', ('expr', '4', '+', '5')),
    ]
    # Deterministic holdout by distinct program, not repeated random invocations
    test_names = {'od','fold','paste','comm','diff','cmp','sleep','uname','whoami','groups','df','test','expr'}
    return [Recipe(name, args, name in test_names) for name, args in cases]


def parse_trace(path: Path) -> set[str]:
    seen = set()
    for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
        m = SYSCALL_LINE.match(line)
        if m: seen.add(m.group(1))
    return seen


def run_trace(recipe: Recipe, output: Path, input_file: Path) -> Optional[set[str]]:
    if not shutil.which(recipe.args[0]):
        print(f'  skip {recipe.name}: command missing')
        return None
    path = output / f'{recipe.name}.strace'
    # No shell, network, sudo, Docker, or privileged operations in the corpus
    stdin = input_file.open('rb') if recipe.name == 'tr' else subprocess.DEVNULL
    try:
        p = subprocess.run(['strace','-qq','-f','-o',str(path),'--',*recipe.args],
                           stdin=stdin, stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,
                           timeout=25,check=False)
    except subprocess.TimeoutExpired:
        print(f'  skip {recipe.name}: timeout')
        return None
    finally:
        if recipe.name == 'tr': stdin.close()
    if p.returncode != 0:
        print(f'  skip {recipe.name}: returncode={p.returncode} {p.stderr.decode(errors="replace")[:150]}')
        return None
    seen = parse_trace(path)
    if len(seen) < 3:
        print(f'  skip {recipe.name}: only {len(seen)} parsed syscalls')
        return None
    return seen


def mine_rules(train: list[set[str]], min_support:float, min_conf:float) -> tuple[list[dict],set[str]]:
    # Pairwise two-item rules X -> Y, faithful to paper's 2-item pruning idea;
    # no FP-Growth implementation because mining only 2-itemsets here.
    if not train: raise ValueError('Empty corpus')
    n=len(train)
    singles=Counter(s for tr in train for s in tr)
    pairs=Counter()
    for tr in train:
        ordered=sorted(tr)
        for i,x in enumerate(ordered):
            for y in ordered[i+1:]: pairs[(x,y)]+=1
    rules=[]
    for (x,y),joint in sorted(pairs.items()):
        support=joint/n
        if support<min_support:continue
        for antecedent, consequent in ((x,y),(y,x)):
            confidence=joint/singles[antecedent]
            if confidence>=min_conf:
                rules.append({'if':antecedent,'then':consequent,
                              'support':round(support,5),'confidence':round(confidence,5),
                              'joint_count':joint,'antecedent_count':singles[antecedent]})
    rules.sort(key=lambda r:(r['if'],r['then']))
    return rules,set(singles)


def candidates(allowed:set[str], rules:list[dict]) -> set[str]:
    return {r['then'] for r in rules if r['if'] in allowed}-allowed


def rates(base:set[str], new_legit:set[str], new_noise:set[str], rules:list[dict]) -> dict:
    candidate=candidates(base,rules)
    return {'candidate_count':len(candidate), 'candidate_names':sorted(candidate),
            'legit_accepted':sorted(new_legit & candidate), 'legit_rejected':sorted(new_legit-candidate),
            'noise_accepted':sorted(new_noise & candidate), 'noise_rejected':sorted(new_noise-candidate),
            'legit_recall':(len(new_legit&candidate)/len(new_legit) if new_legit else None),
            'noise_accept_rate':(len(new_noise&candidate)/len(new_noise) if new_noise else None)}


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exp1',default='results/exp1/report.json')
    ap.add_argument('--out',default='results/exp3')
    ap.add_argument('--support',type=float,default=.40,help='min joint support (Optimus: 0.40)')
    ap.add_argument('--confidence',type=float,default=.80,help='min conditional confidence (Optimus: 0.80)')
    a=ap.parse_args()
    if not 0<a.support<=1 or not 0<a.confidence<=1:ap.error('support/confidence must be within (0,1]')
    if sys.platform!='linux' or not shutil.which('strace'):ap.error('Linux and strace are required')
    report_path=Path(a.exp1)
    if not report_path.is_file():ap.error(f'Exp1 report not found: {report_path}')
    exp1=json.loads(report_path.read_text())
    base=set(exp1['profile']['startup_plus_steady'])
    legit=set(exp1['profile']['unseen_legitimate_calls'])
    noise=set(exp1['profile']['untrusted_extra_calls'])
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    fixtures=out/'fixtures';fixtures.mkdir(exist_ok=True)
    input_file=fixtures/'input.txt'
    input_file.write_text('alpha apple\nbeta banana\nalpha apricot\ngamma grape\nbeta berry\nalpha almond\n',encoding='utf-8')
    (fixtures/'sorted.txt').write_text('a\nb\nc\n',encoding='utf-8')
    (fixtures/'sample.csv').write_text('id,name\n1,alpha\n2,beta\n',encoding='utf-8')
    trace_dir=out/'traces';trace_dir.mkdir(exist_ok=True)
    train=[]; test=[];rows=[]
    print('=== COLLECTING INDEPENDENT REAL PROCESS TRACES ===')
    for recipe in recipes(fixtures):
        seen=run_trace(recipe,trace_dir,input_file)
        if seen is None:continue
        (test if recipe.test else train).append((recipe.name,seen))
        rows.append({'recipe':recipe.name,'split':'test' if recipe.test else 'train', 'syscalls':sorted(seen)})
        print(f'  {recipe.name:<14} {"test" if recipe.test else "train":<5} {len(seen):2} syscalls')
    if len(train)<12 or len(test)<5:
        ap.error(f'Insufficient independent commands: train={len(train)}, test={len(test)}. Check strace and installed coreutils.')
    rules, vocab=mine_rules([s for _,s in train],a.support,a.confidence)
    exp1_audit=rates(base,legit,noise,rules)
    held=[]
    for name, calls in test:
        truly_new=calls-base
        accepted=candidates(base,rules)&truly_new
        held.append({'recipe':name,'actual_extra':sorted(truly_new),'extra_accepted':sorted(accepted),
                     'extra_rejected':sorted(truly_new-accepted),
                     'extra_recall':len(accepted)/len(truly_new) if truly_new else None})
    total_new=sum(len(x['actual_extra']) for x in held)
    total_accepted=sum(len(x['extra_accepted']) for x in held)
    held_recall=(total_accepted/total_new) if total_new else None
    print('\n=== ASSOCIATION RULES (offline actual Linux processes) ===')
    print(f'train={len(train)} independent commands | heldout={len(test)} | vocabulary={len(vocab)}')
    print(f'min_support={a.support:.2f} min_confidence={a.confidence:.2f} pairwise_rules={len(rules)}')
    print('\n=== EXP1 POLICY UPDATE AUDIT ===')
    print(f'Baseline allowed: {len(base)} | inferred candidates: {len(exp1_audit["candidate_names"])}')
    print(f'rare legitimate accepted {exp1_audit["legit_accepted"]} rejected {exp1_audit["legit_rejected"]}')
    print(f'poison extras     accepted {exp1_audit["noise_accepted"]} rejected {exp1_audit["noise_rejected"]}')
    print('\n=== HELD-OUT EXECUTIONS (not used to learn rules) ===')
    print(f'true new syscall occurrences={total_new} | association-accepted={total_accepted} | recall={held_recall if held_recall is not None else "N/A"}')
    for x in held:
        print(f' {x["recipe"]:<12} new={len(x["actual_extra"]):2d} accepted={len(x["extra_accepted"]):2d} missed={x["extra_rejected"]}')
    print('\nWARNING: This experiment only audits rules; it DOES NOT enforce seccomp.')
    print('A legitimate rare syscall absent in the offline corpus cannot be inferred by co-occurrence.')
    print('These traces are host CLI utilities, NOT independent Docker images; not the paper benchmark.')
    result={'title':'Deep Dive 18 Exp3 Optimus-inspired association audit','settings':{'support':a.support,'confidence':a.confidence},
            'train_count':len(train),'heldout_count':len(test),'syscall_vocabulary_size':len(vocab),
            'number_of_rules':len(rules),'exp1_probe':exp1_audit,'heldout':held,
            'heldout_extra_recall':held_recall,'corpus':rows,'rules':rules,
            'limits':['Independent Linux CLI program traces, not Docker images',
                      'Pairwise rules only, no FP-growth',
                      'No seccomp enforcement in Exp3',
                      'The paper uses a richer container-based training corpus and runtime updating']}
    (out/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    (out/'rules.json').write_text(json.dumps(rules,indent=2)+'\n')
    print(f'Report: {out/"report.json"}')

if __name__=='__main__':main()
