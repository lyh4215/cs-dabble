"""ProSys UnitCon-inspired Exp 3: source-derived backward dependency slicing.

A deliberately LIMITED Java parser: flat methods, simple assignments, if guards,
no aliasing, method calls, loops, exceptions, or complex expressions.
Read/write dependency results are sound only for the included fixture/DSL.
"""
from __future__ import annotations
import re
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from typing import LiteralString
from unitcon_exp1 import verify_java

JAVA_SOURCE = r'''
class Service {
    boolean authorized = false;
    boolean enabled = false;
    String payload = "ready";
    int noise = 0;

    void reset() {
        authorized = false;
        enabled = false;
        payload = "ready";
        noise = 0;
    }
    void revoke() {
        authorized = false;
        enabled = false;
    }
    void restore() { payload = "ready"; }
    void disable() { enabled = false; }
    void ping() { noise = noise + 1; }
    void authorize() { authorized = true; }
    void enable() { if (authorized) enabled = true; }
    void nullify() { if (enabled) payload = null; }
    int target() {
        if (!enabled) return -1;
        return payload.length(); // Target NPE
    }
}
'''

FIELDS = {"authorized", "enabled", "payload", "noise"}
ACTIONS = ("CREATE", "RESET", "REVOKE", "RESTORE", "DISABLE", "PING", "AUTHORIZE", "ENABLE", "NULLIFY")


def body_of(source: str, name: str) -> str:
    # Locate method opening brace, then match balanced braces.
    pat = rf'\b(?:void|int)\s+{re.escape(name)}\s*\(\s*\)\s*\{{'
    m = re.search(pat, source)
    if not m:
        raise ValueError(f"Method not found: {name}")
    start = m.end()
    depth = 1
    for i in range(start, len(source)):
        if source[i] == '{': depth += 1
        if source[i] == '}':
            depth -= 1
            if depth == 0: return source[start:i]
    raise ValueError('Unbalanced braces')


def refs(text: str) -> set[str]:
    # Limited lexer: fixture has no field-name occurrences inside string literals.
    return set(re.findall(r'\b[A-Za-z_][A-Za-z_0-9]*\b', text)) & FIELDS


def extract_dependencies(body: str) -> list[tuple[str, set[str]]]:
    """Each (written field, read fields incl. if guard) => dependency edges.
    Supports `if (field) assignment;` and unguarded `field = expr;`.
    """
    deps = []
    # Strip comments and scan each semicolon-delimited statement.
    body = re.sub(r'//[^\n]*', '', body)
    for stmt in body.split(';'):
        stmt = stmt.strip()
        if not stmt or '=' not in stmt:
            continue
        guard_reads: set[str] = set()
        guard = re.search(r'\bif\s*\(([^()]*)\)', stmt)
        if guard:
            guard_reads = refs(guard.group(1))
            stmt = stmt[guard.end():].strip()
        m = re.fullmatch(r'(\w+)\s*=\s*(.+)', stmt, flags=re.S)
        if m and m.group(1) in FIELDS:
            deps.append((m.group(1), refs(m.group(2)) | guard_reads))
        elif '=' in stmt:
            raise ValueError(f"Unsupported assignment: {stmt!r}")
    return deps


def backward_slice(source: str, seed: set[str]):
    all_edges : list[tuple[LiteralString ,str, set[str]]]= []
    for action in ACTIONS:
        if action == 'CREATE': continue
        method = action.lower()
        for target, dependencies in extract_dependencies(body_of(source, method)):
            all_edges.append((method, target, dependencies))

    relevant = set(seed)
    rounds = []
    while True:
        added = set()
        reasons = []
        for method, written, reads in all_edges:
            if written in relevant:
                for r in sorted(reads - relevant):
                    added.add(r)
                    reasons.append(f"{written} <- {r} (method {method})")
        if not added:
            return relevant, rounds, all_edges
        relevant |= added
        rounds.append((sorted(added), reasons))


@dataclass(frozen=True)
class State:
    exists: bool = False
    authorized: bool = False
    enabled: bool = False
    payload: bool = False  # True means null
    noise: int = 0


def step(s: State, action: str) -> State:
    if action == 'CREATE': return State(exists=True)
    if not s.exists: return s
    if action == 'RESET': return State(exists=True)
    if action == 'REVOKE': return replace(s, authorized=False, enabled=False)
    if action == 'RESTORE': return replace(s, payload=False)
    if action == 'DISABLE': return replace(s, enabled=False)
    if action == 'PING': return replace(s, noise=(s.noise+1)%16)
    if action == 'AUTHORIZE': return replace(s, authorized=True)
    if action == 'ENABLE' and s.authorized: return replace(s, enabled=True)
    if action == 'NULLIFY' and s.enabled: return replace(s, payload=True)
    return s


def search(relevant: set[str] | None, depth=7):
    queue = deque([((), State())])
    seen = set()
    explored = 0
    pruned = 0
    field_order = ('authorized', 'enabled', 'payload', 'noise')
    if relevant is not None:
        seen.add((False,) + tuple(getattr(State(), f) for f in field_order if f in relevant))
    while queue:
        seq, state = queue.popleft(); explored += 1
        if state.exists and state.enabled and state.payload:
            return seq, explored, pruned
        if len(seq) >= depth: continue
        for a in ACTIONS:
            new = step(state, a)
            if relevant is not None:
                key = (new.exists,) + tuple(getattr(new, f) for f in field_order if f in relevant)
                if key in seen:
                    pruned += 1
                    continue
                seen.add(key)
            queue.append((seq + (a,), new))
    return None, explored, pruned


def main():
    target_body = body_of(JAVA_SOURCE, 'target')
    # Target reads under approximated execution conditions; a Java source field
    # named payload is represented as a boolean nullness in our search model.
    seed = refs(target_body)
    relevant, rounds, edges = backward_slice(JAVA_SOURCE, seed)
    print('Target read set:', sorted(seed))
    for i, (added, reasons) in enumerate(rounds, 1):
        print(f'Fixed point round {i}: +{added}; reasons={reasons}')
    print('Relevant fields:', sorted(relevant))
    print('Excluded:', sorted(FIELDS - relevant))
    assert relevant == {'authorized', 'enabled', 'payload'}

    for method, written, reads in edges:
        print(f'edge: {method:10} writes {written:10} reads {sorted(reads)}')

    for label, fields in [('none', None), ('target_only_UNSAFE', seed), ('source_slice', relevant), ('full', FIELDS)]:
        witness, explored, pruned = search(fields)
        print(f'{label:19} explored={explored:5} pruned={pruned:5} witness={witness}')
        if witness:
            assert verify_java(witness) == 'TARGET_NPE'

    assert search(seed)[0] is None, 'Target-only slice should fail on this model'
    assert search(relevant)[0] == ('CREATE', 'AUTHORIZE', 'ENABLE', 'NULLIFY')
    print('Assertions passed: slicing is effective; direct-only unsound.')


if __name__ == '__main__': main()
