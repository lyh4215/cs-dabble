"""UnitCon deep dive exp4: heap alias abstraction, safe vs unsafe merging.
Pure Python 3.11, optional Java witness checking if javac/java available.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import argparse, subprocess, tempfile, shutil
from pathlib import Path

ACTIONS = ('NEW_A', 'ALIAS_B_A', 'NEW_B', 'SET_B_NULL', 'RESTORE_A')

@dataclass(frozen=True)
class State:
    a: int = -1
    b: int = -1
    obj1_null: bool = False
    obj2_null: bool = False
    next_obj: int = 1

    def slot_null(self, ref: int) -> bool:
        return self.obj1_null if ref == 1 else self.obj2_null if ref == 2 else False


def transition(s: State, action: str) -> State:
    if action == 'NEW_A':
        # Limit to two abstract allocation objects, immutable identities.
        if s.a != -1 or s.next_obj > 2:
            return s
        return State(s.next_obj, s.b, s.obj1_null, s.obj2_null, s.next_obj + 1)
    if action == 'NEW_B':
        if s.b != -1 or s.next_obj > 2:
            return s
        return State(s.a, s.next_obj, s.obj1_null, s.obj2_null, s.next_obj + 1)
    if action == 'ALIAS_B_A':
        if s.a == -1: return s
        return State(s.a, s.a, s.obj1_null, s.obj2_null, s.next_obj)
    if action in ('SET_B_NULL', 'SET_A_NULL', 'RESTORE_A'):
        ref = s.b if action == 'SET_B_NULL' else s.a
        if ref == -1: return s
        value = action != 'RESTORE_A'
        if ref == 1:
            return State(s.a, s.b, value, s.obj2_null, s.next_obj)
        return State(s.a, s.b, s.obj1_null, value, s.next_obj)
    return s


def target(s: State) -> bool:
    return s.a != -1 and s.slot_null(s.a)


def key(s: State, mode: str):
    if mode == 'full': return s
    if mode == 'name_only_UNSAFE':
        # Incorrectly assumes SET_B_NULL cannot affect target through a.
        # Deliberately drops b and the pointee's nullness; this is not a sound key.
        return (s.a != -1,)
    if mode == 'heap':
        # Canonicalize heap objects by first visible reference (a, then b).
        # Keep alias relation and per-object target field, ignore unreachable objects.
        mapping = {}
        for r in (s.a, s.b):
            if r != -1 and r not in mapping:
                mapping[r] = len(mapping) + 1
        aa = mapping.get(s.a, 0)
        bb = mapping.get(s.b, 0)
        vals = tuple(s.slot_null(old) for old in mapping)
        return (aa, bb, vals, s.next_obj)
    raise ValueError(mode)


def bfs(mode: str, max_depth: int = 5):
    q = deque([((), State())])
    visited = {key(State(), mode)} if mode != 'none' else set()
    explored = generated = pruned = 0
    while q:
        seq, st = q.popleft(); explored += 1
        if target(st):
            return {'witness':seq, 'explored':explored, 'generated':generated, 'pruned':pruned}
        if len(seq) >= max_depth: continue
        for a in ACTIONS:
            nxt = transition(st,a); generated += 1
            if mode != 'none':
                k = key(nxt,mode)
                if k in visited:
                    pruned += 1; continue
                visited.add(k)
            q.append((seq+(a,), nxt))
    return {'witness':None, 'explored':explored, 'generated':generated, 'pruned':pruned}

JAVA = '''
public class AliasFixture {
  static class Box { String value = "ok"; }
  public static void main(String[] args) {
    Box a = null, b = null;
    for (String action : args) {
      switch (action) {
        case "NEW_A": if (a == null) a = new Box(); break;
        case "NEW_B": if (b == null) b = new Box(); break;
        case "ALIAS_B_A": if (a != null) b = a; break;
        case "SET_B_NULL": if (b != null) b.value = null; break;
        case "SET_A_NULL": if (a != null) a.value = null; break;
        case "RESTORE_A": if (a != null) a.value = "ok"; break;
      }
    }
    try { if (a != null) a.value.length(); System.out.println("NO_TARGET_NPE"); }
    catch (NullPointerException e) { System.out.println("TARGET_NPE"); }
  }
}
'''

def java_check(seq):
    if not shutil.which('javac') or not shutil.which('java'): return 'JAVA_UNAVAILABLE'
    with tempfile.TemporaryDirectory() as d:
        p=Path(d); (p/'AliasFixture.java').write_text(JAVA)
        subprocess.run(['javac','AliasFixture.java'], cwd=p,check=True,capture_output=True,text=True,timeout=20)
        r=subprocess.run(['java','-cp',str(p),'AliasFixture',*seq],cwd=p,check=True,capture_output=True,text=True,timeout=20)
        return r.stdout.strip()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--depth',type=int,default=5)
    ap.add_argument('--skip-java',action='store_true')
    args=ap.parse_args()
    for mode in ('none','full','heap','name_only_UNSAFE'):
        r=bfs(mode,args.depth)
        print(mode, r)
        if r['witness'] and not args.skip_java:
            print('  JVM:',java_check(r['witness']))
    # A particularly important alias witness and its negative control.
    aliased=('NEW_A','ALIAS_B_A','SET_B_NULL')
    separate=('NEW_A','NEW_B','SET_B_NULL')

    for name,seq in [('alias',aliased),('separate',separate)]:
        s=State()
        for action in seq:
            s=transition(s,action)
        result=target(s)
        print(name,seq,'target:',result,'heap key:',key(s,'heap'),'unsafe key:',key(s,'name_only_UNSAFE'))
        if not args.skip_java:
            print('  JVM:',java_check(seq))

    assert target(transition(transition(transition(State(),'NEW_A'),'ALIAS_B_A'),'SET_B_NULL'))
    assert not target(transition(transition(transition(State(),'NEW_A'),'NEW_B'),'SET_B_NULL'))
    print('Assertions passed')

if __name__=='__main__':main()
