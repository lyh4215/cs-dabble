from collections import defaultdict, deque
from dataclasses import dataclass
from engine import PROGRAM, Const, Call, Div, BranchPositive, Return, Jump, Function
from typing import Any

@dataclass(frozen=True)
class Result:
    warnings: frozenset
    contexts: int
    reanalyses: int
    transfer_count: int
    provenance_count: int
    inputs: dict

type History = tuple[str, ...]
type Context = tuple[str, ...]
type Key = tuple[str, Context]
type Program = dict[str, Function]

class Analyzer:
    def __init__(self, policy, program: Program | None=None):
        self.program: Program = PROGRAM if program is None else program
        self.policy: dict[str, int] = policy
        self.inputs: dict[Key, frozenset] = {}
        self.returns: dict[Key, frozenset] = {}
        self.histories: dict[Key, set[History]] = defaultdict(set)
        self.dependents: dict[Key, set[Key]] = defaultdict(set)
        self.queue: deque[Key] = deque()
        self.pending: set[Key] = set()
        self.warnings = set()
        self.runs = 0
        self.transfers = 0
        self._check_acyclic()

    def _check_acyclic(self):
        graph = {f: set() for f in self.program}
        for f, fn in self.program.items():
            for instructions, _ in fn.blocks.values():
                for ins in instructions:
                    if isinstance(ins, Call):
                        if ins.callee not in self.program:
                            raise ValueError(f'Unknown callee: {ins.callee}')
                        graph[f].add(ins.callee)
        visiting, done = set(), set()
        def visit(f):
            if f in visiting:
                raise ValueError('Recursive call graphs are not supported in Experiment 3')
            if f in done:
                return
            visiting.add(f)
            for g in graph[f]: visit(g)
            visiting.remove(f)
            done.add(f)
        for f in graph: visit(f)

    def key(self, fn: str, history: list) -> tuple[Any, Any | tuple[()]]:
        k = self.policy.get(fn, 0)
        if not isinstance(k, int) or k < 0:
            raise ValueError('Context depth must be nonnegative integer')
        return (fn, history[-k:] if k else ())

    def enqueue(self, key: Key):
        if key not in self.pending:
            self.pending.add(key)
            self.queue.append(key)

    def add_input(self, key: Key, history: History, values: frozenset):
        if history not in self.histories[key]:
            self.histories[key].add(history)
            self.enqueue(key)
        previous = self.inputs.get(key)
        current = values if previous is None else previous | values
        if current != previous:
            self.inputs[key] = current
            self.enqueue(key)

    def analyze(self, key: Key):
        fn_name, _ = key
        fn = self.program[fn_name]
        self.runs += 1
        output = set()

        # All complete invocation histories sharing one abstract context
        # use the SAME merged input and summary.
        # 현재 key의 history들을 모두 추적
        for history in tuple(self.histories[key]):
            env0 = {fn.param: self.inputs[key]} if fn.param is not None else {}
            blocks = deque([('entry', env0)])
            while blocks:
                block, env = blocks.popleft()
                env = env.copy()
                instructions, terminal = fn.blocks[block]
                paused = False
                for instr in instructions:
                    self.transfers += 1
                    if isinstance(instr, Const):
                        env[instr.dst] = frozenset([instr.value])
                    elif isinstance(instr, Call): # Call Instruction
                        history2 = history + (instr.site,) #자기 자신을 history에 추가
                        callee: Key = self.key(instr.callee, history2) #불리는 애의 Key
                        args = env[instr.arg] if instr.arg is not None else frozenset()
                        self.dependents[callee].add(key) # callee의 dependant의 자신 추가
                        self.add_input(callee, history2, args) # callee 를 다음으로 실행
                        if callee not in self.returns:
                            paused = True
                            break
                        env[instr.dst] = self.returns[callee] # env에 리턴값 저장
                    elif isinstance(instr, Div):
                        denoms = env[instr.denominator]
                        if 0 in denoms:
                            self.warnings.add(instr.site)
                        env[instr.dst] = frozenset(instr.numerator // d for d in denoms if d)
                if paused:
                    continue
                if isinstance(terminal, Return):
                    output.update(env[terminal.var])
                elif isinstance(terminal, BranchPositive):
                    values = env[terminal.var]
                    yes = frozenset(v for v in values if v > 0)
                    no = values - yes
                    if yes:
                        nxt = env.copy(); nxt[terminal.var] = yes
                        blocks.append((terminal.yes, nxt))
                    if no:
                        nxt = env.copy(); nxt[terminal.var] = no
                        blocks.append((terminal.no, nxt))
                elif isinstance(terminal, Jump):
                    blocks.append((terminal.target, env))
                else:
                    raise TypeError(terminal)

        # Key 연산값을 returns[key]에 추가
        if output:
            old = self.returns.get(key, frozenset())
            new = old | frozenset(output)
            if old != new:
                self.returns[key] = new
                # 바뀐에 있으면, caller 재분석
                for caller in self.dependents[key]:
                    self.enqueue(caller)

    def run(self):
        self.add_input(self.key('main', ()), (), frozenset())
        while self.queue:
            key = self.queue.popleft(); self.pending.remove(key)
            self.analyze(key)
        return Result(
            warnings=frozenset(self.warnings),
            contexts=len(self.inputs), reanalyses=self.runs,
            transfer_count=self.transfers,
            provenance_count=sum(map(len, self.histories.values())),
            inputs=dict(self.inputs),
        )

if __name__ == '__main__':
    for label, policy in [
        ('uniform-0', {}),
        ('uniform-1', {name: 1 for name in PROGRAM}),
        ('uniform-2', {name: 2 for name in PROGRAM}),
        ('selective-transform-1', {'transform': 1}),
        ('selective-transform-2', {'transform': 2}),
    ]:
        r = Analyzer(policy).run()
        print(label, r.contexts, r.reanalyses, r.transfer_count, r.provenance_count, sorted(r.warnings))
