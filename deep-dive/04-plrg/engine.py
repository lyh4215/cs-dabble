
from collections import defaultdict, deque
from dataclasses import dataclass


# ==========================
# 1. Intermediate Representation
# ==========================

@dataclass(frozen=True)
class Const:
    dst: str
    value: int


@dataclass(frozen=True)
class Call:
    dst: str
    callee: str
    arg: str | None
    site: str


@dataclass(frozen=True)
class Div:
    dst: str
    numerator: int
    denominator: str
    site: str


@dataclass(frozen=True)
class BranchPositive:
    var: str
    yes: str
    no: str


@dataclass(frozen=True)
class Return:
    var: str


@dataclass(frozen=True)
class Jump:
    target: str


@dataclass
class Function:
    param: str | None
    blocks: dict[str, tuple[list, object]]


# ==========================
# 2. Test program
# ==========================

PROGRAM = {
    "transform": Function("x", {
        "entry": (
            [],
            BranchPositive("x", "positive", "zero")
        ),
        "positive": (
            [],
            Return("x")
        ),
        "zero": (
            [Const("result", 0)],
            Return("result")
        ),
    }),

    "caller_a": Function(None, {
        "entry": (
            [
                Const("x", 10),
                Call("r", "transform", "x", "a->transform"),
            ],
            Return("r")
        ),
    }),

    "caller_b": Function(None, {
        "entry": (
            [
                Const("x", -10),
                Call("r", "transform", "x", "b->transform"),
            ],
            Return("r")
        ),
    }),

    "main": Function(None, {
        "entry": (
            [
                Call("a", "caller_a", None, "main->a"),
                Call("b", "caller_b", None, "main->b"),
                Div("z", 100, "a", "main:divide"),
            ],
            Return("z")
        ),
    }),
}


# ==========================
# 3. Abstract domain
# ==========================

def merge(a, b):
    return a | b


# ==========================
# 4. Interprocedural solver
# ==========================

class Analyzer:
    def __init__(self, k):
        self.k = k

        # (function, context) -> abstract arguments
        self.entries = {}

        # (function, context) -> abstract return
        self.returns = {}

        # callee -> dependent callers
        self.dependents = defaultdict(set)

        self.todo = deque()
        self.pending = set()

        self.warnings = set()
        self.runs = 0

    def enqueue(self, key):
        if key not in self.pending:
            self.pending.add(key)
            self.todo.append(key)

    def add_entry(self, key, incoming):
        old = self.entries.get(key)

        updated = (
            incoming if old is None
            else merge(old, incoming)
        )

        if old != updated:
            self.entries[key] = updated
            self.enqueue(key)

    def analyze_function(self, key):
        name, context = key
        fn = PROGRAM[name]

        self.runs += 1

        initial = (
            {fn.param: self.entries[key]}
            if fn.param else {}
        )

        states = {"entry": initial}
        blocks = deque(["entry"])

        def propagate(label, state):
            previous = states.get(label)

            if previous is None:
                states[label] = state.copy()
                blocks.append(label)
                return

            new = {
                v: merge(
                    previous.get(v, frozenset()),
                    state.get(v, frozenset())
                )
                for v in previous.keys() | state.keys()
            }

            if previous != new:
                states[label] = new
                blocks.append(label)

        while blocks:
            block_name = blocks.popleft()
            env = states[block_name].copy()

            instructions, terminal = (
                fn.blocks[block_name]
            )

            paused = False

            for instr in instructions:
                if isinstance(instr, Const):
                    env[instr.dst] = frozenset({
                        instr.value
                    })

                elif isinstance(instr, Call):
                    if self.k:
                        callee_context = (
                            context + (instr.site,)
                        )[-self.k:]
                    else:
                        callee_context = ()

                    callee_key = (
                        instr.callee,
                        callee_context
                    )

                    args = (
                        env[instr.arg]
                        if instr.arg else frozenset()
                    )

                    self.dependents[callee_key].add(key)
                    self.add_entry(callee_key, args)

                    if callee_key not in self.returns:
                        paused = True
                        break

                    env[instr.dst] = (
                        self.returns[callee_key]
                    )

                elif isinstance(instr, Div):
                    denominators = env[
                        instr.denominator
                    ]

                    if 0 in denominators:
                        self.warnings.add(instr.site)

                    env[instr.dst] = frozenset(
                        instr.numerator // d
                        for d in denominators
                        if d != 0
                    )

                else:
                    raise TypeError(instr)

            if paused:
                continue

            if isinstance(terminal, BranchPositive):
                vals = env[terminal.var]

                branches = (
                    (
                        terminal.yes,
                        frozenset(
                            v for v in vals if v > 0
                        )
                    ),
                    (
                        terminal.no,
                        frozenset(
                            v for v in vals if v <= 0
                        )
                    ),
                )

                for target, sub in branches:
                    if not sub:
                        continue

                    next_env = env.copy()
                    next_env[terminal.var] = sub
                    propagate(target, next_env)

            elif isinstance(terminal, Return):
                outgoing = env[terminal.var]

                old = self.returns.get(key)

                new = (
                    outgoing if old is None
                    else merge(old, outgoing)
                )

                if old != new:
                    self.returns[key] = new

                    for caller in self.dependents[key]:
                        self.enqueue(caller)

            elif isinstance(terminal, Jump):
                propagate(terminal.target, env)

            else:
                raise TypeError(terminal)

    def run(self):
        self.add_entry(
            ("main", ()),
            frozenset()
        )

        while self.todo:
            key = self.todo.popleft()
            self.pending.remove(key)

            self.analyze_function(key)

        return self


# ==========================
# 5. Experiment
# ==========================

def experiment(k):
    analyzer = Analyzer(k).run()

    transform = {
        context: sorted(values)
        for (name, context), values
        in analyzer.entries.items()
        if name == "transform"
    }

    main_result = sorted(
        analyzer.returns.get(
            ("main", ()),
            frozenset()
        )
    )

    print(f"\n=== CONTEXT DEPTH: {k} ===")
    print(f"Contexts: {len(analyzer.entries)}")
    print(f"Reanalyses: {analyzer.runs}")
    print(f"Warnings: {sorted(analyzer.warnings)}")
    print(f"Transform inputs: {transform}")
    print(f"Main return: {main_result}")


if __name__ == "__main__":
    for k in [0, 1, 2]:
        experiment(k)
