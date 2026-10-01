from dataclasses import dataclass
from collections import defaultdict, deque


# ============================================================
# IR
# ============================================================

@dataclass(frozen=True)
class IO:
    op: str
    stream: str


@dataclass(frozen=True)
class Check:
    op: str
    stream: str


@dataclass(frozen=True)
class Clear:
    stream: str


@dataclass(frozen=True)
class Call:
    callee: str
    args: tuple


@dataclass(frozen=True)
class Nop:
    pass


@dataclass
class Function:
    name: str
    params: tuple
    entry: str
    exit: str
    stmts: dict
    succ: dict


# ============================================================
# foo -> bar
#
# source is inside CALLEE
# ============================================================

foo = Function(
    name="foo",
    params=("f",),
    entry="F0",
    exit="F3",

    stmts={
        "F0": Nop(),
        "F1": Call("bar", ("f",)),
        "F2": Check("ferror", "f"),
        "F3": Nop(),
    },

    succ={
        "F0": ["F1"],
        "F1": ["F2"],
        "F2": ["F3"],
        "F3": [],
    },
)


bar = Function(
    name="bar",
    params=("g",),
    entry="B0",
    exit="B2",

    stmts={
        "B0": Nop(),
        "B1": IO("fread", "g"),
        "B2": Nop(),
    },

    succ={
        "B0": ["B1"],
        "B1": ["B2"],
        "B2": [],
    },
)


# ============================================================
# outer -> check
#
# source is inside CALLER
# ============================================================

outer = Function(
    name="outer",
    params=("h",),
    entry="O0",
    exit="O3",

    stmts={
        "O0": Nop(),
        "O1": IO("fwrite", "h"),
        "O2": Call("check", ("h",)),
        "O3": Nop(),
    },

    succ={
        "O0": ["O1"],
        "O1": ["O2"],
        "O2": ["O3"],
        "O3": [],
    },
)


check = Function(
    name="check",
    params=("p",),
    entry="C0",
    exit="C2",

    stmts={
        "C0": Nop(),
        "C1": Check("ferror", "p"),
        "C2": Nop(),
    },

    succ={
        "C0": ["C1"],
        "C1": ["C2"],
        "C2": [],
    },
)


FUNCTIONS = {
    f.name: f
    for f in [foo, bar, outer, check]
}


SOURCES = {
    "fread",
    "fwrite",
    "fseek",
}


# ============================================================
# predecessor CFG
# ============================================================

PREDS = {}

for fn in FUNCTIONS.values():

    pred = defaultdict(list)

    for u, vs in fn.succ.items():
        for v in vs:
            pred[v].append(u)

    PREDS[fn.name] = pred


# ============================================================
# callsite index
#
# callee ->
#     [(caller function, call label, Call stmt), ...]
# ============================================================

CALLERS = defaultdict(list)

for fn in FUNCTIONS.values():

    for label, stmt in fn.stmts.items():

        if isinstance(stmt, Call):
            CALLERS[stmt.callee].append(
                (fn.name, label, stmt)
            )


# ============================================================
# stack frame
#
# callee 분석을 끝내고 어디로 돌아갈지 기억
# ============================================================

@dataclass(frozen=True)
class Frame:
    caller: str
    call_label: str
    caller_stream: str


# ============================================================
# Backward interprocedural analysis
# ============================================================

def find_sources(function_name, check_label):

    start_fn = FUNCTIONS[function_name]
    check_stmt = start_fn.stmts[check_label]

    assert isinstance(check_stmt, Check)

    target_stream = check_stmt.stream

    # state:
    #
    # (
    #   current function,
    #   current label,
    #   tracked stream,
    #   call stack
    # )

    worklist = deque()

    for p in PREDS[function_name][check_label]:
        worklist.append(
            (
                function_name,
                p,
                target_stream,
                tuple()
            )
        )

    visited = set()
    sources = set()

    while worklist:

        fn_name, label, stream, stack = worklist.popleft()

        state = (
            fn_name,
            label,
            stream,
            stack
        )

        if state in visited:
            continue

        visited.add(state)

        fn = FUNCTIONS[fn_name]
        stmt = fn.stmts[label]

        indent = "  " * len(stack)

        print(
            f"{indent}visit "
            f"{fn_name}:{label} "
            f"stream={stream} "
            f"stmt={stmt}"
        )

        # ----------------------------------------------------
        # 1. clearerr
        # ----------------------------------------------------

        if isinstance(stmt, Clear):

            if stmt.stream == stream:

                print(
                    f"{indent}  -> clearerr: "
                    f"stop path"
                )

                continue


        # ----------------------------------------------------
        # 2. I/O source
        # ----------------------------------------------------

        if isinstance(stmt, IO):

            if (
                stmt.stream == stream
                and stmt.op in SOURCES
            ):

                source = (
                    fn_name,
                    label,
                    stmt.op,
                    stream
                )

                sources.add(source)

                print(
                    f"{indent}  -> FOUND SOURCE "
                    f"{fn_name}:{label} "
                    f"{stmt.op}({stream})"
                )

                # nearest source on this path
                continue


        # ----------------------------------------------------
        # 3. Function call
        #
        # backward:
        #
        # caller actual argument
        #       ↓
        # callee formal parameter
        # ----------------------------------------------------

        if isinstance(stmt, Call):

            callee = FUNCTIONS[stmt.callee]

            entered_callee = False

            for arg_index, actual in enumerate(stmt.args):

                if actual != stream:
                    continue

                formal = callee.params[arg_index]

                print(
                    f"{indent}  -> enter callee "
                    f"{callee.name}: "
                    f"{stream} -> {formal}"
                )

                frame = Frame(
                    caller=fn_name,
                    call_label=label,
                    caller_stream=stream,
                )

                # callee exit에서 backward 시작
                worklist.append(
                    (
                        callee.name,
                        callee.exit,
                        formal,
                        stack + (frame,)
                    )
                )

                entered_callee = True

            # tracked stream을 callee에 넘겼다면
            # callee 분석 후 필요한 경우 caller로 복귀.
            if entered_callee:
                continue


        # ----------------------------------------------------
        # 4. normal backward CFG traversal
        # ----------------------------------------------------

        preds = PREDS[fn_name][label]

        if preds:

            for p in preds:

                worklist.append(
                    (
                        fn_name,
                        p,
                        stream,
                        stack
                    )
                )

            continue


        # ----------------------------------------------------
        # 5. function ENTRY에 도달
        # ----------------------------------------------------

        if label == fn.entry:

            # -----------------------------------------------
            # Case A:
            # 다른 함수 call을 따라 들어온 상태
            #
            # callee 안에서 source를 못 찾았으므로
            # caller의 call 직전으로 돌아감.
            # -----------------------------------------------

            if stack:

                frame = stack[-1]
                rest_stack = stack[:-1]

                print(
                    f"{indent}  -> leave callee, "
                    f"return to "
                    f"{frame.caller}:{frame.call_label}"
                )

                for p in PREDS[
                    frame.caller
                ][frame.call_label]:

                    worklist.append(
                        (
                            frame.caller,
                            p,
                            frame.caller_stream,
                            rest_stack
                        )
                    )

                continue


            # -----------------------------------------------
            # Case B:
            # 분석을 시작한 함수 자체의 parameter
            #
            # caller들을 찾아 올라감.
            # -----------------------------------------------

            if stream in fn.params:

                param_index = fn.params.index(stream)

                for (
                    caller_name,
                    call_label,
                    call_stmt
                ) in CALLERS[fn_name]:

                    actual = call_stmt.args[param_index]

                    print(
                        f"{indent}  -> ascend caller "
                        f"{caller_name}:{call_label}: "
                        f"{stream} -> {actual}"
                    )

                    for p in PREDS[
                        caller_name
                    ][call_label]:

                        worklist.append(
                            (
                                caller_name,
                                p,
                                actual,
                                tuple()
                            )
                        )


    return sources


# ============================================================
# RUN
# ============================================================

print("\n======================================")
print("CASE A: source inside callee")
print("======================================")

sources = find_sources(
    "foo",
    "F2"
)

print("\nSOURCES:")
for s in sorted(sources):
    print(" ", s)


print("\n======================================")
print("CASE B: source inside caller")
print("======================================")

sources = find_sources(
    "check",
    "C1"
)

print("\nSOURCES:")
for s in sorted(sources):
    print(" ", s)


# ============================================================
# Transformation planner
# ============================================================

from collections import defaultdict, deque


# ------------------------------------------------------------
# Call graph
#
# caller -> [(callee, call_label), ...]
# ------------------------------------------------------------

CALL_GRAPH = defaultdict(list)

for fn in FUNCTIONS.values():
    for label, stmt in fn.stmts.items():

        if isinstance(stmt, Call):
            CALL_GRAPH[fn.name].append(
                (stmt.callee, label)
            )


# ------------------------------------------------------------
# caller -> ... -> target
# 함수 경로 하나 찾기
#
# ex)
# foo -> bar -> baz
# => ["foo", "bar", "baz"]
# ------------------------------------------------------------

def find_function_path(start, target):

    q = deque([
        (start, [start])
    ])

    visited = {start}

    while q:

        current, path = q.popleft()

        if current == target:
            return path

        for callee, _ in CALL_GRAPH[current]:

            if callee in visited:
                continue

            visited.add(callee)

            q.append(
                (
                    callee,
                    path + [callee]
                )
            )

    return None


# ------------------------------------------------------------
# 특정 caller -> callee call label 찾기
# ------------------------------------------------------------

def find_call_label(caller, callee):

    for target, label in CALL_GRAPH[caller]:

        if target == callee:
            return label

    return None


# ------------------------------------------------------------
# Pretty plan
# ------------------------------------------------------------

def build_transformation_plan(
    check_function,
    check_label,
    source
):

    source_fn, source_label, source_op, source_stream = source

    check_stmt = FUNCTIONS[
        check_function
    ].stmts[check_label]

    assert isinstance(check_stmt, Check)

    check_stream = check_stmt.stream

    plan = []

    # ========================================================
    # CASE 0
    #
    # source와 check가 같은 함수
    # ========================================================

    if source_fn == check_function:

        plan.append({
            "function": source_fn,
            "action":
                f"create local error flag for stream "
                f"{source_stream}"
        })

        plan.append({
            "function": source_fn,
            "action":
                f"rewrite {source_label}:{source_op} "
                f"to capture Rust Result"
        })

        plan.append({
            "function": source_fn,
            "action":
                f"replace {check_label}:"
                f"{check_stmt.op}({check_stream}) "
                f"with local error flag check"
        })

        return plan


    # ========================================================
    # CASE A
    #
    # check function
    #      ↓ calls
    # source function
    #
    # error가 CALLEE에서 발생
    #
    # → return value를 통해 위로 올린다.
    # ========================================================

    downward_path = find_function_path(
        check_function,
        source_fn
    )

    if downward_path is not None:

        plan.append({
            "function": source_fn,
            "action":
                f"create local error flag "
                f"for {source_label}:{source_op}"
        })

        plan.append({
            "function": source_fn,
            "action":
                f"rewrite {source_label}:{source_op} "
                f"to inspect Rust Result"
        })

        # source → ... → check 방향으로 올라간다
        reverse_path = list(
            reversed(downward_path)
        )

        for i in range(
            len(reverse_path) - 1
        ):

            callee = reverse_path[i]
            caller = reverse_path[i + 1]

            call_label = find_call_label(
                caller,
                callee
            )

            # source 함수 또는 중간 함수:
            # error를 return에 포함
            plan.append({
                "function": callee,
                "action":
                    f"return error indicator "
                    f"to {caller}"
            })

            # caller:
            # tuple/error를 받음
            plan.append({
                "function": caller,
                "action":
                    f"receive error indicator from "
                    f"{call_label}:{callee}(...)"
            })

        plan.append({
            "function": check_function,
            "action":
                f"replace {check_label}:"
                f"{check_stmt.op}({check_stream}) "
                f"with received error indicator"
        })

        return plan


    # ========================================================
    # CASE B
    #
    # source function
    #      ↓ calls
    # check function
    #
    # error가 CALLER에서 발생
    #
    # → parameter를 통해 아래로 전달한다.
    # ========================================================

    downward_path = find_function_path(
        source_fn,
        check_function
    )

    if downward_path is not None:

        plan.append({
            "function": source_fn,
            "action":
                f"create local error flag "
                f"for {source_label}:{source_op}"
        })

        plan.append({
            "function": source_fn,
            "action":
                f"rewrite {source_label}:{source_op} "
                f"to inspect Rust Result"
        })

        for i in range(
            len(downward_path) - 1
        ):

            caller = downward_path[i]
            callee = downward_path[i + 1]

            call_label = find_call_label(
                caller,
                callee
            )

            plan.append({
                "function": callee,
                "action":
                    "add error indicator parameter"
            })

            plan.append({
                "function": caller,
                "action":
                    f"pass error indicator at "
                    f"{call_label}:{callee}(...)"
            })

        plan.append({
            "function": check_function,
            "action":
                f"replace {check_label}:"
                f"{check_stmt.op}({check_stream}) "
                f"with error parameter check"
        })

        return plan


    # ========================================================
    # unrelated
    # ========================================================

    plan.append({
        "function": None,
        "action":
            "no call-graph path between "
            "source and check"
    })

    return plan


def print_plan(plan):

    print("\nTRANSFORMATION PLAN")
    print("------------------------------")

    for i, step in enumerate(plan, 1):

        fn = step["function"]

        if fn is None:
            print(
                f"{i:2d}. {step['action']}"
            )
        else:
            print(
                f"{i:2d}. [{fn}] "
                f"{step['action']}"
            )

# ============================================================
# CASE A
# foo:ferror(f)
# source = bar:fread(g)
# ============================================================

sources = find_sources(
    "foo",
    "F2"
)

for source in sources:

    plan = build_transformation_plan(
        check_function="foo",
        check_label="F2",
        source=source
    )

    print_plan(plan)

# ============================================================
# CASE B
# outer:fwrite(h)
# check:ferror(p)
# ============================================================

sources = find_sources(
    "check",
    "C1"
)

for source in sources:

    plan = build_transformation_plan(
        check_function="check",
        check_label="C1",
        source=source
    )

    print_plan(plan)

# ============================================================
# THREE-LEVEL CALL CHAIN
#
# foo -> mid -> bar
# bar contains fread()
# foo contains ferror()
# ============================================================

mid = Function(
    name="mid",
    params=("m",),
    entry="M0",
    exit="M2",

    stmts={
        "M0": Nop(),
        "M1": Call("bar", ("m",)),
        "M2": Nop(),
    },

    succ={
        "M0": ["M1"],
        "M1": ["M2"],
        "M2": [],
    },
)


foo3 = Function(
    name="foo",
    params=("f",),
    entry="F0",
    exit="F3",

    stmts={
        "F0": Nop(),

        # bar가 아니라 mid 호출
        "F1": Call("mid", ("f",)),

        "F2": Check("ferror", "f"),
        "F3": Nop(),
    },

    succ={
        "F0": ["F1"],
        "F1": ["F2"],
        "F2": ["F3"],
        "F3": [],
    },
)

FUNCTIONS = {
    f.name: f
    for f in [
        foo3,
        mid,
        bar,
        outer,
        check
    ]
}


# rebuild predecessors
PREDS = {}

for fn in FUNCTIONS.values():

    pred = defaultdict(list)

    for u, vs in fn.succ.items():
        for v in vs:
            pred[v].append(u)

    PREDS[fn.name] = pred


# rebuild callers
CALLERS = defaultdict(list)

for fn in FUNCTIONS.values():

    for label, stmt in fn.stmts.items():

        if isinstance(stmt, Call):

            CALLERS[stmt.callee].append(
                (
                    fn.name,
                    label,
                    stmt
                )
            )


# rebuild call graph
CALL_GRAPH = defaultdict(list)

for fn in FUNCTIONS.values():

    for label, stmt in fn.stmts.items():

        if isinstance(stmt, Call):

            CALL_GRAPH[fn.name].append(
                (
                    stmt.callee,
                    label
                )
            )

print("\n======================================")
print("THREE LEVEL: foo -> mid -> bar")
print("======================================")

sources = find_sources(
    "foo",
    "F2"
)

print("\nSOURCES:")
for source in sorted(sources):
    print(" ", source)

    plan = build_transformation_plan(
        check_function="foo",
        check_label="F2",
        source=source
    )

    print_plan(plan)