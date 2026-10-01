from dataclasses import dataclass
from collections import defaultdict, deque


# ---------------------------------------------
# IR
# ---------------------------------------------

@dataclass
class IO:
    op: str
    stream: str


@dataclass
class Check:
    op: str
    stream: str


@dataclass
class Clear:
    stream: str


@dataclass
class Nop:
    pass


# ---------------------------------------------
# CFG
# ---------------------------------------------

stmts = {
    "L0": Nop(),

    "L1": IO("fwrite", "f"),

    # branch
    "L2": Nop(),

    "L3": IO("fread", "f"),
    "L4": Nop(),

    # join
    "L5": Check("ferror", "f"),

    "L6": Clear("f"),

    "L7": IO("fseek", "f"),

    "L8": Check("ferror", "f"),
}


# forward CFG edge
succ = {
    "L0": ["L1"],
    "L1": ["L2"],

    # branch
    "L2": ["L3", "L4"],

    "L3": ["L5"],
    "L4": ["L5"],

    "L5": ["L6"],
    "L6": ["L7"],
    "L7": ["L8"],
    "L8": [],
}


# predecessor graph
pred = defaultdict(list)

for u, vs in succ.items():
    for v in vs:
        pred[v].append(u)


SOURCES = {
    "fread",
    "fwrite",
    "fseek",
}


# ---------------------------------------------
# backward source analysis
# ---------------------------------------------

def find_sources(check_label):
    check = stmts[check_label]

    assert isinstance(check, Check)

    target_stream = check.stream

    worklist = deque(pred[check_label])

    visited = set()
    sources = set()

    while worklist:

        label = worklist.popleft()

        if label in visited:
            continue

        visited.add(label)

        stmt = stmts[label]

        print(f"visit {label}: {stmt}")

        # ------------------------
        # clearerr kills old errors
        # ------------------------
        if isinstance(stmt, Clear):

            if stmt.stream == target_stream:
                print("  -> clearerr: stop this path")
                continue

        # ------------------------
        # failable I/O source
        # ------------------------
        if isinstance(stmt, IO):

            if (
                stmt.stream == target_stream
                and stmt.op in SOURCES
            ):
                sources.add(label)

                print(
                    f"  -> found source: "
                    f"{label} ({stmt.op})"
                )

                # 핵심:
                # 이 execution path에서는
                # 가장 가까운 source를 찾았으므로
                # 더 뒤로 가지 않는다.
                continue

        # 계속 backward
        for p in pred[label]:
            worklist.append(p)

    return sources


# ---------------------------------------------
# run
# ---------------------------------------------

print("\n===== CHECK L5 =====")
sources1 = find_sources("L5")
print("sources:", sources1)


print("\n===== CHECK L8 =====")
sources2 = find_sources("L8")
print("sources:", sources2)