from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Cmd:
    name: str
    defines: str
    uses: set[str]
    text: str


program = [
    Cmd(
        name="c1",
        defines="Y",
        uses=set(),
        text="Y = 1"
    ),

    Cmd(
        name="c2",
        defines="Z",
        uses={"X"},
        text="Z = X - 1"
    ),

    Cmd(
        name="c3",
        defines="X",
        uses={"Y", "Z"},
        text="X = Y + Z"
    ),

    Cmd(
        name="c4",
        defines="V",
        uses={"X", "Y"},
        text="V = X + Y"
    ),
]

# ============================================================
# Correct def-use for a straight-line program
# ============================================================

definition = {}
def_use = defaultdict(list)

for cmd in program:

    # 먼저 현재 명령이 사용하는 변수들을
    # "지금까지 등장한 definition"과 연결한다.
    for var in cmd.uses:

        if var in definition:

            def_cmd = definition[var]

            def_use[def_cmd].append(
                (var, cmd.name)
            )

    # 그 다음에 현재 명령의 definition을 등록한다.
    definition[cmd.defines] = cmd.name

print("DEF-USE EDGES")

for src, edges in def_use.items():

    for var, dst in edges:

        print(
            f"{src} --{var}--> {dst}"
        )

delta = "c3"
def get_def_predecessors(delta):
    result = []

    for src, edges in def_use.items():
        for var, dst in edges:

            if dst == delta:
                result.append(
                    (src, var)
                )

    return result

def get_use_successors(delta):
    result = []

    for var, dst in def_use.get(delta, []):
        result.append(
            (dst, var)
        )

    return result

print("\nLAST MUTATED NODE:")
print(delta)


print("\nDEFINITIONS FEEDING DELTA:")

for node, var in get_def_predecessors(delta):

    print(
        f"{node} defines {var} "
        f"used by {delta}"
    )


print("\nUSERS OF DELTA:")

for node, var in get_use_successors(delta):

    print(
        f"{node} uses {var} "
        f"defined by {delta}"
    )