import json
import random
import urllib.request
from collections import Counter


SPEC_URL = (
    "https://raw.githubusercontent.com/"
    "SoftSec-KAIST/SatFuzz/main/specs/cfs/spec.json"
)

N = 5000

rng = random.Random(0)


# ============================================================
# 1. Load REAL SatFuzz cFS specification
# ============================================================

def load_spec():

    print("loading official SatFuzz cFS spec...")

    with urllib.request.urlopen(
        SPEC_URL
    ) as response:

        return json.load(
            response
        )


spec = load_spec()

commands = spec["commands"]

command_map = {
    cmd["name"]: cmd
    for cmd in commands
}


print(
    "commands:",
    len(commands)
)


parameterized = [
    cmd
    for cmd in commands
    if cmd.get("parameters")
]


print(
    "commands with parameters:",
    len(parameterized)
)


# ============================================================
# 2. Parameter type helpers
#
# 실제 packet encoder는 아니고,
# spec에 나온 logical command representation을 사용.
#
# 즉 이 실험은:
#
#   "parser/input gate를 얼마나 잘 통과하나?"
#
# 를 압축해서 보는 것.
# ============================================================

def type_family(param):

    t = (
        param
        .get("type", "")
        .lower()
    )

    if "string" in t:
        return "string"

    if "float" in t or "double" in t:
        return "float"

    if "bool" in t:
        return "int"

    if (
        "int" in t
        or
        "uint" in t
        or
        "enum" in t
    ):
        return "int"

    return "unknown"


def default_value(param):

    possible = (
        param.get(
            "possible_values",
            []
        )
    )

    if possible:
        return possible[0]

    family = type_family(
        param
    )

    if family == "string":
        return "A"

    if family == "float":
        return 0.0

    if family == "int":
        return 0

    return 0


def type_ok(
    param,
    value,
):

    family = type_family(
        param
    )

    if family == "string":

        return isinstance(
            value,
            str
        )

    if family == "float":

        return (
            isinstance(
                value,
                (int, float)
            )
            and
            not isinstance(
                value,
                bool
            )
        )

    if family == "int":

        return (
            isinstance(
                value,
                int
            )
            and
            not isinstance(
                value,
                bool
            )
        )

    # unknown type:
    # 이 toy gate에서는 허용
    return True


# ============================================================
# 3. Valid seed generation
#
# SatFuzz spec에서 실제 command와 parameter structure를 사용.
# ============================================================

def make_seed(cmd):

    args = [

        default_value(
            p
        )

        for p in cmd.get(
            "parameters",
            []
        )
    ]

    return {
        "name": cmd["name"],
        "args": args,
    }


# ============================================================
# 4. Structural validator
#
# 실제 cFS가 아니라,
# spec-derived input gate.
#
#
# depth:
#
# 0 : command name부터 실패
# 1 : command recognized
# 2 : argument count correct
# 3 : parameter types parseable
#
# ============================================================

def validate(inp):

    name = inp["name"]

    if name not in command_map:

        return False, 0


    cmd = command_map[
        name
    ]


    params = cmd.get(
        "parameters",
        []
    )


    if len(inp["args"]) != len(
        params
    ):

        return False, 1


    for param, value in zip(
        params,
        inp["args"],
    ):

        if not type_ok(
            param,
            value,
        ):

            return False, 2


    return True, 3


# ============================================================
# 5. Semantic class
#
# syntax를 통과한 이후:
#
# NORMAL
#     possible_values
#
# OUT_OF_RANGE
#     spec이 deliberate violation candidate로
#     기록한 값
#
# OTHER
#     type은 맞지만 catalog에 없는 값
#
# 이건 실제 code coverage가 아니라
# semantic input diversity proxy.
# ============================================================

def value_class(
    param,
    value,
):

    possible = param.get(
        "possible_values",
        []
    )

    out = param.get(
        "out_of_range_values",
        []
    )


    if value in possible:
        return "NORMAL"

    if value in out:
        return "OUT_OF_RANGE"

    return "OTHER"


def semantic_signature(
    inp,
):

    cmd = command_map[
        inp["name"]
    ]

    params = cmd.get(
        "parameters",
        []
    )


    classes = [

        value_class(
            p,
            v,
        )

        for p, v in zip(
            params,
            inp["args"],
        )
    ]


    return (
        inp["name"],
        tuple(classes),
    )


# ============================================================
# 6. Conventional-ish mutation
#
# valid seed를 얻었다고 하더라도,
# structure를 모르는 mutator가
# name / arity / value를 마구 바꾼다고 가정.
#
# "AFL++ 자체 구현"이 아니라
# syntax-unaware mutation을 압축한 실험.
# ============================================================

def random_junk():

    choice = rng.randrange(
        4
    )

    if choice == 0:
        return rng.randint(
            -2**31,
            2**31 - 1,
        )

    if choice == 1:

        return "".join(

            rng.choice(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789"
            )

            for _ in range(
                rng.randint(
                    1,
                    16,
                )
            )
        )

    if choice == 2:
        return rng.random()

    return None


def syntax_unaware_mutate(
    seed,
):

    x = {
        "name": seed["name"],
        "args": list(seed["args"]),
    }


    # 1~3 independent mutations
    for _ in range(
        rng.randint(
            1,
            3,
        )
    ):

        action = rng.choice(
            [
                "name",
                "value",
                "drop",
                "append",
            ]
        )


        if action == "name":

            x["name"] = (
                "CMD_"
                + str(
                    rng.randint(
                        0,
                        999999,
                    )
                )
            )


        elif (
            action == "value"
            and x["args"]
        ):

            idx = rng.randrange(
                len(x["args"])
            )

            x["args"][idx] = (
                random_junk()
            )


        elif (
            action == "drop"
            and x["args"]
        ):

            idx = rng.randrange(
                len(x["args"])
            )

            del x["args"][idx]


        elif action == "append":

            x["args"].append(
                random_junk()
            )


    return x


# ============================================================
# 7. SatFuzz-like command-aware mutation
#
# command name 보존
# arity 보존
# parameter type 보존
#
# 대신:
#
# possible_values
# out_of_range_values
#
# 를 이용해 의미 있는 mutation 수행.
#
# 실제 SatFuzz 전체 algorithm은 아니고
# artifact의 spec-aware mutation 아이디어를
# 압축한 실험.
# ============================================================

def boundary_value(
    param,
):

    family = type_family(
        param
    )

    if family == "string":

        return rng.choice(
            [
                "",
                "A",
                "test",
                "X" * 64,
            ]
        )

    if family == "float":

        return rng.choice(
            [
                0.0,
                -1.0,
                1.0,
                1e6,
            ]
        )

    if family == "int":

        return rng.choice(
            [
                -1,
                0,
                1,
                255,
                256,
                65535,
            ]
        )

    return 0


def command_aware_mutate(
    seed,
):

    x = {
        "name": seed["name"],
        "args": list(seed["args"]),
    }


    cmd = command_map[
        x["name"]
    ]

    params = cmd.get(
        "parameters",
        []
    )


    if not params:

        return x


    idx = rng.randrange(
        len(params)
    )


    param = params[
        idx
    ]


    possible = param.get(
        "possible_values",
        []
    )

    out = param.get(
        "out_of_range_values",
        []
    )


    choices = []


    if possible:

        choices.append(
            ("NORMAL", possible)
        )


    if out:

        # type이 맞는 out-of-range만 사용
        compatible_out = [

            v

            for v in out

            if type_ok(
                param,
                v,
            )
        ]


        if compatible_out:

            choices.append(
                (
                    "OUT_OF_RANGE",
                    compatible_out,
                )
            )


    if choices:

        _, values = rng.choice(
            choices
        )

        x["args"][idx] = (
            rng.choice(
                values
            )
        )

    else:

        x["args"][idx] = (
            boundary_value(
                param
            )
        )


    return x


# ============================================================
# 8. Experiment
# ============================================================

def experiment(
    name,
    mutator,
):

    depths = Counter()

    valid = 0

    semantic_signatures = set()

    normal_hits = 0
    out_hits = 0
    other_hits = 0


    for _ in range(N):

        cmd = rng.choice(
            parameterized
        )


        seed = make_seed(
            cmd
        )


        inp = mutator(
            seed
        )


        ok, depth = validate(
            inp
        )


        depths[depth] += 1


        if not ok:
            continue


        valid += 1


        sig = semantic_signature(
            inp
        )


        semantic_signatures.add(
            sig
        )


        actual_cmd = command_map[
            inp["name"]
        ]


        for p, v in zip(

            actual_cmd.get(
                "parameters",
                []
            ),

            inp["args"],
        ):

            c = value_class(
                p,
                v,
            )


            if c == "NORMAL":
                normal_hits += 1

            elif c == "OUT_OF_RANGE":
                out_hits += 1

            else:
                other_hits += 1


    print(
        "\n================================"
    )

    print(name)

    print(
        "================================"
    )


    print(
        "trials:",
        N
    )


    print(
        "valid:",
        valid
    )


    print(
        "valid rate:",
        f"{valid/N:.2%}"
    )


    print(
        "\nparser depth:"
    )


    for depth in range(
        4
    ):

        print(
            f"depth {depth}: "
            f"{depths[depth]:5d} "
            f"({depths[depth]/N:.2%})"
        )


    print(
        "\nunique semantic signatures:",
        len(
            semantic_signatures
        )
    )


    print(
        "NORMAL values:",
        normal_hits
    )


    print(
        "OUT_OF_RANGE values:",
        out_hits
    )


    print(
        "OTHER type-valid values:",
        other_hits
    )


    return {
        "valid_rate":
        valid / N,

        "semantic_signatures":
        len(
            semantic_signatures
        ),

        "out_hits":
        out_hits,
    }


# ============================================================
# MAIN
# ============================================================

print(
    "\n################################"
)

print(
    "REAL SatFuzz SPEC EXPERIMENT"
)

print(
    "################################"
)


a = experiment(

    "SYNTAX-UNAWARE MUTATION",

    syntax_unaware_mutate,
)


b = experiment(

    "COMMAND-AWARE MUTATION",

    command_aware_mutate,
)


print(
    "\n================================"
)

print(
    "COMPARISON"
)

print(
    "================================"
)


print(
    "valid-rate gain:",
    f"{b['valid_rate'] - a['valid_rate']:+.2%}"
)


print(
    "semantic signatures:"
)

print(
    "  unaware:",
    a[
        "semantic_signatures"
    ]
)


print(
    "  aware:  ",
    b[
        "semantic_signatures"
    ]
)


print(
    "out-of-range hits:"
)

print(
    "  unaware:",
    a[
        "out_hits"
    ]
)


print(
    "  aware:  ",
    b[
        "out_hits"
    ]
)