import re
import subprocess
from pathlib import Path


GROUND_TRUTH_BIN = "cet"
STRIPPED_BIN = "cet_stripped"


# ============================================================
# command helper
# ============================================================

def run(*args):

    return subprocess.check_output(
        args,
        text=True,
    )


# ============================================================
# 1. Ground truth
#
# symbol이 살아 있는 binary의 T/t symbol을 함수 정답으로 사용.
#
# 단, .text 안에 실제 instruction address가 존재하는 symbol만
# 남긴다.
# ============================================================

def get_text_disassembly(binary):

    return run(
        "objdump",
        "-d",
        "-j",
        ".text",
        "-M",
        "intel",
        "--no-show-raw-insn",
        binary,
    )


def parse_instruction_addresses(disasm):

    addresses = set()

    for line in disasm.splitlines():

        m = re.match(
            r"^\s*([0-9a-fA-F]+):",
            line,
        )

        if m:

            addresses.add(
                int(
                    m.group(1),
                    16,
                )
            )

    return addresses


def get_ground_truth(binary):

    disasm = get_text_disassembly(
        binary
    )

    text_addresses = (
        parse_instruction_addresses(
            disasm
        )
    )


    nm = run(
        "nm",
        "-n",
        binary,
    )


    funcs = {}


    for line in nm.splitlines():

        parts = line.split()

        if len(parts) < 3:
            continue


        addr_s, typ, name = (
            parts[0],
            parts[1],
            parts[2],
        )


        if typ not in {
            "T",
            "t",
        }:
            continue


        try:

            addr = int(
                addr_s,
                16,
            )

        except ValueError:

            continue


        # .text에 실제 instruction으로 존재하는 것만
        if addr not in text_addresses:
            continue


        funcs[addr] = name


    return funcs


# ============================================================
# 2. Parse stripped assembly
# ============================================================

def parse_stripped(binary):

    disasm = get_text_disassembly(
        binary
    )


    instructions = []


    for line in disasm.splitlines():

        m = re.match(

            r"^\s*"
            r"([0-9a-fA-F]+):"
            r"\s+"
            r"(\S+)"
            r"(?:\s+(.*))?$",

            line,
        )


        if not m:
            continue


        addr = int(
            m.group(1),
            16,
        )


        mnemonic = (
            m.group(2)
            .lower()
        )


        operands = (
            m.group(3)
            or ""
        ).strip()


        instructions.append(
            (
                addr,
                mnemonic,
                operands,
            )
        )


    return instructions


# ============================================================
# 3. CALL-only heuristic
#
# call 11e0
#
# 처럼 목적지가 immediate address로 명시된 direct call만 사용.
#
# call rax
# call [rax]
#
# 같은 indirect call은 목적지를 알 수 없으므로 제외.
# ============================================================

def call_candidates(
    instructions,
):

    valid_text_addresses = {

        addr

        for addr, _, _
        in instructions
    }


    candidates = set()


    for (
        addr,
        mnemonic,
        operands,
    ) in instructions:


        if mnemonic != "call":
            continue


        # 예:
        #
        # call 1240
        # call 1240 <ordinary>
        #
        m = re.match(
            r"^([0-9a-fA-F]+)\b",
            operands,
        )


        if not m:

            # call rax
            # call QWORD PTR [...]
            continue


        target = int(
            m.group(1),
            16,
        )


        # PLT 같은 .text 밖 target은 제외
        if target not in (
            valid_text_addresses
        ):

            continue


        candidates.add(
            target
        )


    return candidates


# ============================================================
# 4. ENDBR-only heuristic
# ============================================================

def endbr_candidates(
    instructions,
):

    return {

        addr

        for (
            addr,
            mnemonic,
            operands,
        ) in instructions

        if mnemonic == "endbr64"
    }


# ============================================================
# 5. Evaluation
# ============================================================

def evaluate(
    name,
    predicted,
    ground_truth,
):

    gt = set(
        ground_truth.keys()
    )


    tp = predicted & gt
    fp = predicted - gt
    fn = gt - predicted


    precision = (

        len(tp)
        /
        len(predicted)

        if predicted
        else 0.0
    )


    recall = (

        len(tp)
        /
        len(gt)

        if gt
        else 0.0
    )


    f1 = (

        2
        * precision
        * recall
        /
        (
            precision
            + recall
        )

        if (
            precision
            + recall
        )
        else 0.0
    )


    print(
        "\n================================"
    )

    print(name)

    print(
        "================================"
    )


    print(
        "predicted:",
        len(predicted)
    )

    print(
        "ground truth:",
        len(gt)
    )

    print(
        "TP:",
        len(tp)
    )

    print(
        "FP:",
        len(fp)
    )

    print(
        "FN:",
        len(fn)
    )


    print(
        f"precision: {precision:.2%}"
    )

    print(
        f"recall:    {recall:.2%}"
    )

    print(
        f"F1:        {f1:.4f}"
    )


    print(
        "\nTRUE POSITIVES:"
    )


    for addr in sorted(tp):

        print(
            f"  0x{addr:x} "
            f"{ground_truth[addr]}"
        )


    print(
        "\nFALSE NEGATIVES:"
    )


    for addr in sorted(fn):

        print(
            f"  0x{addr:x} "
            f"{ground_truth[addr]}"
        )


    print(
        "\nFALSE POSITIVES:"
    )


    for addr in sorted(fp):

        print(
            f"  0x{addr:x}"
        )


    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# MAIN
# ============================================================

if not Path(
    GROUND_TRUTH_BIN
).exists():

    raise FileNotFoundError(
        GROUND_TRUTH_BIN
    )


if not Path(
    STRIPPED_BIN
).exists():

    raise FileNotFoundError(
        STRIPPED_BIN
    )


ground_truth = (
    get_ground_truth(
        GROUND_TRUTH_BIN
    )
)


instructions = (
    parse_stripped(
        STRIPPED_BIN
    )
)


call_only = (
    call_candidates(
        instructions
    )
)


endbr_only = (
    endbr_candidates(
        instructions
    )
)


union = (
    call_only
    |
    endbr_only
)


print(
    "================================"
)

print(
    "GROUND TRUTH FUNCTIONS"
)

print(
    "================================"
)


for addr, name in sorted(
    ground_truth.items()
):

    print(
        f"0x{addr:x}  {name}"
    )


print(
    "\ntext instructions:",
    len(instructions)
)


a = evaluate(
    "CALL-ONLY",
    call_only,
    ground_truth,
)


b = evaluate(
    "ENDBR-ONLY",
    endbr_only,
    ground_truth,
)


c = evaluate(
    "CALL ∪ ENDBR",
    union,
    ground_truth,
)


print(
    "\n================================"
)

print(
    "SUMMARY"
)

print(
    "================================"
)


print(
    f"{'method':15s}"
    f"{'precision':>12s}"
    f"{'recall':>12s}"
    f"{'F1':>10s}"
)


for name, m in [

    ("CALL", a),

    ("ENDBR", b),

    ("UNION", c),

]:

    print(

        f"{name:15s}"

        f"{m['precision']:12.2%}"

        f"{m['recall']:12.2%}"

        f"{m['f1']:10.4f}"
    )