
import argparse
import struct
from collections import deque
from pathlib import Path

from capstone import Cs, CS_ARCH_X86, CS_MODE_64, CS_GRP_RET, CS_GRP_JUMP
from capstone.x86 import (
    X86_OP_IMM,
    X86_OP_REG,
    X86_OP_MEM,
    X86_REG_RIP,
    X86_INS_JMP,
    X86_INS_JA,
    X86_INS_CMP,
    X86_INS_LEA,
    X86_INS_MOVSXD,
    X86_INS_ADD,
    X86_INS_MOV,
)
from elftools.elf.elffile import ELFFile


parser = argparse.ArgumentParser()
parser.add_argument("binary", type=Path)
parser.add_argument("--entry", required=True, type=lambda s: int(s, 0))
parser.add_argument("--output", default="auto_cfg.dot")
args = parser.parse_args()

md = Cs(CS_ARCH_X86, CS_MODE_64)
md.detail = True

with args.binary.open("rb") as f:
    elf = ELFFile(f)
    text_section = elf.get_section_by_name(".text")

    if text_section is None:
        raise RuntimeError("Missing .text section")

    base = text_section["sh_addr"]
    code = text_section.data()
    end = base + len(code)

    # Keep all file-backed sections for virtual-address reads.
    sections = [
        (s["sh_addr"], s["sh_addr"] + s["sh_size"],
         s.name, s.data())
        for s in elf.iter_sections()
        if s["sh_type"] != "SHT_NOBITS"
        and s["sh_size"] > 0
        and s["sh_flags"] & 0x2  # SHF_ALLOC
    ]


def read_memory(address, size):
    for start, stop, name, data in sections:
        if start <= address and address + size <= stop:
            offset = address - start
            return data[offset:offset + size], name
    return None, None


def decode(address):
    if not (base <= address < end):
        return None

    offset = address - base
    return next(
        md.disasm(code[offset:offset + 15], address, count=1),
        None,
    )


def imm_target(ins):
    if (
        len(ins.operands) == 1
        and ins.operands[0].type == X86_OP_IMM
    ):
        return ins.operands[0].imm
    return None


def same_register(a, b):
    """Compare register families, including 32/64-bit aliases."""
    if not a or not b:
        return False

    # Normalize common x86-64 GPR aliases.
    aliases = {
        "eax": "rax", "ax": "rax", "al": "rax",
        "ebx": "rbx", "bx": "rbx", "bl": "rbx",
        "ecx": "rcx", "cx": "rcx", "cl": "rcx",
        "edx": "rdx", "dx": "rdx", "dl": "rdx",
        "edi": "rdi", "di": "rdi", "dil": "rdi",
        "esi": "rsi", "si": "rsi", "sil": "rsi",
        "ebp": "rbp", "bp": "rbp", "bpl": "rbp",
        "esp": "rsp", "sp": "rsp", "spl": "rsp",
    }

    for i in range(8, 16):
        aliases[f"r{i}d"] = f"r{i}"
        aliases[f"r{i}w"] = f"r{i}"
        aliases[f"r{i}b"] = f"r{i}"

    na = md.reg_name(a)
    nb = md.reg_name(b)

    return aliases.get(na, na) == aliases.get(nb, nb)


instructions = {}
successors = {}
leaders = {args.entry}
previous_by_end = {}
recovered_tables = []


def recover_jump_table(jmp_ins):
    """
    Recognize the concrete GCC PIE pattern:

      cmp     index32, max_index
      ja      default
      lea     base64, [rip+disp]
      [mov    index32, index32]
      movsxd  dst64, dword ptr [base64+index64*4]
      add     dst64, base64
      notrack jmp dst64

    Returns targets or None.
    """

    # Walk backward through contiguous decoded instructions.
    history = []
    cursor = jmp_ins.address

    for _ in range(7):
        prev = previous_by_end.get(cursor)
        if prev is None:
            break

        history.append(prev)
        cursor = prev.address

    history.reverse()

    # Strip the optional "mov edi, edi" from this fixture.
    filtered = []

    for ins in history:
        if (
            ins.id == X86_INS_MOV
            and len(ins.operands) == 2
            and all(o.type == X86_OP_REG for o in ins.operands)
            and ins.operands[0].reg == ins.operands[1].reg
        ):
            continue

        filtered.append(ins)

    if len(filtered) < 5:
        return None

    # Require the instructions immediately before jmp
    # to match the table dispatch pattern.
    cmp_ins, ja_ins, lea_ins, load_ins, add_ins = filtered[-5:]

    if (
        cmp_ins.id != X86_INS_CMP
        or ja_ins.id != X86_INS_JA
        or lea_ins.id != X86_INS_LEA
        or load_ins.id != X86_INS_MOVSXD
        or add_ins.id != X86_INS_ADD
        or jmp_ins.id != X86_INS_JMP
    ):
        return None

    if not (
        len(cmp_ins.operands) == 2
        and cmp_ins.operands[0].type == X86_OP_REG
        and cmp_ins.operands[1].type == X86_OP_IMM
    ):
        return None

    index_reg = cmp_ins.operands[0].reg
    max_index = cmp_ins.operands[1].imm

    # This fixture uses unsigned x <= 7.
    if not (0 <= max_index < 4096):
        return None

    # The JA must fall through into the dispatch path.
    if ja_ins.address + ja_ins.size != lea_ins.address:
        return None

    # lea table_base_reg, [rip + displacement]
    if not (
        len(lea_ins.operands) == 2
        and lea_ins.operands[0].type == X86_OP_REG
        and lea_ins.operands[1].type == X86_OP_MEM
        and lea_ins.operands[1].mem.base == X86_REG_RIP
    ):
        return None

    base_reg = lea_ins.operands[0].reg
    table_base = (
        lea_ins.address
        + lea_ins.size
        + lea_ins.operands[1].mem.disp
    )

    # movsxd target_reg, DWORD PTR [base_reg + index_reg*4]
    if not (
        len(load_ins.operands) == 2
        and load_ins.operands[0].type == X86_OP_REG
        and load_ins.operands[1].type == X86_OP_MEM
    ):
        return None

    target_reg = load_ins.operands[0].reg
    mem = load_ins.operands[1].mem

    if not (
        same_register(mem.base, base_reg)
        and same_register(mem.index, index_reg)
        and mem.scale == 4
        and mem.disp == 0
        and load_ins.operands[1].size == 4
    ):
        return None

    # add target_reg, base_reg
    if not (
        len(add_ins.operands) == 2
        and add_ins.operands[0].type == X86_OP_REG
        and add_ins.operands[1].type == X86_OP_REG
        and same_register(add_ins.operands[0].reg, target_reg)
        and same_register(add_ins.operands[1].reg, base_reg)
    ):
        return None

    # jmp target_reg
    if not (
        len(jmp_ins.operands) == 1
        and jmp_ins.operands[0].type == X86_OP_REG
        and same_register(jmp_ins.operands[0].reg, target_reg)
    ):
        return None

    count = max_index + 1
    raw, section_name = read_memory(table_base, count * 4)

    if raw is None:
        return None

    offsets = struct.unpack("<" + "i" * count, raw)
    targets = [table_base + relative for relative in offsets]

    # Validate that every target starts in executable .text.
    if any(decode(target) is None for target in targets):
        return None

    recovered_tables.append({
        "jmp": jmp_ins.address,
        "base": table_base,
        "count": count,
        "section": section_name,
        "offsets": offsets,
        "targets": targets,
    })

    return targets


# Phase 1: Reachability-driven instruction discovery
worklist = deque([args.entry])
MAX_INSNS = 10000

while worklist:
    address = worklist.popleft()

    if address in instructions:
        continue

    ins = decode(address)
    if ins is None:
        continue

    instructions[address] = ins
    previous_by_end[address + ins.size] = ins

    if len(instructions) > MAX_INSNS:
        raise RuntimeError("Too many decoded instructions")

    next_address = address + ins.size
    edges = []

    if ins.group(CS_GRP_RET):
        pass

    elif ins.id == X86_INS_JMP:
        target = imm_target(ins)

        if target is not None:
            edges.append((target, "branch"))
            leaders.add(target)
        else:
            recovered = recover_jump_table(ins)

            if recovered is None:
                print(
                    f"[UNRESOLVED] 0x{address:x}: "
                    f"{ins.mnemonic} {ins.op_str}"
                )
            else:
                for target in recovered:
                    edges.append((target, "jump-table"))
                    leaders.add(target)

        # No fallthrough for an unconditional JMP!

    elif ins.group(CS_GRP_JUMP):  # CS_GRP_JUMP
        target = imm_target(ins)

        if target is not None:
            edges.append((target, "branch"))
            leaders.add(target)

        edges.append((next_address, "fallthrough"))
        leaders.add(next_address)

    else:
        # CALL is treated as intraprocedural fallthrough.
        edges.append((next_address, "next"))

    successors[address] = edges

    for target, kind in edges:
        if base <= target < end:
            worklist.append(target)


# Phase 2: Basic Block construction
leaders = {a for a in leaders if a in instructions}
blocks = {}
instruction_to_block = {}

for leader in sorted(leaders):
    body = []
    current = leader

    while current in instructions:
        if current != leader and current in leaders:
            break

        if current in instruction_to_block:
            break

        ins = instructions[current]
        body.append(ins)
        instruction_to_block[current] = leader

        if ins.group(CS_GRP_RET) or ins.group(1):
            break

        current += ins.size

    if body:
        blocks[leader] = body


# Phase 3: CFG edge construction
block_edges = set()

for leader, body in blocks.items():
    last = body[-1]

    for target, kind in successors[last.address]:
        dst = instruction_to_block.get(target)

        if dst is not None:
            block_edges.add((leader, dst, kind))


# Phase 4: Report recovered tables and CFG
print("\n=== Recovered Jump Tables ===")

for jt in recovered_tables:
    print(f"Indirect jump: 0x{jt['jmp']:x}")
    print(f"Section:       {jt['section']}")
    print(f"Table base:    0x{jt['base']:x}")
    print(f"Entry count:   {jt['count']}")

    for i, (relative, target) in enumerate(
        zip(jt["offsets"], jt["targets"])
    ):
        print(
            f"  case {i}: "
            f"offset={relative:+d}, "
            f"target=0x{target:x}"
        )

print("\n=== CFG Summary ===")
print(f"Instructions: {len(instructions)}")
print(f"Basic blocks: {len(blocks)}")
print(f"CFG edges:    {len(block_edges)}")

for leader, body in sorted(blocks.items()):
    print(f"\n[Block 0x{leader:x}]")

    for ins in body:
        print(
            f"  0x{ins.address:x}: "
            f"{ins.mnemonic} {ins.op_str}"
        )

    for src, dst, kind in sorted(block_edges):
        if src == leader:
            print(f"    -> 0x{dst:x} [{kind}]")


# Phase 5: Graphviz DOT export
dot = [
    "digraph CFG {",
    '  node [shape=box, fontname="monospace"];',
]

for leader, body in sorted(blocks.items()):
    lines = [f"0x{leader:x}"]

    for ins in body:
        lines.append(
            f"0x{ins.address:x}: "
            f"{ins.mnemonic} {ins.op_str}"
        )

    label = "\\l".join(
        line.replace("\\", "\\\\").replace('"', '\\"')
        for line in lines
    ) + "\\l"

    dot.append(f'  b{leader:x} [label="{label}"];')

for src, dst, kind in sorted(block_edges):
    dot.append(
        f'  b{src:x} -> b{dst:x} [label="{kind}"];'
    )

dot.append("}")

Path(args.output).write_text("\n".join(dot))
print(f"\nSaved: {args.output}")
