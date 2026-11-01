
import argparse
from collections import deque
from pathlib import Path

from capstone import (
    Cs, CS_ARCH_X86, CS_MODE_64,
    CS_GRP_JUMP, CS_GRP_RET,
)
from capstone.x86 import X86_OP_IMM
from elftools.elf.elffile import ELFFile


parser = argparse.ArgumentParser()
parser.add_argument("binary", type=Path)
parser.add_argument("--entry", type=lambda s: int(s, 0), required=True)
parser.add_argument("--output", default="cfg.dot")
args = parser.parse_args()

md = Cs(CS_ARCH_X86, CS_MODE_64)
md.detail = True

with args.binary.open("rb") as f:
    elf = ELFFile(f)
    section = elf.get_section_by_name(".text")

    base = section["sh_addr"]
    code = section.data()

end = base + len(code)


def decode(addr):
    if not (base <= addr < end):
        return None

    offset = addr - base
    return next(
        md.disasm(code[offset:offset + 15], addr, count=1),
        None,
    )


def direct_target(ins):
    if (
        len(ins.operands) == 1
        and ins.operands[0].type == X86_OP_IMM
    ):
        return ins.operands[0].imm
    return None


# Phase 1: Instruction-level reachable code discovery
worklist = deque([args.entry])
instructions = {}
successors = {}
leaders = {args.entry}

MAX_INSNS = 10000

while worklist:
    addr = worklist.popleft()

    if addr in instructions:
        continue

    ins = decode(addr)
    if ins is None:
        continue

    instructions[addr] = ins

    if len(instructions) > MAX_INSNS:
        raise RuntimeError("Instruction limit reached")

    next_addr = addr + ins.size
    edges = []

    if ins.group(CS_GRP_RET):
        # Function return: no intraprocedural successor
        pass

    elif ins.group(CS_GRP_JUMP):
        target = direct_target(ins)

        if target is not None:
            edges.append((target, "branch"))
            leaders.add(target)

        if ins.mnemonic != "jmp":
            # Conditional branch: both taken and not-taken paths
            edges.append((next_addr, "fallthrough"))
            leaders.add(next_addr)

        # Indirect jmp target is unresolved in this demo.

    else:
        # For CALL, follow the return site, not the callee.
        edges.append((next_addr, "next"))

    successors[addr] = edges

    for target, _ in edges:
        if base <= target < end:
            worklist.append(target)


# Phase 2: Build Basic Blocks from leaders.
leaders = {a for a in leaders if a in instructions}
blocks = {}
instruction_to_block = {}

for leader in sorted(leaders):
    current = leader
    body = []

    while current in instructions:
        if current != leader and current in leaders:
            break

        if current in instruction_to_block:
            break

        ins = instructions[current]
        body.append(ins)
        instruction_to_block[current] = leader

        edges = successors[current]

        # Stop at branches and returns.
        if ins.group(CS_GRP_JUMP) or ins.group(CS_GRP_RET):
            break

        current += ins.size

    if body:
        blocks[leader] = body


# Phase 3: Connect Basic Blocks.
block_edges = set()

for leader, body in blocks.items():
    last = body[-1]

    for target, kind in successors[last.address]:
        dst = instruction_to_block.get(target)

        if dst is not None:
            block_edges.add((leader, dst, kind))


print(f"\nEntry: 0x{args.entry:x}")
print(f"Decoded instructions: {len(instructions)}")
print(f"Basic blocks: {len(blocks)}")
print(f"CFG edges: {len(block_edges)}")

for leader, body in blocks.items():
    print(f"\n[Basic Block 0x{leader:x}]")

    for ins in body:
        print(
            f"  0x{ins.address:x}: "
            f"{ins.mnemonic:8} {ins.op_str}"
        )

    outgoing = [
        (dst, kind)
        for src, dst, kind in sorted(block_edges)
        if src == leader
    ]

    print("  successors:", [
        f"0x{dst:x} ({kind})"
        for dst, kind in outgoing
    ])


# Phase 4: Save Graphviz DOT.
dot = [
    "digraph CFG {",
    '  node [shape=box, fontname="monospace"];',
]

for leader, body in blocks.items():
    lines = [f"0x{leader:x}"]

    for ins in body:
        lines.append(
            f"0x{ins.address:x}: "
            f"{ins.mnemonic} {ins.op_str}"
        )

    label = "\\l".join(
        line.replace('"', '\\"')
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
