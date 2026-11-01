
import argparse
import struct
from io import BytesIO
from pathlib import Path

from capstone import Cs, CS_ARCH_X86, CS_MODE_64, CS_GRP_JUMP
from capstone.x86 import X86_OP_MEM, X86_REG_RIP
from elftools.elf.elffile import ELFFile


parser = argparse.ArgumentParser()
parser.add_argument("binary", type=Path)
parser.add_argument("--table-base", required=True,
                    type=lambda x: int(x, 0))
parser.add_argument("--case", type=int, default=0)
parser.add_argument("--output", default="jump7_rewritten")
args = parser.parse_args()

blob = bytearray(args.binary.read_bytes())
elf = ELFFile(BytesIO(blob))

text = elf.get_section_by_name(".text")
rodata = elf.get_section_by_name(".rodata")

if text is None or rodata is None:
    raise RuntimeError("Expected .text and .rodata sections")

text_va = text["sh_addr"]
text_off = text["sh_offset"]
text_size = text["sh_size"]

ro_va = rodata["sh_addr"]
ro_off = rodata["sh_offset"]
ro_size = rodata["sh_size"]

md = Cs(CS_ARCH_X86, CS_MODE_64)
md.detail = True

ENDBR64 = bytes.fromhex("f3 0f 1e fa")
NOP_COUNT = 64
CAVE_PATTERN = ENDBR64 + b"\x90" * NOP_COUNT


def text_file_offset(va, size=1):
    if not (text_va <= va and va + size <= text_va + text_size):
        raise RuntimeError(
            f"Address 0x{va:x} is outside .text"
        )
    return text_off + va - text_va


def rodata_file_offset(va, size=1):
    if not (ro_va <= va and va + size <= ro_va + ro_size):
        raise RuntimeError(
            f"Address 0x{va:x} is outside .rodata"
        )
    return ro_off + va - ro_va


def signed32(value):
    if not -(1 << 31) <= value < (1 << 31):
        raise RuntimeError(
            f"Relative offset out of 32-bit range: {value}"
        )
    return value


def find_code_cave():
    data = bytes(blob[text_off:text_off + text_size])

    matches = []
    start = 0

    while True:
        index = data.find(CAVE_PATTERN, start)
        if index < 0:
            break

        matches.append(index)
        start = index + 1

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one code cave, found {len(matches)}"
        )

    index = matches[0]

    return (
        text_va + index,
        text_off + index,
        len(CAVE_PATTERN),
    )


def read_case_target(table_base, case_index):
    entry_va = table_base + case_index * 4
    entry_off = rodata_file_offset(entry_va, 4)

    relative = struct.unpack_from("<i", blob, entry_off)[0]

    return entry_off, table_base + relative


def extract_case(target):
    """
    This experiment moves a simple, linear case block.
    Reject branches, and stop at the first RET.
    """
    offset = text_file_offset(target)
    remaining = bytes(
        blob[offset:text_off + text_size]
    )

    result = []
    length = 0

    for ins in md.disasm(remaining, target):
        if ins.group(CS_GRP_JUMP):
            raise RuntimeError(
                "Case contains a jump: unsupported in Exp7"
            )

        if ins.mnemonic.startswith("call"):
            raise RuntimeError(
                "Case contains a call: unsupported in Exp7"
            )

        result.append(ins)
        length += ins.size

        if ins.mnemonic.startswith("ret"):
            return result, length

        if length > 64:
            raise RuntimeError("Case block is too large")

    raise RuntimeError("Could not find terminating RET")


# 1. Locate the original case using the jump table.
entry_offset, old_target = read_case_target(
    args.table_base,
    args.case,
)

instructions, code_size = extract_case(old_target)

print("=== ORIGINAL CASE ===")
print(f"Case index:       {args.case}")
print(f"Original target:  0x{old_target:x}")
print(f"Code size:        {code_size} bytes")

for ins in instructions:
    print(
        f"  0x{ins.address:x}: "
        f"{ins.mnemonic} {ins.op_str}"
    )


# 2. Find the reserved executable cave.
cave_va, cave_off, cave_capacity = find_code_cave()

# Keep ENDBR64 at the new entry.
new_entry = cave_va
new_code_va = cave_va + len(ENDBR64)
new_code_off = cave_off + len(ENDBR64)

if code_size > cave_capacity - len(ENDBR64):
    raise RuntimeError("Not enough code cave space")

print("\n=== NEW LOCATION ===")
print(f"Cave entry:       0x{new_entry:x}")
print(f"Copied code VA:   0x{new_code_va:x}")


# 3. Copy machine code.
original_offset = text_file_offset(old_target, code_size)
copied = bytearray(
    blob[original_offset:original_offset + code_size]
)


# 4. Repair RIP-relative memory references.
print("\n=== RIP-RELATIVE REPAIR ===")

for ins in instructions:
    for operand in ins.operands:
        if not (
            operand.type == X86_OP_MEM
            and operand.mem.base == X86_REG_RIP
        ):
            continue

        if ins.disp_size != 4:
            raise RuntimeError(
                "Expected 32-bit RIP-relative displacement"
            )

        # Original data reference target.
        data_target = (
            ins.address
            + ins.size
            + operand.mem.disp
        )

        # Instruction position within the copied case.
        relative_position = ins.address - old_target

        new_ins_address = new_code_va + relative_position

        new_disp = signed32(
            data_target - (new_ins_address + ins.size)
        )

        patch_offset = relative_position + ins.disp_offset

        struct.pack_into(
            "<i",
            copied,
            patch_offset,
            new_disp,
        )

        print(
            f"0x{ins.address:x} -> 0x{new_ins_address:x}: "
            f"data target 0x{data_target:x}, "
            f"new disp {new_disp:+d}"
        )


# 5. Write the relocated instructions into the cave.
blob[new_code_off:new_code_off + code_size] = copied


# 6. Repair the jump table entry.
new_relative = signed32(
    new_entry - args.table_base
)

struct.pack_into(
    "<i",
    blob,
    entry_offset,
    new_relative,
)

print("\n=== JUMP TABLE REPAIR ===")
print(f"Old target:       0x{old_target:x}")
print(f"New target:       0x{new_entry:x}")
print(f"New table offset: {new_relative:+d}")


# 7. Save without changing ELF layout.
output = Path(args.output)
output.write_bytes(blob)
output.chmod(args.binary.stat().st_mode & 0o777)

print(f"\nSaved: {output}")
