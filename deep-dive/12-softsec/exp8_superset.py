
import argparse
import struct
from io import BytesIO
from pathlib import Path

from elftools.elf.elffile import ELFFile


parser = argparse.ArgumentParser()
parser.add_argument("binary", type=Path)
parser.add_argument("--table", type=lambda x: int(x, 0),
                    default=0x2004)
parser.add_argument("--count", type=int, default=8)
parser.add_argument("--extra", type=int, default=4)
args = parser.parse_args()

original = bytearray(args.binary.read_bytes())
elf = ELFFile(BytesIO(original))

rodata = elf.get_section_by_name(".rodata")
text = elf.get_section_by_name(".text")

if rodata is None or text is None:
    raise RuntimeError("Missing .rodata or .text")

table = args.table
ro_base = rodata["sh_addr"]
ro_offset = rodata["sh_offset"]
ro_end = ro_base + rodata["sh_size"]

text_begin = text["sh_addr"]
text_end = text_begin + text["sh_size"]

if not (ro_base <= table < ro_end):
    raise RuntimeError("Table is outside .rodata")


def read_offset(blob, index):
    address = table + index * 4

    if address + 4 > ro_end:
        return None

    file_offset = ro_offset + address - ro_base
    return struct.unpack_from("<i", blob, file_offset)[0]


def target_of(relative):
    return table + relative


def executable_address(address):
    return text_begin <= address < text_end


def scan(blob, count):
    candidates = []

    for i in range(count):
        relative = read_offset(blob, i)
        if relative is None:
            break

        target = target_of(relative)

        candidates.append({
            "index": i,
            "relative": relative,
            "target": target,
            "in_text": executable_address(target),
        })

    return candidates


def show(title, candidates):
    print(f"\n=== {title} ===")

    for c in candidates:
        status = (
            "CODE-CANDIDATE"
            if c["in_text"]
            else "NOT-IN-TEXT"
        )

        print(
            f"[{c['index']:2}] "
            f"offset={c['relative']:+11} "
            f"target=0x{c['target']:x} "
            f"{status}"
        )


# 1. Ground truth for this specific fixture:
# the previously recovered 8 jump-table entries.
bounded = scan(original, args.count)

show("BOUNDED TABLE", bounded)

# 2. Scan beyond the known boundary.
expanded = scan(original, args.count + args.extra)

show("EXPANDED CANDIDATES", expanded)

print("\nActual valid entries:", len(bounded))
print(
    "Expanded executable-address candidates:",
    sum(c["in_text"] for c in expanded)
)

# 3. Inject an artificial false-positive candidate.
#
# The original binary is NOT modified.
# We temporarily interpret the next 4 bytes as though
# they contained an offset to an executable address.
#
# 0x11d8 is an actual code location in our old fixture,
# but we use a target found from the table instead.
if not bounded:
    raise RuntimeError("No table entries found")

fake_target = bounded[0]["target"]

fake_index = args.count
fake_va = table + fake_index * 4

if fake_va + 4 > ro_end:
    raise RuntimeError("Not enough trailing .rodata bytes")

synthetic = bytearray(original)
fake_file_offset = ro_offset + fake_va - ro_base

struct.pack_into(
    "<i",
    synthetic,
    fake_file_offset,
    fake_target - table,
)

synthetic_candidates = scan(
    synthetic, args.count + 1
)

show(
    "SYNTHETIC FALSE-POSITIVE EXPERIMENT",
    synthetic_candidates,
)

print("\nInjected data location:", hex(fake_va))
print("Injected apparent target:", hex(fake_target))
print("This entry is NOT part of the real jump table.")
print("No executable file was modified.")
