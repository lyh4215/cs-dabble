
from pathlib import Path
import re
import struct
import subprocess

SOURCE = Path("example")
DEST = Path("example_reassembled")
raw = bytearray(SOURCE.read_bytes())


def symbols():
    output = subprocess.check_output(
        ["nm", "-n", str(SOURCE)], text=True
    )
    result = {}

    for line in output.splitlines():
        m = re.fullmatch(
            r"([0-9a-fA-F]+)\s+\w\s+(\S+)", line
        )
        if m:
            result[m.group(2)] = int(m.group(1), 16)

    return result


def vaddr_to_offset(vaddr):
    # ELF64, little-endian only
    assert raw[:4] == b"\x7fELF"
    assert raw[4] == 2
    assert raw[5] == 1

    phoff = struct.unpack_from("<Q", raw, 0x20)[0]
    phentsize = struct.unpack_from("<H", raw, 0x36)[0]
    phnum = struct.unpack_from("<H", raw, 0x38)[0]

    for i in range(phnum):
        pos = phoff + i * phentsize

        p_type, p_flags = struct.unpack_from(
            "<II", raw, pos
        )

        (
            p_offset,
            p_vaddr,
            _,
            p_filesz,
            _,
            _,
        ) = struct.unpack_from("<QQQQQQ", raw, pos + 8)

        if (
            p_type == 1
            and p_flags & 1
            and p_vaddr <= vaddr < p_vaddr + p_filesz
        ):
            return p_offset + (vaddr - p_vaddr)

    raise ValueError(
        f"Address 0x{vaddr:x} not in executable LOAD segment"
    )


sym = symbols()

old = sym["calculate"]
new = sym["code_cave"]
main = sym["main"]

# calculate consists of exactly 13 bytes in this demo.
old_off = vaddr_to_offset(old)
orig = bytes(raw[old_off:old_off + 13])

# endbr64; mov eax,edi; add eax,[rip+disp32]; ret
assert orig[:8] == bytes.fromhex(
    "f3 0f 1e fa 89 f8 03 05"
), orig.hex(" ")
assert orig[-1] == 0xC3

# 1. Recover the original data target.
old_disp = struct.unpack_from("<i", orig, 8)[0]
data_target = old + 12 + old_disp

# 2. Copy function and repair RIP-relative reference.
new_disp = data_target - (new + 12)
assert -(2**31) <= new_disp < 2**31

copy = bytearray(orig)
struct.pack_into("<i", copy, 8, new_disp)

new_off = vaddr_to_offset(new)

# Ensure that we are writing into the reserved cave.
assert raw[new_off:new_off + 4] == bytes.fromhex(
    "f3 0f 1e fa"
)
assert all(
    b == 0x90
    for b in raw[new_off + 4:new_off + len(copy)]
), "Code cave is too small"

raw[new_off:new_off + len(copy)] = copy

# 3. Find the existing call to calculate inside main.
call_candidates = []

for call_va in range(main, main + 96):
    try:
        call_off = vaddr_to_offset(call_va)
    except ValueError:
        break

    if raw[call_off] != 0xE8:
        continue

    disp = struct.unpack_from("<i", raw, call_off + 1)[0]
    target = call_va + 5 + disp

    if target == old:
        call_candidates.append((call_va, call_off))

assert len(call_candidates) == 1, call_candidates
call_va, call_off = call_candidates[0]

# 4. Redirect main's call to the copied function.
new_call_disp = new - (call_va + 5)
assert -(2**31) <= new_call_disp < 2**31

struct.pack_into(
    "<i", raw, call_off + 1, new_call_disp
)

DEST.write_bytes(raw)
DEST.chmod(SOURCE.stat().st_mode)

print(f"calculate:         0x{old:x}")
print(f"code_cave:         0x{new:x}")
print(f"global_value:      0x{data_target:x}")
print(f"RIP displacement:  {old_disp:+d} -> {new_disp:+d}")
print(f"call site:         0x{call_va:x} -> 0x{new:x}")
print(f"written:           {DEST}")
