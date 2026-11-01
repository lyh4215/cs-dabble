
from pathlib import Path
import struct
import subprocess

original = Path("example").read_bytes()

# User-observed instruction:
# add eax, DWORD PTR [rip+0x2ebb]
instruction = bytes.fromhex("03 05 bb 2e 00 00")

assert original.count(instruction) == 1, (
    "Expected exactly one matching instruction"
)

offset = original.index(instruction)

patched = bytearray(original)

# 03 05: opcode and ModRM
# Following four bytes: signed displacement
old_disp = struct.unpack_from("<i", patched, offset + 2)[0]
new_disp = old_disp + 1

struct.pack_into("<i", patched, offset + 2, new_disp)

Path("example_patched").write_bytes(patched)
Path("example_patched").chmod(0o755)

print(f"File offset: 0x{offset:x}")
print(f"Old displacement: 0x{old_disp:x}")
print(f"New displacement: 0x{new_disp:x}")

for binary in ["./example", "./example_patched"]:
    result = subprocess.run(
        [binary], capture_output=True, text=True
    )
    print(
        binary,
        "returncode =", result.returncode,
        "stdout =", result.stdout.strip()
    )
