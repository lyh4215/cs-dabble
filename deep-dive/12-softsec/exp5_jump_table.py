
import struct
from pathlib import Path

from elftools.elf.elffile import ELFFile

BINARY = Path("jump_stripped")

# Recovered from machine code:
TABLE_BASE = 0x2004
ENTRY_COUNT = 8

with BINARY.open("rb") as f:
    elf = ELFFile(f)

    # Find the section containing the jump table.
    section = next(
        s for s in elf.iter_sections()
        if s["sh_type"] != "SHT_NOBITS"
        and s["sh_addr"] <= TABLE_BASE
        < s["sh_addr"] + s["sh_size"]
    )

    data = section.data()
    offset = TABLE_BASE - section["sh_addr"]

    # 8 signed, little-endian 32-bit offsets.
    entries = struct.unpack_from(
        "<" + "i" * ENTRY_COUNT,
        data,
        offset,
    )

    print(f"Table section: {section.name}")
    print(f"Table base: 0x{TABLE_BASE:x}")
    print()

    for index, relative in enumerate(entries):
        target = TABLE_BASE + relative

        print(
            f"case {index}: "
            f"relative={relative:+d}, "
            f"target=0x{target:x}"
        )
