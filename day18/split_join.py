NUM_BANKS = 4


def split_block(data: bytes):
    """
    block을 bank 수만큼 interleaving해서 분리
    """

    parts = [
        bytearray()
        for _ in range(NUM_BANKS)
    ]

    for i, byte in enumerate(data):
        bank = i % NUM_BANKS
        parts[bank].append(byte)

    return [
        bytes(part)
        for part in parts
    ]


def join_block(parts):
    """
    split의 역연산
    """

    result = bytearray()

    max_len = max(
        len(part)
        for part in parts
    )

    for i in range(max_len):
        for bank in range(NUM_BANKS):
            if i < len(parts[bank]):
                result.append(parts[bank][i])

    return bytes(result)


def pim_process(bank_id, part):
    """
    실제 논문에서는 여기서
    각 bank의 DPU가 자기 ORAM tree를 처리한다고 생각.
    """

    print(
        f"Bank {bank_id}:",
        part,
        "processing..."
    )

    return part


data = b"ABCDEFGHIJKLMNOP"

print("Original:")
print(data)

parts = split_block(data)

print("\n=== Split ===")

for bank, part in enumerate(parts):
    print(
        f"Bank {bank}:",
        part,
    )


print("\n=== Parallel PIM-side processing ===")

processed = [
    pim_process(bank, part)
    for bank, part in enumerate(parts)
]


result = join_block(processed)

print("\n=== Join ===")
print(result)

print(
    "\nCorrect:",
    result == data,
)