BLOCK_SIZES_KB = [
    1, 2, 4, 8, 16,
    32, 64, 128, 256,
]

CHUNK_BYTES = 128

# 논문에서 관측된 DPU-side ORAM 시간
PER_DPU_MS = 2.41


print(
    f"{'Block(KB)':>10}"
    f"{'DPUs':>8}"
    f"{'Each DPU':>12}"
    f"{'Serial(ms)':>14}"
    f"{'Parallel(ms)':>16}"
)

for kb in BLOCK_SIZES_KB:

    block_bytes = kb * 1024

    num_dpus = block_bytes // CHUNK_BYTES

    # 만약 DPU들이 순차 실행됐다면
    serial = num_dpus * PER_DPU_MS

    # 모두 동시에 실행된다면 critical path는
    # 가장 느린 DPU 하나의 시간
    parallel = PER_DPU_MS

    print(
        f"{kb:10d}"
        f"{num_dpus:8d}"
        f"{CHUNK_BYTES:9d} B"
        f"{serial:14.2f}"
        f"{parallel:16.2f}"
    )