block_kb = [
    1, 2, 4, 8, 16,
    32, 64, 128, 256,
]

dpus = [
    8, 16, 32, 64, 128,
    256, 512, 1024, 2048,
]

# 논문 Table 1의 POMC thread 수
threads = [
    1, 1, 2, 2, 4,
    4, 8, 8, 8,
]

# 논문의 실제 POMC latency
pomc_ms = [
    0.43, 0.51, 0.57, 0.69,
    1.07, 1.73, 4.64, 8.68, 15.65,
]


print(
    f"{'Block(KB)':>10}"
    f"{'DPUs':>8}"
    f"{'Threads':>10}"
    f"{'Pieces/thread':>16}"
    f"{'POMC(ms)':>12}"
)

for b, d, t, latency in zip(
    block_kb,
    dpus,
    threads,
    pomc_ms,
):
    pieces_per_thread = d / t

    print(
        f"{b:10d}"
        f"{d:8d}"
        f"{t:10d}"
        f"{pieces_per_thread:16.1f}"
        f"{latency:12.2f}"
    )