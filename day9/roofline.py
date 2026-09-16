import statistics
import torch


DEVICE = "cuda"

print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# CUDA timing helper
# ============================================================

def benchmark(fn, warmup=10, repeats=30):
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []

    for _ in range(repeats):
        start.record()

        fn()

        end.record()

        torch.cuda.synchronize()

        times.append(
            start.elapsed_time(end)
        )

    return statistics.median(times)


# ============================================================
# 1. MEMORY-BOUND
# A + B
# ============================================================

print()
print("=== MEMORY BOUND: VECTOR ADD ===")

VECTOR_SIZES = [
    1_000_000,
    4_000_000,
    16_000_000,
    32_000_000,
]

print(
    f"{'elements':>12} "
    f"{'latency(ms)':>14} "
    f"{'GB/s':>12}"
)

print("-" * 42)


for n in VECTOR_SIZES:

    A = torch.randn(
        n,
        device=DEVICE,
        dtype=torch.float32,
    )

    B = torch.randn(
        n,
        device=DEVICE,
        dtype=torch.float32,
    )

    C = torch.empty_like(A)

    def run():
        torch.add(
            A,
            B,
            out=C,
        )

    latency_ms = benchmark(run)

    # float32 = 4 bytes
    #
    # A read : 4N
    # B read : 4N
    # C write: 4N
    #
    # total = 12N bytes

    bytes_moved = (
        n * 12
    )

    seconds = (
        latency_ms / 1000
    )

    bandwidth_gbs = (
        bytes_moved
        / seconds
        / 1e9
    )

    print(
        f"{n:>12,} "
        f"{latency_ms:>14.4f} "
        f"{bandwidth_gbs:>12.2f}"
    )


# ============================================================
# 2. COMPUTE-BOUND
# Matrix multiplication
# ============================================================

print()
print("=== COMPUTE BOUND: MATMUL ===")

MATRIX_SIZES = [
    256,
    512,
    1024,
    2048,
    4096,
]

print(
    f"{'N':>8} "
    f"{'latency(ms)':>14} "
    f"{'TFLOP/s':>12} "
    f"{'ideal AI':>12}"
)

print("-" * 52)


for n in MATRIX_SIZES:

    A = torch.randn(
        n,
        n,
        device=DEVICE,
        dtype=torch.float32,
    )

    B = torch.randn(
        n,
        n,
        device=DEVICE,
        dtype=torch.float32,
    )

    C = torch.empty(
        n,
        n,
        device=DEVICE,
        dtype=torch.float32,
    )

    def run():
        torch.mm(
            A,
            B,
            out=C,
        )

    latency_ms = benchmark(run)

    # GEMM:
    #
    # C = A @ B
    #
    # N^3 multiply
    # N^3 add
    #
    # ~2N^3 FLOPs

    flops = (
        2 * (n ** 3)
    )

    seconds = (
        latency_ms / 1000
    )

    tflops = (
        flops
        / seconds
        / 1e12
    )


    # 이상적으로:
    #
    # A read: 4N²
    # B read: 4N²
    # C write: 4N²
    #
    # total ≈ 12N² bytes
    #
    # AI ≈ 2N³ / 12N²
    #    = N / 6

    ideal_ai = (
        n / 6
    )


    print(
        f"{n:>8} "
        f"{latency_ms:>14.4f} "
        f"{tflops:>12.3f} "
        f"{ideal_ai:>12.1f}"
    )