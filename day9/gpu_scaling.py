import time
import statistics

import torch


# ============================================================
# Device
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("device:", DEVICE)

if DEVICE == "cuda":
    print(
        "gpu:",
        torch.cuda.get_device_name(0),
    )


# ============================================================
# Settings
# ============================================================

SIZES = [
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
]

REPEATS = 30
WARMUP = 10


# ============================================================
# 정확한 GPU timer
# ============================================================

def benchmark_cpu(
    A,
    B,
):
    times = []

    # warmup
    for _ in range(WARMUP):
        _ = A @ B

    for _ in range(REPEATS):
        start = time.perf_counter()

        _ = A @ B

        elapsed = (
            time.perf_counter()
            - start
        )

        times.append(
            elapsed * 1000
        )

    return statistics.median(
        times
    )


def benchmark_gpu(
    A,
    B,
):
    # GPU operation은 asynchronous라
    # perf_counter만 바로 쓰면 안 됨.

    for _ in range(WARMUP):
        _ = A @ B

    torch.cuda.synchronize()


    start_event = (
        torch.cuda.Event(
            enable_timing=True
        )
    )

    end_event = (
        torch.cuda.Event(
            enable_timing=True
        )
    )


    times = []

    for _ in range(REPEATS):

        start_event.record()

        _ = A @ B

        end_event.record()

        torch.cuda.synchronize()

        elapsed_ms = (
            start_event.elapsed_time(
                end_event
            )
        )

        times.append(
            elapsed_ms
        )

    return statistics.median(
        times
    )


# ============================================================
# Benchmark
# ============================================================

print()

print(
    f"{'N':>8} "
    f"{'CPU(ms)':>12} "
    f"{'GPU(ms)':>12} "
    f"{'speedup':>10}"
)

print("-" * 50)


for n in SIZES:

    A_cpu = torch.randn(
        n,
        n,
        dtype=torch.float32,
    )

    B_cpu = torch.randn(
        n,
        n,
        dtype=torch.float32,
    )

    cpu_ms = benchmark_cpu(
        A_cpu,
        B_cpu,
    )


    if DEVICE == "cuda":

        A_gpu = A_cpu.cuda()
        B_gpu = B_cpu.cuda()

        gpu_ms = benchmark_gpu(
            A_gpu,
            B_gpu,
        )

        speedup = (
            cpu_ms
            / gpu_ms
        )

        print(
            f"{n:>8} "
            f"{cpu_ms:>12.4f} "
            f"{gpu_ms:>12.4f} "
            f"{speedup:>9.2f}x"
        )

    else:

        print(
            f"{n:>8} "
            f"{cpu_ms:>12.4f} "
            f"{'-':>12} "
            f"{'-':>10}"
        )