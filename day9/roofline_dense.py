import statistics

import matplotlib.pyplot as plt
import numpy as np
import torch


DEVICE = "cuda"

print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Settings
# ============================================================

# 작은 GEMM → GPU가 차오르는 과정 → 포화까지 촘촘히
MATRIX_SIZES = [
    64,
    96,
    128,
    160,
    192,
    224,
    256,
    320,
    384,
    448,
    512,
    640,
    768,
    896,
    1024,
    1280,
    1536,
    1792,
    2048,
    2304,
    2560,
    3072,
    3584,
    4096,
]

VECTOR_SIZES = [
    250_000,
    500_000,
    1_000_000,
    2_000_000,
    4_000_000,
    8_000_000,
    16_000_000,
    32_000_000,
]

WARMUP = 10
REPEATS = 20


# ============================================================
# CUDA benchmark helper
# ============================================================

def benchmark(fn):
    for _ in range(WARMUP):
        fn()

    torch.cuda.synchronize()

    start_event = torch.cuda.Event(
        enable_timing=True
    )

    end_event = torch.cuda.Event(
        enable_timing=True
    )

    times = []

    for _ in range(REPEATS):
        start_event.record()

        fn()

        end_event.record()

        torch.cuda.synchronize()

        times.append(
            start_event.elapsed_time(
                end_event
            )
        )

    return statistics.median(times)


# ============================================================
# 1. Vector Add benchmark
# ============================================================

print()
print("=== VECTOR ADD ===")

vector_results = []

print(
    f"{'elements':>12} "
    f"{'latency(ms)':>14} "
    f"{'GB/s':>12} "
    f"{'GFLOP/s':>12}"
)

print("-" * 56)


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

    seconds = latency_ms / 1000


    # ---------------------------------------------
    # Memory traffic
    #
    # read A : 4 bytes
    # read B : 4 bytes
    # write C: 4 bytes
    #
    # = 12 bytes / element
    # ---------------------------------------------

    bytes_moved = (
        n * 12
    )

    bandwidth_gbs = (
        bytes_moved
        / seconds
        / 1e9
    )


    # ---------------------------------------------
    # FLOPs
    #
    # one add / element
    # ---------------------------------------------

    flops = n

    gflops = (
        flops
        / seconds
        / 1e9
    )


    vector_results.append({
        "n": n,
        "latency_ms": latency_ms,
        "bandwidth_gbs": bandwidth_gbs,
        "gflops": gflops,
    })


    print(
        f"{n:>12,} "
        f"{latency_ms:>14.4f} "
        f"{bandwidth_gbs:>12.2f} "
        f"{gflops:>12.3f}"
    )


# ============================================================
# 2. Matmul benchmark
# ============================================================

print()
print("=== MATMUL ===")

matmul_results = []

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

    seconds = (
        latency_ms / 1000
    )


    # ---------------------------------------------
    # GEMM FLOPs
    #
    # approximately 2N^3
    # ---------------------------------------------

    flops = (
        2 * n**3
    )

    tflops = (
        flops
        / seconds
        / 1e12
    )


    # ---------------------------------------------
    # Ideal arithmetic intensity
    #
    # FLOPs ≈ 2N^3
    #
    # ideal memory traffic:
    #
    # A = 4N^2
    # B = 4N^2
    # C = 4N^2
    #
    # ≈ 12N^2 bytes
    #
    # AI = 2N^3 / 12N^2
    #    = N / 6
    # ---------------------------------------------

    ideal_ai = (
        n / 6
    )


    matmul_results.append({
        "n": n,
        "latency_ms": latency_ms,
        "tflops": tflops,
        "ai": ideal_ai,
    })


    print(
        f"{n:>8} "
        f"{latency_ms:>14.4f} "
        f"{tflops:>12.3f} "
        f"{ideal_ai:>12.1f}"
    )


# ============================================================
# 3. Empirical ceilings
# ============================================================

# 큰 vector-add 결과 몇 개의 median을 사용
large_vector_bandwidths = [
    r["bandwidth_gbs"]
    for r in vector_results[-3:]
]

MEMORY_BW_GBS = statistics.median(
    large_vector_bandwidths
)


# 큰 GEMM 결과 몇 개의 최대/근접 성능
large_matmul_tflops = [
    r["tflops"]
    for r in matmul_results[-5:]
]

COMPUTE_TFLOPS = max(
    large_matmul_tflops
)


ridge_ai = (
    COMPUTE_TFLOPS * 1000
    / MEMORY_BW_GBS
)


print()
print("=== EMPIRICAL ROOFLINE ===")

print(
    f"Memory ceiling : "
    f"{MEMORY_BW_GBS:.2f} GB/s"
)

print(
    f"Compute ceiling: "
    f"{COMPUTE_TFLOPS:.3f} TFLOP/s"
)

print(
    f"Ridge point    : "
    f"{ridge_ai:.2f} FLOP/byte"
)


# ============================================================
# 4. Build Roofline
# ============================================================

ai_range = np.logspace(
    -2,
    3.2,
    1000,
)


# Memory roof
#
# GB/s * FLOP/byte
# = GFLOP/s
#
# /1000
# = TFLOP/s

memory_roof = (
    MEMORY_BW_GBS
    * ai_range
    / 1000
)


roofline = np.minimum(
    memory_roof,
    COMPUTE_TFLOPS,
)


# ============================================================
# 5. Plot Roofline
# ============================================================

plt.figure(
    figsize=(12, 8)
)


plt.loglog(
    ai_range,
    roofline,
    linewidth=3,
    label="Empirical Roofline",
)


plt.loglog(
    ai_range,
    memory_roof,
    linestyle="--",
    alpha=0.5,
    label="Memory bandwidth roof",
)


plt.axhline(
    COMPUTE_TFLOPS,
    linestyle="--",
    alpha=0.6,
    label=(
        f"Compute ceiling "
        f"({COMPUTE_TFLOPS:.2f} TFLOP/s)"
    ),
)


plt.axvline(
    ridge_ai,
    linestyle=":",
    alpha=0.7,
    label=(
        f"Ridge point "
        f"({ridge_ai:.1f} FLOP/B)"
    ),
)


# ============================================================
# Vector Add points
# ============================================================

VECTOR_ADD_AI = (
    1 / 12
)

vector_tflops = np.array([
    r["gflops"] / 1000
    for r in vector_results
])

vector_ai = np.full(
    len(vector_results),
    VECTOR_ADD_AI,
)


plt.scatter(
    vector_ai,
    vector_tflops,
    s=55,
    label="Vector Add",
    zorder=5,
)


# ============================================================
# Matmul points
# ============================================================

matmul_ai = np.array([
    r["ai"]
    for r in matmul_results
])

matmul_tflops = np.array([
    r["tflops"]
    for r in matmul_results
])


plt.scatter(
    matmul_ai,
    matmul_tflops,
    s=55,
    label="Measured GEMM",
    zorder=5,
)


# label 일부만 표시
LABEL_SIZES = {
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
}


for r in matmul_results:

    if r["n"] in LABEL_SIZES:

        plt.annotate(
            f"N={r['n']}",
            (
                r["ai"],
                r["tflops"],
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )


plt.xlabel(
    "Arithmetic Intensity (FLOP / byte)"
)

plt.ylabel(
    "Performance (TFLOP / s)"
)

plt.title(
    "Dense Empirical Roofline - Tesla T4"
)

plt.grid(
    True,
    which="both",
    alpha=0.25,
)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 6. Extra plot:
# GEMM size vs performance
# ============================================================

plt.figure(
    figsize=(11, 6)
)


plt.plot(
    MATRIX_SIZES,
    matmul_tflops,
    marker="o",
)


plt.axhline(
    COMPUTE_TFLOPS,
    linestyle="--",
    alpha=0.6,
)


plt.xlabel(
    "Matrix size N"
)

plt.ylabel(
    "Performance (TFLOP / s)"
)

plt.title(
    "GEMM Performance Scaling - Tesla T4"
)

plt.grid(
    True,
    alpha=0.25,
)

plt.tight_layout()

plt.show()


# ============================================================
# 7. Extra plot:
# Vector size vs memory bandwidth
# ============================================================

plt.figure(
    figsize=(11, 6)
)


plt.plot(
    [
        r["n"]
        for r in vector_results
    ],
    [
        r["bandwidth_gbs"]
        for r in vector_results
    ],
    marker="o",
)


plt.axhline(
    MEMORY_BW_GBS,
    linestyle="--",
    alpha=0.6,
)


plt.xscale("log")


plt.xlabel(
    "Number of elements"
)

plt.ylabel(
    "Effective memory bandwidth (GB/s)"
)

plt.title(
    "Vector Add Memory Bandwidth - Tesla T4"
)

plt.grid(
    True,
    alpha=0.25,
)

plt.tight_layout()

plt.show()