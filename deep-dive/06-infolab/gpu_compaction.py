"""InfoLab Deep Dive 06: GPU two-pass scan/scatter vs one-pass atomic append.

CPU: NumPy reference/correctness only (not a simulated GPU latency result).
GPU: CuPy RawKernel (CUDA). Timings include kernel launches and scan, but exclude
input/output allocations, host transfers, and correctness validation.

Install GPU CuPy wheel matching CUDA toolkit, e.g. `pip install cupy-cuda12x`.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
import numpy as np

CUDA_SOURCE = r'''
extern "C" {
__device__ __forceinline__ unsigned int mix32(unsigned int x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16; return x;
}
__global__ void mark(const int n, const unsigned int seed,
                     const unsigned long long threshold, int* flags) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) flags[i] = ((unsigned long long)mix32(((unsigned int)i) ^ seed) < threshold);
}
__global__ void scatter(const int n, const int* flags,
                        const int* inclusive_scan, int* output) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n && flags[i]) output[inclusive_scan[i] - 1] = i;
}
__global__ void atomic_append(const int n, const unsigned int seed,
                              const unsigned long long threshold,
                              const int capacity, int* count, int* overflow,
                              int* output) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n && (unsigned long long)mix32(((unsigned int)i) ^ seed) < threshold) {
        int slot = atomicAdd(count, 1);
        if (slot < capacity) output[slot] = i;
        else atomicExch(overflow, 1);
    }
}
}
'''


def threshold_for(selectivity: float) -> int:
    if not 0 <= selectivity <= 1:
        raise ValueError('selectivity must be within [0, 1]')
    return int(selectivity * (1 << 32))


def cpu_reference(n: int, seed: int, selectivity: float) -> np.ndarray:
    """32-bit integer hashing with the same unsigned overflow as the CUDA code."""
    x = np.arange(n, dtype=np.uint32) ^ np.uint32(seed)
    with np.errstate(over='ignore'):
        x ^= x >> np.uint32(16)
        x *= np.uint32(0x7FEB352D)
        x ^= x >> np.uint32(15)
        x *= np.uint32(0x846CA68B)
        x ^= x >> np.uint32(16)
    return np.flatnonzero(x.astype(np.uint64) < threshold_for(selectivity)).astype(np.int32)


def cpu_sanity(n: int, seed: int, selectivities: list[float]):
    print('CPU correctness/reference (NOT GPU kernel timing):')
    for p in selectivities:
        start = time.perf_counter()
        ref = cpu_reference(n, seed, p)
        elapsed = (time.perf_counter() - start) * 1e3
        print(f'  p={p:5.2%}  count={len(ref):>9}  reference={elapsed:7.2f} ms')
    print('  GPU measurements require --device cuda and a working CuPy/CUDA installation.')


def gpu_benchmark(n: int, seed: int, selectivities: list[float],
                  repeats: int, warmup: int, capacity_factor: float, block: int):
    try:
        import cupy as cp
    except ImportError as ex:
        raise SystemExit('GPU benchmark requires CuPy: install a wheel matching your CUDA toolkit') from ex
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise SystemExit('No CUDA GPU visible to CuPy')

    module = cp.RawModule(code=CUDA_SOURCE, options=('-std=c++11',),
                          name_expressions=('mark', 'scatter', 'atomic_append'))
    mark = module.get_function('mark')
    scatter = module.get_function('scatter')
    atomic = module.get_function('atomic_append')
    grid = ((n + block - 1) // block,)
    blocks = (block,)
    seed32 = np.uint32(seed)
    rows = []

    def timed(fn):
        # Each trial starts after a synchronization; no allocations in the measured function.
        for _ in range(warmup):
            fn()
        cp.cuda.Stream.null.synchronize()
        readings = []
        for _ in range(repeats):
            start = cp.cuda.Event()
            stop = cp.cuda.Event()
            start.record()
            fn()
            stop.record()
            stop.synchronize()
            readings.append(cp.cuda.get_elapsed_time(start, stop))
        return float(np.median(readings))

    for p in selectivities:
        threshold = np.uint64(threshold_for(p))
        flags = cp.empty(n, dtype=cp.int32)
        scan = cp.empty(n, dtype=cp.int32)
        scan_output = cp.empty(n, dtype=cp.int32)  # preallocated workspace
        capacity = max(1, min(n, int(n * capacity_factor)))
        atomic_output = cp.empty(capacity, dtype=cp.int32)
        count = cp.zeros(1, dtype=cp.int32)
        overflow = cp.zeros(1, dtype=cp.int32)

        def scan_run():
            mark(grid, blocks, (np.int32(n), seed32, threshold, flags))
            # cp.cumsum(out=...) keeps scan allocation outside the timed region.
            cp.cumsum(flags, dtype=cp.int32, out=scan)
            scatter(grid, blocks, (np.int32(n), flags, scan, scan_output))

        def atomic_run():
            count.fill(0)
            overflow.fill(0)
            atomic(grid, blocks, (np.int32(n), seed32, threshold,
                                  np.int32(capacity), count, overflow, atomic_output))

        scan_ms = timed(scan_run)
        atomic_ms = timed(atomic_run)
        scan_run()
        atomic_run()
        cp.cuda.Stream.null.synchronize()

        ref = cpu_reference(n, seed, p)
        got_count = int(count.get()[0])
        did_overflow = bool(overflow.get()[0])
        expected_overflow = len(ref) > capacity
        scan_count = int(scan[-1].get())
        got_scan = cp.asnumpy(scan_output[:scan_count])
        assert np.array_equal(got_scan, ref), 'scan/scatter mismatch'
        assert scan_count == len(ref), 'scan count mismatch'
        assert got_count == len(ref), 'atomic counter mismatch'
        assert did_overflow == expected_overflow, 'atomic overflow flag mismatch'
        if not did_overflow:
            got_atomic = cp.asnumpy(cp.sort(atomic_output[:got_count]))
            assert np.array_equal(got_atomic, ref), 'atomic output mismatch'

        rows.append(dict(n=n, selectivity=p, actual_count=len(ref),
                         capacity=capacity, atomic_overflow=did_overflow,
                         scan_ms=round(scan_ms, 4), atomic_ms=round(atomic_ms, 4),
                         correctness='pass'))
        print(f'p={p:5.1%} count={len(ref):>9} cap={capacity:>9} '
              f'scan={scan_ms:8.3f}ms atomic={atomic_ms:8.3f}ms '
              f'atomic_overflow={did_overflow}')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    ap.add_argument('--n', type=int, default=600_000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--selectivities', nargs='+', type=float, default=[.02, .2, .8])
    ap.add_argument('--repeats', type=int, default=20)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--capacity-factor', type=float, default=1.0)
    ap.add_argument('--block', type=int, default=256)
    ap.add_argument('--output', type=Path, default=Path('gpu_compaction_results.csv'))
    args = ap.parse_args()
    if args.n <= 0 or args.n >= 2 ** 31:
        ap.error('n must be 1..(2^31-1)')
    if not 0 < args.capacity_factor <= 1:
        ap.error('--capacity-factor must be in (0,1]')
    if args.block <= 0 or args.block > 1024:
        ap.error('--block must be 1..1024')
    if args.device == 'cpu':
        cpu_sanity(args.n, args.seed, args.selectivities)
        return
    rows = gpu_benchmark(args.n, args.seed, args.selectivities,
                         args.repeats, args.warmup, args.capacity_factor, args.block)
    with args.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
