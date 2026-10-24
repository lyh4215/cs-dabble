"""InfoLab 06, experiment 5: estimated output cardinality -> GPU capacity.

CUDA run compares end-to-end (synchronized wall-clock) strategies:
  * scan: exact count and scatter, with exact-size output allocation
  * full: atomic append with worst-case N-slot output allocation
  * sample: sampled Wilson upper count -> atomic append; on overflow double and retry

This is a toy filter/stream-compaction benchmark, not a GPU OOM guarantee.
CUDA support: CuPy RawKernel. Python 3.11+.
"""
from __future__ import annotations
import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
from gpu_compaction import cpu_reference, threshold_for

CUDA = r'''
extern "C" {
__device__ __forceinline__ unsigned int mix32(unsigned int x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16; return x;
}
__device__ __forceinline__ int accept(int i, int n, unsigned int seed,
      unsigned long long threshold, int distribution) {
    if (distribution == 0) return (unsigned long long)mix32(((unsigned int)i) ^ seed) < threshold;
    // Clustered: contiguous circular interval of length floor(n * p).
    int start = (int)(seed % (unsigned int)n);
    int relative = (i - start + n) % n;
    int num = (int)(((unsigned long long)n * threshold) >> 32);
    return relative < num;
}
__global__ void sample_count(int n, unsigned int seed, unsigned long long threshold,
      int distribution, int samples, int offset, int* counter) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= samples) return;
    // Stratified, one sample in each contiguous interval.
    int i = (int)(((unsigned long long)j * n) / samples);
    int end = (int)(((unsigned long long)(j+1) * n) / samples);
    int width = end - i;
    i += (offset + j * 1103515245U) % width;
    if (accept(i, n, seed, threshold, distribution)) atomicAdd(counter, 1);
}
__global__ void mark(int n, unsigned int seed, unsigned long long threshold,
      int distribution, int* flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) flags[i] = accept(i, n, seed, threshold, distribution);
}
__global__ void scatter(int n, const int* flags, const int* inclusive_scan,
      int* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && flags[i]) output[inclusive_scan[i]-1] = i;
}
__global__ void atomic_append(int n, unsigned int seed, unsigned long long threshold,
      int distribution, int capacity, int* count, int* overflow, int* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && accept(i, n, seed, threshold, distribution)) {
        int slot = atomicAdd(count, 1);
        if (slot < capacity) output[slot] = i;
        else if (slot == capacity) atomicExch(overflow, 1);
    }
}
}
'''


def wilson_upper(k: int, m: int, z: float) -> float:
    p = k / m
    denom = 1 + z * z / m
    center = (p + z * z / (2 * m)) / denom
    radius = z * math.sqrt((p*(1-p) + z*z/(4*m))/m) / denom
    return min(1., center + radius)


def choose_capacity(n: int, k: int, m: int, z: float, slack: float) -> int:
    upper = wilson_upper(k, m, z)
    return max(1, min(n, math.ceil(n * upper * (1 + slack))))


def reference(n: int, seed: int, p: float, distribution: str) -> np.ndarray:
    if distribution == 'uniform':
        return cpu_reference(n, seed, p)
    count = (n * threshold_for(p)) >> 32
    start = seed % n
    return np.sort((start + np.arange(count, dtype=np.int64)) % n).astype(np.int32)


def run_cpu(args):
    # CPU-only estimator/capacity validation, not GPU timing.
    rng = np.random.default_rng(args.seed)
    rows=[]
    for dist in args.distributions:
        for p in args.selectivities:
            ref=reference(args.n,args.seed,p,dist)
            flags=np.zeros(args.n,dtype=np.bool_)
            flags[ref]=True
            for m in args.samples:
                # Independent stratified random sample within m equal-width buckets.
                edges=np.linspace(0,args.n,m+1,dtype=np.int64)
                idx=edges[:-1]+np.array([rng.integers(0,int(b-a)) for a,b in zip(edges[:-1],edges[1:])])
                k=int(flags[idx].sum())
                cap=choose_capacity(args.n,k,m,args.z,args.slack)
                rows.append(dict(distribution=dist,selectivity=p,n=args.n,samples=m,
                    sample_hits=k,estimate_pct=round(100*k/m,3),actual=len(ref),
                    initial_capacity=cap,overflow=len(ref)>cap,capacity_over_actual=round(cap/max(1,len(ref)),3)))
    return rows


def run_cuda(args):
    try:
        import cupy as cp
    except ImportError as ex:
        raise SystemExit('Install compatible CuPy for CUDA GPU') from ex
    try:
        if cp.cuda.runtime.getDeviceCount()<1:
            raise RuntimeError('No CUDA device')
    except Exception as ex:
        raise SystemExit(f'CuPy CUDA initialization failed: {ex}') from ex
    module=cp.RawModule(code=CUDA,options=('-std=c++11',),name_expressions=('sample_count','mark','scatter','atomic_append'))
    sample_fn=module.get_function('sample_count')
    mark_fn=module.get_function('mark')
    scatter_fn=module.get_function('scatter')
    atomic_fn=module.get_function('atomic_append')
    n=args.n; block=args.block; grid=((n+block-1)//block,); blocks=(block,)
    seed=np.uint32(args.seed); rng=np.random.default_rng(args.seed)
    rows=[]
    def atomic_call(p,dist,cap):
        output=cp.empty(cap,dtype=cp.int32)
        count=cp.zeros(1,dtype=cp.int32)
        overflow=cp.zeros(1,dtype=cp.int32)
        atomic_fn(grid,blocks,(np.int32(n),seed,np.uint64(threshold_for(p)),np.int32(dist),np.int32(cap),count,overflow,output))
        count_host=int(count.get()[0])  # also synchronizes
        return count_host, output, bool(overflow.get()[0])
    def once(method,p,dist,m):
        start=time.perf_counter()
        if method=='scan':
            flags=cp.empty(n,dtype=cp.int32)
            scan=cp.empty(n,dtype=cp.int32)
            mark_fn(grid,blocks,(np.int32(n),seed,np.uint64(threshold_for(p)),np.int32(dist),flags))
            cp.cumsum(flags,dtype=cp.int32,out=scan)
            count=int(scan[-1].get())
            out=cp.empty(max(1,count),dtype=cp.int32)
            scatter_fn(grid,blocks,(np.int32(n),flags,scan,out))
            cp.cuda.Stream.null.synchronize()
            capacity=count; retries=0; estimated=None; sample_hits=None
        elif method=='full':
            capacity=n; count,out,overflow=atomic_call(p,dist,capacity)
            assert not overflow
            retries=0; estimated=None; sample_hits=None
        else:
            samples_counter=cp.zeros(1,dtype=cp.int32)
            offset=int(rng.integers(0,2**30))
            sample_fn(((m+block-1)//block,),blocks,(np.int32(n),seed,np.uint64(threshold_for(p)),np.int32(dist),np.int32(m),np.int32(offset),samples_counter))
            sample_hits=int(samples_counter.get()[0])
            capacity=choose_capacity(n,sample_hits,m,args.z,args.slack)
            estimated=n*sample_hits/m
            retries=0
            while True:
                count,out,overflow=atomic_call(p,dist,capacity)
                if not overflow: break
                retries+=1
                if retries>args.max_retries:
                    raise RuntimeError('Exceeded max retries; increase --max-retries')
                capacity=min(n,max(capacity+1,capacity*2))
            cp.cuda.Stream.null.synchronize()
        elapsed=(time.perf_counter()-start)*1000
        return dict(ms=elapsed,actual=count,capacity=capacity,retries=retries,estimated=estimated,sample_hits=sample_hits),out[:count]

    for dist_name in args.distributions:
        dist=0 if dist_name=='uniform' else 1
        for p in args.selectivities:
            expected=reference(n,args.seed,p,dist_name)
            for method in ('scan','full','sample'):
                m_values=args.samples if method=='sample' else [0]
                for m in m_values:
                    measurements=[]
                    for rep in range(args.warmup+args.repeats):
                        stats,out=once(method,p,dist,m)
                        if rep==args.warmup:
                            got=np.sort(cp.asnumpy(out))
                            assert np.array_equal(got,expected),f'{method}: incorrect output'
                        if rep>=args.warmup: measurements.append(stats)
                    median_ms=float(np.median([x['ms'] for x in measurements]))
                    median_retries=float(np.median([x['retries'] for x in measurements]))
                    overflows=sum(x['retries']>0 for x in measurements)
                    cap_median=int(np.median([x['capacity'] for x in measurements]))
                    row=dict(distribution=dist_name,selectivity=p,method=method,samples=m,
                        actual_count=len(expected),median_ms=round(median_ms,4),
                        median_capacity=cap_median,median_retries=median_retries,
                        overflow_trials=overflows,trials=args.repeats,correctness='pass')
                    rows.append(row)
                    print(f'{dist_name:9} p={p:4.0%} {method:6} m={m:5} '
                          f'time={median_ms:8.4f}ms cap={cap_median:7} retries={median_retries:g} overflow-trials={overflows}')
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--n',type=int,default=600_000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--selectivities',type=float,nargs='+',default=[.02,.2,.8])
    parser.add_argument('--distributions',nargs='+',choices=('uniform','clustered'),default=['uniform','clustered'])
    parser.add_argument('--samples',type=int,nargs='+',default=[128,1024,8192])
    parser.add_argument('--z',type=float,default=2.576,help='Wilson z score (~99% one-sided conservatism)')
    parser.add_argument('--slack',type=float,default=.05,help='Additional capacity margin')
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--max-retries',type=int,default=8)
    parser.add_argument('--block',type=int,default=256)
    parser.add_argument('--output',type=Path,default=Path('sampling_adaptive_results.csv'))
    args=parser.parse_args()
    if args.n<2 or args.n >= 2**31: parser.error('--n must be in [2, 2^31)')
    if args.block<1 or args.block>1024: parser.error('--block must be 1..1024')
    if any(m<1 or m>args.n for m in args.samples): parser.error('--samples each 1..n')
    if any(not 0<=p<=1 for p in args.selectivities): parser.error('selectivities must be 0..1')
    if not 0<=args.slack<=1: parser.error('--slack must be 0..1')
    rows=run_cpu(args) if args.device=='cpu' else run_cuda(args)
    with args.output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print('Saved:',args.output)

if __name__=='__main__': main()
