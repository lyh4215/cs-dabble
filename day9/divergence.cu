#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>


#define CHECK_CUDA(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr,                                       \
                "CUDA error: %s\n",                           \
                cudaGetErrorString(err));                     \
        exit(1);                                              \
    }                                                         \
} while (0)


__device__ __forceinline__
float work_a(float x)
{
    #pragma unroll 1
    for (int i = 0; i < 64; i++) {
        x = x * 1.00001f + 0.00001f;
    }

    return x;
}


__device__ __forceinline__
float work_b(float x)
{
    #pragma unroll 1
    for (int i = 0; i < 64; i++) {
        x = x * 0.99999f - 0.00001f;
    }

    return x;
}


// ============================================================
// 1. Uniform
// 모든 thread가 같은 path
// ============================================================

__global__
void uniform_kernel(
    const float* input,
    float* output,
    int n
)
{
    int i =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (i >= n)
        return;

    float x = input[i];

    if (i >= 0) {
        x = work_a(x);
    }
    else {
        x = work_b(x);
    }

    output[i] = x;
}


// ============================================================
// 2. Warp-aligned
//
// warp 단위로 A/B가 갈림.
//
// warp 0 : 전부 A
// warp 1 : 전부 B
// warp 2 : 전부 A
// ...
//
// warp 내부 divergence는 없음.
// ============================================================

__global__
void warp_aligned_kernel(
    const float* input,
    float* output,
    int n
)
{
    int i =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (i >= n)
        return;

    float x = input[i];

    int warp_id =
        i / 32;

    if ((warp_id & 1) == 0) {
        x = work_a(x);
    }
    else {
        x = work_b(x);
    }

    output[i] = x;
}


// ============================================================
// 3. Divergent
//
// 같은 warp 안에서
//
// thread 0 → A
// thread 1 → B
// thread 2 → A
// thread 3 → B
// ...
//
// 모든 warp가 분기됨.
// ============================================================

__global__
void divergent_kernel(
    const float* input,
    float* output,
    int n
)
{
    int i =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (i >= n)
        return;

    float x = input[i];

    if ((i & 1) == 0) {
        x = work_a(x);
    }
    else {
        x = work_b(x);
    }

    output[i] = x;
}


template <typename Kernel>
float benchmark(
    Kernel kernel,
    const float* input,
    float* output,
    int n,
    int blocks,
    int threads
)
{
    // warmup
    for (int i = 0; i < 5; i++) {
        kernel<<<blocks, threads>>>(
            input,
            output,
            n
        );
    }

    CHECK_CUDA(
        cudaDeviceSynchronize()
    );


    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(
        cudaEventCreate(&start)
    );

    CHECK_CUDA(
        cudaEventCreate(&stop)
    );


    std::vector<float> times;


    for (int r = 0; r < 20; r++) {

        CHECK_CUDA(
            cudaEventRecord(start)
        );

        kernel<<<blocks, threads>>>(
            input,
            output,
            n
        );

        CHECK_CUDA(
            cudaEventRecord(stop)
        );

        CHECK_CUDA(
            cudaEventSynchronize(stop)
        );


        float ms;

        CHECK_CUDA(
            cudaEventElapsedTime(
                &ms,
                start,
                stop
            )
        );

        times.push_back(ms);
    }


    std::sort(
        times.begin(),
        times.end()
    );


    CHECK_CUDA(
        cudaEventDestroy(start)
    );

    CHECK_CUDA(
        cudaEventDestroy(stop)
    );


    return times[
        times.size() / 2
    ];
}


int main()
{
    cudaDeviceProp prop;

    CHECK_CUDA(
        cudaGetDeviceProperties(
            &prop,
            0
        )
    );


    printf(
        "GPU: %s\n\n",
        prop.name
    );


    const int N =
        16 * 1024 * 1024;

    const size_t bytes =
        (size_t)N
        * sizeof(float);


    float* input;
    float* output;

    CHECK_CUDA(
        cudaMalloc(
            &input,
            bytes
        )
    );

    CHECK_CUDA(
        cudaMalloc(
            &output,
            bytes
        )
    );


    CHECK_CUDA(
        cudaMemset(
            input,
            0,
            bytes
        )
    );


    const int THREADS = 256;

    const int BLOCKS =
        (N + THREADS - 1)
        / THREADS;


    float uniform_ms =
        benchmark(
            uniform_kernel,
            input,
            output,
            N,
            BLOCKS,
            THREADS
        );


    float aligned_ms =
        benchmark(
            warp_aligned_kernel,
            input,
            output,
            N,
            BLOCKS,
            THREADS
        );


    float divergent_ms =
        benchmark(
            divergent_kernel,
            input,
            output,
            N,
            BLOCKS,
            THREADS
        );


    printf(
        "%-20s %12s %12s\n",
        "kernel",
        "latency(ms)",
        "relative"
    );

    printf(
        "-----------------------------------------------\n"
    );


    printf(
        "%-20s %12.4f %11.2fx\n",
        "uniform",
        uniform_ms,
        1.0
    );


    printf(
        "%-20s %12.4f %11.2fx\n",
        "warp-aligned",
        aligned_ms,
        aligned_ms / uniform_ms
    );


    printf(
        "%-20s %12.4f %11.2fx\n",
        "divergent",
        divergent_ms,
        divergent_ms / uniform_ms
    );


    CHECK_CUDA(
        cudaFree(input)
    );

    CHECK_CUDA(
        cudaFree(output)
    );


    return 0;
}