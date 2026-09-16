#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>


#define CHECK_CUDA(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr,                                       \
                "CUDA error: %s (%s:%d)\n",                   \
                cudaGetErrorString(err),                      \
                __FILE__,                                     \
                __LINE__);                                    \
        exit(1);                                              \
    }                                                         \
} while (0)


// ============================================================
// ILP kernel
//
// ILP=1
//   acc[0] 하나에 dependency chain
//
// ILP=8
//   acc[0] ... acc[7]이 서로 독립
//
// compiler는 이 값들을 register에 유지하려고 한다.
// ============================================================

template <int ILP>
__global__
void ilp_kernel(
    const float* input,
    float* output,
    int iterations
)
{
    int idx =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    int total_threads =
        blockDim.x * gridDim.x;


    // compile-time 크기이므로
    // compiler가 scalar register들로 풀 가능성이 높다.
    float acc[ILP];


    #pragma unroll
    for (int j = 0; j < ILP; j++) {

        int pos =
            idx
            + j * total_threads;

        acc[j] =
            input[pos];
    }


    // 각 accumulator 안에서는 dependency chain:
    //
    // acc[j]_(t+1)
    //     depends on
    // acc[j]_t
    //
    // 하지만 서로 다른 j끼리는 독립.
    //
    // → ILP 증가
    for (
        int iter = 0;
        iter < iterations;
        iter++
    ) {

        #pragma unroll
        for (int j = 0; j < ILP; j++) {

            acc[j] = fmaf(
                acc[j],
                1.000001f,
                0.000001f
            );
        }
    }


    float sum = 0.0f;

    #pragma unroll
    for (int j = 0; j < ILP; j++) {
        sum += acc[j];
    }


    output[idx] = sum;
}


// ============================================================
// Benchmark one kernel
// ============================================================

template <int ILP>
float benchmark(
    const float* input,
    float* output,
    int blocks,
    int threads,
    int iterations,
    int repeats
)
{
    // warmup
    for (int i = 0; i < 3; i++) {

        ilp_kernel<ILP>
            <<<blocks, threads>>>(
                input,
                output,
                iterations
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


    for (int r = 0; r < repeats; r++) {

        CHECK_CUDA(
            cudaEventRecord(start)
        );


        ilp_kernel<ILP>
            <<<blocks, threads>>>(
                input,
                output,
                iterations
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


// ============================================================
// Run one ILP configuration
// ============================================================

template <int ILP>
void run_case(
    const float* input,
    float* output,
    const cudaDeviceProp& prop,
    int blocks,
    int threads,
    int iterations
)
{
    // --------------------------------------------------------
    // 실제 compiler register 사용량
    // --------------------------------------------------------

    cudaFuncAttributes attr;

    CHECK_CUDA(
        cudaFuncGetAttributes(
            &attr,
            ilp_kernel<ILP>
        )
    );


    // --------------------------------------------------------
    // register 사용량까지 고려해서
    // 동시에 resident 가능한 block 수 계산
    // --------------------------------------------------------

    int active_blocks = 0;

    CHECK_CUDA(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            ilp_kernel<ILP>,
            threads,
            0
        )
    );


    int warps_per_block =
        threads / prop.warpSize;

    int active_warps =
        active_blocks
        * warps_per_block;

    int max_warps =
        prop.maxThreadsPerMultiProcessor
        / prop.warpSize;

    double occupancy =
        (double) active_warps
        / max_warps;


    // --------------------------------------------------------
    // Benchmark
    // --------------------------------------------------------

    float ms =
        benchmark<ILP>(
            input,
            output,
            blocks,
            threads,
            iterations,
            15
        );


    // --------------------------------------------------------
    // FLOPs
    //
    // fmaf:
    // multiply + add
    // ≈ 2 FLOPs
    //
    // thread 수 × ILP × iterations × 2
    // --------------------------------------------------------

    long long total_threads =
        (long long)blocks
        * threads;

    double flops =
        (double)total_threads
        * ILP
        * iterations
        * 2.0;


    double gflops =
        flops
        / (ms / 1000.0)
        / 1e9;


    printf(
        "%5d "
        "%10d "
        "%10d "
        "%10d "
        "%10.1f%% "
        "%12.4f "
        "%12.1f\n",
        ILP,
        attr.numRegs,
        active_blocks,
        active_warps,
        occupancy * 100.0,
        ms,
        gflops
    );
}


// ============================================================
// Main
// ============================================================

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
        "GPU: %s\n",
        prop.name
    );

    printf(
        "SM count: %d\n",
        prop.multiProcessorCount
    );

    printf(
        "max threads / SM: %d\n",
        prop.maxThreadsPerMultiProcessor
    );

    printf(
        "registers / SM: %d\n",
        prop.regsPerMultiprocessor
    );

    printf("\n");


    const int THREADS = 256;

    // 충분한 block을 만들어
    // 모든 SM에 작업이 계속 존재하도록 한다.
    const int BLOCKS =
        prop.multiProcessorCount
        * 32;

    const int ITERATIONS =
        1024;

    const int MAX_ILP =
        64;


    long long total_threads =
        (long long)BLOCKS
        * THREADS;

    long long input_elements =
        total_threads
        * MAX_ILP;


    size_t input_bytes =
        input_elements
        * sizeof(float);

    size_t output_bytes =
        total_threads
        * sizeof(float);


    float* input;
    float* output;


    CHECK_CUDA(
        cudaMalloc(
            &input,
            input_bytes
        )
    );

    CHECK_CUDA(
        cudaMalloc(
            &output,
            output_bytes
        )
    );


    // 0보다는 non-zero 값을 넣는 게 낫다.
    std::vector<float> host(
        input_elements,
        0.5f
    );


    CHECK_CUDA(
        cudaMemcpy(
            input,
            host.data(),
            input_bytes,
            cudaMemcpyHostToDevice
        )
    );


    printf(
        "%5s "
        "%10s "
        "%10s "
        "%10s "
        "%11s "
        "%12s "
        "%12s\n",
        "ILP",
        "regs/thd",
        "blocks/SM",
        "warps/SM",
        "occupancy",
        "latency(ms)",
        "GFLOP/s"
    );


    printf(
        "-------------------------------------------------------------------------------\n"
    );


    run_case<1>(
        input, output,
        prop,
        BLOCKS, THREADS,
        ITERATIONS
    );

    run_case<2>(
        input, output,
        prop,
        BLOCKS, THREADS,
        ITERATIONS
    );

    run_case<4>(
        input, output,
        prop,
        BLOCKS, THREADS,
        ITERATIONS
    );

    run_case<8>(
        input, output,
        prop,
        BLOCKS, THREADS,
        ITERATIONS
    );

    run_case<16>(
        input, output,
        prop,
        BLOCKS, THREADS,
        ITERATIONS
    );

    run_case<32>(
        input, output,
        prop,
        BLOCKS, THREADS,
        ITERATIONS
    );

    run_case<64>(
        input, output,
        prop,
        BLOCKS, THREADS,
        ITERATIONS
    );


    CHECK_CUDA(
        cudaFree(input)
    );

    CHECK_CUDA(
        cudaFree(output)
    );


    return 0;
}