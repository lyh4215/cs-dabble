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
                "CUDA error: %s (%s:%d)\n",                   \
                cudaGetErrorString(err),                      \
                __FILE__,                                     \
                __LINE__);                                    \
        exit(1);                                              \
    }                                                         \
} while (0)


__global__
void memory_kernel(
    const float* x,
    float* y,
    int n
) {
    // 이 launch에서 지정한 dynamic shared memory를
    // block이 예약하게 된다.
    extern __shared__ unsigned char scratch[];

    // 실제 계산은 매우 단순한 memory-bound operation.
    //
    //  x read
    //  y read
    //  y write
    //
    // arithmetic intensity가 매우 낮음.

    int idx =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    int stride =
        blockDim.x * gridDim.x;


    for (
        int i = idx;
        i < n;
        i += stride
    ) {
        y[i] =
            1.001f * x[i]
            + y[i];
    }
}


float benchmark(
    const float* x,
    float* y,
    int n,
    int blocks,
    int threads,
    size_t shared_bytes,
    int repeats
) {
    // warmup
    for (int i = 0; i < 5; i++) {

        memory_kernel
            <<<blocks,
               threads,
               shared_bytes>>>(
                    x,
                    y,
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

    for (
        int r = 0;
        r < repeats;
        r++
    ) {

        CHECK_CUDA(
            cudaEventRecord(start)
        );


        memory_kernel
            <<<blocks,
               threads,
               shared_bytes>>>(
                    x,
                    y,
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


int main() {

    int device = 0;

    cudaDeviceProp prop;

    CHECK_CUDA(
        cudaGetDeviceProperties(
            &prop,
            device
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
        "warp size: %d\n",
        prop.warpSize
    );

    printf(
        "max threads / SM: %d\n",
        prop.maxThreadsPerMultiProcessor
    );

    printf(
        "shared memory / SM: %.1f KiB\n",
        prop.sharedMemPerMultiprocessor
            / 1024.0
    );

    printf("\n");


    // 약 256 MiB / array
    const int N =
        64 * 1024 * 1024;

    const size_t bytes =
        (size_t) N
        * sizeof(float);


    float* x;
    float* y;

    CHECK_CUDA(
        cudaMalloc(
            &x,
            bytes
        )
    );

    CHECK_CUDA(
        cudaMalloc(
            &y,
            bytes
        )
    );


    CHECK_CUDA(
        cudaMemset(
            x,
            0,
            bytes
        )
    );

    CHECK_CUDA(
        cudaMemset(
            y,
            0,
            bytes
        )
    );


    const int THREADS = 256;

    // grid는 충분히 크게 만들어서
    // 모든 SM에 계속 작업이 존재하게 함.
    const int BLOCKS =
        prop.multiProcessorCount
        * 32;


    // T4에서는 이 정도 범위로
    // resident block 수가 달라지는 모습을
    // 볼 가능성이 높다.
    size_t shared_sizes[] = {
        0,
        8 * 1024,
        16 * 1024,
        24 * 1024,
        32 * 1024,
        40 * 1024,
        48 * 1024
    };


    printf(
        "%10s "
        "%14s "
        "%14s "
        "%12s "
        "%12s\n",
        "shared KB",
        "blocks/SM",
        "warps/SM",
        "occupancy",
        "GB/s"
    );

    printf(
        "-----------------------------------------------------------------\n"
    );


    const int max_warps_per_sm =
        prop.maxThreadsPerMultiProcessor
        / prop.warpSize;


    for (
        size_t shared_bytes :
        shared_sizes
    ) {

        // 해당 block size + shared memory 사용량에서
        // SM 하나에 최대 몇 block이 동시에 resident 가능한지 계산.
        int active_blocks;

        cudaError_t occ_err =
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks,
                memory_kernel,
                THREADS,
                shared_bytes
            );


        if (
            occ_err != cudaSuccess
        ) {
            printf(
                "%10.1f  unsupported\n",
                shared_bytes / 1024.0
            );

            cudaGetLastError();

            continue;
        }


        int warps_per_block =
            THREADS
            / prop.warpSize;


        int active_warps =
            active_blocks
            * warps_per_block;


        double occupancy =
            (double) active_warps
            / max_warps_per_sm;


        float ms =
            benchmark(
                x,
                y,
                N,
                BLOCKS,
                THREADS,
                shared_bytes,
                15
            );


        // x read  = 4N
        // y read  = 4N
        // y write = 4N
        //
        // ≈ 12N bytes

        double bytes_moved =
            12.0
            * N;


        double bandwidth =
            bytes_moved
            / (ms / 1000.0)
            / 1e9;


        printf(
            "%10.1f "
            "%14d "
            "%14d "
            "%11.1f%% "
            "%12.2f\n",
            shared_bytes / 1024.0,
            active_blocks,
            active_warps,
            occupancy * 100.0,
            bandwidth
        );
    }


    CHECK_CUDA(
        cudaFree(x)
    );

    CHECK_CUDA(
        cudaFree(y)
    );


    return 0;
}