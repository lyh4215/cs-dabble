#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

using Clock = std::chrono::steady_clock;

constexpr int N = 512;

// 1. 가장 직관적인 i-j-k
void matmul_ijk(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C
) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;

            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}

// 2. 계산식은 똑같고 loop 순서만 i-k-j
void matmul_ikj(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C
) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float a = A[i * N + k];

            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

template <int BS>
void matmul_tiled(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C
) {
    for (int ii = 0; ii < N; ii += BS) {
        for (int kk = 0; kk < N; kk += BS) {
            for (int jj = 0; jj < N; jj += BS) {

                for (int i = ii; i < ii + BS; ++i) {
                    for (int k = kk; k < kk + BS; ++k) {
                        float a = A[i * N + k];

                        for (int j = jj; j < jj + BS; ++j) {
                            C[i * N + j] += a * B[k * N + j];
                        }
                    }
                }

            }
        }
    }
}

template <typename F>
void benchmark(
    const char* name,
    F func,
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C
) {
    double best_ms = 1e100;
    double checksum = 0.0;

    for (int repeat = 0; repeat < 3; ++repeat) {
        std::fill(C.begin(), C.end(), 0.0f);

        auto start = Clock::now();
        func(A, B, C);
        auto end = Clock::now();

        double ms =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        best_ms = std::min(best_ms, ms);

        // 결과가 실제로 사용되도록.
        checksum = 0.0;
        for (float x : C)
            checksum += x;
    }

    std::cout
        << std::left << std::setw(20) << name
        << ": " << std::setw(10) << best_ms
        << " ms checksum=" << checksum
        << '\n';
}

int main() {
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C(N * N);

    for (std::size_t i = 0; i < A.size(); ++i) {
        A[i] = static_cast<float>((i % 13) + 1) / 13.0f;
        B[i] = static_cast<float>((i % 17) + 1) / 17.0f;
    }

    std::cout << "One matrix: "
              << (A.size() * sizeof(float)) / (1024.0 * 1024.0)
              << " MiB\n\n";

    benchmark("ijk naive", matmul_ijk, A, B, C);
    benchmark("ikj reordered", matmul_ikj, A, B, C);
    benchmark("tile 8",   matmul_tiled<8>,   A, B, C);
    benchmark("tile 16",  matmul_tiled<16>,  A, B, C);
    benchmark("tile 32",  matmul_tiled<32>,  A, B, C);
    benchmark("tile 64",  matmul_tiled<64>,  A, B, C);
    benchmark("tile 128", matmul_tiled<128>, A, B, C);
}