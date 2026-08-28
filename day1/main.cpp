#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

using Clock = std::chrono::steady_clock;

constexpr int N = 4096;

double row_major_sum(const std::vector<float>& a) {
    double sum = 0.0;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            sum += a[i * N + j];
        }
    }

    return sum;
}

double column_major_sum(const std::vector<float>& a) {
    double sum = 0.0;

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            sum += a[i * N + j];
        }
    }

    return sum;
}

template <int B>
double blocked_sum(const std::vector<float>& a) {
    double sum = 0.0;

    for (int jj = 0; jj < N; jj += B) {
        for (int i = 0; i < N; ++i) {
            for (int j = jj; j < jj + B && j < N; ++j) {
                sum += a[i * N + j];
            }
        }
    }

    return sum;
}

double stride_sum(const std::vector<float>& a, std::size_t stride) {
    double sum = 0.0;

    for (std::size_t offset = 0; offset < stride; ++offset) {
        for (std::size_t i = offset; i < a.size(); i += stride) {
            sum += a[i];
        }
    }

    return sum;
}

template <typename F>
void benchmark(const char* name, F func, const std::vector<float>& a) {
    double best_ms = 1e100;
    double checksum = 0.0;

    for (int repeat = 0; repeat < 5; ++repeat) {
        auto start = Clock::now();

        checksum = func(a);

        auto end = Clock::now();

        double ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        if (ms < best_ms) {
            best_ms = ms;
        }
    }

    std::cout << std::left << std::setw(20) << name
              << ": " << std::setw(10) << best_ms
              << " ms"
              << "  checksum=" << checksum << '\n';
}

template <typename F>
void benchmark_stride(
    const char* name,
    F func,
    const std::vector<float>& a,
    std::size_t stride
) {
    double best_ms = 1e100;
    double checksum = 0.0;

    for (int repeat = 0; repeat < 5; ++repeat) {
        auto start = Clock::now();

        checksum = func(a, stride);

        auto end = Clock::now();

        double ms =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        if (ms < best_ms) {
            best_ms = ms;
        }
    }

    std::cout << std::left << std::setw(20)
              << name
              << ": "
              << std::setw(10)
              << best_ms
              << " ms"
              << " checksum="
              << checksum
              << '\n';
}

int main() {
    std::vector<float> a(N * N);

    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(i % 100);
    }

    std::cout << "Matrix size: "
              << (a.size() * sizeof(float)) / (1024.0 * 1024.0)
              << " MiB\n\n";

    benchmark("row-major", row_major_sum, a);
    benchmark("column-major", column_major_sum, a);
    benchmark("block 1",  blocked_sum<1>,  a);
    benchmark("block 2",  blocked_sum<2>,  a);
    benchmark("block 4",  blocked_sum<4>,  a);
    benchmark("block 8",  blocked_sum<8>,  a);
    benchmark("block 16", blocked_sum<16>, a);
    benchmark("block 32", blocked_sum<32>, a);
    benchmark("block 64", blocked_sum<64>, a);

    benchmark_stride("stride 1",    stride_sum, a, 1);
    benchmark_stride("stride 2",    stride_sum, a, 2);
    benchmark_stride("stride 4",    stride_sum, a, 4);
    benchmark_stride("stride 8",    stride_sum, a, 8);
    benchmark_stride("stride 16",   stride_sum, a, 16);
    benchmark_stride("stride 32",   stride_sum, a, 32);
    benchmark_stride("stride 64",   stride_sum, a, 64);
    benchmark_stride("stride 256",  stride_sum, a, 256);
    benchmark_stride("stride 1024", stride_sum, a, 1024);
    benchmark_stride("stride 4096", stride_sum, a, 4096);
}