#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess)                                            \
        {                                                                    \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)         \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

__global__ void add(const float *A, const float *B, float *C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

__global__ void strided_add(const float *A, const float *B, float *C, int N)
{
    int elem = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int idx = elem; idx < N; idx += stride)
    {
        C[idx] = A[idx] + B[idx];
    }
}

static float time_kernel_add(
    bool use_strided,
    const float *A_d,
    const float *B_d,
    float *C_d,
    int N,
    int block_size,
    int grid_size,
    int warmup_iters,
    int timed_iters)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup (also helps JIT/clock ramp effects).
    for (int i = 0; i < warmup_iters; i++)
    {
        if (use_strided)
        {
            strided_add<<<grid_size, block_size>>>(A_d, B_d, C_d, N);
        }
        else
        {
            add<<<grid_size, block_size>>>(A_d, B_d, C_d, N);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < timed_iters; i++)
    {
        if (use_strided)
        {
            strided_add<<<grid_size, block_size>>>(A_d, B_d, C_d, N);
        }
        else
        {
            add<<<grid_size, block_size>>>(A_d, B_d, C_d, N);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / timed_iters; // average ms per kernel
}

static double effective_gbps(int N, float ms_per_iter)
{
    // Vector add does: read A + read B + write C = 3 * 4 bytes per element
    const double bytes = static_cast<double>(N) * 3.0 * sizeof(float);
    const double seconds = static_cast<double>(ms_per_iter) / 1e3;
    return (bytes / seconds) / 1e9;
}

int main()
{
    int devcount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devcount));
    if (devcount == 0)
    {
        std::cerr << "No CUDA devices found.\n";
        return 1;
    }

    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    std::cout << "Device " << dev << ": " << prop.name << "\n";
    std::cout << "  SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "  maxThreadsPerBlock: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "  warpSize: " << prop.warpSize << "\n";
    std::cout << "  totalGlobalMem: " << static_cast<unsigned long long>(prop.totalGlobalMem) << " bytes\n";
    std::cout << "  maxGridSize.x: " << prop.maxGridSize[0] << "\n\n";

    // ----------------------------
    // Decide maximum N that fits in VRAM for 3 float arrays.
    // Use a safety factor so other allocations don't OOM.
    // ----------------------------
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    const double safety = 0.85; // keep headroom
    size_t usable = static_cast<size_t>(static_cast<double>(free_bytes) * safety);

    // Need 3 arrays: A,B,C each N*sizeof(float).
    size_t maxN_by_mem = usable / (3 * sizeof(float));
    if (maxN_by_mem > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        maxN_by_mem = static_cast<size_t>(std::numeric_limits<int>::max());
    }

    std::cout << "CUDA mem free: " << free_bytes << " bytes\n";
    std::cout << "Using up to ~" << static_cast<unsigned long long>(usable) << " bytes (safety=" << safety << ")\n";
    std::cout << "Max N by memory (and int limit): " << static_cast<unsigned long long>(maxN_by_mem) << "\n\n";

    if (maxN_by_mem < 1'000'000)
    {
        std::cerr << "Not enough free VRAM for a meaningful test.\n";
        return 1;
    }

    // Choose a "large N" for block-size sweep.
    int N_large = static_cast<int>(std::min<size_t>(maxN_by_mem, 50'000'000)); // cap to keep runtimes sane
    std::cout << "Large-N used for block sweep: N = " << N_large << " (" << (N_large * sizeof(float) / (1024.0 * 1024.0)) << " MiB per array)\n\n";

    // Allocate device buffers.
    float *A_d = nullptr;
    float *B_d = nullptr;
    float *C_d = nullptr;
    CUDA_CHECK(cudaMalloc(&A_d, static_cast<size_t>(N_large) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B_d, static_cast<size_t>(N_large) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C_d, static_cast<size_t>(N_large) * sizeof(float)));

    // Initialize device buffers quickly (no need to copy host for benchmarking bandwidth kernel).
    CUDA_CHECK(cudaMemset(A_d, 0, static_cast<size_t>(N_large) * sizeof(float)));
    CUDA_CHECK(cudaMemset(B_d, 0, static_cast<size_t>(N_large) * sizeof(float)));
    CUDA_CHECK(cudaMemset(C_d, 0, static_cast<size_t>(N_large) * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());

    const int warmup_iters = 10;
    const int timed_iters = 100;

    // ============================================================
    // 1) Regular add, varying block sizes
    // ============================================================
    std::cout << "=== Benchmark 1: regular add, varying blockDim.x ===\n";
    std::cout << "N = " << N_large << ", timed_iters = " << timed_iters << "\n\n";

    std::cout << std::left
              << std::setw(10) << "block"
              << std::setw(12) << "grid"
              << std::setw(14) << "ms/iter"
              << std::setw(14) << "GB/s"
              << "\n";

    int max_block = prop.maxThreadsPerBlock; // 1024 on your device
    for (int block = 64; block <= max_block; block *= 2)
    {
        int grid = (N_large + block - 1) / block;
        // Grid can be huge; for this benchmark that's fine.
        float ms = time_kernel_add(false, A_d, B_d, C_d, N_large, block, grid, warmup_iters, timed_iters);
        double gbps = effective_gbps(N_large, ms);

        std::cout << std::left
                  << std::setw(10) << block
                  << std::setw(12) << grid
                  << std::setw(14) << std::fixed << std::setprecision(4) << ms
                  << std::setw(14) << std::fixed << std::setprecision(2) << gbps
                  << "\n";
    }
    std::cout << "\n";

    // ============================================================
    // 2) Strided add vs regular add across varying N
    // ============================================================
    std::cout << "=== Benchmark 2: regular vs strided across N ===\n";

    // Pick a reasonable block size.
    int block = 256;

    // Pick a reasonable grid size for strided kernel:
    // start with ~8 blocks/SM, and cap by maxGridSize.x.
    int grid_strided = prop.multiProcessorCount * 8;
    grid_strided = std::max(grid_strided, 1);
    grid_strided = std::min(grid_strided, prop.maxGridSize[0]);

    std::cout << "Using block = " << block << "\n";
    std::cout << "Strided grid = " << grid_strided << " (~8 blocks/SM)\n";
    std::cout << "Regular grid = ceil(N/block)\n\n";

    std::cout << std::left
              << std::setw(14) << "N"
              << std::setw(14) << "regular_ms"
              << std::setw(14) << "regular_GB/s"
              << std::setw(14) << "strided_ms"
              << std::setw(14) << "strided_GB/s"
              << "\n";

    // We'll sweep N geometrically up to maxN_by_mem (but also cap runtime).
    // Reallocate buffers when N grows beyond current allocation.
    auto ensure_capacity = [&](int N)
    {
        static int cap = N_large;
        if (N <= cap)
        {
            return;
        }
        CUDA_CHECK(cudaFree(A_d));
        CUDA_CHECK(cudaFree(B_d));
        CUDA_CHECK(cudaFree(C_d));
        cap = N;
        CUDA_CHECK(cudaMalloc(&A_d, static_cast<size_t>(cap) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&B_d, static_cast<size_t>(cap) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&C_d, static_cast<size_t>(cap) * sizeof(float)));
        CUDA_CHECK(cudaMemset(A_d, 0, static_cast<size_t>(cap) * sizeof(float)));
        CUDA_CHECK(cudaMemset(B_d, 0, static_cast<size_t>(cap) * sizeof(float)));
        CUDA_CHECK(cudaMemset(C_d, 0, static_cast<size_t>(cap) * sizeof(float)));
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    // N values: from 1K up to max (with a few hand-picked points near the top).
    std::vector<int> Ns;
    for (int N = 1'024; N <= 16'777'216; N *= 2)
    { // up to ~16M
        Ns.push_back(N);
    }
    // Add a few large points if memory allows.
    if (maxN_by_mem >= 50'000'000)
        Ns.push_back(50'000'000);
    if (maxN_by_mem >= 100'000'000)
        Ns.push_back(100'000'000);
    Ns.push_back(static_cast<int>(std::min<size_t>(maxN_by_mem, 200'000'000))); // cap
    // Dedup and sort
    std::sort(Ns.begin(), Ns.end());
    Ns.erase(std::unique(Ns.begin(), Ns.end()), Ns.end());

    for (int N : Ns)
    {
        if (N <= 0)
            continue;
        if (static_cast<size_t>(N) > maxN_by_mem)
            continue;

        ensure_capacity(N);

        int grid_regular = (N + block - 1) / block;
        // For fairness, cap regular grid to maxGridSize.x (rarely needed here but correct).
        grid_regular = std::min(grid_regular, prop.maxGridSize[0]);

        float ms_regular = time_kernel_add(false, A_d, B_d, C_d, N, block, grid_regular, warmup_iters, timed_iters);
        float ms_strided = time_kernel_add(true, A_d, B_d, C_d, N, block, grid_strided, warmup_iters, timed_iters);

        double gbps_regular = effective_gbps(N, ms_regular);
        double gbps_strided = effective_gbps(N, ms_strided);

        std::cout << std::left
                  << std::setw(14) << N
                  << std::setw(14) << std::fixed << std::setprecision(4) << ms_regular
                  << std::setw(14) << std::fixed << std::setprecision(2) << gbps_regular
                  << std::setw(14) << std::fixed << std::setprecision(4) << ms_strided
                  << std::setw(14) << std::fixed << std::setprecision(2) << gbps_strided
                  << "\n";
    }

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));

    return 0;
}
