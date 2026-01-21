// A = MxK, B = KxN, C = MxN
template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_1d_blocktiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    static_assert(BM == BN, "This kernel assumes BM == BN for 1-element-per-thread loads.");
    static_assert(BK * TM == BM, "This kernel assumes BK * TM == BM so blockDim matches tile sizes.");

    const unsigned int block_row = blockIdx.x;
    const unsigned int block_col = blockIdx.y;

    __shared__ float tile_a[BM][BK];
    __shared__ float tile_b[BK][BN];

    // linearized index -> local row/col in output block. set up the pointers right now to use later
    const unsigned int output_row = (threadIdx.x / BN) * TM;
    const unsigned int output_col = threadIdx.x % BN;

    // this is the mapping for the computation (later). Notice we're now multiplying rows by TM, because every thread calculates a column of TM elements.
    // const unsigned int global_row = block_row * BM + thread_row * TM;
    // const unsigned int global_col = block_col * BN + thread_col;

    // move the matrix pointers
    A += block_row * BM * K;
    B += block_col * BN;
    C += (block_row * BM) * N + (block_col * BN);

    // at this point, we can think of all threads as being indexed locally (besides when we update A/B)

    // if TM is small enough, the compiler will scalarize the array and put it in the register file.
    // if TM gets big enough, the compiler will force it to spill into gmem (very bad)
    float thread_results[TM] = {0.0f};  
    float tmp = 0.0;

    for (int phase = 0; phase < K; phase += BK) {
        // now we're indexing into different spots in A. we can use the linearized threadIdx.x to reindex easily
        const unsigned int a_row = threadIdx.x / BK;
        const unsigned int a_col = threadIdx.x % BK;
        tile_a[a_row][a_col] = A[a_row * K + a_col];

        // and B
        const unsigned int b_row = threadIdx.x / BN;
        const unsigned int b_col = threadIdx.x % BN;
        tile_b[b_row][b_col] = B[b_row * N + b_col];

        // notice the above assumed that blockDim.x (which is BM*BN/TM) = BN*BK = BK*BN. 
        // if we had blockDim.x > the other two, we'd need boundary checks. if we had <, we'd need strided loads.
        // but in general, reindexing is why a 1d block is just easier to work with.

        __syncthreads();

        A += BK;
        B += BK * N;
        
        // ignoring all bounds checks (basically assuming BM%TM == 0)
        // We're caching a tile_b element from SMEM->register so we can reuse it while walking down tile_a (and reduce MIO pressure)
        // now each thread does TM*BK work, but only 1/TM threads are doing that work.
        for (int k = 0; k < BK; k++) {
            float b_tmp = tile_b[k][output_col];
            for (int i = 0; i < TM; i++) {
                thread_results[i] += tile_a[output_row + i][k] * b_tmp;
            }
        }

        __syncthreads();
    }

    // now all the phases are done, and all thread_results are ready. just write to C

    for (int i = 0; i < TM; i++) {
        C[(output_row + i) * N + output_col] = alpha * thread_results[i] + beta * C[(output_row + i) * N + output_col];
    }
}

/*
void sgemm_blocktiling_1d(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                          torch::Tensor &output_matrix, float alpha, float beta)
{
    // ... rest of the code is similar
    // ...
    // Template parameters for kernel
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;

    // Configure kernel launch
    // Tiling strategy :
    // - BM x BK from A, BK x BN from B
    // - TM values per thread
    // Number of threads = (BM / TM) * BN = (64 / 8) * 64 = 512 threads per block. 1/TM fewer threads.
    dim3 block_dim((BM / TM) * BN);
    dim3 grid_dim(CEIL_DIV(num_rows_a, BM),
                  CEIL_DIV(num_cols_b, BN));

    sgemm_blocktiling_1d_kernel<BM, BN, BK, TM><<<grid_dim, block_dim>>>(
        num_rows_a, num_cols_b, num_cols_a,
        alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}
*/

// the solution/improvement from the last one is to give more work to do to each thread.
// so, instead of launching BM*BN threads, we launch BM*BN/TM threads and let each thread handle TM global loads and outputs.
// this increases the arithmetic intensity by TM times, so we should see much less MIO throttling.
// we can take this even further by letting each thread handle a whole mini-tile (next)