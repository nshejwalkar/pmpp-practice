// A = MxK, B = KxN, C = MxN
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_2d_blocktiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    const unsigned int block_row = blockIdx.x;
    const unsigned int block_col = blockIdx.y;

    __shared__ float tile_a[BM][BK];
    __shared__ float tile_b[BK][BN];

    // linearized index -> local row/col in output block. set up the pointers right now to use later
    const unsigned int micro_N = BN / TN;
    const unsigned int micro_row = threadIdx.x / micro_N;
    const unsigned int micro_col = threadIdx.x % micro_N;
    const unsigned int output_row = micro_row * TM;
    const unsigned int output_col = micro_col * TN;

    // move the matrix pointers
    A += block_row * BM * K;
    B += block_col * BN;
    C += (block_row * BM) * N + (block_col * BN);

    // at this point, we can think of all threads as being indexed locally (besides when we update A/B)

    float thread_results[TM][TN] = {0.0f};
    float a_tmp[TM];  // these are for later during the main computation loop
    float b_tmp[TN];
    float tmp = 0.0;

    for (int phase = 0; phase < K; phase += BK) {
        // strided loads
        // the pragmas are to force the compiler to unroll the loop (it might do it anyway, but this makes sure it does)
#pragma unroll
        for (int load_offset = 0; load_offset < BM*BK; load_offset += blockDim.x) {
            const unsigned int strided_index = threadIdx.x + load_offset;
            const unsigned int a_row = strided_index / BK;
            const unsigned int a_col = strided_index % BK;
            tile_a[a_row][a_col] = A[a_row * K + a_col];
        }

#pragma unroll
        for (int load_offset = 0; load_offset < BK*BN; load_offset += blockDim.x) {
            const unsigned int strided_index = threadIdx.x + load_offset;
            const unsigned int b_row = strided_index / BN;
            const unsigned int b_col = strided_index % BN;
            tile_b[b_row][b_col] = B[b_row * N + b_col];
        }

        // in the 1d case, we assumed that blockDim.x = BN*BK = BK*BN, which was restrictive but okay.
        // now, enforcing that is somewhat unusable, as the number of threads/block is << tile sizes, so we need strided loads.
        // we still have an assumption here that blockDim.x is a divisor of BM*BK and BK*BN (otherwise we'd need guards).

        __syncthreads();

        A += BK;
        B += BK * N;

        // ignoring all bounds checks (basically assuming BM%TM and BN%TN == 0)
        // same-ish as before, cache TM elems in a mini-col of A and TN elems in a mini-row of B, so the TM*TN accumulations happen very fast.
        // now each thread does TN*TM*BK work, but only 1/(TM*TN) threads are doing that work.
        for (int k = 0; k < BK; k++) {
#pragma unroll
            for (int ai = 0; ai < TM; ai++) {
                a_tmp[ai] = tile_a[output_row + ai][k];
            }
#pragma unroll
            for (int bi = 0; bi < TN; bi++) {    
                b_tmp[bi] = tile_b[k][output_col + bi];
            }
#pragma unroll
            for (int ai = 0; ai < TM; ai++) {
#pragma unroll
                for (int bi = 0; bi < TN; bi++) {
                    thread_results[ai][bi] += a_tmp[ai] * b_tmp[bi];
                }
            }
        }
        __syncthreads();
    }

    // now all the phases are done, and all thread_results are ready. just write to C

    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            C[(output_row + i) * N + (output_col + j)] = alpha * thread_results[i][j] + beta * C[(output_row + i) * N + (output_col + j)];
        }
    }
}

// theres more register pressure now, TM*TN for thread_results, and TM+TN for a_tmp and b_tmp.