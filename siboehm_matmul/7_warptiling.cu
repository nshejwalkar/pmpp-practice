// A = MxK, B = KxN, C = MxN

// this is called every phase to load in one block.
template <const int BM, const int BN, const int BK,
          const int row_stride_a, const int row_stride_b>
__device__ void load_from_gmem(int M, int N, int K, const float *A, const float *B,
                                int load_row_a, int load_col_a, int load_row_b, int load_col_b,
                                float** tile_a, float** tile_b) {

    // vectorized GMEM loads with float4 (LDG.E.128), we're going to store in SMEM as A transpose.
    // notice we're now striding explicitly across rows, 
    for (int offset = 0; offset < BM; offset += row_stride_a) {
        float4 tmp_a = *reinterpret_cast<float4 *>(&A[(load_row_a + offset) * K + load_col_a]);
        tile_a[load_col_a + 0][load_row_a + offset] = tmp_a.x;
        tile_a[load_col_a + 1][load_row_a + offset] = tmp_a.y;
        tile_a[load_col_a + 2][load_row_a + offset] = tmp_a.z;
        tile_a[load_col_a + 3][load_row_a + offset] = tmp_a.w;
    }

    // this actually does the same thing and is faster, because only 1 STS.128 instruction and not 4 stores.
    for (int offset = 0; offset < BK; offset += row_stride_b) {
        *reinterpret_cast<float4 *>(&tile_b[load_row_b + offset][load_col_b]) = *reinterpret_cast<float4 *>(&B[load_row_b + offset * N + load_col_b]);
    }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void process_warp_tile(float *register_a, float *register_b, float *thread_results,
                                  float **tile_a, float **tile_b, const unsigned int warp_row, const unsigned int warp_col,
                                  const unsigned int thread_row_in_warp, const unsigned int thread_col_in_warp) {
    for (int k = 0; k < BK; k++) {
        for (int w_sub_row = 0; w_sub_row < WMITER; w_sub_row++) {
            for (int i = 0; i < TM; i++) {
                // register_a[w_sub_row * TM + i] = tile_a[]
                // finish later
            }
        }
    }
}

template <const int BM, const int BN, const int BK, 
        const int WM, const int WN,
        const int TM, const int TN>
__global__ void sgemm_warptiling_kernel(int M, int N, int K, 
                                        float alpha, const float *A, const float *B, float beta, float *C) {
    const unsigned int block_row = blockIdx.x;
    const unsigned int block_col = blockIdx.y;

    // warp level indexing for computation later
    const uint warp_idx = threadIdx.x / 32;
    const uint warp_col = warp_idx % (BN / WN);
    const uint warp_row = warp_idx / (BN / WN);

    constexpr uint WMITER = (WM * WN) / (32 * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;

    const uint thread_idx_in_warp = threadIdx.x % 32;
    const uint thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN);
    const uint thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN);

    __shared__ float tile_a[BM][BK];
    __shared__ float tile_b[BK][BN];

    // indices simplified a bit. think of it as threadIdx.x reshaped in the smaller grid, then expanded by TM/TN.
    const unsigned int output_row = (threadIdx.x / (BN / TN)) * TM;
    const unsigned int output_col = (threadIdx.x % (BN / TN)) * TN;

    // indices for loading a and b - remember, the pattern is to reshape to smaller grid, then expand (if needed).
    const unsigned int load_row_a = threadIdx.x / (BK / 4);
    const unsigned int load_col_a = threadIdx.x % (BK / 4) * 4;
    const unsigned int load_row_b = threadIdx.x / (BN / 4);
    const unsigned int load_col_b = threadIdx.x % (BN / 4) * 4;

    // will be used for strided gmem->smem loading.
    const unsigned int row_stride_a = blockDim.x / (BK / 4);
    const unsigned int row_stride_b = blockDim.x / (BN / 4);

    // move the matrix pointers
    A += block_row * BM * K;
    B += block_col * BN;
    C += (block_row * BM) * N + (block_col * BN);

    // at this point, we can think of all threads as being indexed locally (besides when we update A/B)
                  
    // expanded register usage
    float thread_results[WMITER * TM * WNITER * TN] = {0.0f};
    float register_m[WMITER * TM] = {0.0f};
    float register_n[WNITER * TN] = {0.0f};
    float tmp = 0.0;

    for (int phase = 0; phase < K; phase += BK) {
        load_from_gmem<BM,BN,BK,row_stride_a,row_stride_b>(M, N, K, A, B, load_row_a, load_col_a, load_row_b, load_col_b, tile_a, tile_b)

        __syncthreads();

        process_warp_tile<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(register_a, register_b, thread_results, tile_a, tile_b, warp_row, warp_col, thread_row_in_warp, thread_col_in_warp);

        A += BK;
        B += BK * N;

        // ignoring all bounds checks
        // now we're vectorizing the SMEM loads - this was the point of transposing A before
        for (int k = 0; k < BK; k++) {
            // TM = 8
            *reinterpret_cast<float4*>(&register_a[0]) = *reinterpret_cast<const float4*>(tile_a[output_row + 0][k]);
            *reinterpret_cast<float4*>(&register_a[4]) = *reinterpret_cast<const float4*>(tile_a[output_row + 4][k]);
            
            // TN = 8
            *reinterpret_cast<float4*>(&register_b[0]) = *reinterpret_cast<const float4*>(tile_b[k][output_col + 0]);
            *reinterpret_cast<float4*>(&register_b[4]) = *reinterpret_cast<const float4*>(tile_b[k][output_col + 4]);
#pragma unroll
            for (int ai = 0; ai < TM; ai++) {
#pragma unroll
                for (int bi = 0; bi < TN; bi++) {
                    thread_results[ai][bi] += register_a[ai] * register_b[bi];
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

/*

*/