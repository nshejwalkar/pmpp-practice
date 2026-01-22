// A = MxK, B = KxN, C = MxN
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_2d_blocktiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    const unsigned int block_row = blockIdx.x;
    const unsigned int block_col = blockIdx.y;

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

    // move the matrix pointers
    A += block_row * BM * K;
    B += block_col * BN;
    C += (block_row * BM) * N + (block_col * BN);

    // at this point, we can think of all threads as being indexed locally (besides when we update A/B)

    float thread_results[TM][TN] = {0.0f};
    float register_a[TM];  // these are for later during the main computation loop
    float register_b[TN];
    float tmp = 0.0;

    for (int phase = 0; phase < K; phase += BK) {
        // vectorized GMEM loads with float4 (LDG.E.128), we're going to store in SMEM as A transpose.
        float4 tmp_a = *reinterpret_cast<float4*>(&A[load_row_a * K + load_col_a]);
        tile_a[load_col_a + 0][load_row_a] = tmp_a.x;
        tile_a[load_col_a + 1][load_row_a] = tmp_a.y;
        tile_a[load_col_a + 2][load_row_a] = tmp_a.z;
        tile_a[load_col_a + 3][load_row_a] = tmp_a.w;

        // we're going to load B in the same way (without the transpose now)
        // float4 tmp_b = *reinterpret_cast<float4 *>(&B[load_row_b * N + load_col_b]);
        // tile_b[load_row_b + 0][load_col_b] = tmp_b.x;
        // tile_b[load_row_b + 1][load_col_b] = tmp_b.y;
        // tile_b[load_row_b + 2][load_col_b] = tmp_b.z;
        // tile_b[load_row_b + 3][load_col_b] = tmp_b.w;

        // this actually does the same thing and is faster, because only 1 STS.128 instruction and not 4 stores.
        *reinterpret_cast<float4*>(&tile_b[load_row_b][load_col_b]) = *reinterpret_cast<float4*>(&B[load_row_b * N + load_col_b]);

        // very strongly assuming that each thread loads 4 elements and not more, so blockDim.x * 4 = BM*BK, otherwise we'd need striding.
        // now we have a assumption here that BK and BN are multiples of 4 (and ofc N%BN=0, K%BK=0), or we'll need tail handling.

        __syncthreads();

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
the big idea here and with matmul kernels in general is that loading is decoupled from computation. TM/TN don't show up in loading ever.

its good to list out all the assumptions we've made by not including error handling.
vectorized float4 loads/stores in GMEM: every phase, one thread loads one float4. if not, need striding
- blockDim.x * 4 = BM * BK = BK * BN
vectorized float4 loads/stores in GMEM: floats are actually aligned in memory in groups of 4. if not, need tail handling
- BK % 4 = 0, BN % 4 = 0
general oob handling
- N % BN = 0, K % BK = 0, M % BM = 0
vectorized SMEM loads
- TM = 8, TN = 8
- BM % TM = 0, BN % TN = 0
*/