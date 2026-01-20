// A = MxK, B = KxN, C = MxN
template <const unsigned int block_size>
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    const unsigned int x = blockIdx.x * block_size + (threadIdx.x / block_size);
    const unsigned int y = blockIdx.y * block_size + (threadIdx.x % block_size);

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; i++) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

// the blog more or less forces the mapping x = rows and y = cols, so changing the x/y math is the fix here.
// if we weren't bound to this, just doing it the regular 2d way would work just fine.
// instead, we should now launch this kernel using a 1d block shape.
// block_size is now a template parameter (instead of just blockDim.x) because we might want to tile the indices differently.
// (will come later, but imagine threads not being mapped 1 to 1 to output elements, but rather given responsibility to calculate a whole tile, say 16x16.
// then, block_size = 16 != blockDim.x.)

// 32x32 block, block_size 4 -> x=0,0,0,0,1,1,1,1 ... 255,255,255,255
//                              y=0,1,2,3,0,1,2,3 ... 0,1,2,3

// so actually, in this case the formula isn't for general tiling, but it's just to set up the concept for later