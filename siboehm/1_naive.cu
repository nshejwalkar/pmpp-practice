// A = MxK, B = KxN, C = MxN
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; i++) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}
// this is even worse than the "regular naive gemm" because array accesses vary along rows, not columns, so no gmem coalescing

// linearization indexing:
// 2d:
// threadIdx.y * blockDim.x + threadIdx.x
// 3d:
// threadId = threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z)
// threadId = threadIdx.x
//          + blockDim.x * threadIdx.y
//          + blockDim.x * blockDim.y * threadIdx.z