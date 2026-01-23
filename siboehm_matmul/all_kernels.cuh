__global__ void sgemm_naive(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C);
__global__ void sgemm_coalescing(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);
__global__ void sgemm_cache_blocking(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);
__global__ void sgemm_1d_blocktiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);
__global__ void sgemm_2d_blocktiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);
