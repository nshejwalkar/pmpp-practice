#include <cuda_runtime.h>

// 1a
// A*B = C. A has shape mxn, B has shape nxk. launch only m threads
// matmul_1a<<<(m+15)/16, 16>>>(A_d, B_d, C_d, m, n, k);
__global__ void matmul_1a(float* A, float* B, float* C, int m, int n, int k) {
   int row = blockDim.x * blockIdx.x + threadIdx.x;

   if (row < m) {
      for (int col = 0; col < k; col++) {
         int cur_idx = row*k+col;
         float cur_sum = 0;
         for (int i = 0; i < n; i++) {
            cur_sum += A[row*n+i]*B[i*k+col];
         }
         C[cur_idx] = cur_sum;
      }
   }
}

// 1b
// A*B = C. A has shape mxn, B has shape nxk. launch only k threads
// matmul_1a<<<(k+15)/16, 16>>>(A_d, B_d, C_d, m, n, k);
__global__ void matmul_1b(float *A, float *B, float *C, int m, int n, int k)
{
   int col = blockDim.x * blockIdx.x + threadIdx.x;

   if (col < k) {
      for (int row = 0; row < m; row++) {
         int cur_idx = row * k + col;
         float cur_sum = 0;
         for (int i = 0; i < n; i++) {
            cur_sum += A[row*n+i]*B[i*k+col];
         }
         C[cur_idx] = cur_sum;
      }
   }
}

// 2
// dimxdim times dimx1
__global__ void matvecmul(float* mat_in, float* vec_out, float* vec, int dim) {
   int row = blockDim.x * blockIdx.x + threadIdx.x;

   if (row < dim) {
      float sum = 0.f;
      for (int col = 0; col < dim; col++) {
         sum += mat_in[row*dim+col] * vec[col];
      }
      vec_out[row] = sum;
   }
}

int main() {
   int dim = 1024;
   float* mat_h = new float[dim * dim];
   float* vec_h = new float[dim];
   float* out_h = new float[dim];
   float* mat_in;
   float* vec_out;
   float* vec;
   
   cudaMalloc((void **)&mat_in, dim*dim*sizeof(*mat_in));
   cudaMalloc((void **)&vec_out, dim*sizeof(*vec_out));
   cudaMalloc((void **)&vec, dim*sizeof(*vec));

   cudaMemcpy(mat_in, mat_h, dim*dim*sizeof(*mat_in), cudaMemcpyHostToDevice);
   cudaMemcpy(vec, vec_h, dim*sizeof(*vec), cudaMemcpyHostToDevice);

   matvecmul<<<dim3((dim+15)/16), 16>>>(mat_in, vec_out, vec, dim);
   cudaDeviceSynchronize();  // wait for kernel to finish

   cudaMemcpy(out_h, vec_out, dim*sizeof(*out_h), cudaMemcpyDeviceToHost);

   cudaFree(mat_in);
   cudaFree(vec);
   cudaFree(vec_out);
   delete[] mat_h;
   delete[] vec_h;
   delete[] out_h;
}