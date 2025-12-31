#include <cuda_runtime.h>
#include <time.h>
#include <iostream>

// two separate boundary checks are needed. One for making sure reading from the input is in bounds
// and one for making sure the thread is calculating valid output element. Notice that these checks
// are not mutually exclusive or redundant.
#define TILE_WIDTH 16
__global__ void matmul(float* M, float* N, float* P, int mat_width) {
   __shared__ float M_shared[TILE_WIDTH][TILE_WIDTH];
   __shared__ float N_shared[TILE_WIDTH][TILE_WIDTH];

   int row = blockIdx.y * TILE_WIDTH + threadIdx.y;  // tile width is blockdim
   int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

   float accum = 0.f;
   for (int phase = 0; phase < (mat_width + TILE_WIDTH - 1)/TILE_WIDTH; phase++) {
      // first load M and N into the shared mem. lasts only for this phase

      // this is to check if thread is inside input matrix bounds
      if (row < mat_width && phase * TILE_WIDTH + threadIdx.x < mat_width)
         M_shared[threadIdx.y][threadIdx.x] = M[row*mat_width + phase*TILE_WIDTH+threadIdx.x];     // M[row][phase][col]
      else
         M_shared[threadIdx.y][threadIdx.x] = 0.0f;

      // same thing as above
      if (col < mat_width && phase * TILE_WIDTH + threadIdx.y < mat_width)
         N_shared[threadIdx.y][threadIdx.x] = N[(phase*TILE_WIDTH+threadIdx.y) * mat_width + col]; // N[row][phase][col]
      else
         N_shared[threadIdx.y][threadIdx.x] = 0.0f;

      __syncthreads();  // ensures writes are finished

      for (int k = 0; k < TILE_WIDTH; k++) {
         accum += M_shared[threadIdx.y][k] * N_shared[k][threadIdx.x];
      }

      __syncthreads();  // ensures reads are finished before overwriting shared mem in next phase
   }

   // this is to check if thread is inside output matrix bounds
   if (row < mat_width && col < mat_width)
      P[row*mat_width+col] = accum;
}

int main() {
   const int dim = 1024;
   float* mat_h = new float[dim * dim];
   float* vec_h = new float[dim];
   float* out_h = new float[dim];

   float* mat_d;
   float* vec_d;
   float* out_d;

   // Initialize mat_h and vec_h
   for (int i = 0; i < dim * dim; i++) {
      mat_h[i] = static_cast<float>(i % 100);
   }
   for (int i = 0; i < dim; i++) {
      vec_h[i] = static_cast<float>(i % 100);
   }

   cudaMalloc((void**)&mat_d, dim * dim * sizeof(float));
   cudaMalloc((void**)&vec_d, dim * sizeof(float));
   cudaMalloc((void**)&out_d, dim * sizeof(float));

   cudaMemcpy(mat_d, mat_h, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(vec_d, vec_h, dim * sizeof(float), cudaMemcpyHostToDevice);

   dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
   dim3 gridSize((dim + TILE_WIDTH - 1) / TILE_WIDTH, (dim + TILE_WIDTH - 1) / TILE_WIDTH);

   // start timing
   clock_t start = clock();
   matmul<<<gridSize, blockSize>>>(mat_d, mat_d, mat_d, dim);
   clock_t end = clock();
   double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
   std::cout << "Time taken: " << time_taken << " seconds" << std::endl;

   cudaMemcpy(out_h, out_d, dim * sizeof(float), cudaMemcpyDeviceToHost);

   delete[] mat_h;
   delete[] vec_h;
   delete[] out_h;
   cudaFree(mat_d);
   cudaFree(vec_d);
   cudaFree(out_d);

   return 0;
}