#include <cuda_runtime.h>
#include <iostream>

// kernel
__global__ void rgb_to_gray(unsigned char* rgb_d, unsigned char* gray_d, int n, int m) {
   int col = blockDim.x*blockIdx.x + threadIdx.x;
   int row = blockDim.y*blockIdx.y + threadIdx.y;

   if (row < n && col < m) {
      int gray_idx = row*m + col;
      int rgb_idx = gray_idx*3;

      unsigned char r = rgb_d[rgb_idx];
      unsigned char g = rgb_d[rgb_idx+1];
      unsigned char b = rgb_d[rgb_idx+2];

      gray_d[gray_idx] = 0.21f*r + 0.71f*g + 0.07f*b;
   }
}

int main() {
   unsigned char* rgb_h;
   unsigned char* rgb_d;
   unsigned char* gray_d;
   unsigned char* gray_h;
   int n = 800;
   int m = 600;

   cudaError_t err = cudaMalloc((void**)&rgb_d, 3*n*m*sizeof(*rgb_d));
   cudaError_t err2 = cudaMalloc((void**)&gray_d, n*m*sizeof(*gray_d));

   if (err != cudaSuccess || err2 != cudaSuccess) {
      return -1;
   }

   cudaMemcpy(rgb_d, rgb_h, 3*n*m*sizeof(*rgb_d), cudaMemcpyHostToDevice);

   dim3 block(16, 16);
   dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
   rgb_to_gray<<<grid, block>>>(rgb_d, gray_d, n, m);

   cudaMemcpy(gray_h, gray_d, n*m*sizeof(*gray_d), cudaMemcpyDeviceToHost);

   cudaFree(rgb_d);
   cudaFree(gray_d);
   delete[] rgb_h;
   delete[] gray_h;
}