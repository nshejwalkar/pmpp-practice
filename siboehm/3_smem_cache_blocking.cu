// A = MxK, B = KxN, C = MxN
template <const unsigned int block_size>
__global__ void sgemm_cache_blocking(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    const unsigned int block_row = blockIdx.x;
    const unsigned int block_col = blockIdx.y;

    __shared__ float tile_a[block_size*block_size];
    __shared__ float tile_b[block_size*block_size];

    const unsigned int thread_row = threadIdx.x / block_size;
    const unsigned int thread_col = threadIdx.x % block_size;

    const unsigned int global_row = block_row * block_size + thread_row;
    const unsigned int global_col = block_col * block_size + thread_col;

    // get A,B,C to correct starting positions (all on first column, first row for A and B)
    A += block_row * block_size * K; // remember, this will implicitly multiply by sizeof(float)
    B += block_col * block_size;
    C += (block_row * block_size) * N + (block_col * block_size);

    float tmp = 0.0;

    for (int phase = 0; phase < K; phase += block_size) {
        tile_a[thread_row * block_size + thread_col] = A[thread_row * K + thread_col];
        tile_b[thread_row * block_size + thread_col] = B[thread_row * N + thread_col];  // it doesnt matter that B is off the first col, indexing works the same

        __syncthreads();

        A += block_size;
        B += block_size * N;
        
        // ignoring all bounds checks
        for (int i = 0; i < block_size; i++) {
            tmp += tile_a[thread_row * block_size + i] * tile_b[i * block_size + thread_col];
        }

        __syncthreads();
    }

    if (x < M && y < N) {
        C[thread_row * block_size + thread_col] = alpha * tmp + beta * C[thread_row * block_size + thread_col];
    }
}

// launched the same as 2.

// this is the same thing as the tiled matmul from pmpp, except we're manually doing the work of advancing pointers.
// 2*32*32*4b = 8kb of smem used, but more isn't better. if you increase too high, you reach the limit of 
// (total smem in sm)/(total smem per block) = max blocks per sm when bounded by per block smem usage. less blocks = less occupancy.
// main bounds are smem/block, threads/block, regs/block -> 11,1,1 according to the blog, so more smem wouldnt have changed the occupancy.
// apparently, the main bottleneck is MIO, a pipeline for shared memory operations.

/*
very roughly, every cycle, one instruction per warp scheduler is selected from an active (running) warp, decoded, then issued in the right pipeline.
the MIO pipeline is relatively shallow (only a couple of in flight slots for instructions) because the point of gpu pipelines is just to hide latency. shared mem accesses dont take that long.
for comparison, LG pipelines are probably hundreds deep! CPUs dont do this at all, because they dont "hide latency." this is what people mean when they say gpus have deep pipelines.
mio throttle is a sign to increase the arithmetic intensity, because we've hit the top of the line in the roofline model (for smem, not gmem anymore).
this is bc bandwidth is roughly bounded by in-flight ops * bytes per op / latency per op (< than the physical limit of the wire im guessing)
*/