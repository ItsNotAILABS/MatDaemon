import cupy as cp
from .base import BaseBackend

# Highly optimized tiled + register-tiled GEMM kernel
GEMM_KERNEL = r'''
extern "C" __global__ void gemm_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K)
{
    const int TILE = 128;
    const int THREAD_TILE = 8;

    __shared__ float As[TILE][TILE + 4];  // +4 to avoid bank conflicts
    __shared__ float Bs[TILE][TILE + 4];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_start = by * TILE + ty * THREAD_TILE;
    int col_start = bx * TILE + tx * THREAD_TILE;

    float acc[THREAD_TILE][THREAD_TILE] = {0.0f};

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // Load A tile (coalesced)
        for (int i = 0; i < THREAD_TILE; ++i) {
            for (int j = 0; j < THREAD_TILE; ++j) {
                int r = by * TILE + ty * THREAD_TILE + i;
                int c = t * TILE + tx * THREAD_TILE + j;
                As[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = 
                    (r < M && c < K) ? A[r * K + c] : 0.0f;
            }
        }

        // Load B tile (coalesced)
        for (int i = 0; i < THREAD_TILE; ++i) {
            for (int j = 0; j < THREAD_TILE; ++j) {
                int r = t * TILE + ty * THREAD_TILE + i;
                int c = bx * TILE + tx * THREAD_TILE + j;
                Bs[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = 
                    (r < K && c < N) ? B[r * N + c] : 0.0f;
            }
        }

        __syncthreads();

        // Compute using registers (register tiling)
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float a_frag[THREAD_TILE];
            float b_frag[THREAD_TILE];

            #pragma unroll
            for (int i = 0; i < THREAD_TILE; ++i)
                a_frag[i] = As[ty * THREAD_TILE + i][k];

            #pragma unroll
            for (int j = 0; j < THREAD_TILE; ++j)
                b_frag[j] = Bs[k][tx * THREAD_TILE + j];

            #pragma unroll
            for (int i = 0; i < THREAD_TILE; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE; ++j) {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }

    // Write back results
    #pragma unroll
    for (int i = 0; i < THREAD_TILE; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE; ++j) {
            int r = row_start + i;
            int c = col_start + j;
            if (r < M && c < N)
                C[r * N + c] = acc[i][j];
        }
    }
}
'''

class CUDABackend(BaseBackend):
    def __init__(self, tile_size=128):
        self.tile_size = tile_size
        self.kernel = cp.RawKernel(GEMM_KERNEL, 'gemm_tiled')

    def matmul(self, A, B):
        if not isinstance(A, cp.ndarray):
            A = cp.asarray(A)
        if not isinstance(B, cp.ndarray):
            B = cp.asarray(B)

        M, K = A.shape
        K2, N = B.shape
        assert K == K2

        C = cp.zeros((M, N), dtype=cp.float32)

        block = (16, 16)  # 256 threads (good for 8x8 thread tile)
        grid = ((N + self.tile_size - 1) // self.tile_size,
                (M + self.tile_size - 1) // self.tile_size)

        self.kernel(grid, block, (A, B, C, M, N, K))
        return cp.asnumpy(C)  # Return as NumPy for compatibility
