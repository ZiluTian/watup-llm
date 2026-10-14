// The LLM Prefill Hot Path
//
// What this simulates: one linear layer during prefill -- the prompt's
// activation matrix X [M x in_dim] multiplied against the weight matrix
// W [out_dim x in_dim] to produce Y [M x out_dim]:  Y = X @ W^T.  This is
// the same layer as decode_hotpath.cu, but with M = prompt length instead
// of M = 1. During prefill the GPU ingests the whole prompt in one shot,
// and for any M above a few dozen this is a large GEMM: 2*M*N*K FLOPs,
// COMPUTE bound, not bandwidth bound. (In llama.cpp this is what
// mul_mat / the matrix-matrix path does.)
//
// It is the other side of the roofline from decode:
//   M = 1    (decode):  ~1 FLOP per HBM byte  -> bandwidth bound, GEMV wins
//   M = 4096 (prefill):  ~1365 FLOPs per HBM byte -> compute bound, GEMM wins
// The self-tuning prompt-length sweep at the end makes that crossover
// visible: a warp-per-row GEMV (the decode v2 kernel, included here as a
// candidate) beats every GEMM kernel at small M and loses at large M.
//
// Seven hand-written kernels, each one optimization on top of the last:
//   v0  naive fp32            -- one thread per output element, no tiling:
//                                 the global-memory thrash baseline
//   v1  fp32 smem tile 32x32  -- shared-memory tiling: global traffic drops
//                                 from O(M*N*K) to O(M*N*K / tile)
//   v2  fp16 smem tile        -- half the bytes, fp32 accumulation
//   v3  fp16 reg-blocked 4x4  -- each thread owns 4x4 outputs (64x32 block
//                                 tile): smem traffic cut, 16 independent
//                                 FMA streams per thread
//   v4  fp16 cp.async dblbuf  -- async global->smem copy of the NEXT k tile
//                                 while the current one is being consumed
//   v5  fp16 tensor cores     -- mma.sync m16n8k16 + ldmatrix + cp.async
//                                 double buffering: the compute-bound payoff
//   v6  int8 tensor cores     -- mma.sync m16n8k8, per-row scales on both
//                                 X and W, fp32 rescale in the epilogue:
//                                 2x the MAC rate of fp16, at an accuracy cost
// Plus two reference rows:
//   gemv-fp16                 -- warp-per-row fp16 GEMV (decode-hotpath v2),
//                                 included so the M sweep can show decode
//                                 still winning at M = 1
//   cuBLAS sgemm / hgemm      -- what you would actually ship
//
// The M sweep is self-tuning: for each prompt length M the harness probes
// the applicable fp16 designs (10 iterations each) and times the winner
// (50 iterations), so the table shows which structure wins on YOUR GPU.
//
// Watch v5 vs v2 closely: v2 halved the bytes (like decode v1->v2) but the
// payoff stops there -- once the kernel is compute bound, bytes are no
// longer the wall. v5 wins by issuing the FMAs on the tensor cores, and
// v6 (int8 TC) adds ~1.26x on top -- NOT the full 2x int8 TC rate, because
// v6's narrow 4-byte smem fragment loads (12 per k-tile vs v5's 6
// ldmatrix.x4) throttle the pipeline before the TC issue rate does.
// Quantization helps prefill at the compute-rate level, not the bandwidth
// level; its price is accuracy (v6: 1.23% vs 0.03% for the fp16 kernels),
// and the table measures it.
//
// Build:  make pfill            (or: nvcc -O3 -arch=sm_120 -lcublas -o bin/prefill_hotpath src/prefill_hotpath.cu)
// Run:    ./bin/prefill_hotpath [M] [N] [K]   (defaults 4096 4096 4096)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <functional>
#include <algorithm>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)
#define CUBLAS_CHECK(call)                                                   \
    do {                                                                     \
        cublasStatus_t st = (call);                                          \
        if (st != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,  \
                    (int)st);                                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// ============================================================
// PTX helpers (sm_80+ features; sm_120 supports all of these)
// ============================================================

__device__ __forceinline__ unsigned saddr(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

// 16-byte async global->shared copy; src_bytes < 16 zero-fills the rest
// (used to keep OOB tile edges at zero instead of touching OOB addresses)
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(saddr(dst)), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_commit()  { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait0()   { asm volatile("cp.async.wait_group 0;\n"); }
__device__ __forceinline__ void cp_wait1()   { asm volatile("cp.async.wait_group 1;\n"); }

// ldmatrix.x4: four 8x8 fp16 matrices -> a 16x16 fragment set
__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1,
                                            unsigned& r2, unsigned& r3,
                                            const void* smem) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr(smem)));
}

// mma.sync m16n8k16: D = A(16x16 fp16, row) * B(16x8 fp16, col) + C (fp32)
__device__ __forceinline__ void mma_16816(float& c0, float& c1, float& c2, float& c3,
                                          unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                          unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// mma.sync m16n8k16 s8/s32 -- the int8 tensor-core shape in the CUDA 13
// PTX ISA (the legacy m8n8k4 s8 shape is gone; this one is 2x the MAC
// count of the fp16 m16n8k16, which is the int8 rate bump). A = 2 regs
// (8 s8 per lane), B = 1 reg (4 s8 per lane), C = 4 regs (4 s32).
__device__ __forceinline__ void mma_16816_s32(int& c0, int& c1, int& c2, int& c3,
                                              int a0, int a1, int b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
}

// ============================================================
// Kernels
//
// Conventions:  X is [M][K] row-major, W is [N][K] row-major, Y is [M][N].
// Y = X @ W^T, i.e. Y[m][n] = sum_k X[m][k] * W[n][k].
// ============================================================

// v0: one thread per output element, no tiling. For a warp at fixed m,
// X[m][k] is broadcast (fine) but W[n][k] with n varying across the warp
// is K*2 bytes apart per lane: zero coalescing on the weights. Every
// element of W is re-read from memory M/... times -- the classic
// un-tiled GEMM. This is the baseline the whole ladder is a fix for.
__global__ void gemm_v0_naive_fp32(const float* __restrict__ X,
                                   const float* __restrict__ W,
                                   float* __restrict__ Y,
                                   int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;   // column (W row)
    int m = blockIdx.y * blockDim.y + threadIdx.y;   // row
    if (m >= M || n >= N) return;
    const float* xrow = X + (size_t)m * K;
    const float* wrow = W + (size_t)n * K;
    float sum = 0.f;
    for (int k = 0; k < K; k++) sum += xrow[k] * wrow[k];
    Y[(size_t)m * N + n] = sum;
}

// v1: textbook shared-memory tiling, fp32. 32x32 k tile, 16x16 blocks
// (256 threads), one thread per output element. Global traffic per tile:
// 32*(32+32) floats instead of 32*32*32 -- a 32x reduction.
#define V1_TILE 32
__global__ void gemm_v1_smem_fp32(const float* __restrict__ X,
                                  const float* __restrict__ W,
                                  float* __restrict__ Y,
                                  int M, int N, int K) {
    __shared__ float As[V1_TILE][V1_TILE + 1];
    __shared__ float Bs[V1_TILE][V1_TILE + 1];
    const int m0 = blockIdx.y * V1_TILE;
    const int n0 = blockIdx.x * V1_TILE;
    const int tx = threadIdx.x, ty = threadIdx.y;

    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += V1_TILE) {
        // 256 threads load a 32x32 tile: 4 elements each (2 row-halves x
        // 2 col-halves). The col offset must follow the col-half index,
        // not the row-half index.
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int r = m0 + ty + 16 * j;
                int c = k0 + tx + 16 * i;
                As[ty + 16 * j][tx + 16 * i] = (r < M) ? X[(size_t)r * K + c] : 0.f;
            }
        }
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int r = n0 + ty + 16 * j;
                int c = k0 + tx + 16 * i;
                Bs[ty + 16 * j][tx + 16 * i] = (r < N) ? W[(size_t)r * K + c] : 0.f;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < V1_TILE; k++) acc += As[ty][k] * Bs[tx][k];
        __syncthreads();
    }
    int m = m0 + ty, n = n0 + tx;
    if (m < M && n < N) Y[(size_t)m * N + n] = acc;
}

// v2: same structure, fp16 data with fp32 accumulation. Half the bytes
// of X/W in HBM and in smem; the FMA count is unchanged (fp32 accumulate).
#define V2_TILE 32
#define V2_PAD 8
__global__ void gemm_v2_smem_fp16(const __half* __restrict__ X,
                                  const __half* __restrict__ W,
                                  float* __restrict__ Y,
                                  int M, int N, int K) {
    __shared__ __half As[V2_TILE][V2_TILE + V2_PAD];
    __shared__ __half Bs[V2_TILE][V2_TILE + V2_PAD];
    const int m0 = blockIdx.y * V2_TILE;
    const int n0 = blockIdx.x * V2_TILE;
    const int tx = threadIdx.x, ty = threadIdx.y;

    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += V2_TILE) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int r = m0 + ty + 16 * j;
                int c = k0 + tx + 16 * i;
                As[ty + 16 * j][tx + 16 * i] = (r < M) ? X[(size_t)r * K + c] : __half(0.f);
            }
        }
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int r = n0 + ty + 16 * j;
                int c = k0 + tx + 16 * i;
                Bs[ty + 16 * j][tx + 16 * i] = (r < N) ? W[(size_t)r * K + c] : __half(0.f);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < V2_TILE; k++)
            acc += __half2float(As[ty][k]) * __half2float(Bs[tx][k]);
        __syncthreads();
    }
    int m = m0 + ty, n = n0 + tx;
    if (m < M && n < N) Y[(size_t)m * N + n] = acc;
}

// v3: register blocking. 64x32 output tile per 128-thread block; each
// thread accumulates 4x4 outputs in registers. smem reads drop 4x per
// thread (one As row + one Bs row feed 16 FMAs instead of 1 FMA), and
// the 16 accumulators give the FMA pipe independent streams.
#define V3_MTILE 64
#define V3_NTile 32
#define V3_KTILE 32
#define V3_PAD 8
__global__ void gemm_v3_regblock_fp16(const __half* __restrict__ X,
                                      const __half* __restrict__ W,
                                      float* __restrict__ Y,
                                      int M, int N, int K) {
    __shared__ __half As[V3_MTILE][V3_KTILE + V3_PAD];
    __shared__ __half Bs[V3_NTile][V3_KTILE + V3_PAD];
    const int m0 = blockIdx.y * V3_MTILE;
    const int n0 = blockIdx.x * V3_NTile;
    const int tx = threadIdx.x & 7;      // 8 n-groups x 4
    const int ty = threadIdx.x >> 3;     // 16 m-groups x 4
    const int my = ty * 4;               // this thread's 4 m rows
    const int ny = tx * 4;               // this thread's 4 n cols

    float c[4][4];
    #pragma unroll
    for (int a = 0; a < 4; a++)
        #pragma unroll
        for (int b = 0; b < 4; b++) c[a][b] = 0.f;

    for (int k0 = 0; k0 < K; k0 += V3_KTILE) {
        // A tile: 64 rows x 32 halves = 2048 halves -> 16 per thread:
        //   2 threads per row, 16 halves (8 x half2) each
        // B tile: 32 rows x 32 halves = 1024 halves -> 8 per thread:
        //   4 threads per row, 8 halves (4 x half2) each
        {
            int ar = m0 + (threadIdx.x >> 1);
            int ac = k0 + (threadIdx.x & 1) * 16;
            if (ar < M) {
                const __half* src = X + (size_t)ar * K + ac;
                #pragma unroll
                for (int h = 0; h < 8; h++)
                    *(__half2*)(&As[threadIdx.x >> 1][ac - k0 + h * 2]) =
                        *(__half2*)(src + h * 2);
            }
            // OOB rows: cells stay unwritten, but only threads computing
            // outputs for those (discarded) rows ever read them.
        }
        {
            int br = n0 + (threadIdx.x >> 2);
            int bc = k0 + (threadIdx.x & 3) * 8;
            if (br < N) {
                const __half* src = W + (size_t)br * K + bc;
                #pragma unroll
                for (int h = 0; h < 4; h++)
                    *(__half2*)(&Bs[threadIdx.x >> 2][bc - k0 + h * 2]) =
                        *(__half2*)(src + h * 2);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < V3_KTILE; k++) {
            #pragma unroll
            for (int a = 0; a < 4; a++) {
                float x = __half2float(As[my + a][k]);
                #pragma unroll
                for (int b = 0; b < 4; b++) c[a][b] += x * __half2float(Bs[ny + b][k]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int a = 0; a < 4; a++) {
        int m = m0 + my + a;
        if (m >= M) continue;
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            int n = n0 + ny + b;
            if (n < N) Y[(size_t)m * N + n] = c[a][b];
        }
    }
}

// v4: cp.async + double buffering on top of v3. The k tile is fetched with
// cp.async.cg (16B chunks, L2-bypassing) into a second smem buffer while the
// first is being consumed; cp.async.wait_group 1 + __syncthreads hands the
// buffers over. The global->smem latency of tile k+1 overlaps the FMA work
// of tile k, so the smem load no longer sits on the critical path.
#define V4_MTILE 64
#define V4_NTile 32
#define V4_KTILE 32
#define V4_PAD 8
__global__ void gemm_v4_cpasync_fp16(const __half* __restrict__ X,
                                     const __half* __restrict__ W,
                                     float* __restrict__ Y,
                                     int M, int N, int K) {
    extern __shared__ __half smem[];
    // two buffers, each: A 64 x (32+8) halves + B 32 x (32+8) halves
    const int ASZ = V4_MTILE * (V4_KTILE + V4_PAD);
    const int BSZ = V4_NTile * (V4_KTILE + V4_PAD);
    // Buffer 1 starts right after buffer 0 (ASZ + BSZ, NOT 2*ASZ -- the
    // A and B tiles have different sizes, so 2*ASZ would make As1 and Bs1
    // overlap and the prefetch would clobber the B tile with A data).
    __half* As0 = smem;
    __half* Bs0 = smem + ASZ;
    __half* As1 = smem + ASZ + BSZ;
    __half* Bs1 = smem + 2 * ASZ + BSZ;

    const int m0 = blockIdx.y * V4_MTILE;
    const int n0 = blockIdx.x * V4_NTile;
    const int tx = threadIdx.x & 7;
    const int ty = threadIdx.x >> 3;
    const int my = ty * 4;
    const int ny = tx * 4;

    float c[4][4];
    #pragma unroll
    for (int a = 0; a < 4; a++)
        #pragma unroll
        for (int b = 0; b < 4; b++) c[a][b] = 0.f;

    // A tile: 64 rows x 32 halves x 2B = 4KB = 256 chunks of 16B -> 2/thread
    //   row = t/4 (2 rows per chunk-row... 4 threads per row), col16 = (t%4)*8
    // B tile: 32 rows x 64B = 1KB = 128 chunks -> 1/thread
    auto issue_load = [&](int buf, int k0) {
        __half* A = buf ? As1 : As0;
        __half* B = buf ? Bs1 : Bs0;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int t = threadIdx.x + j * 128;
            int ar = m0 + (t >> 2);
            int ac = k0 + (t & 3) * 8;
            int valid = (ar < M) ? 16 : 0;
            cp_async16(&A[(t >> 2) * (V4_KTILE + V4_PAD) + (t & 3) * 8],
                       X + (size_t)(ar < M ? ar : 0) * K + ac, valid);
        }
        {
            int t = threadIdx.x;
            int br = n0 + (t >> 2);
            int bc = k0 + (t & 3) * 8;
            int valid = (br < N) ? 16 : 0;
            cp_async16(&B[(t >> 2) * (V4_KTILE + V4_PAD) + (t & 3) * 8],
                       W + (size_t)(br < N ? br : 0) * K + bc, valid);
        }
        cp_commit();
    };

    int ntiles = K / V4_KTILE;
    issue_load(0, 0);
    if (ntiles > 1) issue_load(1, V4_KTILE);

    for (int k0 = 0; k0 < K; k0 += V4_KTILE) {
        int buf = (k0 / V4_KTILE) & 1;
        if (ntiles > 1) cp_wait1(); else cp_wait0();
        __syncthreads();
        __half* As = buf ? As1 : As0;
        __half* Bs = buf ? Bs1 : Bs0;
        #pragma unroll
        for (int k = 0; k < V4_KTILE; k++) {
            #pragma unroll
            for (int a = 0; a < 4; a++) {
                float x = __half2float(As[(my + a) * (V4_KTILE + V4_PAD) + k]);
                #pragma unroll
                for (int b = 0; b < 4; b++)
                    c[a][b] += x * __half2float(Bs[(ny + b) * (V4_KTILE + V4_PAD) + k]);
            }
        }
        __syncthreads();
        // Prefetch tile k0+64 into the buffer we just consumed (it holds
        // tile k0, which is now fully read out); tile k0+32 sits in the
        // other buffer.
        if (k0 + 2 * V4_KTILE < K) issue_load(buf, k0 + 2 * V4_KTILE);
        else cp_commit();  // keep the group count balanced for wait_group
    }
    #pragma unroll
    for (int a = 0; a < 4; a++) {
        int m = m0 + my + a;
        if (m >= M) continue;
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            int n = n0 + ny + b;
            if (n < N) Y[(size_t)m * N + n] = c[a][b];
        }
    }
}

// v5: fp16 tensor cores. mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32.
//
// Block: 128 threads (4 warps), 64(M) x 32(N) output tile.
// Warp:  32(M) x 16(N) = two 16x8 mma tiles per 16-k step.
// K tile: 32 halves, double-buffered with cp.async (the v4 pipeline).
//
// Fragment loading via ldmatrix.x4 (four 8x8 matrices per 16x16 tile):
//   A (row-major X tile):  lane l supplies the row address
//       row = (l%8) + 8*((l>>3)&1),  col = 16*h + 8*(l>>4)
//     so matrix 0 = (m0-7,k0-7), matrix 1 = (m8-15,k0-7),
//     matrix 2 = (m0-7,k8-15), matrix 3 = (m8-15,k8-15).  The m16n8k16
//     A-fragment needs a0 = top-left, a1 = bottom-left, a2 = top-right,
//     a3 = bottom-right -- which is exactly ldmatrix's reg0..reg3, so the
//     A operand goes in natural register order.
//   B (W tile, stored [n][k] row-major = the mma's col-major B): the SAME
//     address pattern on the W tile already yields the B fragment, because
//     a W row IS an n-row of the fragment and the 8 in-row halves are the
//     k dimension: n-tile 0 uses {reg0, reg2}, n-tile 1 uses {reg1, reg3}.
//     (No ldmatrix.trans needed -- that variant is for k-major B storage.)
#define V5_MTILE 64
#define V5_NTile 32
#define V5_KTILE 32
#define V5_PAD 8   // row stride 40 halves keeps ldmatrix bank-conflict-free
__global__ void gemm_v5_tc_fp16(const __half* __restrict__ X,
                                const __half* __restrict__ W,
                                float* __restrict__ Y,
                                int M, int N, int K) {
    extern __shared__ __half smem[];
    const int ASZ = V5_MTILE * (V5_KTILE + V5_PAD);
    const int BSZ = V5_NTile * (V5_KTILE + V5_PAD);
    // Buffer 1 starts right after buffer 0 (ASZ + BSZ, NOT 2*ASZ -- the
    // A and B tiles have different sizes, so 2*ASZ would make As1 and Bs1
    // overlap and the prefetch would clobber the B tile with A data).
    __half* As0 = smem;
    __half* Bs0 = smem + ASZ;
    __half* As1 = smem + ASZ + BSZ;
    __half* Bs1 = smem + 2 * ASZ + BSZ;

    const int m0 = blockIdx.y * V5_MTILE;
    const int n0 = blockIdx.x * V5_NTile;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wm = (warp >> 1) * 32;   // warp's 32-row M window
    const int wn = (warp & 1) * 16;    // warp's 16-col N window

    float c[2][2][4];  // [m-tile][n-tile][4 fragment regs]
    #pragma unroll
    for (int j = 0; j < 2; j++)
        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int r = 0; r < 4; r++) c[j][i][r] = 0.f;

    // ldmatrix row/col offsets for the 4 8x8 matrices of one 16x16 tile
    const int lrow = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int lcol = 8 * (lane >> 4);

    auto issue_load = [&](int buf, int k0) {
        __half* A = buf ? As1 : As0;
        __half* B = buf ? Bs1 : Bs0;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int t = threadIdx.x + j * 128;
            int ar = m0 + (t >> 2);
            int ac = k0 + (t & 3) * 8;
            int valid = (ar < M) ? 16 : 0;
            cp_async16(&A[(t >> 2) * (V5_KTILE + V5_PAD) + (t & 3) * 8],
                       X + (size_t)(ar < M ? ar : 0) * K + ac, valid);
        }
        {
            int t = threadIdx.x;
            int br = n0 + (t >> 2);
            int bc = k0 + (t & 3) * 8;
            int valid = (br < N) ? 16 : 0;
            cp_async16(&B[(t >> 2) * (V5_KTILE + V5_PAD) + (t & 3) * 8],
                       W + (size_t)(br < N ? br : 0) * K + bc, valid);
        }
        cp_commit();
    };

    const int ntiles = K / V5_KTILE;
    issue_load(0, 0);
    if (ntiles > 1) issue_load(1, V5_KTILE);

    for (int k0 = 0; k0 < K; k0 += V5_KTILE) {
        int buf = (k0 / V5_KTILE) & 1;
        if (ntiles > 1) cp_wait1(); else cp_wait0();
        __syncthreads();
        __half* As = buf ? As1 : As0;
        __half* Bs = buf ? Bs1 : Bs0;

        #pragma unroll
        for (int h = 0; h < 2; h++) {   // two 16-k steps per 32-k tile
            unsigned ra[2][4], rb[4];
            #pragma unroll
            for (int j = 0; j < 2; j++)  // 2 m-tiles (32-row window)
                ldmatrix_x4(ra[j][0], ra[j][1], ra[j][2], ra[j][3],
                            &As[(wm + j * 16 + lrow) * (V5_KTILE + V5_PAD) + h * 16 + lcol]);
            ldmatrix_x4(rb[0], rb[1], rb[2], rb[3],
                        &Bs[(wn + lrow) * (V5_KTILE + V5_PAD) + h * 16 + lcol]);
            #pragma unroll
            for (int j = 0; j < 2; j++)
                #pragma unroll
                for (int i = 0; i < 2; i++)
                    mma_16816(c[j][i][0], c[j][i][1], c[j][i][2], c[j][i][3],
                              ra[j][0], ra[j][1], ra[j][2], ra[j][3],   // A: {TL, BL, TR, BR}
                              i ? rb[1] : rb[0], i ? rb[3] : rb[2]);   // B per n-tile
        }
        __syncthreads();
        // Prefetch tile k0+64 into the buffer we just consumed (it holds
        // tile k0, which is now fully read out); tile k0+32 sits in the
        // other buffer.
        if (k0 + 2 * V5_KTILE < K) issue_load(buf, k0 + 2 * V5_KTILE);
        else cp_commit();
    }

    // epilogue: m16n8k16 C-fragment (PTX ISA, m16n8k16 accumulator):
    // lane l: c0 = C[l/4][2*(l%4)],  c1 = C[l/4][2*(l%4)+1],
    //         c2 = C[l/4+8][2*(l%4)], c3 = C[l/4+8][2*(l%4)+1].
    // (NOT the m16n8k8 layout c0=C[l/4][l%4], c1=C[l/4+8][l%4], ... --
    // the row and column pairing is different.)  halfrow selects the +8
    // row (reg 0/1 -> reg 2/3), halfcol the +1 col (reg 0/2 -> reg 1/3),
    // so the register index is halfrow*2 + halfcol.
    const int cr = lane >> 2, cc = lane & 3;
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        #pragma unroll
        for (int halfrow = 0; halfrow < 2; halfrow++) {
            int m = m0 + wm + j * 16 + cr + halfrow * 8;
            if (m >= M) continue;
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int halfcol = 0; halfcol < 2; halfcol++) {
                    int n = n0 + wn + i * 8 + 2 * cc + halfcol;
                    if (n < N) Y[(size_t)m * N + n] = c[j][i][halfrow * 2 + halfcol];
                }
            }
        }
    }
}

// v6: int8 tensor cores. mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32
//
// Same structure as v5 (64x32 block tile, 32x16 warp tile, cp.async
// double buffering) with int8 operands: half the smem bytes of v5 at the
// same 32-element k tile, and the int8 TC rate is 2x the fp16 rate -- so
// v6 is the "how far does quantization get you on prefill" data point.
// Fragment layout (m16n8k16 s8: 4 int8 per register, A = 2 regs, B = 1):
//   a0 = A[l/4][4(l%4)+0..3],   a1 = A[l/4+8][4(l%4)+0..3]
//        (4 CONSECUTIVE k bytes from one row; the 16 k columns are
//         covered by the 4 lanes of each row group)
//   b0 = B[4*(l%4) + 0..3][l/4]  (doc layout: B row (k) = 4*(l%4)+i,
//        B col (n) = l/4 -- i.e. lane (cr=l/4, cc=l%4) owns the 4
//        consecutive k-bytes 4cc..4cc+3 of W row cr)
//   c0 = C[l/4][2*(l%4)], c1 = C[l/4][2*(l%4)+1], c2 = C[l/4+8][2*(l%4)],
//    c3 = C[l/4+8][2*(l%4)+1]      (same as the fp16 C fragment)
// Plain aligned 4-byte smem reads, no ldmatrix needed.
//
// Quantization: per-ROW scales on both operands (the same idea as decode
// v3's per-row W scale, extended to X):
//   qX = round(X / sx[m])  in [-127,127],  sx[m] = max|X[m][:]| / 127
//   qW = round(W / sw[n])  in [-127,127],  sw[n] = max|W[n][:]| / 127
// The kernel accumulates the int8 product in int32 (exact, no rounding)
// and rescales in the epilogue:  Y = acc * sx[m] * sw[n].
#define V6_MTILE 64
#define V6_NTile 32
#define V6_KTILE 32
#define V6_PAD 16  // row stride 48 int8: 16B multiple (cp.async dst alignment),
                  // 12-word stride keeps the fragment loads conflict-light
__global__ void gemm_v6_tc_int8(const int8_t* __restrict__ qX,
                                const int8_t* __restrict__ qW,
                                const float* __restrict__ sx,
                                const float* __restrict__ sw,
                                float* __restrict__ Y,
                                int M, int N, int K) {
    extern __shared__ int8_t smem8[];
    const int ASZ = V6_MTILE * (V6_KTILE + V6_PAD);
    const int BSZ = V6_NTile * (V6_KTILE + V6_PAD);
    // (same buffer-1 offset rule as v4/v5: ASZ + BSZ, not 2*ASZ)
    int8_t* As0 = smem8;
    int8_t* Bs0 = smem8 + ASZ;
    int8_t* As1 = smem8 + ASZ + BSZ;
    int8_t* Bs1 = smem8 + 2 * ASZ + BSZ;

    const int m0 = blockIdx.y * V6_MTILE;
    const int n0 = blockIdx.x * V6_NTile;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wm = (warp >> 1) * 32;   // warp's 32-row M window
    const int wn = (warp & 1) * 16;    // warp's 16-col N window
    const int S = V6_KTILE + V6_PAD;

    int c[2][2][4];  // int32 accumulators [m-tile][n-tile][4 frag regs]
    #pragma unroll
    for (int j = 0; j < 2; j++)
        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int r = 0; r < 4; r++) c[j][i][r] = 0;

    const int cr = lane >> 2;    // fragment row (m / n index)
    const int cc = lane & 3;     // column group within the k dimension
    const int k4 = 4 * cc;       // 4 consecutive k bytes owned by this lane

    auto issue_load = [&](int buf, int k0) {
        int8_t* A = buf ? As1 : As0;
        int8_t* B = buf ? Bs1 : Bs0;
        // A tile: 64 rows x 32 B = 2KB = 128 chunks of 16B -> 1/thread
        // B tile: 32 rows x 32 B = 1KB = 64 chunks -> threads 0..63
        {
            int t = threadIdx.x;
            int ar = m0 + (t >> 1);
            int ac = k0 + (t & 1) * 16;
            int valid = (ar < M) ? 16 : 0;
            cp_async16(&A[(t >> 1) * S + (t & 1) * 16],
                       qX + (size_t)(ar < M ? ar : 0) * K + ac, valid);
        }
        if (threadIdx.x < 64) {
            int t = threadIdx.x;
            int br = n0 + (t >> 1);
            int bc = k0 + (t & 1) * 16;
            int valid = (br < N) ? 16 : 0;
            cp_async16(&B[(t >> 1) * S + (t & 1) * 16],
                       qW + (size_t)(br < N ? br : 0) * K + bc, valid);
        }
        cp_commit();
    };

    const int ntiles = K / V6_KTILE;
    issue_load(0, 0);
    if (ntiles > 1) issue_load(1, V6_KTILE);

    for (int k0 = 0; k0 < K; k0 += V6_KTILE) {
        int buf = (k0 / V6_KTILE) & 1;
        if (ntiles > 1) cp_wait1(); else cp_wait0();
        __syncthreads();
        int8_t* As = buf ? As1 : As0;
        int8_t* Bs = buf ? Bs1 : Bs0;

        #pragma unroll
        for (int h = 0; h < 2; h++) {   // two 16-k steps per 32-k tile
            int kt = h * 16;
            int a[2][2], b[2];
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                // a0: 4 k-bytes of row cr; a1: same 4 k-bytes of row cr+8
                a[j][0] = *(const unsigned*)(As + (size_t)(wm + j * 16 + cr) * S + kt + k4);
                a[j][1] = *(const unsigned*)(As + (size_t)(wm + j * 16 + cr + 8) * S + kt + k4);
            }
            #pragma unroll
            for (int i = 0; i < 2; i++)
                b[i] = *(const unsigned*)(Bs + (size_t)(wn + i * 8 + cr) * S + kt + k4);
            #pragma unroll
            for (int j = 0; j < 2; j++)
                #pragma unroll
                for (int i = 0; i < 2; i++)
                    mma_16816_s32(c[j][i][0], c[j][i][1], c[j][i][2], c[j][i][3],
                                  a[j][0], a[j][1], b[i]);
        }
        __syncthreads();
        // Prefetch tile k0+64 into the buffer we just consumed (it holds
        // tile k0, which is now fully read out); tile k0+32 sits in the
        // other buffer.
        if (k0 + 2 * V6_KTILE < K) issue_load(buf, k0 + 2 * V6_KTILE);
        else cp_commit();
    }

    // epilogue: dequantize. s32 layout (same as the f32 m16n8k16 C):
    // c0 = C[l/4][2*(l%4)], c1 = C[l/4][2*(l%4)+1],
    // c2 = C[l/4+8][2*(l%4)], c3 = C[l/4+8][2*(l%4)+1]
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        #pragma unroll
        for (int halfrow = 0; halfrow < 2; halfrow++) {
            int m = m0 + wm + j * 16 + cr + halfrow * 8;
            if (m >= M) continue;
            float row_scale = sx[m];
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int halfcol = 0; halfcol < 2; halfcol++) {
                    int n = n0 + wn + i * 8 + 2 * cc + halfcol;
                    if (n < N)
                        Y[(size_t)m * N + n] = (float)c[j][i][halfrow * 2 + halfcol]
                                               * row_scale * sw[n];
                }
            }
        }
    }
}

// gemv-fp16: the decode-hotpath v2 kernel, included as a sweep candidate.
// One WARP per output row: the 32 lanes sweep a W row in 8-half (16B)
// chunks, so the weight stream is fully coalesced. This is what M=1 "should"
// look like -- the M sweep shows where GEMM structure takes over from it.
__global__ void gemm_gemv_fp16(const __half* __restrict__ X,
                               const __half* __restrict__ W,
                               float* __restrict__ Y,
                               int M, int N, int K) {
    // warp w handles output (m = w / N, n = w % N): y[m][n] = dot(x[m], w[n])
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int m = warp / N;
    int n = warp % N;
    if (m >= M) return;
    const __half* x = X + (size_t)m * K;
    const __half* w = W + (size_t)n * K;
    float acc = 0.f;
    // 8 halves per lane per pass; a pass covers 32*8 = 256 k values
    for (int k = lane * 8; k < K; k += 32 * 8) {
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            __half2 xh = *(__half2*)(x + k + q * 2);
            __half2 wh = *(__half2*)(w + k + q * 2);
            acc += __half2float(xh.x) * __half2float(wh.x)
                 + __half2float(xh.y) * __half2float(wh.y);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) Y[(size_t)m * N + n] = acc;
}

// ============================================================
// Benchmark harness
// ============================================================

struct Result {
    std::string name;
    float ms;
    double tflops;
    double pct_peak;      // % of measured cuBLAS hgemm peak (same shape)
    double speedup;       // vs v0
    double rel_err;       // vs cuBLAS sgemm fp32 reference
};

// If you launch the same GEMM against the same buffers 50 times, X and W go
// warm in L2 after the first launch (at 4096^3 fp16 that's 64 MB -- inside
// the 100.7 MB L2) and you measure cache, not HBM. Rotate through several
// physically distinct copies sized so the footprint clearly exceeds L2.
template <typename T>
struct RotatingBuffer {
    std::vector<T*> ptrs;
    int n = 1;

    void init(const std::vector<T>& host_data, size_t l2_bytes) {
        size_t buf_bytes = host_data.size() * sizeof(T);
        n = std::max(2, (int)std::ceil((4.0 * l2_bytes) / (double)buf_bytes));
        n = std::min(n, 16);
        ptrs.resize(n);
        for (int i = 0; i < n; i++) {
            CUDA_CHECK(cudaMalloc(&ptrs[i], buf_bytes));
            CUDA_CHECK(cudaMemcpy(ptrs[i], host_data.data(), buf_bytes, cudaMemcpyHostToDevice));
        }
    }
    T* get(int iter) const { return ptrs[iter % n]; }
    void free_all() { for (auto p : ptrs) cudaFree(p); }
};

float time_kernel(std::function<void(int)> launch, int iters = 50, const char* what = "?") {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    launch(0);  // warmup
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("FAULT in '%s' warmup: %s (last-launch: %s)\n",
               what, cudaGetErrorString(e), cudaGetErrorString(cudaGetLastError()));
        exit(1);
    }
    (void)0;
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++) launch(i);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / iters;
}

double relative_error(const std::vector<float>& a, const std::vector<float>& ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); i++) {
        num += (a[i] - ref[i]) * (a[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return std::sqrt(num / (den + 1e-30));
}

// Y[M,N] = X[M,K] @ W[N,K]^T  (all row-major, fp32) via cuBLAS (col-major).
void cublas_sgemm_ref(cublasHandle_t h, const float* X, const float* W, float* Y,
                      int M, int N, int K) {
    float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                             &alpha, W, K, X, K, &beta, Y, N));
}

// Y16[M,N] = X16 @ W16^T, fp16 in/out, fp32 compute (the shipping hgemm)
void cublas_hgemm(cublasHandle_t h, const __half* X, const __half* W, __half* Y,
                  int M, int N, int K) {
    float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                              &alpha, W, CUDA_R_16F, K, X, CUDA_R_16F, K,
                              &beta, Y, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));
}

// Per-row int8 quantization: q = round(x / s), s = max|x| / 127
void quantize_row_int8(const std::vector<float>& X, int rows, int K,
                       std::vector<int8_t>& q, std::vector<float>& s) {
    q.resize((size_t)rows * K);
    s.resize(rows);
    for (int r = 0; r < rows; r++) {
        float mx = 0.f;
        for (int k = 0; k < K; k++) mx = std::max(mx, std::fabs(X[(size_t)r * K + k]));
        float scale = (mx > 0.f) ? mx / 127.f : 1.f;
        s[r] = scale;
        for (int k = 0; k < K; k++) {
            float v = std::round(X[(size_t)r * K + k] / scale);
            if (v > 127.f) v = 127.f;
            if (v < -127.f) v = -127.f;
            q[(size_t)r * K + k] = (int8_t)v;
        }
    }
}

// shared-mem footprint for the cp.async / tensor-core kernels (double buffered)
int smem_bytes_v45(int ktile, int pad) {
    int asz = 64 * (ktile + pad), bsz = 32 * (ktile + pad);
    return 2 * (asz + bsz) * (int)sizeof(__half);
}
int smem_bytes_v6(int ktile, int pad) {
    int asz = 64 * (ktile + pad), bsz = 32 * (ktile + pad);
    return 2 * (asz + bsz) * (int)sizeof(int8_t);
}

double flops_for(int m, int n, int k) { return 2.0 * m * n * k; }
dim3 grid_gemv_for(int m, int n) { return dim3((m * (size_t)n + 7) / 8); }
dim3 grid_for(int m, int n, int m_tile) {
    return dim3((n + 31) / 32, (m + m_tile - 1) / m_tile);
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 4096;
    int N = (argc > 2) ? atoi(argv[2]) : 4096;
    int K = (argc > 3) ? atoi(argv[3]) : 4096;
    if (M <= 0 || N <= 0 || K <= 0 || K % 32 != 0) {
        fprintf(stderr, "dims must be positive and K a multiple of 32: "
                        "usage: %s [M] [N] [K]\n", argv[0]);
        return 1;
    }
    printf("Prefill hot path: Y = X @ W^T,  X is %d x %d, W is %d x %d "
           "(%.1f MB in fp16 each)\n\n", M, K, N, K,
           (double)std::max(M, N) * K * 2 / 1e6);

    // --- device info ---
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int mem_clock_khz = 0, bus_width_bits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
    double peak_gbps = 2.0 * mem_clock_khz * (bus_width_bits / 8) / 1e6;
    size_t l2_bytes = (size_t)prop.l2CacheSize;
    printf("Device: %s   Peak HBM BW: %.0f GB/s   L2 cache: %.1f MB\n\n",
           prop.name, peak_gbps, l2_bytes / 1e6);

    // --- data: X [M][K], W [N][K], typical LLM activation/weight scale ---
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.f, 0.02f);
    std::vector<float> X((size_t)M * K), W((size_t)N * K);
    for (auto& v : X) v = dist(rng);
    for (auto& v : W) v = dist(rng);
    std::vector<__half> X16(X.size()), W16(W.size());
    for (size_t i = 0; i < X.size(); i++) X16[i] = __float2half(X[i]);
    for (size_t i = 0; i < W.size(); i++) W16[i] = __float2half(W[i]);
    std::vector<int8_t> qX8, qW8;
    std::vector<float> sx, sw;
    quantize_row_int8(X, M, K, qX8, sx);
    quantize_row_int8(W, N, K, qW8, sw);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // --- reference: cuBLAS sgemm (fp32) on the primary copies ---
    float *d_X32_0, *d_W32_0, *d_Yref;
    CUDA_CHECK(cudaMalloc(&d_X32_0, X.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W32_0, W.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Yref, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_X32_0, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W32_0, W.data(), W.size() * sizeof(float), cudaMemcpyHostToDevice));
    cublas_sgemm_ref(handle, d_X32_0, d_W32_0, d_Yref, M, N, K);
    std::vector<float> Yref((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(Yref.data(), d_Yref, Yref.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- device buffers ---
    float *d_Y;
    CUDA_CHECK(cudaMalloc(&d_Y, (size_t)M * N * sizeof(float)));
    __half *d_Y16;
    CUDA_CHECK(cudaMalloc(&d_Y16, (size_t)M * N * sizeof(__half)));
    float *d_sx, *d_sw;
    CUDA_CHECK(cudaMalloc(&d_sx, sx.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sw, sw.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_sx, sx.data(), sx.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sw, sw.data(), sw.size() * sizeof(float), cudaMemcpyHostToDevice));

    RotatingBuffer<float> X32rb, W32rb;
    RotatingBuffer<__half> X16rb, W16rb;
    RotatingBuffer<int8_t> qX8rb, qW8rb;
    X32rb.init(X, l2_bytes);
    W32rb.init(W, l2_bytes);
    X16rb.init(X16, l2_bytes);
    W16rb.init(W16, l2_bytes);
    qX8rb.init(qX8, l2_bytes);
    qW8rb.init(qW8, l2_bytes);
    printf("  [ring] rotating over %d x %.1f MB copies (X+W) to defeat L2\n\n",
           X16rb.n, (X16.size() + W16.size()) * sizeof(__half) / 1e6);

    double flops = 2.0 * M * N * K;
    dim3 grid_v0(N / 16 + 1, M / 16 + 1), block_v0(16, 16);
    dim3 grid_tile32((N + 31) / 32, (M + 31) / 32), block_tile32(16, 16);
    dim3 grid_v345((N + 31) / 32, (M + 63) / 64), block_128(128);
    dim3 grid_gemv((M * (size_t)N + 7) / 8), block_gemv(256);

    int smem_v45 = smem_bytes_v45(32, 8);
    int smem_v6  = smem_bytes_v6(32, 16);
    CUDA_CHECK(cudaFuncSetAttribute(gemm_v4_cpasync_fp16,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_v45));
    CUDA_CHECK(cudaFuncSetAttribute(gemm_v5_tc_fp16,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_v45));
    CUDA_CHECK(cudaFuncSetAttribute(gemm_v6_tc_int8,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_v6));

    // cuBLAS hgemm = the measured peak for this shape; everything is % of it
    float ms_hgemm = time_kernel([&](int i) {
        cublas_hgemm(handle, X16rb.get(i), W16rb.get(i), d_Y16, M, N, K);
    }, 50, "hgemm-peak");
    double peak_tflops = flops / (ms_hgemm / 1e3) / 1e12;

    std::vector<Result> results;
    std::vector<float> y_out((size_t)M * N);

    auto pull_y = [&]() {
        CUDA_CHECK(cudaMemcpy(y_out.data(), d_Y, y_out.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    };
    auto add_result = [&](const std::string& name, float ms, double rel_err,
                          double speedup) {
        double tf = flops / (ms / 1e3) / 1e12;
        results.push_back({name, ms, tf, 100.0 * tf / peak_tflops, speedup, rel_err});
    };

    // ---------- v0: naive fp32 (10 iters -- it is intentionally slow) ----------
    {
        float ms = time_kernel([&](int i) {
            gemm_v0_naive_fp32<<<grid_v0, block_v0>>>(X32rb.get(i), W32rb.get(i), d_Y, M, N, K);
        }, 10, "v0");
        gemm_v0_naive_fp32<<<grid_v0, block_v0>>>(X32rb.get(0), W32rb.get(0), d_Y, M, N, K);
        pull_y();
        add_result("v0 naive fp32", ms, relative_error(y_out, Yref), 1.0);
    }

    // ---------- v1: fp32 smem tile ----------
    {
        float ms = time_kernel([&](int i) {
            gemm_v1_smem_fp32<<<grid_tile32, block_tile32>>>(X32rb.get(i), W32rb.get(i), d_Y, M, N, K);
        }, 50, "v1");
        gemm_v1_smem_fp32<<<grid_tile32, block_tile32>>>(X32rb.get(0), W32rb.get(0), d_Y, M, N, K);
        pull_y();
        add_result("v1 fp32 smem tile", ms, relative_error(y_out, Yref),
                   results[0].ms / ms);
    }

    // ---------- v2: fp16 smem tile ----------
    {
        float ms = time_kernel([&](int i) {
            gemm_v2_smem_fp16<<<grid_tile32, block_tile32>>>(X16rb.get(i), W16rb.get(i), d_Y, M, N, K);
        }, 50, "v2");
        gemm_v2_smem_fp16<<<grid_tile32, block_tile32>>>(X16rb.get(0), W16rb.get(0), d_Y, M, N, K);
        pull_y();
        add_result("v2 fp16 smem tile", ms, relative_error(y_out, Yref),
                   results[0].ms / ms);
    }

    // ---------- v3: fp16 reg-blocked 4x4 ----------
    {
        float ms = time_kernel([&](int i) {
            gemm_v3_regblock_fp16<<<grid_v345, block_128>>>(X16rb.get(i), W16rb.get(i), d_Y, M, N, K);
        }, 50, "v3");
        gemm_v3_regblock_fp16<<<grid_v345, block_128>>>(X16rb.get(0), W16rb.get(0), d_Y, M, N, K);
        pull_y();
        add_result("v3 fp16 reg-block 4x4", ms, relative_error(y_out, Yref),
                   results[0].ms / ms);
    }

    // ---------- v4: fp16 cp.async double buffer ----------
    {
        float ms = time_kernel([&](int i) {
            gemm_v4_cpasync_fp16<<<grid_v345, block_128, smem_v45>>>(X16rb.get(i), W16rb.get(i), d_Y, M, N, K);
        }, 50, "v4");
        gemm_v4_cpasync_fp16<<<grid_v345, block_128, smem_v45>>>(X16rb.get(0), W16rb.get(0), d_Y, M, N, K);
        pull_y();
        add_result("v4 fp16 cp.async dblbuf", ms, relative_error(y_out, Yref),
                   results[0].ms / ms);
    }

    // ---------- v5: fp16 tensor cores ----------
    {
        float ms = time_kernel([&](int i) {
            gemm_v5_tc_fp16<<<grid_v345, block_128, smem_v45>>>(X16rb.get(i), W16rb.get(i), d_Y, M, N, K);
        }, 50, "v5");
        gemm_v5_tc_fp16<<<grid_v345, block_128, smem_v45>>>(X16rb.get(0), W16rb.get(0), d_Y, M, N, K);
        pull_y();
        add_result("v5 fp16 tensor core", ms, relative_error(y_out, Yref),
                   results[0].ms / ms);
    }

    // ---------- v6: int8 tensor cores ----------
    {
        float ms = time_kernel([&](int i) {
            gemm_v6_tc_int8<<<grid_v345, block_128, smem_v6>>>(qX8rb.get(i), qW8rb.get(i), d_sx, d_sw, d_Y, M, N, K);
        }, 50, "v6");
        gemm_v6_tc_int8<<<grid_v345, block_128, smem_v6>>>(qX8rb.get(0), qW8rb.get(0), d_sx, d_sw, d_Y, M, N, K);
        pull_y();
        add_result("v6 int8 tensor core", ms, relative_error(y_out, Yref),
                   results[0].ms / ms);
    }

    // ---------- gemv-fp16: the decode-hotpath v2 kernel ----------
    {
        float ms = time_kernel([&](int i) {
            gemm_gemv_fp16<<<grid_gemv, block_gemv>>>(X16rb.get(i), W16rb.get(i), d_Y, M, N, K);
        }, 50, "gemv");
        gemm_gemv_fp16<<<grid_gemv, block_gemv>>>(X16rb.get(0), W16rb.get(0), d_Y, M, N, K);
        pull_y();
        add_result("gemv-fp16 (decode v2)", ms, relative_error(y_out, Yref),
                   results[0].ms / ms);
    }

    // ---------- cuBLAS sgemm (fp32 reference upper bound) ----------
    {
        float ms = time_kernel([&](int i) {
            cublas_sgemm_ref(handle, X32rb.get(i), W32rb.get(i), d_Y, M, N, K);
        }, 50, "sgemm");
        add_result("cuBLAS sgemm (fp32)", ms, 0.0, results[0].ms / ms);
    }

    // ---------- cuBLAS hgemm (the measured peak) ----------
    {
        std::vector<__half> y16((size_t)M * N);
        std::vector<float> y16f((size_t)M * N);
        cublas_hgemm(handle, X16rb.get(0), W16rb.get(0), d_Y16, M, N, K);
        CUDA_CHECK(cudaMemcpy(y16.data(), d_Y16, y16.size() * sizeof(__half),
                              cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < y16.size(); i++) y16f[i] = __half2float(y16[i]);
        double rel = relative_error(y16f, Yref);
        add_result("cuBLAS hgemm (peak)", ms_hgemm, rel, results[0].ms / ms_hgemm);
    }

    // ---------- print the ladder ----------
    printf("%-24s %9s %9s %9s %9s %12s\n", "kernel", "ms", "TFLOPs", "% of peak",
           "speedup", "rel.err vs fp32");
    for (auto& r : results)
        printf("%-24s %9.4f %9.1f %8.1f%% %8.2fx %12.4f%%\n",
               r.name.c_str(), r.ms, r.tflops, r.pct_peak, r.speedup, r.rel_err * 100);

    // ---------- self-tuning prompt-length sweep ----------
    {
        std::vector<int> Ms = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096};
        printf("\nPrompt-length sweep (N = %d, K = %d, fp16, self-tuning):\n", N, K);
        printf("%-7s %-18s %9s %9s %10s %11s\n", "M", "winner", "ms", "TFLOPs",
               "% of peak", "rel.err");

        for (int m : Ms) {
            if (m > M) continue;
            std::mt19937 srng(1000 + m);
            std::vector<float> Xs((size_t)m * K);
            std::vector<__half> X16s(Xs.size());
            for (auto& v : Xs) v = dist(srng);
            for (size_t i = 0; i < Xs.size(); i++) X16s[i] = __float2half(Xs[i]);

            float *dXs, *dYs;
            CUDA_CHECK(cudaMalloc(&dXs, Xs.size() * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dYs, (size_t)m * N * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(dXs, Xs.data(), Xs.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            cublas_sgemm_ref(handle, dXs, d_W32_0, dYs, m, N, K);

            RotatingBuffer<__half> Xsrb;
            Xsrb.init(X16s, l2_bytes);

            struct Cand { const char* name; std::function<void(int)> launch; };
            std::vector<Cand> cands;
            cands.push_back({"gemv-fp16", [&](int i) {
                gemm_gemv_fp16<<<grid_gemv_for(m, N), block_gemv>>>(Xsrb.get(i), W16rb.get(i), d_Y, m, N, K);
            }});
            cands.push_back({"v2 fp16 smem", [&](int i) {
                gemm_v2_smem_fp16<<<grid_for(m, N, 32), block_tile32>>>(Xsrb.get(i), W16rb.get(i), d_Y, m, N, K);
            }});
            cands.push_back({"v3 reg-block", [&](int i) {
                gemm_v3_regblock_fp16<<<grid_for(m, N, 64), block_128>>>(Xsrb.get(i), W16rb.get(i), d_Y, m, N, K);
            }});
            cands.push_back({"v4 cp.async", [&](int i) {
                gemm_v4_cpasync_fp16<<<grid_for(m, N, 64), block_128, smem_v45>>>(Xsrb.get(i), W16rb.get(i), d_Y, m, N, K);
            }});
            cands.push_back({"v5 tensor core", [&](int i) {
                gemm_v5_tc_fp16<<<grid_for(m, N, 64), block_128, smem_v45>>>(Xsrb.get(i), W16rb.get(i), d_Y, m, N, K);
            }});

            int best = 0;
            float best_probe = 1e30f;
            for (size_t i = 0; i < cands.size(); i++) {
                float p = time_kernel(cands[i].launch, 10);
                if (p < best_probe) { best_probe = p; best = (int)i; }
            }
            float ms = time_kernel(cands[best].launch, 50);

            // correctness of the winner: relaunch on ring 0 (both operands)
            cands[best].launch(0);
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<float> ychk((size_t)m * N), yref_m((size_t)m * N);
            CUDA_CHECK(cudaMemcpy(ychk.data(), d_Y, (size_t)m * N * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(yref_m.data(), dYs, (size_t)m * N * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            double rel = relative_error(ychk, yref_m);

            double tf = flops_for(m, N, K) / (ms / 1e3) / 1e12;
            printf("%-7d %-18s %9.4f %9.1f %9.1f%% %10.4f%%\n", m, cands[best].name,
                   ms, tf, 100.0 * tf / peak_tflops, rel * 100);

            Xsrb.free_all();
            cudaFree(dXs); cudaFree(dYs);
        }
    }

    // ---------- roofline summary ----------
    {
        double intensity = 2.0 * M * N * K / (2.0 * ((double)M * K + (double)N * K + (double)M * N));
        double knee = peak_tflops * 1e12 / (peak_gbps * 1e9);
        int m_cross = 1;
        while (m_cross < M) {
            double i_m = 2.0 * m_cross * N * K /
                         (2.0 * ((double)m_cross * K + (double)N * K + (double)m_cross * N));
            if (i_m >= knee) break;
            m_cross *= 2;
        }
        printf("\nRoofline (measured): peak %.0f GB/s HBM, %.0f TFLOP/s (cuBLAS hgemm) "
               "-> knee at %.0f FLOP/byte\n", peak_gbps, peak_tflops, knee);
        printf("At M = %d the layer is at %.0f FLOP/byte (%s bound); the crossover "
               "M* ~= %d.\n", M, intensity, intensity >= knee ? "compute" : "bandwidth", m_cross);
    }

    X32rb.free_all(); W32rb.free_all(); X16rb.free_all(); W16rb.free_all();
    qX8rb.free_all(); qW8rb.free_all();
    cudaFree(d_X32_0); cudaFree(d_W32_0); cudaFree(d_Yref);
    cudaFree(d_Y); cudaFree(d_Y16); cudaFree(d_sx); cudaFree(d_sw);
    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
