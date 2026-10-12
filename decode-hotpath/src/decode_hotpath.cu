// The LLM Decode Hot Path
//
// What this simulates: one linear layer during autoregressive decode --
// a single token's activation vector x [in_dim] multiplied against a
// weight matrix W [out_dim x in_dim] to produce y [out_dim]. This is
// THE operation that dominates single-user LLM inference: batch=1
// matrix-vector multiply, repeated once per layer per token. It is
// memory-bandwidth bound, not compute bound -- every byte of W has to
// come from HBM exactly once, and gets used in exactly one multiply.
// (In llama.cpp this is what mul_mat_vec / mul_mat_vec_q do.)
//
// Twelve kernels, each one optimization on top of the last:
//   v0  naive fp32            -- one thread per output row (bad coalescing)
//   v1  warp-per-row fp32     -- fixes coalescing, same bytes moved
//   v2  warp-per-row fp16     -- half the bytes moved
//   v3  warp-per-row int8     -- per-row scale, quarter the bytes moved
//   v4  warp-per-row int4     -- grouped scale (group=32, GGUF-Q4-style),
//                                 eighth the bytes moved
//   v5  block-per-row fp16    -- 256 threads/row + shared-mem x staging
//   v6  warp-per-row int8     -- hardware DP4A, both W and x quantized
//   v7  warp-per-row int4     -- 4-way unrolled, 4 independent accumulators
//   v8  warp-per-row int4     -- vectorized: uint32_t word loads (8 cols at
//                                 once), float4 x loads, single shift-based
//                                 group-index lookup instead of two divisions
//   v11 reg-blocked fp16       -- batched decode, B <= 16: B fp32 accumulators
//                                  per lane in registers (B a template
//                                  constant), each weight feeds all B requests
//   v12 lane-owned fp16        -- batched decode, B >= 32: each lane computes
//                                  complete dot products for the B/32 entries
//                                  it owns, sweeping all columns (no smem,
//                                  no reductions)
//   v13 smem-tiled fp16        -- batched decode, B >= 32: 8-row output tiles
//                                  (one warp per row), 64-col k tiles, W tile
//                                  fetched from HBM once per k tile
//
// The batched sweep is self-tuning: for B >= 8 the harness probes every
// applicable design (10 iterations each) and times the winner (50
// iterations), so the table shows which structure wins on YOUR GPU.
//
// Watch v4 vs v7 vs v8 closely: if v7 (more ILP, same work) doesn't beat v4,
// that's the tell that the kernel is compute-bound on instruction count, not
// latency-bound -- and the fix is v8's "do less total work per element," not
// more independent accumulators.
//
// Each step is a real, measurable win. v0->v1 is "fix the bug," v1-v4 are
// "trade accuracy for bandwidth" -- the same trade-off quantization-aware
// training makes, here applied to inference kernels instead of training.
// v5-v7 are a second axis entirely: same bytes moved as v2/v3/v4, but
// restructured to hide latency and use hardware instructions better -- the
// "everything's already bandwidth-bound, now squeeze compute" phase.
//
// Build:  make                     (or: nvcc -O3 -arch=sm_120 -o bin/decode_hotpath src/decode_hotpath.cu)
// Run:    ./bin/decode_hotpath [out_dim] [in_dim]     (defaults 4096 x 4096)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <functional>
#include <cuda_fp16.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                \
            exit(1);                                                        \
        }                                                                    \
    } while (0)

constexpr int GROUP_SIZE = 32;  // matches GGUF Q4_K-style grouping

// ============================================================
// Kernels
// ============================================================

// v0: one thread per output row. Thread i and thread i+1 read
// W[i*in_dim + col] and W[(i+1)*in_dim + col] -- addresses in_dim*4
// bytes apart. Zero coalescing. This is the mistake everyone makes first.
__global__ void matvec_v0_naive_fp32(const float* __restrict__ W,
                                      const float* __restrict__ x,
                                      float* __restrict__ y,
                                      int out_dim, int in_dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;
    float sum = 0.f;
    const float* row_ptr = W + (size_t)row * in_dim;
    for (int col = 0; col < in_dim; col++) sum += row_ptr[col] * x[col];
    y[row] = sum;
}

// v1: one WARP per output row. All 32 lanes read consecutive columns of
// the SAME row -> fully coalesced 128-byte transactions. Same bytes
// moved as v0, but now the hardware can actually deliver them.
__global__ void matvec_v1_warp_fp32(const float* __restrict__ W,
                                     const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int out_dim, int in_dim) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= out_dim) return;
    const float* row_ptr = W + (size_t)warp_id * in_dim;
    float sum = 0.f;
    for (int col = lane; col < in_dim; col += 32) sum += row_ptr[col] * x[col];
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) y[warp_id] = sum;
}

// v2: same warp-per-row pattern, weights stored as fp16 -> half the
// bytes read from HBM per element. x stays fp32 (it's tiny, doesn't matter).
__global__ void matvec_v2_warp_fp16(const __half* __restrict__ W,
                                     const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int out_dim, int in_dim) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= out_dim) return;
    const __half* row_ptr = W + (size_t)warp_id * in_dim;
    float sum = 0.f;
    for (int col = lane; col < in_dim; col += 32)
        sum += __half2float(row_ptr[col]) * x[col];
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) y[warp_id] = sum;
}

// v3: int8 weights, ONE scale per row (per-channel quantization -- same
// idea as per-channel QAT scales, applied at inference time here).
// dequant is `(float)qval * scale`, done in-register, never materialized.
__global__ void matvec_v3_warp_int8(const int8_t* __restrict__ W,
                                     const float* __restrict__ row_scale,
                                     const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int out_dim, int in_dim) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= out_dim) return;
    const int8_t* row_ptr = W + (size_t)warp_id * in_dim;
    float scale = row_scale[warp_id];
    float sum = 0.f;
    for (int col = lane; col < in_dim; col += 32)
        sum += (float)row_ptr[col] * scale * x[col];
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) y[warp_id] = sum;
}

// v4: int4 weights, packed 2-per-byte, one scale per GROUP_SIZE-element
// group (not per row) -- coarser than int8's per-row scale would be too
// lossy at 4 bits, so GGUF-style formats scale in small groups instead.
// Each lane reads one byte -> decodes 2 output columns -> 2 FMAs per byte
// read. Byte reads across lanes are still consecutive -> still coalesced.
__global__ void matvec_v4_warp_int4(const uint8_t* __restrict__ W_packed,
                                     const float* __restrict__ group_scale,
                                     const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int out_dim, int in_dim) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= out_dim) return;

    int bytes_per_row = in_dim / 2;
    int groups_per_row = in_dim / GROUP_SIZE;
    const uint8_t* row_bytes = W_packed + (size_t)warp_id * bytes_per_row;
    const float* row_scales = group_scale + (size_t)warp_id * groups_per_row;

    float sum = 0.f;
    for (int b = lane; b < bytes_per_row; b += 32) {
        uint8_t byte = row_bytes[b];
        int c0 = b * 2, c1 = b * 2 + 1;
        // nibbles stored as unsigned 0..15, dequant back to signed -8..7
        int lo = (int)(byte & 0x0F) - 8;
        int hi = (int)((byte >> 4) & 0x0F) - 8;
        sum += lo * row_scales[c0 / GROUP_SIZE] * x[c0];
        sum += hi * row_scales[c1 / GROUP_SIZE] * x[c1];
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) y[warp_id] = sum;
}

// v5: one BLOCK per row (256 threads) + shared memory staging for x, fp16 weights.
// Note on the "why": x is only in_dim floats (tens of KB) and would already sit
// comfortably in L1/read-only cache under v2's warp-per-row scheme -- so the
// shared-memory copy itself isn't the main lever here. The bigger effect is
// that a block gets 8x more threads (256 vs. 32) cooperating on ONE row, which
// means 8x more concurrent global-memory requests for W in flight at once,
// which is what actually hides HBM latency better. Staging x in shared memory
// is a real but secondary win: it guarantees every warp in the block reads x
// from a single on-chip, conflict-free location instead of each warp separately
// hitting the cache hierarchy for the same bytes.
__global__ void matvec_v5_block_smem_fp16(const __half* __restrict__ W,
                                           const float* __restrict__ x,
                                           float* __restrict__ y,
                                           int out_dim, int in_dim) {
    extern __shared__ float smem_x[];  // size = in_dim * sizeof(float), set at launch

    for (int i = threadIdx.x; i < in_dim; i += blockDim.x) smem_x[i] = x[i];
    __syncthreads();

    int row = blockIdx.x;  // one block per row
    if (row >= out_dim) return;

    const __half* row_ptr = W + (size_t)row * in_dim;
    float sum = 0.f;
    for (int col = threadIdx.x; col < in_dim; col += blockDim.x)
        sum += __half2float(row_ptr[col]) * smem_x[col];

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);

    __shared__ float warp_sums[32];  // supports up to 1024 threads/block
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
        if (lane == 0) y[row] = sum;
    }
}

// v6: int8 x int8 via the hardware DP4A instruction (needs sm_61+, fine on the
// sm_80 build target used elsewhere in this file). DP4A does 4 int8x int8
// multiplies + accumulate into an int32 in one instruction, replacing what
// would otherwise be ~4 separate mul+add sequences per group of 4 elements.
// This is the first kernel where BOTH operands are quantized -- v3 only
// quantized W and kept x in fp32. Quantizing the tiny activation vector too
// barely helps bandwidth (x is small either way) but it's what unlocks DP4A,
// since the instruction requires both operands to be int8.
__global__ void matvec_v6_dp4a_int8(const int8_t* __restrict__ W,
                                     const int8_t* __restrict__ x_q,
                                     const float* __restrict__ row_scale,
                                     const float x_scale,
                                     float* __restrict__ y,
                                     int out_dim, int in_dim) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= out_dim) return;

    // in_dim is always a multiple of 64 in this file (see GROUP_SIZE check in
    // main), so in_dim/4 is exact and these 4-byte-aligned reinterpret casts
    // are safe -- cudaMalloc'd buffers are always sufficiently aligned.
    const int32_t* W_row = (const int32_t*)(W + (size_t)warp_id * in_dim);
    const int32_t* x_vec = (const int32_t*)x_q;

    int sum_int = 0;
    for (int i = lane; i < in_dim / 4; i += 32)
        sum_int = __dp4a(W_row[i], x_vec[i], sum_int);

    for (int off = 16; off > 0; off >>= 1) sum_int += __shfl_down_sync(0xffffffff, sum_int, off);

    if (lane == 0) y[warp_id] = (float)sum_int * row_scale[warp_id] * x_scale;
}

// v7: same int4 grouped layout as v4, but with 4-way manual loop unrolling and
// 4 independent accumulators. A single running `sum` forces each iteration's
// add to wait on the previous one (serial dependency chain); 4 separate
// accumulators give the scheduler 4 independent chains it can interleave,
// hiding the latency of the shift/mask/convert sequence behind other work.
// NOTE: the original version of this kernel only filled in 2 of the 4
// unrolled iterations ("omitted for brevity") -- with a stride of 128 bytes
// but only 2 of 4 sub-offsets actually read, it would have silently skipped
// half the row's data and produced wrong output. All 4 are implemented below.
__global__ void matvec_v7_ilp_int4(const uint8_t* __restrict__ W_packed,
                                    const float* __restrict__ group_scale,
                                    const float* __restrict__ x,
                                    float* __restrict__ y,
                                    int out_dim, int in_dim) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= out_dim) return;

    int bytes_per_row = in_dim / 2;
    int groups_per_row = in_dim / GROUP_SIZE;
    const uint8_t* row_bytes = W_packed + (size_t)warp_id * bytes_per_row;
    const float* row_scales = group_scale + (size_t)warp_id * groups_per_row;

    auto decode_add = [&](float& acc, int b) {
        uint8_t byte = row_bytes[b];
        int c0 = b * 2, c1 = b * 2 + 1;
        acc += ((int)(byte & 0x0F) - 8) * row_scales[c0 / GROUP_SIZE] * x[c0];
        acc += ((int)((byte >> 4) & 0x0F) - 8) * row_scales[c1 / GROUP_SIZE] * x[c1];
    };

    float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f, sum3 = 0.f;
    for (int b = lane; b < bytes_per_row; b += 128) {
        decode_add(sum0, b);
        if (b + 32 < bytes_per_row) decode_add(sum1, b + 32);
        if (b + 64 < bytes_per_row) decode_add(sum2, b + 64);
        if (b + 96 < bytes_per_row) decode_add(sum3, b + 96);
    }

    float sum = sum0 + sum1 + sum2 + sum3;
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) y[warp_id] = sum;
}

// v8: fixes what v4/v7 actually cost time on. v7 showed ILP gave ~zero
// improvement over v4 -- that's the signal that v4/v7 aren't latency-bound
// (waiting on memory), they're INSTRUCTION-bound: byte-at-a-time reads only
// use 1/4 of a 32-bit load's payload, and computing `col / GROUP_SIZE` twice
// per byte (once per nibble, even though both nibbles are always in the same
// group) forces the compiler into real division sequences since `col` is a
// signed int it can't prove is non-negative.
//
// Fix: read 4 packed bytes (8 columns) per lane per iteration via a uint32_t
// load -- full 128-byte coalesced transactions per warp instead of 32-byte
// ones -- compute the group index ONCE per word using an unsigned shift, and
// pull the matching 8 x-values with two float4 loads instead of 8 scalar ones.
__global__ void matvec_v8_vectorized_int4(const uint8_t* __restrict__ W_packed,
                                           const float* __restrict__ group_scale,
                                           const float* __restrict__ x,
                                           float* __restrict__ y,
                                           int out_dim, int in_dim) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= out_dim) return;

    int bytes_per_row = in_dim / 2;
    unsigned words_per_row = bytes_per_row / 4;  // 1 word = 4 bytes = 8 columns
    // GROUP_SIZE(32) elements / 2 per byte / 4 bytes per word = 4 words/group,
    // so group index = w / 4 -- always exact since a word's 8 columns never
    // straddle a 32-element group boundary (verified: 32 is a multiple of 8).
    const uint32_t* row_words = reinterpret_cast<const uint32_t*>(
        W_packed + (size_t)warp_id * bytes_per_row);
    const float* row_scales = group_scale + (size_t)warp_id * (in_dim / GROUP_SIZE);

    float sum = 0.f;
    for (unsigned w = lane; w < words_per_row; w += 32) {
        uint32_t word = row_words[w];
        float gscale = row_scales[w >> 2];  // single load, shared by all 8 columns
        int col_base = (int)w * 8;

        float4 xa = *reinterpret_cast<const float4*>(x + col_base);
        float4 xb = *reinterpret_cast<const float4*>(x + col_base + 4);

        int n0 = (int)(word & 0xF) - 8,        n1 = (int)((word >> 4) & 0xF) - 8;
        int n2 = (int)((word >> 8) & 0xF) - 8, n3 = (int)((word >> 12) & 0xF) - 8;
        int n4 = (int)((word >> 16) & 0xF) - 8, n5 = (int)((word >> 20) & 0xF) - 8;
        int n6 = (int)((word >> 24) & 0xF) - 8, n7 = (int)((word >> 28) & 0xF) - 8;

        sum += gscale * (n0 * xa.x + n1 * xa.y + n2 * xa.z + n3 * xa.w +
                          n4 * xb.x + n5 * xb.y + n6 * xb.z + n7 * xb.w);
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) y[warp_id] = sum;
}

// v11: register-blocked batched decode, small B (B <= 4).
//
// Every kernel above computes y = W @ x for ONE token (B=1): every byte of W
// is read from HBM, used in exactly one multiply, and discarded. Batching B
// simultaneous requests amortizes that cost -- same bytes read from HBM, B
// times more FMAs done with them.
//
// One warp per output row. Each lane strides the columns and keeps B fp32
// accumulators (one per batch entry) IN REGISTERS: each weight value is
// loaded once and multiplied into all B accumulators. This is the "register
// blocking" microkernel a real GEMM uses, applied along the batch axis
// instead of a second matrix dimension.
//
// B is a template constant for a reason: with a runtime B, the compiler must
// allocate sum[] in LOCAL memory (it can't prove a register array fits), and
// every FMA then becomes a local load + local store through L1 -- the
// kernel collapses as B grows (this is exactly how an earlier version of
// this experiment died at B >= 8, with the classic register-spilling
// fingerprint: time growing super-linearly in the work). A fixed B lets the
// compiler keep everything in registers and fully unroll the inner loop.
//
// This design wins up to B ~ 4. Beyond that, every output row must read the
// whole batched activation vector (in_dim x B floats) from L2 -- 512 MB per
// launch at B=8 on the 4096x4096 shape -- and the x traffic, not the weight
// traffic or the FLOPs, becomes the bottleneck. Watch the sweep: throughput
// is still B x B=1 at B=2, but the %peak column already sagging at B=4 tells
// you where the wall is. For B >= 32, v12/v13 restructure the work (they are
// still x-traffic and instruction-issue limited -- see their comments).
//
// x is laid out [in_dim][B] (not [B][in_dim]) so that for a fixed column,
// the B values a lane needs are contiguous floats.

template <int B>
__global__ void matvec_v11a_regblock_fp16(const __half* __restrict__ W,
                                          const float* __restrict__ x_batched,  // [in_dim][B]
                                          float* __restrict__ y_batched,        // [out_dim][B]
                                          int out_dim, int in_dim) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= out_dim) return;

    const __half* row_ptr = W + (size_t)warp_id * in_dim;
    float sum[B];
    #pragma unroll
    for (int b = 0; b < B; b++) sum[b] = 0.f;

    for (int col = lane; col < in_dim; col += 32) {
        float w_val = __half2float(row_ptr[col]);  // read once, reused B times
        const float* xcol = x_batched + (size_t)col * B;
        #pragma unroll
        for (int b = 0; b < B; b++) sum[b] += w_val * xcol[b];
    }

    #pragma unroll
    for (int b = 0; b < B; b++) {
        float s = sum[b];
        for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffff, s, off);
        if (lane == 0) y_batched[(size_t)warp_id * B + b] = s;
    }
}

// v12: lane-owned batch lanes, large B (B >= 32).
//
// Why a new kernel at all: v11's register-blocked design hits a wall at
// B >= 8 that is NOT register spilling (ptxas reports zero spills at B=16)
// -- it is ACTIVATION traffic. Every output row must read the whole batched
// activation vector (in_dim x B floats) from L2: 512 MB per launch at B=8,
// 1 GB at B=16 on the 4096x4096 shape. B >= 32 on top of that would also
// need 32+ accumulators per lane.
//
// The restructure: each lane owns a FIXED slice of the batch -- lane l
// computes the complete dot product for entries b = l, l+32, l+64, ... -- so
// it needs only B/32 accumulators (two for B=64) and, because every lane
// sweeps ALL columns, no cross-lane reduction at all: lane l alone produces
// y[row, l], y[row, l+32], ...
//
// Consequences:
//  - W HBM traffic is unchanged by B (all 32 lanes load the same W element
//    per column -- an L1 broadcast, one real fetch per column per row).
//  - x reads stay fully coalesced (for a fixed column the 32 lanes touch 32
//    consecutive floats) and x stays L2-resident.
//  - The cost: one global load + address computation per column per lane
//    (in_dim of them), which starts to show as instruction-issue pressure
//    at large B -- that is v13's motivation.
//  - Lanes whose entries are all >= B own nothing and exit early; for
//    B < 32 that leaves idle lanes, which is why B <= 16 uses v11.

__global__ void matvec_v12_laneowned_fp16(const __half* __restrict__ W,
                                          const float* __restrict__ x_batched,  // [in_dim][B]
                                          float* __restrict__ y_batched,        // [out_dim][B]
                                          int out_dim, int in_dim, int B) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= out_dim) return;

    // Batch entries owned by this lane: lane, lane+32, lane+64, ...
    // For B in [32, 96] that is 1..3 entries; lanes with lane >= B own none.
    int n_owned = (B - lane + 31) / 32;
    if (n_owned <= 0) return;

    const __half* row_ptr = W + (size_t)warp_id * in_dim;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f;
    for (int col = 0; col < in_dim; col++) {
        float w_val = __half2float(row_ptr[col]);
        const float* xcol = x_batched + (size_t)col * B;
        acc0 += w_val * xcol[lane];
        if (n_owned >= 2) acc1 += w_val * xcol[lane + 32];
        if (n_owned >= 3) acc2 += w_val * xcol[lane + 64];
    }

    size_t base = (size_t)warp_id * B + lane;
    y_batched[base] = acc0;
    if (n_owned >= 2) y_batched[base + 32] = acc1;
    if (n_owned >= 3) y_batched[base + 64] = acc2;
}

// v13: shared-memory-tiled batched decode, B >= 32 (B is a template constant
// so the compiler can keep the accumulators in registers).
//
// v12 pays one global load + address computation per column per lane
// (in_dim of them). v13 is the standard GEMM tiling pattern applied to
// decode:
//   - A block (256 threads, 8 warps) owns an 8-row output tile; one warp
//     per row, so all 32 lanes of the warp work on the same row.
//   - The inner dimension is walked in K tiles of 64 columns. For each k
//     tile the 8x64 W tile is loaded from HBM once into shared memory, and
//     the 64 x 64 x tile is loaded from L2 (x is small, stays L2-resident)
//     into shared memory.
//   - Within a k tile each lane sweeps all 64 columns for the two batch
//     entries it owns (b = 2*lane, 2*lane+1 within the b sub-tile): 32
//     lanes x 2 entries cover the 64-wide b sub-tile exactly, every (row,
//     b) output is produced by exactly one lane, and no cross-lane
//     reduction is needed.
//   - For B <= 64 that single b sub-tile IS the whole batch, so W is
//     fetched from HBM exactly once, regardless of B. (B = 128 needs two
//     sub-tiles and re-fetches W twice.)
//
// What v13 still cannot fix: the per-row x traffic (in_dim x B floats per
// row, now served from smem but loaded from L2 in_dim/64 times per row) and
// the instruction cost of ~2 smem loads + 2 FMAs per (column, entry) pair.
// Tensor cores are the escape: one m16n8k16 MMA does 2048 fp16 FMA per
// instruction and reuses each x-tile row across 16 output rows at once --
// which is what cuBLAS / vLLM actually run at large B.

template <int B>
__global__ void matvec_v13_smem_tile_fp16(const __half* __restrict__ W,
                                          const float* __restrict__ x_batched,  // [in_dim][B]
                                          float* __restrict__ y_batched,        // [out_dim][B]
                                          int out_dim, int in_dim) {
    constexpr int OUT_TILE = 8;   // output rows per block (one warp each)
    constexpr int K_TILE   = 64;  // columns per k tile
    constexpr int B_TILE   = 64;  // batch entries per b sub-tile (smem cap)
    constexpr int N_B_TILES = (B + B_TILE - 1) / B_TILE;

    __shared__ __half smem_W[OUT_TILE][K_TILE];
    __shared__ float  smem_x[K_TILE][B_TILE];

    int row_base = blockIdx.x * OUT_TILE;
    int tid      = threadIdx.x;
    int row      = tid / 32;      // 8 rows, one warp each
    int lane     = tid % 32;
    int b_base   = lane * 2;      // this lane's entries within a b sub-tile

    for (int bt = 0; bt < N_B_TILES; bt++) {
        int b0 = bt * B_TILE;
        int n_b = (B - b0 < B_TILE) ? (B - b0) : B_TILE;
        float acc0 = 0.f, acc1 = 0.f;

        for (int k0 = 0; k0 < in_dim; k0 += K_TILE) {
            // 1. W tile from HBM -- fetched once per (b sub-tile, k tile).
            //    8 x 64 halves, 256 threads -> 2 per thread.
            for (int i = tid; i < OUT_TILE * K_TILE; i += blockDim.x) {
                int r = i / K_TILE, c = i % K_TILE;
                int grow = row_base + r;
                smem_W[r][c] = (grow < out_dim) ? W[(size_t)grow * in_dim + k0 + c] : __half(0.f);
            }
            // 2. x tile from L2: 64 x 64 floats, 256 threads -> 16 per thread.
            for (int i = tid; i < K_TILE * B_TILE; i += blockDim.x) {
                int k_off = i / B_TILE, b_off = i % B_TILE;
                smem_x[k_off][b_off] = (b_off < n_b)
                    ? x_batched[(size_t)(k0 + k_off) * B + b0 + b_off]
                    : 0.f;
            }
            __syncthreads();

            // 3. This lane computes its two entries over all 64 columns.
            if (b_base < n_b) {
                #pragma unroll
                for (int k_off = 0; k_off < K_TILE; k_off++) {
                    float w = __half2float(smem_W[row][k_off]);
                    acc0 += w * smem_x[k_off][b_base];
                    if (b_base + 1 < n_b) acc1 += w * smem_x[k_off][b_base + 1];
                }
            }
            __syncthreads();
        }

        // 4. Epilogue: this lane owns y[row, b0 + b_base + {0, 1}].
        if (row_base + row < out_dim) {
            size_t base = (size_t)(row_base + row) * B + b0 + b_base;
            if (b_base < n_b)     y_batched[base]     = acc0;
            if (b_base + 1 < n_b) y_batched[base + 1] = acc1;
        }
    }
}





// Per-tensor int8 quantization for a single vector (used to quantize x for v6)
void quantize_int8_tensor(const std::vector<float>& v, std::vector<int8_t>& q, float& scale) {
    float max_abs = 1e-8f;
    for (float f : v) max_abs = std::max(max_abs, std::fabs(f));
    scale = max_abs / 127.0f;
    q.resize(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        int qi = (int)std::lround(v[i] / scale);
        qi = std::max(-127, std::min(127, qi));
        q[i] = (int8_t)qi;
    }
}

// Per-row int8 quantization: scale = max(|row|) / 127
void quantize_int8_per_row(const std::vector<float>& W, int out_dim, int in_dim,
                            std::vector<int8_t>& qW, std::vector<float>& scale) {
    qW.resize((size_t)out_dim * in_dim);
    scale.resize(out_dim);
    for (int r = 0; r < out_dim; r++) {
        float max_abs = 1e-8f;
        for (int c = 0; c < in_dim; c++)
            max_abs = std::max(max_abs, std::fabs(W[(size_t)r * in_dim + c]));
        float s = max_abs / 127.0f;
        scale[r] = s;
        for (int c = 0; c < in_dim; c++) {
            int q = (int)std::lround(W[(size_t)r * in_dim + c] / s);
            q = std::max(-127, std::min(127, q));
            qW[(size_t)r * in_dim + c] = (int8_t)q;
        }
    }
}


// Grouped int4 quantization: scale = max(|group|) / 8, group = GROUP_SIZE elems
void quantize_int4_grouped(const std::vector<float>& W, int out_dim, int in_dim,
                            std::vector<uint8_t>& packed, std::vector<float>& group_scale) {
    int groups_per_row = in_dim / GROUP_SIZE;
    packed.resize((size_t)out_dim * in_dim / 2);
    group_scale.resize((size_t)out_dim * groups_per_row);

    for (int r = 0; r < out_dim; r++) {
        for (int g = 0; g < groups_per_row; g++) {
            float max_abs = 1e-8f;
            for (int i = 0; i < GROUP_SIZE; i++) {
                int c = g * GROUP_SIZE + i;
                max_abs = std::max(max_abs, std::fabs(W[(size_t)r * in_dim + c]));
            }
            float s = max_abs / 8.0f;
            group_scale[(size_t)r * groups_per_row + g] = s;
        }
        for (int c = 0; c < in_dim; c += 2) {
            auto quant_nibble = [&](int col) -> uint8_t {
                int g = col / GROUP_SIZE;
                float s = group_scale[(size_t)r * groups_per_row + g];
                int q = (int)std::lround(W[(size_t)r * in_dim + col] / s);
                q = std::max(-8, std::min(7, q));
                return (uint8_t)(q + 8);  // store as unsigned 0..15
            };
            uint8_t lo = quant_nibble(c);
            uint8_t hi = quant_nibble(c + 1);
            packed[(size_t)r * (in_dim / 2) + c / 2] = lo | (hi << 4);
        }
    }
}

// ============================================================
// Benchmark harness
// ============================================================

struct Result {
    std::string name;
    float ms;
    double gb_moved;
    bool correct;
    double rel_err_vs_fp32;  // accuracy cost relative to the true fp32 output
};

struct BatchResult {
    int B;
    std::string kernel;       // which design won for this B
    float ms;
    double gbps;              // weight bytes moved / time -- constant-ish across B
    double throughput;        // B / time -- "requests decoded per second"
    double pct_peak;
    double rel_err_vs_fp32;
};

// IMPORTANT: if you launch the same kernel against the same buffer 50 times
// in a row with nothing else touching the GPU in between, the weight
// matrix goes warm in L2 after the first launch and every subsequent
// launch is served from cache, not HBM -- you end up "measuring" numbers
// that can exceed 100% of theoretical HBM peak bandwidth, which is a sign
// something's wrong, not a sign your kernel is unreasonably fast.
//
// Fix: allocate several physically distinct copies of each buffer and
// rotate through them so every timed launch reads an address range that
// wasn't just touched, sized so the combined footprint clearly exceeds L2.
template <typename T>
struct RotatingBuffer {
    std::vector<T*> ptrs;
    int n = 1;

    void init(const std::vector<T>& host_data, size_t l2_bytes) {
        size_t buf_bytes = host_data.size() * sizeof(T);
        // aim for total footprint >= 4x L2 so cache can't hold the working set
        n = std::max(2, (int)std::ceil((4.0 * l2_bytes) / (double)buf_bytes));
        n = std::min(n, 32);  // cap so we don't blow through device memory
        ptrs.resize(n);
        for (int i = 0; i < n; i++) {
            CUDA_CHECK(cudaMalloc(&ptrs[i], buf_bytes));
            CUDA_CHECK(cudaMemcpy(ptrs[i], host_data.data(), buf_bytes, cudaMemcpyHostToDevice));
        }
    }
    T* get(int iter) const { return ptrs[iter % n]; }
    void free_all() { for (auto p : ptrs) cudaFree(p); }
};

// launch takes the iteration index so it can pick a rotating-buffer copy
float time_kernel(std::function<void(int)> launch, int iters = 50) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    launch(0);  // warmup
    CUDA_CHECK(cudaDeviceSynchronize());
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
    return std::sqrt(num / den);
}

// Self-tuning batched decode: probes each applicable design (10 iterations),
// then times the winner (50 iterations). v11 is only a candidate for B <= 16
// (beyond that, its per-lane B-accumulator array starts spilling).
template <int B>
static float time_best_batched(const RotatingBuffer<__half>& W16rb, float* d_xb, float* d_yb,
                               int out_dim, int in_dim, int blocks, int blocks_v13, int threads,
                               std::string& kernel_name) {
    std::vector<std::string> names;
    std::vector<std::function<void(int)>> launches;
    if constexpr (B <= 16) {
        names.push_back("v11 reg-block");
        launches.push_back([&](int i) {
            matvec_v11a_regblock_fp16<B><<<blocks, threads>>>(W16rb.get(i), d_xb, d_yb, out_dim, in_dim);
        });
    }
    names.push_back("v12 lane-owned");
    launches.push_back([&](int i) {
        matvec_v12_laneowned_fp16<<<blocks, threads>>>(W16rb.get(i), d_xb, d_yb, out_dim, in_dim, B);
    });
    names.push_back("v13 smem-tile");
    launches.push_back([&](int i) {
        matvec_v13_smem_tile_fp16<B><<<blocks_v13, threads>>>(W16rb.get(i), d_xb, d_yb, out_dim, in_dim);
    });

    int best = 0;
    float best_probe = 1e30f;
    for (size_t i = 0; i < launches.size(); i++) {
        float p = time_kernel(launches[i], 10);
        if (p < best_probe) { best_probe = p; best = (int)i; }
    }
    kernel_name = names[best];
    return time_kernel(launches[best], 50);
}

// Runs the batched decode kernels for a fixed batch size B: generates B distinct request vectors
// (not B copies of the same vector -- a real batch has different requests),
// computes an independent fp32 reference for each, runs the optimized batched kernel
// once, and reports both raw bandwidth and effective request throughput.
BatchResult run_batched_decode(int out_dim, int in_dim, const std::vector<float>& W,
                             size_t l2_bytes, int threads, double peak_gbps, int B) {
    std::vector<__half> W16(W.size());
    for (size_t i = 0; i < W.size(); i++) W16[i] = __float2half(W[i]);
    RotatingBuffer<__half> W16rb;
    W16rb.init(W16, l2_bytes);

    std::mt19937 rng(1000 + B);
    std::normal_distribution<float> dist(0.f, 0.02f);
    std::vector<float> x_rows((size_t)B * in_dim);  // [B][in_dim], B distinct requests
    for (auto& v : x_rows) v = dist(rng);

    // device wants [in_dim][B] so each lane's B values are contiguous
    std::vector<float> x_dev((size_t)in_dim * B);
    for (int b = 0; b < B; b++)
        for (int c = 0; c < in_dim; c++)
            x_dev[(size_t)c * B + b] = x_rows[(size_t)b * in_dim + c];

    std::vector<float> y_true((size_t)out_dim * B);
    for (int r = 0; r < out_dim; r++) {
        for (int b = 0; b < B; b++) {
            double s = 0;
            for (int c = 0; c < in_dim; c++)
                s += (double)W[(size_t)r * in_dim + c] * x_rows[(size_t)b * in_dim + c];
            y_true[(size_t)r * B + b] = (float)s;
        }
    }

    float *d_xb, *d_yb;
    CUDA_CHECK(cudaMalloc(&d_xb, x_dev.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_yb, y_true.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_xb, x_dev.data(), x_dev.size() * sizeof(float), cudaMemcpyHostToDevice));

    int blocks = (out_dim * 32 + threads - 1) / threads;   // v11/v12: one warp per row
    int blocks_v13 = (out_dim + 7) / 8;                    // v13: 8 rows per block
    float ms = 0.f;
    std::string kernel_name;
    switch (B) {
        case 1:  kernel_name = "v11 reg-block";
                 ms = time_kernel([&](int i){ matvec_v11a_regblock_fp16<1><<<blocks, threads>>>(W16rb.get(i), d_xb, d_yb, out_dim, in_dim); }); break;
        case 2:  kernel_name = "v11 reg-block";
                 ms = time_kernel([&](int i){ matvec_v11a_regblock_fp16<2><<<blocks, threads>>>(W16rb.get(i), d_xb, d_yb, out_dim, in_dim); }); break;
        case 4:  kernel_name = "v11 reg-block";
                 ms = time_kernel([&](int i){ matvec_v11a_regblock_fp16<4><<<blocks, threads>>>(W16rb.get(i), d_xb, d_yb, out_dim, in_dim); }); break;
        case 8:  ms = time_best_batched<8>(W16rb, d_xb, d_yb, out_dim, in_dim, blocks, blocks_v13, threads, kernel_name); break;
        case 16: ms = time_best_batched<16>(W16rb, d_xb, d_yb, out_dim, in_dim, blocks, blocks_v13, threads, kernel_name); break;
        case 32: ms = time_best_batched<32>(W16rb, d_xb, d_yb, out_dim, in_dim, blocks, blocks_v13, threads, kernel_name); break;
        case 64: ms = time_best_batched<64>(W16rb, d_xb, d_yb, out_dim, in_dim, blocks, blocks_v13, threads, kernel_name); break;
        case 128: ms = time_best_batched<128>(W16rb, d_xb, d_yb, out_dim, in_dim, blocks, blocks_v13, threads, kernel_name); break;
        default:
            fprintf(stderr, "batched decode: unsupported B=%d (need 1,2,4,8,16,32,64,128)\n", B);
            exit(1);
    }

    std::vector<float> y_out(y_true.size());
    CUDA_CHECK(cudaMemcpy(y_out.data(), d_yb, y_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double gb = (double)W16.size() * sizeof(__half) / 1e9;  // weight bytes: same regardless of B
    double gbps = gb / (ms / 1e3);

    BatchResult r;
    r.B = B;
    r.kernel = kernel_name;
    r.ms = ms;
    r.gbps = gbps;
    r.throughput = B / (ms / 1e3);  // requests decoded per second
    r.pct_peak = 100.0 * gbps / peak_gbps;
    r.rel_err_vs_fp32 = relative_error(y_out, y_true);
    
    W16rb.free_all();
    cudaFree(d_xb); cudaFree(d_yb);
    return r;
}


int main(int argc, char** argv) {
    int out_dim = (argc > 1) ? atoi(argv[1]) : 4096;
    int in_dim  = (argc > 2) ? atoi(argv[2]) : 4096;
    if (out_dim <= 0 || in_dim <= 0) {
        fprintf(stderr, "dimensions must be positive: usage: %s [out_dim] [in_dim]\n", argv[0]);
        return 1;
    }
    if (in_dim % (GROUP_SIZE * 2) != 0) {
        fprintf(stderr, "in_dim must be a multiple of %d\n", GROUP_SIZE * 2);
        return 1;
    }
    printf("Decode hot path: y = W @ x,  W is %d x %d (%.1f MB in fp32)\n\n",
           out_dim, in_dim, (double)out_dim * in_dim * 4 / 1e6);

    // --- generate weights + activation vector ---
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.f, 0.02f);  // typical LLM weight scale
    std::vector<float> W(( size_t)out_dim * in_dim), x(in_dim);
    for (auto& w : W) w = dist(rng);
    for (auto& v : x) v = dist(rng);

    // --- true fp32 reference (double accumulation for a trustworthy ground truth) ---
    std::vector<float> y_true(out_dim);
    for (int r = 0; r < out_dim; r++) {
        double sum = 0;
        for (int c = 0; c < in_dim; c++) sum += (double)W[(size_t)r * in_dim + c] * x[c];
        y_true[r] = (float)sum;
    }

    // --- device info: need L2 size up front to size the rotating buffers ---
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int mem_clock_khz = 0, bus_width_bits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
    double peak_gbps = 2.0 * mem_clock_khz * (bus_width_bits / 8) / 1e6;
    size_t l2_bytes = (size_t)prop.l2CacheSize;
    printf("Device: %s   Peak HBM BW: %.0f GB/s   L2 cache: %.1f MB\n\n",
           prop.name, peak_gbps, l2_bytes / 1e6);

    // --- device buffers ---
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, x.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, out_dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    // x is tiny (in_dim floats) and legitimately stays resident/reused every
    // real decode step too -- no need to rotate it.

    int threads = 256;
    std::vector<float> y_out(out_dim);
    std::vector<Result> results;

    auto pull_y = [&]() {
        CUDA_CHECK(cudaMemcpy(y_out.data(), d_y, out_dim * sizeof(float), cudaMemcpyDeviceToHost));
    };

    // ---------- v0: naive fp32 ----------
    {
        RotatingBuffer<float> W32rb; W32rb.init(W, l2_bytes);
        int blocks = (out_dim + threads - 1) / threads;
        float ms = time_kernel([&](int i){
            matvec_v0_naive_fp32<<<blocks, threads>>>(W32rb.get(i), d_x, d_y, out_dim, in_dim);
        });
        pull_y();
        double gb = (double)W.size() * sizeof(float) / 1e9;
        results.push_back({"v0 naive fp32", ms, gb, true, relative_error(y_out, y_true)});
        printf("  [v0] rotating over %d x %.1f MB copies to defeat L2\n", W32rb.n, W.size()*sizeof(float)/1e6);
        W32rb.free_all();
    }

    // ---------- v1: warp-per-row fp32 ----------
    {
        RotatingBuffer<float> W32rb; W32rb.init(W, l2_bytes);
        int blocks = (out_dim * 32 + threads - 1) / threads;
        float ms = time_kernel([&](int i){
            matvec_v1_warp_fp32<<<blocks, threads>>>(W32rb.get(i), d_x, d_y, out_dim, in_dim);
        });
        pull_y();
        double gb = (double)W.size() * sizeof(float) / 1e9;
        results.push_back({"v1 warp fp32", ms, gb, true, relative_error(y_out, y_true)});
        W32rb.free_all();
    }

    // ---------- v2: warp-per-row fp16 ----------
    {
        std::vector<__half> W16(W.size());
        for (size_t i = 0; i < W.size(); i++) W16[i] = __float2half(W[i]);
        RotatingBuffer<__half> W16rb; W16rb.init(W16, l2_bytes);

        int blocks = (out_dim * 32 + threads - 1) / threads;
        float ms = time_kernel([&](int i){
            matvec_v2_warp_fp16<<<blocks, threads>>>(W16rb.get(i), d_x, d_y, out_dim, in_dim);
        });
        pull_y();
        double gb = (double)W16.size() * sizeof(__half) / 1e9;
        results.push_back({"v2 warp fp16", ms, gb, true, relative_error(y_out, y_true)});
        W16rb.free_all();
    }

    // ---------- v3: warp-per-row int8, per-row scale ----------
    {
        std::vector<int8_t> qW8; std::vector<float> scale8;
        quantize_int8_per_row(W, out_dim, in_dim, qW8, scale8);
        RotatingBuffer<int8_t> qW8rb; qW8rb.init(qW8, l2_bytes);
        // scale array is tiny relative to the weights -- a single copy is fine,
        // it contributes negligible bytes to the bandwidth measurement either way
        float* d_scale8;
        CUDA_CHECK(cudaMalloc(&d_scale8, scale8.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_scale8, scale8.data(), scale8.size() * sizeof(float), cudaMemcpyHostToDevice));

        int blocks = (out_dim * 32 + threads - 1) / threads;
        float ms = time_kernel([&](int i){
            matvec_v3_warp_int8<<<blocks, threads>>>(qW8rb.get(i), d_scale8, d_x, d_y, out_dim, in_dim);
        });
        pull_y();
        double gb = ((double)qW8.size() + scale8.size() * sizeof(float)) / 1e9;
        results.push_back({"v3 warp int8", ms, gb, true, relative_error(y_out, y_true)});
        qW8rb.free_all();
        cudaFree(d_scale8);
    }

    // ---------- v4: warp-per-row int4, grouped scale ----------
    {
        std::vector<uint8_t> packed; std::vector<float> gscale;
        quantize_int4_grouped(W, out_dim, in_dim, packed, gscale);
        RotatingBuffer<uint8_t> packedrb; packedrb.init(packed, l2_bytes);
        float* d_gscale;
        CUDA_CHECK(cudaMalloc(&d_gscale, gscale.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_gscale, gscale.data(), gscale.size() * sizeof(float), cudaMemcpyHostToDevice));

        int blocks = (out_dim * 32 + threads - 1) / threads;
        float ms = time_kernel([&](int i){
            matvec_v4_warp_int4<<<blocks, threads>>>(packedrb.get(i), d_gscale, d_x, d_y, out_dim, in_dim);
        });
        pull_y();
        double gb = ((double)packed.size() + gscale.size() * sizeof(float)) / 1e9;
        results.push_back({"v4 warp int4", ms, gb, true, relative_error(y_out, y_true)});
        packedrb.free_all();
        cudaFree(d_gscale);
    }

    // ---------- v5: block-per-row fp16, shared-mem x staging ----------
    {
        std::vector<__half> W16(W.size());
        for (size_t i = 0; i < W.size(); i++) W16[i] = __float2half(W[i]);
        RotatingBuffer<__half> W16rb; W16rb.init(W16, l2_bytes);

        int smem_bytes = in_dim * sizeof(float);
        // Dynamic shared memory above the 48KB default needs an explicit
        // opt-in per function (e.g. in_dim=16384 -> 64KB).
        if (smem_bytes > 48 * 1024) {
            CUDA_CHECK(cudaFuncSetAttribute(matvec_v5_block_smem_fp16,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        }
        // one block per row -> grid = out_dim blocks, fixed 256 threads/block
        float ms = time_kernel([&](int i){
            matvec_v5_block_smem_fp16<<<out_dim, threads, smem_bytes>>>(
                W16rb.get(i), d_x, d_y, out_dim, in_dim);
        });
        pull_y();
        double gb = (double)W16.size() * sizeof(__half) / 1e9;
        results.push_back({"v5 block smem fp16", ms, gb, true, relative_error(y_out, y_true)});
        W16rb.free_all();
    }

    // ---------- v6: warp-per-row int8, hardware DP4A ----------
    {
        std::vector<int8_t> qW8; std::vector<float> scale8;
        quantize_int8_per_row(W, out_dim, in_dim, qW8, scale8);
        RotatingBuffer<int8_t> qW8rb; qW8rb.init(qW8, l2_bytes);

        std::vector<int8_t> xq; float x_scale;
        quantize_int8_tensor(x, xq, x_scale);

        float* d_scale8; int8_t* d_xq;
        CUDA_CHECK(cudaMalloc(&d_scale8, scale8.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_xq, xq.size() * sizeof(int8_t)));
        CUDA_CHECK(cudaMemcpy(d_scale8, scale8.data(), scale8.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_xq, xq.data(), xq.size(), cudaMemcpyHostToDevice));

        int blocks = (out_dim * 32 + threads - 1) / threads;
        float ms = time_kernel([&](int i){
            matvec_v6_dp4a_int8<<<blocks, threads>>>(
                qW8rb.get(i), d_xq, d_scale8, x_scale, d_y, out_dim, in_dim);
        });
        pull_y();
        double gb = ((double)qW8.size() + scale8.size() * sizeof(float)) / 1e9;
        results.push_back({"v6 dp4a int8", ms, gb, true, relative_error(y_out, y_true)});
        qW8rb.free_all();
        cudaFree(d_scale8); cudaFree(d_xq);
    }

    // ---------- v7: warp-per-row int4, 4-way unrolled ----------
    {
        std::vector<uint8_t> packed; std::vector<float> gscale;
        quantize_int4_grouped(W, out_dim, in_dim, packed, gscale);
        RotatingBuffer<uint8_t> packedrb; packedrb.init(packed, l2_bytes);
        float* d_gscale;
        CUDA_CHECK(cudaMalloc(&d_gscale, gscale.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_gscale, gscale.data(), gscale.size() * sizeof(float), cudaMemcpyHostToDevice));

        int blocks = (out_dim * 32 + threads - 1) / threads;
        float ms = time_kernel([&](int i){
            matvec_v7_ilp_int4<<<blocks, threads>>>(packedrb.get(i), d_gscale, d_x, d_y, out_dim, in_dim);
        });
        pull_y();
        double gb = ((double)packed.size() + gscale.size() * sizeof(float)) / 1e9;
        results.push_back({"v7 ilp int4", ms, gb, true, relative_error(y_out, y_true)});
        packedrb.free_all();
        cudaFree(d_gscale);
    }

    // ---------- v8: vectorized int4, uint32_t word loads + float4 x loads ----------
    {
        std::vector<uint8_t> packed; std::vector<float> gscale;
        quantize_int4_grouped(W, out_dim, in_dim, packed, gscale);
        RotatingBuffer<uint8_t> packedrb; packedrb.init(packed, l2_bytes);
        float* d_gscale;
        CUDA_CHECK(cudaMalloc(&d_gscale, gscale.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_gscale, gscale.data(), gscale.size() * sizeof(float), cudaMemcpyHostToDevice));

        int blocks = (out_dim * 32 + threads - 1) / threads;
        float ms = time_kernel([&](int i){
            matvec_v8_vectorized_int4<<<blocks, threads>>>(packedrb.get(i), d_gscale, d_x, d_y, out_dim, in_dim);
        });
        pull_y();
        double gb = ((double)packed.size() + gscale.size() * sizeof(float)) / 1e9;
        results.push_back({"v8 vec int4", ms, gb, true, relative_error(y_out, y_true)});
        packedrb.free_all();
        cudaFree(d_gscale);
    }

    printf("%-20s %10s %12s %10s %12s %12s\n",
           "kernel", "ms", "GB/s", "%peak", "speedup", "rel.err vs fp32");
    float baseline_ms = results[0].ms;
    for (auto& r : results) {
        double gbps = r.gb_moved / (r.ms / 1e3);
        printf("%-20s %10.4f %12.1f %9.1f%% %11.2fx %12.4f%%\n",
               r.name.c_str(), r.ms, gbps, 100.0 * gbps / peak_gbps,
               baseline_ms / r.ms, r.rel_err_vs_fp32 * 100.0);
    }

    printf("\nRead this as: v0->v1 is pure engineering (same bytes, fixed access\n"
           "pattern). v1->v2->v3->v4 is bytes-moved reduction, each step trading\n"
           "some accuracy (rightmost column) for bandwidth. v5-v7 move the SAME\n"
           "bytes as v2/v3/v4 respectively -- any speedup there is purely from\n"
           "better latency hiding (v5's extra threads/row, v6's DP4A instruction,\n"
           "v7's independent accumulators), not from reading less data. If v5-v7\n"
           "don't beat their v2/v3/v4 counterparts on your GPU, that's a real\n"
           "result too: it means those kernels were already latency-hidden well\n"
           "enough that the extra complexity wasn't the bottleneck.\n");

    // ---------- v11: batched decode, sweep batch size ----------
    // Every kernel above is B=1 -- one request's worth of FMAs per byte of W
    // read. This section reuses each weight byte across B simultaneous
    // requests. The expected signature: GB/s stays roughly flat across B (the
    // same weight bytes are read either way), while throughput(req/s) climbs
    // in proportion to B -- that ratio is the entire point of batching in
    // real serving: the fixed cost of reading W is amortized over more useful
    // work.
    printf("\n--- v11/v12/v13: batched decode (fp16), sweeping batch size ---\n");
    printf("        B <= 4: v11 register-blocked (B accumulators per lane in registers)\n"
           "        B >= 8: v11 (if B<=16) / v12 lane-owned / v13 smem-tiled probed, winner shown\n");
    printf("%-4s %-14s %10s %12s %10s %18s %12s\n",
           "B", "kernel", "ms", "GB/s", "%peak", "throughput(req/s)", "rel.err");
    std::vector<BatchResult> v11_results;
    v11_results.push_back(run_batched_decode(out_dim, in_dim, W, l2_bytes, threads, peak_gbps, 1));
    v11_results.push_back(run_batched_decode(out_dim, in_dim, W, l2_bytes, threads, peak_gbps, 2));
    v11_results.push_back(run_batched_decode(out_dim, in_dim, W, l2_bytes, threads, peak_gbps, 4));
    v11_results.push_back(run_batched_decode(out_dim, in_dim, W, l2_bytes, threads, peak_gbps, 8));
    v11_results.push_back(run_batched_decode(out_dim, in_dim, W, l2_bytes, threads, peak_gbps, 16));
    v11_results.push_back(run_batched_decode(out_dim, in_dim, W, l2_bytes, threads, peak_gbps, 32));
    v11_results.push_back(run_batched_decode(out_dim, in_dim, W, l2_bytes, threads, peak_gbps, 64));
    v11_results.push_back(run_batched_decode(out_dim, in_dim, W, l2_bytes, threads, peak_gbps, 128));
    for (auto& r : v11_results) {
        printf("%-4d %-14s %10.4f %12.1f %9.1f%% %18.0f %11.4f%%\n",
               r.B, r.kernel.c_str(), r.ms, r.gbps, r.pct_peak, r.throughput, r.rel_err_vs_fp32 * 100.0);
    }
    printf("\nRead the sweep as: at B=1-4 GB/s stays near the B=1 fp16 level (weights read\n"
           "once per launch) and throughput(req/s) climbs ~linearly in B -- the cheap win\n"
           "of continuous batching. From B=8 on, every output row has to touch the whole\n"
           "batched activation vector (in_dim x B floats), and that x traffic -- plus\n"
           "SIMT instruction issue, not weight bandwidth or FLOPs -- becomes the wall:\n"
            "watch throughput per request (req/s / B) fall off and %%peak sag. The designs\n"
           "in this file all stop at the same wall from different angles (v11: x re-read\n"
           "per row from L2; v12: in_dim address computations per lane; v13: smem loads\n"
           "per (column, entry) pair). What breaks through it is tensor-core GEMM -- the\n"
           "same tiling as v13 with m16n8k16 MMAs reusing each x row across 16 output\n"
           "rows per instruction -- which is what cuBLAS / vLLM actually run at large B.\n");



    cudaFree(d_x); cudaFree(d_y);
    return 0;
}

