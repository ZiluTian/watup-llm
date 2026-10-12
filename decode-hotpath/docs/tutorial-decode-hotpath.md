# The LLM Decode Hot Path: A CUDA GEMV Tutorial

A guided walkthrough of `src/decode_hotpath.cu`. All numbers below are from
an RTX 5090 (1792 GB/s HBM peak, 100.7 MB L2) unless noted; the same binary
re-measures everything on whatever GPU you run it on.

## 1. Why decode is a bandwidth problem

Autoregressive generation works in two phases. *Prefill* crunches the whole
prompt in one big GEMM — compute-bound, tensor cores, done. *Decode* then
generates one token at a time: for each layer, the current activation
`x [in_dim]` is multiplied by that layer's weight matrix `W [out_dim × in_dim]`:

```
y[r] = sum over c of W[r, c] * x[c]        (one dot product per output row)
```

This is a **batch-1 matrix-vector multiply** (GEMV). Its arithmetic intensity
is ~1 FLOP per byte of weight: every byte of `W` is fetched from HBM, used in
exactly one multiply, and discarded. No matter how much compute the GPU has,
the time is bounded below by `bytes(W) / HBM bandwidth`. A 4096×4096 fp32
weight matrix is 67 MB, so the floor on a 1792 GB/s part is ~37 µs per layer.
The whole discipline of this experiment is: *how close can we get to that
floor, and what do we trade to get there?*

## 2. The measurement trap (and the fix)

The first mistake to avoid is timing a kernel against warm L2. Launch the
same kernel 50 times against the same buffer and, after the first launch,
every launch is served from L2 instead of HBM. At 4096×4096 this produces
"bandwidth" numbers that **exceed 100% of theoretical HBM peak** — a sign
something is wrong, not that your kernel is magic. (This experiment's first
run logged v1 at 272% of peak. It was L2.)

The fix, implemented in the `RotatingBuffer` template in the source: allocate
`n = max(2, ceil(4 × L2_size / buffer_size))` physically distinct copies of
each weight buffer (capped at 32) and rotate through them, so every timed
launch reads an address range that was not just touched:

```cpp
void init(const std::vector<T>& host_data, size_t l2_bytes) {
    size_t buf_bytes = host_data.size() * sizeof(T);
    // aim for total footprint >= 4x L2 so cache can't hold the working set
    n = std::max(2, (int)std::ceil((4.0 * l2_bytes) / (double)buf_bytes));
    n = std::min(n, 32);
    ...
}
```

At 4096×4096 the fp32 buffers rotate over 6 × 67.1 MB copies and the fp16
ones over 12 × 33.6 MB — footprints of 403 MB, well past the 100.7 MB L2.
(The tiny activation vector `x` is *not* rotated: in real decode it stays
hot in cache too, which is correct behavior, not a measurement artifact.)

Each kernel is timed over 50 launches with CUDA events after one warmup, and
checked against an fp32 reference computed with **double-precision
accumulation** on the host, so the "rel. error" column is a true accuracy
cost, not noise.

## 3. The B=1 ladder: nine kernels

### v0 → v1: coalescing (the bug fix)

v0 gives each thread a whole output row: thread `i` reads
`W[i*in_dim + col]` for col = 0…in_dim. Thread `i` and thread `i+1` read
addresses `in_dim × 4` bytes apart — zero memory coalescing, ~1/32 of a
transaction used at a time. Result: 90 GB/s, 5% of peak.

v1 gives each **warp** a row: lane `l` reads `col = l, l+32, l+64, …`.
Consecutive lanes read consecutive elements, so the 32 lanes' loads merge
into full 128-byte HBM transactions. Same bytes moved, 8.2× faster (740
GB/s, 41% of peak). The partial sums then combine with a five-step
`__shfl_down_sync` butterfly — the standard warp reduction.

Two things to notice in the numbers:

- v1 (fp32) moves **twice** the bytes of v2 (fp16) but takes almost **4×**
  the time (0.091 ms vs 0.023 ms) — 41% of peak vs 82%. On this part the
  fp32 path costs more than its bytes; the per-byte efficiency of a 4-byte-
  per-lane GEMV is simply worse. Practical takeaway: if you keep fp32
  weights, you keep both the bytes *and* the efficiency tax.
- v2 reaches 82% of HBM peak. That is close to what a large contiguous
  stream achieves; the remaining 18% is the kernel being short-lived (33 MB
  per launch) and not fully ramping the memory pipeline. Batching — section 5
  — is partly about buying more ramp-up time.

### v2 → v3 → v4: the quantization ladder (bytes for accuracy)

Each step keeps the v1 access pattern and stores `W` in a smaller format,
dequantizing in registers (the dequantized value is never materialized in
memory):

- **v3, int8, per-row scale.** One scale per output row: `scale = max|row| /
  127`. Each lane loads its row's scale once (an L1 broadcast — all lanes
  read the same address) and multiplies it into its partial products.
  Quarter the bytes of v2, 0.85% relative error, 72.6% of peak.
- **v4, int4, per-group scale.** 4-bit weights can't survive a per-row scale
  (32 bits of dynamic range in 2 bits of mantissa), so GGUF-style formats
  scale in small groups instead: here every 32 elements share one fp32
  scale, and two nibbles are packed per byte. Each lane reads one byte,
  decodes two signed nibbles (`(byte & 0xF) - 8`), and does two FMAs against
  *two different* group scales. Eighth the bytes of v2, 9.24% relative
  error — the cost of 4 bits, measured, not assumed.

The accuracy column is the whole point of the ladder: every step's price is
visible. int8 buys 4× fewer bytes at ~1% error; int4 buys 8× at ~9%. Which
you can live with is a product decision (it is the same decision
quantization-aware training optimizes, just made here at inference time on
synthetic data).

### v5: block-per-row + shared memory (what *doesn't* win)

v5 stages `x` in shared memory and gives each row 256 threads (a full block)
instead of 32 (a warp), on the theory that more concurrent threads per row
hide HBM latency better. Result: 0.090 ms — the same 8.2× as v1, *not*
better than v2's 0.023 ms. The extra synchronization and the per-row x
re-staging cost more than the extra latency hiding earns, because the warp-
per-row kernels were already latency-hidden well enough. Shipped as an
honest negative result: "add more threads + shared memory" is not a free
win. (It also exercises the `cudaFuncSetAttribute` opt-in for dynamic shared
memory above 48 KB — at `in_dim = 16384` the x staging needs 64 KB.)

### v6: DP4A — hardware int8 dot product

v6 quantizes `x` too (per-tensor int8) so both operands of the multiply are
int8, and uses `__dp4a`: one instruction does four int8×int8 multiplies
accumulated into int32. It replaces a 4-wide mul+add chain with a single
hardware instruction and lands at 74.2% of peak (beating v3's 72.6%) with
1.20% error — the extra x-quantization error is visible and, again, priced.

### v7: independent accumulators — the null result that matters

v4/v7 both read int4 weights, but v4 accumulates into a *single* running sum:
every iteration's add waits on the previous one (a serial dependency chain
through shift/mask/convert/add). v7 unrolls 4× and uses four independent
accumulators so the scheduler can interleave them. It helps — v7 (46.8% of
peak) beats v4 (39.3%) — but not much, and v7 moves exactly the same bytes
as v4. That gap is the diagnostic: **if more ILP on the same data gives you
almost nothing, you are not latency-bound, you are instruction-bound** — the
cost is in the total instruction count per element, not in waiting on
memory. The fix for that is not more accumulators; it's *less work per
element*.

### v8: vectorization — less total work per element

v8 attacks the instruction count directly, three ways:

1. **uint32 word loads on W**: one lane loads 4 packed bytes (8 columns) per
   iteration instead of 1 byte — full 128-byte warp transactions instead of
   32-byte ones, a quarter of the load instructions.
2. **float4 loads on x**: the 8 x-values a word's 8 columns need arrive in
   two `float4` loads instead of 8 scalar loads.
3. **One group lookup instead of two divisions**: v4 computed
   `col / GROUP_SIZE` once *per nibble* (twice per byte) with signed
   division the compiler couldn't strength-reduce. v8 notes that 32-element
   groups align to 4-byte words (32 / 2 per byte / 4 bytes = 4 words), so
   the group index is `w >> 2` — one unsigned shift, shared by all 8 columns
   in the word.

Result: 0.0084 ms, 70% of peak, **89× the naive kernel** — while v4's 9.24%
error is *identical to the byte* (same quantization, same dequant, cheaper
instructions). v8 reaches v3's per-byte efficiency with half of v3's bytes.
That is the whole B=1 story in one row: fix the access pattern, cut the
bytes, then cut the instructions per byte.

### Reading the B=1 table

| kernel | bytes moved | ms | GB/s | %peak | rel.err |
|--------|-------------|-----:|-----:|-----:|-----:|
| v0 naive fp32 | 67.1 MB | 0.744 | 90 | 5.0% | 0.0001% |
| v1 warp fp32 | 67.1 MB | 0.091 | 740 | 41.3% | 0.0000% |
| v2 warp fp16 | 33.6 MB | 0.023 | 1470 | 82.0% | 0.021% |
| v3 warp int8 | 16.8 MB | 0.013 | 1301 | 72.6% | 0.85% |
| v4 warp int4 | 8.4 MB | 0.015 | 705 | 39.3% | 9.24% |
| v5 block smem fp16 | 33.6 MB | 0.090 | 372 | 20.7% | 0.021% |
| v6 dp4a int8 | 16.8 MB | 0.013 | 1329 | 74.2% | 1.20% |
| v7 ilp int4 | 8.4 MB | 0.012 | 839 | 46.8% | 9.24% |
| v8 vec int4 | 8.4 MB | 0.008 | 1255 | 70.0% | 9.24% |

There are two independent axes here:

- **v0→v1** and **v4→v7→v8**: same bytes, more efficiency (access pattern,
  then instruction count).
- **v1→v2→v3→v4**: same pattern, fewer bytes (quantization) — each step
  buys bandwidth and pays a measured accuracy price.

v5/v6 show the "same bytes as a v2/v3, restructured" axis: v6 (DP4A) wins
its matchup, v5 (more threads + smem) loses it. Both are legitimate
experiments; the table is what decides.

## 4. Why batch, and the layout choice

At B=1 every byte of W does exactly one FMA. Real serving doesn't decode one
request at a time — it runs a batch of B requests through the same layer,
and each weight byte then does B FMAs: same HBM traffic, B× the useful
work. That ratio *is* the point of continuous batching.

The one layout decision that matters: store the batched activation as
`x [in_dim][B]` (batch-contiguous), not `[B][in_dim]`. For a fixed column
`c`, the B values every row needs are `x[c*B + 0..B-1]` — contiguous
floats, so a kernel that walks columns reads coalesced strides of B.
(With `[B][in_dim]` those B values are `in_dim` floats apart.)

## 5. The batched kernels: v11, v12, v13

### v11 — register blocking (wins at B ≤ 4)

One warp per output row; each lane strides the columns and keeps **B fp32
accumulators in registers**, one per batch entry:

```cpp
for (int col = lane; col < in_dim; col += 32) {
    float w_val = __half2float(row_ptr[col]);      // read once...
    const float* xcol = x_batched + (size_t)col * B;
    #pragma unroll
    for (int b = 0; b < B; b++) sum[b] += w_val * xcol[b];  // ...reused B times
}
```

This is the "register blocking" microkernel a real GEMM uses, applied along
the batch axis instead of a second matrix dimension. `B` is a **template
constant** for a reason: with a runtime B the compiler cannot keep `sum[]`
in registers — it allocates the array in *local memory* (L1-backed, but a
load+store per FMA). The earlier, pre-portfolio version of this experiment
made exactly that mistake, and its B-sweep fingerprint was the classic
register-spill signature — time growing *super-linearly* in B (0.66 → 2.5 →
10 → 20.5 ms for B = 8 → 16 → 32 → 64). If you see a kernel that is fine at
B=4 and suddenly 4× slower at B=8 for the same access pattern, check
`ptxas -v` for spill stores before looking anywhere else.

Measured: B=1…4 give 43.8k → 86.9k → 130k req/s at 82% → 81% → 61% of peak.
That is the cheap win of batching — and note B=4 already sags from linear
scaling (3.0× instead of 4×), which is the x-traffic wall below starting to
show.

### The x-traffic wall

Here is the subtlety the B=1 kernels never meet: **every output row must
read the entire batched activation vector** — `in_dim × B` floats — no
matter what. At 4096 in_dim, B=8, that is 128 KB of x per row; × 4096 rows
is **512 MB of L2 traffic per launch** (x itself is L2-resident; HBM weight
traffic stays at 33.6 MB). At B=16 it is 1 GB per launch. From B=8 on, that
x traffic — plus the SIMT instruction cost of addressing and consuming it —
dominates. This is the wall, and it shows up in the sweep as: throughput
per request (`req/s / B`) collapsing while the kernel is demonstrably *not*
re-reading W from HBM.

### v12 — lane-owned batches (B ≥ 32)

Register blocking caps out around B=16 (16 accumulators per lane is the last
comfortable size; B=32 would need 32). The restructure: **each lane owns a
fixed slice of the batch** — lane `l` computes *complete* dot products for
entries `b = l, l+32, l+64, …`, sweeping **all** columns:

```cpp
for (int col = 0; col < in_dim; col++) {
    float w_val = __half2float(row_ptr[col]);
    const float* xcol = x_batched + (size_t)col * B;
    acc0 += w_val * xcol[lane];
    if (n_owned >= 2) acc1 += w_val * xcol[lane + 32];
}
y_batched[base] = acc0;      // lane l alone produces y[row, l]
```

Two properties make it correct and cheap in registers: only B/32
accumulators per lane (two for B=64), and **no cross-lane reduction** —
each output `(row, b)` is produced by exactly one lane, which writes it
directly. The price is `in_dim` global-load-plus-address-compute iterations
per lane, so at large B it becomes instruction-issue limited (on this GPU:
547 µs at B=32, 2.85 ms at B=64 — measured, then beaten by v13).

### v13 — GEMM-style shared-memory tiling (wins at B ≥ 16 here)

The standard GEMM pattern, applied to decode:

- A block (256 threads, 8 warps) owns an **8-row output tile** — one warp
  per row, so all 32 lanes of a warp work on the same row.
- The inner dimension is walked in **k-tiles of 64 columns**. Per k-tile,
  the `8×64` W tile is loaded from HBM **once** into shared memory, and the
  `64×64` x tile is loaded from L2 into shared memory.
- Within the k-tile, each lane computes the two batch entries it owns
  (`b = 2*lane, 2*lane+1` within the 64-wide b sub-tile) over **all 64
  columns**: 32 lanes × 2 entries cover the sub-tile exactly, and every
  `(row, b)` output is produced by exactly one lane. No shuffles, no
  reductions — the lane's register accumulators *are* the answer.

```cpp
#pragma unroll
for (int k_off = 0; k_off < K_TILE; k_off++) {
    float w = __half2float(smem_W[row][k_off]);
    acc0 += w * smem_x[k_off][b_base];
    if (b_base + 1 < n_b) acc1 += w * smem_x[k_off][b_base + 1];
}
```

For `B ≤ 64` the whole batch is one b sub-tile, so **W is fetched from HBM
exactly once, regardless of B**; the x tile is loaded from L2
`in_dim/64` times per row instead of `in_dim` times. (B=128 needs two
sub-tiles and re-fetches W twice — visible as the B=128 row's per-request
cost rising again.)

**The trap that costs you a day:** it is tempting to let the 32 lanes of a
row *split the k-columns* (8 columns each) to cut per-lane work. It is
wrong: if lanes split k and also each own a different b, then lane `l` holds
a partial sum for *its* b over only 1/32 of the columns, and there is no
legal way to combine lanes — every b's partials live in different lanes than
the lane that owns b. The split must go one way only: either one lane sweeps
all k for a fixed b (v13), or all b's for a k-slice (which pushes B
accumulators per lane — back to v11's spill). v13's original draft hit this
and produced ~93% relative error with a "plausible-looking" output; the bug
was not a typo, it was the mapping.

### What the sweep says (RTX 5090, 4096×4096)

| B | kernel (winner) | ms | GB/s | %peak | req/s | req/s per B |
|--:|-----------------|-----:|-----:|-----:|-----:|-----:|
| 1 | v11 reg-block | 0.023 | 1470 | 82.0% | 43.8k | 43.8k |
| 2 | v11 reg-block | 0.023 | 1457 | 81.3% | 86.9k | 43.4k |
| 4 | v11 reg-block | 0.031 | 1086 | 60.6% | 130k | 32.4k |
| 8 | v11 reg-block | 0.193 | 174 | 9.7% | 41.6k | 5.2k |
| 16 | v13 smem-tile | 0.535 | 63 | 3.5% | 29.9k | 1.9k |
| 32 | v13 smem-tile | 0.533 | 63 | 3.5% | 60.0k | 1.9k |
| 64 | v13 smem-tile | 0.472 | 71 | 4.0% | 135.7k | 2.1k |
| 128 | v13 smem-tile | 1.445 | 23 | 1.3% | 88.6k | 0.7k |

Read the **req/s per B** column: at B=2 it matches the B=1 per-request cost
(free win); by B=8 it is 8× worse than B=1; at B=64, 20× worse. The weight
bandwidth was *not* the constraint any of the time — W is read once per
launch in every row above. The constraint is the `in_dim × B` floats every
row must touch (x traffic), plus the instruction cost of a SIMT kernel
consuming it. Three different structures (register-blocked, lane-owned,
smem-tiled) all stop at the same wall from different angles; v13 simply
stops closest to it.

**What breaks through the wall is tensor cores.** m16n8k16 MMA does 2048
fp16 FMA per instruction and reuses each x-tile row across 16 output rows
at once — the x-traffic-per-FMA ratio drops by an order of magnitude and
the instruction count drops with it. That is what cuBLAS and vLLM actually
run at large B, and it is the natural "v14" of this experiment: v13's
tiling, with the inner `k_off` loop replaced by `mma.sync` on the smem W
tile.

## 6. Debugging notes (what actually went wrong, in order)

- **272%-of-peak bandwidth in the first 4096² run.** L2 was warm across the
  50 timed launches. Fixed with the rotating buffers (§2). Any "bandwidth"
  number above ~100% of peak HBM is a measurement bug until proven
  otherwise.
- **Super-linear time vs B in the first batched kernel.** It looked like
  register spilling and the fix (template B) was right *for that kernel* —
  but at B=8/16 the *correct* kernels still sag, and that is a different
  beast: x-traffic, not spills (§5). Same symptom, different cause; `ptxas
  -v` (zero spills at B=16) is what separates them.
- **v5 silently failed at `in_dim=16384`.** 64 KB of dynamic shared memory
  exceeds the 48 KB default; the launch error only surfaces at the next
  sync, and the harness reported a near-zero time and a 16,000%-of-peak
  "bandwidth". Fixed with `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`;
  the harness now CUDA_CHECKs around every event so a failed launch can't
  masquerade as a fast kernel.
- **v13's first version: 93% relative error, "plausible" outputs.** The
  k-split/b-split mapping trap (§5). The fix was structural, not a typo.
- **`(nil)eak sag` in the output text.** A `%peak` inside a printf format
  string — `%p` wants a pointer. `%` in format strings must be `%%`.

## 7. What to try next

In rough order of effort-to-insight:

1. **Tensor-core v14**: v13's tiling with `mma.sync.aligned.m16n8k16` on the
   smem W tile (fp16×fp16→fp32 acc). Expect to break the x-traffic wall;
   compare against `cublasGemmEx` for the same shapes as the reference.
2. **Async copy / double-buffered smem** in v13 (`cp.async` or Blackwell TMA)
   so the next k-tile's HBM load overlaps the current tile's compute.
3. **int8 W + int8 x tensor cores** (like real GGUF/llama.cpp Q8_0 kernels)
   with the per-row/per-group dequant folded into the epilogue.
4. **Autotune**: tile sizes, `num_stages`-style pipelining, and the v11→v12→v13
   crossover point are all shape- and GPU-dependent; the self-tuning probe in
   the harness is the seed of a real auto-tuner.
5. **End-to-end**: wire the winning kernel into a small decoder layer
   (RMSNorm → QKV projection → attention → MLP) and measure tokens/s for a
   real model at B=1…128 against an unmodified PyTorch/CuBLAS baseline.
