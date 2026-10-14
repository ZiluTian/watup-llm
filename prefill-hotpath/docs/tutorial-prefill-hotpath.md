# The LLM Prefill Hot Path: A CUDA GEMM Tutorial

A guided walkthrough of `src/prefill_hotpath.cu`. All numbers below are from
an RTX 5090 (1792 GB/s HBM peak, 100.7 MB L2, 217.6 TFLOP/s fp16 TC peak
measured via cuBLAS hgemm on the same shape) unless noted; the same binary
re-measures everything on whatever GPU you run it on.

## 1. Why prefill is a compute problem

Decode and prefill are the same layer, run at opposite ends of the batch
axis. Decode is M = 1 (one activation row per token). Prefill ingests the
whole prompt at once:

```
Y = X @ W^T          X [M x in_dim] fp16,  W [out_dim x in_dim] fp16
```

with M = prompt length. The arithmetic intensity of the layer is

```
I(M) = 2*M*N*K / (2*(M*K + N*K + M*N))     [FLOP / HBM byte, fp16 operands]
```

(2 FLOPs per MAC; the denominator counts one HBM touch per element of X, W
and Y — the binary counts Y as fp16-sized, which slightly overstates the
intensity for the fp32 Y actually written; the conclusion is unchanged.)
At M = N = K = 4096 that is **1365 FLOP/byte**. The roofline knee of this
machine — measured, not datasheet: 217.6 TFLOP/s ÷ 1792 GB/s — sits at
**121 FLOP/byte**. The prefill layer runs ~11× above the knee: every byte
of X and W bought 1365 FLOPs of work.

Two floors, for scale:

```
bandwidth floor:  (33.6 MB X + 33.6 MB W) / 1792 GB/s = 37.5 us
compute floor:    2*4096^3 / 217.6 TFLOP/s          = 631 us
```

The compute floor is **17× higher** than the bandwidth floor. cuBLAS hgemm
measured 0.632 ms — it hits the compute floor. So the entire ladder in this
file is a story about TFLOP/s, not GB/s: the question is not "how fast can
we move the bytes" but "how much of the 217.6 TFLOP/s TC rate can a
hand-written kernel actually issue".

That is the mirror image of `tutorial-decode-hotpath.md`, where the compute
floor is 17× *below* the bandwidth floor and every step of the ladder is a
bandwidth story.

## 2. Measurement method (what's shared, what's different)

Same cache-cold harness as the decode experiment: a `RotatingBuffer` ring of
12 × 67.1 MB copies (X + W) — 805 MB, ~8× the 100.7 MB L2 — so every
launch starts with cold operands. (The ring caps at 16 copies; at 4096³ the
12-copy ring is already 8× L2.) Same CUDA-event timing: 1 warmup + 50
iterations (10 for v0, which is 80 ms per iteration).

Two differences from decode:

- **The reference is cuBLAS sgemm (fp32), not a host double loop.** At
  4096³ a host double loop would take seconds and is the wrong tool: cuBLAS's
  fp32 GEMM is the practical ground truth, and its own error against the
  exact value is far below the smallest number on the table.
- **The peak is cuBLAS hgemm on the same shape**, used as the 100% baseline
  for the "% of peak" column. That is why the roofline knee (121 FLOP/byte)
  and the M\* below are *measured* quantities — the whole table is anchored
  to a measured fp16 TC peak, not a datasheet number.

## 3. The SIMT ladder: v0 → v4

### v0 → v1: tiling (the 13×)

v0 is the classic mistake: one thread per output element, no reuse. Each of
the M·N threads walks K X values and K W values from global memory, so the
launch moves `M*N*K*2*4 B` of global traffic — 1.37 GB at 4096³, at
~0.97% of HBM peak. Result: 1.7 TFLOP/s.

v1 is the 32×32 shared-memory tile (256 threads/block, 8 warps): each block
loads its 32-wide X slab and 32-wide W slab once into smem, then all 256
threads mine them. Global traffic drops by the tile factor: `M*N*K*(2/32)*4 B`
— **13.25× faster**, 23.1 TFLOP/s (10.6% of the hgemm peak). The remaining
gap to the peak is SIMT FMA rate plus smem traffic: this kernel does
2 scalar fp16 smem loads per FMA.

### v2: fp16 — the last bytes step

Store X and W in fp16, accumulate in fp32: every HBM and smem operand byte
is halved. **27.19× vs v0, 47.4 TFLOP/s (21.8% of peak)**, 0.015% rel. err
vs the fp32 reference.

Note the trajectory: v2 is the *last* step that helps by moving fewer
bytes. The next SIMT kernel (v4) will land *below* v2's 21.8%, at 19.0%.
Once the kernel is compute bound, bytes are no longer the wall.

### v3: register blocking 4×4 — the null result

v3 moves to a 64×32 tile with 128 threads (4 warps); each thread owns a 4×4
patch of the output tile (16 fp32 accumulators). The smem-traffic argument
for it is clean: each A element feeds 4 FMAs and each B element feeds 4 FMAs,
so smem loads per FMA drop 4× from v2. The prediction is "faster".

Measured: **4.57 ms vs v2's 2.90 ms — 1.57× slower** (30.1 vs 47.4 TF).

The telling part: v3 *reduces* smem traffic per FMA by 4× and still loses.
So smem byte volume was not v2's bottleneck. What v3 changed besides that
is warps per tile: 8 → 4. With half the warps, there is half the parallelism
available to hide smem load latency, and the inner loop is a chain of
2-byte smem loads feeding scalar FMAs. (Register pressure is *not* the
story — ptxas reports 56 registers for v3 vs 40 for v2, both unremarkable.)

The same structure won in the *batched* decode ladder (v11), where
register blocking reuses weight across batch entries with no extra traffic.
Here it only reshuffles smem traffic and loses the warps-per-tile trade.
Lesson: a micro-optimization that wins in one regime can lose in another;
the table, not the argument, adjudicates.

### v4: cp.async double buffering — the enabler, not the winner

v4 keeps the 64×32 / 128-thread shape and adds `cp.async`: while the block
computes k-tile *i* out of buffer A, the next k-tile's bytes stream
global → smem into buffer B, with `cp.async.wait_group 1` + `__syncthreads()`
at the swap. This is the standard software pipeline.

It still loses: **41.3 TF vs v2's 47.4 TF.** The reason is v3's reason:
cp.async hides *global* latency, but the consume side is still 4 warps
issuing scalar 2-byte smem loads; the pipeline just arrives at the
instruction-issue bottleneck with the door already open.

v4's value is structural: double buffering is the *prerequisite* for
feeding tensor cores (v5), where 4 warps are plenty because the mma pipe
absorbs the FMA work.

Two traps are documented in the source because both are silent:

1. **Buffer-1 offset is `ASZ + BSZ`, not `2*ASZ`.** The A and B tiles have
   different sizes (64×40 vs 32×40 halves). Offsetting buffer 1 by `2*ASZ`
   makes As1 and Bs1 overlap; the prefetch then clobbers the B tile the
   next tile's mmas read. Correctness, not performance.
2. **The trailing `else cp_commit()`.** `cp.async.wait_group 1` means "wait
   until at most one group is in flight". The pending-group count must stay
   balanced across the loop: every iteration that prefetches a tile commits
   a group, and the last iterations (no next tile) commit nothing — the
   `else cp_commit()` keeps the count honest. Skip it and the final
   `wait_group 1` can return before the *last* tile's bytes have landed:
   silent, shape-dependent corruption that only appears at certain K/tile
   counts.

OOB handling at the M boundary: `cp.async` with `src_bytes = 0` zero-fills
rows past M (the destination is still written), so boundary tiles compute
garbage-free padding without a branch.

## 4. v5: fp16 tensor cores — the payoff

Same block shape (64×32, 128 threads, 4 warps), same cp.async double
buffering; the inner loop now issues `mma.sync` instead of scalar FMAs:

```
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

Each warp owns a 32×16 output patch; per 32-wide k-tile that is 2 m-tiles ×
2 n-tiles × 2 k-halves = **8 mma instructions per warp per k-tile**, each
doing 16×8×16 = 2048 MACs. ptxas: 66 registers, 15,360 B dynamic smem
(two buffers × (64×40 + 32×40) halves × 2 B; the 40-half row stride — pad of
8 — keeps ldmatrix's 8-row reads bank-conflict-free and 16 B aligned).

**Result: 78.1 TFLOP/s = 35.9% of the cuBLAS hgemm peak, 44.87× vs v0.**
The 4-warp shape that lost at v3/v4 now wins, because the bottleneck moved
from "issue scalar FMAs" to "feed the TC pipe", and 4 warps can feed it.

The fragment mechanics (all verified against the PTX ISA 9.4 fragment
tables, §9.7.16.5):

**A via ldmatrix.x4.** The A operand (X tile, 16 rows × 32 halves per mma)
is loaded with one `ldmatrix.x4` per 8×16-half sub-tile, addresses from the
8 consecutive lanes of each 8×8 matrix:

```cpp
int lrow = (lane & 7) + 8 * ((lane >> 3) & 1);   // row within the 8x8
int lcol = 8 * (lane >> 4);                      // which 8x8 matrix
// load 16 B from As[lrow * 40 + lcol]  -> regs a0..a3
```

`ldmatrix` distributes each 8×8 matrix so lane *l* holds row `l/4`, column
pair `l%4` — which is *exactly* the m16n8k16 A-fragment order
(a0/a1: row `l/4`, cols `2(l%4)..+1`; a2/a3: row `l/4+8`, same cols), so
the registers go straight into the mma with no shuffles.

**B via the same ldmatrix.** W is stored row-major `[N][K]`, and mma wants
B column-major `[K][N]` — which is the same memory layout, so the identical
address pattern works on the W tile. Per n-tile, mma takes
`(rb[0], rb[2])` for n-tile 0 and `(rb[1], rb[3])` for n-tile 1, per the
B-fragment register table.

**The C fragment — and the trap.** This is where the first draft burned a
day. The epilogue was written with the **m16n8k8** C layout:

```cpp
// WRONG for k16 (this is the k8 layout):
c1 = C[l/4+8][l%4];
c2 = C[l/4][4+l%4];
```

The m16n8k16 C layout is different — each register pair owns two
*consecutive* columns of one row, and the second register pair is the same
two columns of the row 8 rows down:

```cpp
// RIGHT for k16:
int lr = lane / 4;      // 0..7
int lc = lane % 4;      // 0..3
c0 = C[lr][2*lc];       c1 = C[lr][2*lc + 1];
c2 = C[lr+8][2*lc];     c3 = C[lr+8][2*lc + 1];
```

The failure signature was exactly the kind that misleads: **131–134% rel.
err, but the output looked plausible** — values in the right range, just
permuted within each 16×8 tile. And the M-sweep, which selects the winner by
timing alone, then *crowned the broken kernel* — fastest "winner", garbage
numbers. Two lessons:

1. Fragment layouts differ **across k** within the same mma family. Never
   port an epilogue between k8 and k16 without re-deriving it from the ISA
   tables; the register-to-(row, col) map is not what you think it is.
2. A kernel selector that ignores accuracy will happily promote a broken
   kernel. (§9 lists the fix: gate "winner" on a rel-err threshold.)

## 5. v6: int8 tensor cores — and why it is not 2×

v6 is the same kernel in integer: `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32`.
The fragment shapes: A is **2** s8 registers per lane (4 consecutive k of
row `cr`, then the same k of row `cr+8`), B is **1** s8 register per n-tile
(4 consecutive k of the n row), and C is s32 with the **same k16 layout as
the f32 C fragment** (the §4 layout, not the k8 one).

Quantization is per-row on *both* operands — `x_q[m][k] = round(X[m][k] /
sx[m])`, `w_q[n][k] = round(W[n][k] / sw[n])` — with the scale folded back
in the epilogue: `Y[m][n] = acc_s32 * sx[m] * sw[n]`. The int8 × int8 →
int32 path is exact through the whole K accumulation; the single rounding
happens once, at the final rescale. Same structure as decode v6 (dp4a):
integer accumulation, one rescale, and the same measured accuracy band —
**1.226% rel. err here, 0.85–1.20% in the decode experiment**.

**Result: 98.3 TFLOP/s, 45.2% of the hgemm peak — but only 1.26× over v5.**

The int8 TC rate on this part is 2× the fp16 rate at the hardware level
(~436 TOP/s spec vs 217.6 measured fp16 peak), and v6 issues the *same* 8
mma per k-tile as v5. If the TC pipe were the bottleneck, v6 would be ~2×
v5. It isn't there. The instruction-count accounting points at the fragment
loads:

```
per k-tile, per lane:
  v5:  6 x ldmatrix.x4   =  96 B moved, 16 B per load instruction
  v6: 12 x 4-byte smem ld =  48 B moved,  4 B per load instruction
```

v6 moves *half* the smem bytes but needs **twice the load instructions** for
it, and cp.async now has fewer bytes to hide behind (9.2 KB of operand
stream per two k-tiles vs 15.4 KB), so the consume side is exposed. Leading
hypothesis: v6 is smem-load-instruction-bound, which is why it sits at
~22.5% of the int8 peak while v5 sits at 35.9% of the fp16 peak. (That is
an instruction-count argument, not an ncu profile — verify in §9.)

The 2× is real; this data path cannot reach it. And quantization's price is
on the table: 1.226% vs 0.029% for the fp16 kernels.

## 6. Reading the table

```
kernel                     ms     TFLOP/s  % of hgemm  speedup  rel.err
v0 naive fp32            78.93      1.7      0.8%     1.00x   0.0001%
v1 fp32 smem tile         5.96     23.1     10.6%    13.25x   0.0001%
v2 fp16 smem tile         2.90     47.4     21.8%    27.19x   0.0147%
v3 fp16 reg-block 4x4     4.57     30.1     13.8%    17.26x   0.0293%
v4 fp16 cp.async dblbuf   3.33     41.3     19.0%    23.70x   0.0293%
v5 fp16 tensor core       1.76     78.1     35.9%    44.87x   0.0293%
v6 int8 tensor core       1.40     98.3     45.2%    56.44x   1.2259%
gemv-fp16 (decode v2)    21.51      6.4      2.9%     3.67x   0.0293%
cuBLAS sgemm (fp32)       2.37     58.0     26.6%    33.28x   0.0000%
cuBLAS hgemm (peak)       0.63    217.6    100.0%   124.94x   0.0359%
```

Three axes, in order of appearance:

1. **Bytes** — v1 → v2 is the last step that helps by moving fewer bytes.
   After that, the ladder stops buying bandwidth wins.
2. **SIMT efficiency** — v3 and v4 are null results: the bottleneck is
   warps-per-tile and scalar-load instruction issue, *not* smem byte
   volume (v3 proves that by cutting smem loads 4× and still losing) and
   *not* FMA count.
3. **FMA rate** — v5/v6: moving the FMAs onto the tensor cores is where
   the compute-bound payoff actually is (19.0% → 35.9% → 45.2% of peak).

One line deserves a comment: **cuBLAS sgemm (fp32) at 58.0 TF beats every
SIMT fp16 kernel in this file.** A hand-written scalar-fp16 kernel does not
beat a library fp32 GEMM; only the TC kernels (and v6) do.

The shape of the ladder is the mirror of the decode ladder: there, step one
was access-pattern engineering (8×) and most of the story was bytes; here,
tiling is a single 13× step and the rest of the story is TFLOP/s.

## 7. The M sweep: two crossovers

The harness re-times the fp16 candidates (gemv-fp16 from the decode
experiment + v5) at each prompt length — 10 probe iterations, 50 timed —
and reports the winner per M:

```
M      winner             ms      TFLOP/s   % of peak
1      gemv-fp16        0.0223      1.5       0.7%
2      gemv-fp16        0.0244      2.8       1.3%
4      gemv-fp16        0.0340      3.9       1.8%
8      v5 tensor core   0.0399      6.7       3.1%
16     v5 tensor core   0.0417     12.9       5.9%
64     v5 tensor core   0.0480     44.7      20.6%
128    v5 tensor core   0.0732     58.7      27.0%
256    v5 tensor core   0.1239     69.3      31.9%
512    v5 tensor core   0.2332     73.7      33.9%
1024   v5 tensor core   0.4583     75.0      34.5%
4096   v5 tensor core   1.7730     77.5      35.6%
```

**gemv-fp16 wins at M ≤ 4; v5 takes over from M = 8** and stays the winner
to M = 4096. That crossover is *not* the same object as the roofline knee:

- The **roofline knee** (M\* ≈ 256, where I(M) crosses 121 FLOP/byte) answers
  "at what prompt length does the layer *become* compute bound?" At M = 8
  the layer sits at ~8 FLOP/byte — a tenth of the way to the knee — yet the
  TC kernel already wins.
- The **kernel crossover** (M ≈ 4→8) answers "which kernel should I
  launch?". It comes far earlier because tiling changes the *traffic
  structure*, not just the FMA rate: a GEMV re-reads all of W once per
  output row (W is 33.6 MB, and although it L2-resides, every row's lanes
  pull it again), while v5's 64×32 tiles read W from HBM once and X once.
  The countervailing force at small M is tile padding: a 64×32 block tile
  computes `ceil(M/64)*64 × N × K` MACs, so at M = 4 it wastes 16× and at
  M = 8 it wastes 8×. The crossover is set by *padding waste vs TC rate*:
  at M = 4 the 16× waste kills v5; at M = 8 the 8× waste still beats a GEMV
  that re-reads W.

Both crossovers are measured in the log; a production selector needs both —
launch the GEMV below the kernel crossover, and expect the roofline story
(compute bound, TC-bound) only past the knee.

## 8. Debugging notes (what actually went wrong, in order)

1. **v5/v6: 132% rel. err — the k8 C-fragment epilogue on a k16 kernel.**
   §4 has the full story: plausible-looking output, and the timing-only
   sweep crowned the broken kernel "winner". Re-derived from the PTX ISA
   fragment tables; the fix is the 4-line C layout in §4.
2. **cp.async group-count drift.** Drop the trailing `else cp_commit()` and
   `wait_group 1` can return before the last tile lands — silent corruption
   that appears only at specific K/tile counts. (§3)
3. **Buffer-1 offset `2*ASZ` vs `ASZ+BSZ`.** A and B tiles differ in size;
   `2*ASZ` overlaps As1/Bs1 and the prefetch clobbers B. (§3)
4. **A comment that claimed a number.** The original header said "v6 buys
   exactly 2× over v5 (int8 TC rate)"; the measurement said 1.26× (§5).
   A comment that asserts a number is a testable claim; the table
   adjudicates. The comment now states the measured 1.26× and the
   smem-load-instruction bottleneck.
5. **Register/smem accounting.** Use `nvcc --ptxas-options=-v` on the
   *real* build line (`-O3`): v2 40 regs, v3 56, v4 78, v5 66, v6 46.
   An `-O0` build (there was a stale `bin/prefill_hotpath_O0` around) gives
   misleading counts — and its timings are meaningless.

## 9. What to try next

1. **cuBLAS int8 reference row** (`cublasGemmEx` / `cublasLt` with
   `CUDA_R_8I`): measure the int8 peak directly and v6's true % of it —
   the 22.5% figure above uses the spec peak.
2. **Widen v6's smem loads.** The 12 four-byte loads per k-tile are the
   measured bottleneck: the A fragment is 32 B and the B fragment 16 B per
   lane per k-tile, both contiguous — `uint4` 16-byte loads cut it to 3
   loads. That is the direct path to making the 2× int8 rate show up.
3. **Deeper pipeline.** 3–4 cp.async stages and/or a 64-wide k-tile, to
   amortize the fragment loads and the `__syncthreads()` across more work.
4. **ncu pass on v5/v6.** Confirm the smem-load-instruction hypothesis
   (v6) and audit bank conflicts: V5_PAD = 8 was chosen to be
   ldmatrix-conflict-free by construction; measure it.
5. **Blackwell 5th-gen TC path.** `tcgen05.mma` with tensor memory + TMA
   is the successor to the `mma.sync` register-fragment path on sm_120 and
   has a much higher ceiling; this ladder's 35.9%-of-peak is a 4th-gen
   register-path result.
6. **Accuracy gate in the sweep.** Refuse to report a "winner" whose
   rel. err vs the fp32 reference exceeds a threshold (say 1%). Today
   selection is timing-only — which is precisely how the k8/k16 bug got
   crowned champion.
7. **Non-power-of-two M** (1000, 3000): exercise the OOB zero-fill paths
   (`cp.async` with `src_bytes = 0`, M-boundary tiles). Every measured M
   here divides cleanly.
