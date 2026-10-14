# LLM Prefill Hot Path — CUDA GEMM Optimization

The same layer as the decode study, but with M = prompt length instead of
M = 1: `Y = X @ W^T` with `X [M x in_dim]`. Prefill ingests the whole prompt
in one shot, and at M = 4096 the layer sits at ~1365 FLOP/HBM byte — 11×
above this machine's roofline knee (~121, measured from a cuBLAS hgemm peak
of 217.6 TFLOP/s) — so it is **compute bound**, the mirror image of
decode's bandwidth story.

The repo contains one self-contained experiment (`src/prefill_hotpath.cu`)
with **seven hand-written kernels plus the decode GEMV and two cuBLAS
references**, each a single optimization on top of the last:

| # | Kernel | What it adds |
|---|--------|--------------|
| v0 | naive fp32 | one thread per output element, no tiling: the global-memory thrash baseline |
| v1 | fp32 smem tile 32×32 | shared-memory tiling: global traffic O(M·N·K) → O(M·N·K/tile) |
| v2 | fp16 smem tile | half the operand bytes, fp32 accumulation |
| v3 | fp16 reg-blocked 4×4 | each thread owns 4×4 outputs: 16 independent FMA streams (a null result here) |
| v4 | fp16 cp.async dblbuf | async global→smem copy of the next k-tile while the current one is consumed |
| v5 | fp16 tensor cores | mma.sync m16n8k16 + ldmatrix + cp.async double buffering: the compute-bound payoff |
| v6 | int8 tensor cores | mma.sync m16n8k16 s8/s32, per-row scales on both operands, fp32 rescale in the epilogue |
| gemv-fp16 | (decode v2) | warp-per-row GEMV, in the table so the M sweep shows where GEMM structure takes over |
| cuBLAS | sgemm / hgemm | the fp32 reference and the measured fp16 TC peak |

The prompt-length sweep is **self-tuning**: for each M the harness probes the
fp16 candidates (10 iterations) and times the winner (50 iterations).

## Headline results (RTX 5090, 4096×4096×4096, fp32 reference)

| kernel | ms | TFLOP/s | % of cuBLAS hgemm | speedup vs v0 | rel. error |
|--------|----:|-------:|------:|------:|------:|
| v0 naive fp32 | 78.93 | 1.7 | 0.8% | 1.00× | 0.0001% |
| v1 fp32 smem tile | 5.96 | 23.1 | 10.6% | 13.25× | 0.0001% |
| v2 fp16 smem tile | 2.90 | 47.4 | 21.8% | 27.19× | 0.0147% |
| v3 fp16 reg-block 4×4 | 4.57 | 30.1 | 13.8% | 17.26× | 0.0293% |
| v4 fp16 cp.async dblbuf | 3.33 | 41.3 | 19.0% | 23.70× | 0.0293% |
| v5 fp16 tensor core | 1.76 | 78.1 | 35.9% | 44.87× | 0.0293% |
| v6 int8 tensor core | 1.40 | 98.3 | 45.2% | 56.44× | 1.2259% |
| gemv-fp16 (decode v2) | 21.51 | 6.4 | 2.9% | 3.67× | 0.0293% |
| cuBLAS sgemm (fp32) | 2.37 | 58.0 | 26.6% | 33.28× | 0.0000% |
| cuBLAS hgemm (peak) | 0.63 | 217.6 | 100.0% | 124.94× | 0.0359% |

Three findings are the point of the experiment:

1. **The bytes payoff stops at v2.** v1 → v2 is the last step that helps by
   moving fewer bytes; v3 (register blocking) is *slower* than the plain
   tile even though it cuts smem loads per FMA by 4× — with SIMT fp16 the
   bottleneck is warps-per-tile and scalar-load issue, not smem traffic or
   FMA count. The compute-bound payoff only arrives at v5, when the tensor
   cores absorb the FMA work.
2. **The int8 TC rate does not show up as 2×.** v6 is 1.26× over v5
   (98.3 vs 78.1 TFLOP/s), not 2×: the int8 TC rate is 2× the fp16 rate at
   the hardware level, but v6's narrow 4-byte smem fragment loads (12 per
   k-tile vs v5's 6 ldmatrix.x4) throttle the pipeline before the TC issue
   rate does — and the quantization price is real: 1.23% vs 0.029%.
3. **The decode kernel still wins short prompts.** The self-tuning sweep has
   gemv-fp16 winning for M ≤ 4 and v5 (tensor-core GEMM) taking over from
   M = 8 — while the roofline arithmetic-intensity knee sits at M\* ≈ 256.
   The kernel crossover comes long before the layer is formally
   "compute bound", because GEMM tiling reads W from HBM once while a GEMV
   re-reads it once per output row (at M = 4, though, the 64-row tile's
   16× padding waste kills v5). See the tutorial
   (`docs/tutorial-prefill-hotpath.md`) for the full derivation of both
   crossovers.

## Building & running

Requirements: CUDA toolkit 12+ (developed on CUDA 13.0), one NVIDIA GPU
(developed on an RTX 5090).

```sh
make                  # builds bin/prefill_hotpath (ARCH defaults to sm_120)
./bin/prefill_hotpath                     # 4096 x 4096 x 4096
./bin/prefill_hotpath 8192 4096 4096      # M = 8192 (long-prompt prefill)
make ARCH=sm_89 run                       # e.g. RTX 4090
```

`K` must be a multiple of 32. Raw logs are in `results/`.

## Layout

```
src/prefill_hotpath.cu  the whole experiment: 7-kernel GEMM ladder, int8
                        quantizer, M sweep, roofline summary
Makefile                build (ARCH ?= sm_120)
results/                raw benchmark logs (RTX 5090)
docs/
  tutorial-prefill-hotpath.md
                        the guided walkthrough: why prefill is compute-bound,
                        tensor-core fragment layouts (and the k8/k16
                        C-fragment trap), why int8 TC does not show up as 2x
```

## Skills on display

- GEMM kernel design: smem tiling, cp.async double buffering, ldmatrix /
  mma tensor-core fragments (and the k8/k16 C-fragment trap), int8 TC with
  exact int32 accumulation and an epilogue dequant
- Roofline reasoning across the compute/bandwidth divide: decode (M=1,
  ~1 FLOP/byte) vs prefill (M=4096, ~1365 FLOP/byte), and why the kernel
  crossover (M ≈ 4→8) comes long before the arithmetic-intensity knee
  (M\* ≈ 256)
