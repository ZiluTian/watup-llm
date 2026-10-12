# LLM Decode Hot Path — CUDA GEMV Optimization

A study of the single most important operation in single-user LLM inference:
the **decode linear layer** — `y = W @ x`, a batch-1 matrix-vector multiply
(GEMV) repeated once per layer, per generated token. On a modern GPU it is
*memory-bandwidth bound, not compute bound*: every byte of `W` must be pulled
from HBM and used in exactly one multiply. (In llama.cpp this is what
`mul_mat_vec` / `mul_mat_vec_q` do.)

The repo contains one self-contained experiment (`src/decode_hotpath.cu`) with
**twelve kernels**, each a single optimization on top of the last, benchmarked
side by side on the same data with a cache-cold measurement harness:

| # | Kernel | What it adds |
|---|--------|--------------|
| v0 | naive fp32 | one thread per output row — the classic un-coalesced mistake |
| v1 | warp-per-row fp32 | fixes coalescing; same bytes moved |
| v2 | warp-per-row fp16 | half the bytes moved |
| v3 | warp-per-row int8 | per-row scale; quarter the bytes |
| v4 | warp-per-row int4 | GGUF-style per-group scale (group=32); eighth the bytes |
| v5 | block-per-row fp16 | 256 threads/row + shared-memory x staging (latency hiding) |
| v6 | warp-per-row int8 | hardware DP4A; both operands quantized |
| v7 | warp-per-row int4 | 4-way unroll, independent accumulators (ILP) |
| v8 | warp-per-row int4 | vectorized uint32 word loads + float4 x loads — the instruction-bound fix |
| v11 | register-blocked fp16 | **batched** decode, B ≤ 16: B fp32 accumulators per lane in registers |
| v12 | lane-owned fp16 | **batched** decode, B ≥ 32: each lane owns B/32 batch entries, no shuffles |
| v13 | smem-tiled fp16 | **batched** decode, B ≥ 32: GEMM-style tiling, W fetched from HBM once per k-tile |

The batched sweep (v11–v13, B = 1…128) is **self-tuning**: for each B the
harness probes the applicable designs and times the winner, so the table shows
which structure wins on *your* GPU.

## Headline results (RTX 5090, 4096×4096, fp32 reference)

| kernel | ms | GB/s | % of HBM peak | speedup vs v0 | rel. error vs fp32 |
|--------|----:|-----:|-----:|-----:|-----:|
| v0 naive fp32 | 0.744 | 90 | 5.0% | 1.00× | 0.0001% |
| v1 warp fp32 | 0.091 | 740 | 41.3% | 8.2× | 0.0000% |
| v2 warp fp16 | 0.023 | 1470 | 82.0% | 32.6× | 0.021% |
| v3 warp int8 | 0.013 | 1301 | 72.6% | 57.6× | 0.85% |
| v4 warp int4 | 0.015 | 705 | 39.3% | 50.0× | 9.24% |
| v5 block smem fp16 | 0.090 | 372 | 20.7% | 8.2× | 0.021% |
| v6 dp4a int8 | 0.013 | 1329 | 74.2% | 58.9× | 1.20% |
| v7 ilp int4 | 0.012 | 839 | 46.8% | 59.5× | 9.24% |
| v8 vec int4 | 0.008 | 1255 | 70.0% | 89.0× | 9.24% |

Batched decode (fp16 weights), same GPU:

| B | winning kernel | ms | req/s | rel. error |
|--:|----------------|----:|-----:|-----:|
| 1 | v11 reg-block | 0.023 | 43.8k | 0.021% |
| 2 | v11 reg-block | 0.023 | 86.9k | 0.020% |
| 4 | v11 reg-block | 0.031 | 130k | 0.021% |
| 8 | v11 reg-block | 0.193 | 41.6k | 0.021% |
| 16 | v13 smem-tile | 0.535 | 29.9k | 0.021% |
| 32 | v13 smem-tile | 0.533 | 60.0k | 0.021% |
| 64 | v13 smem-tile | 0.472 | 135.7k | 0.021% |
| 128 | v13 smem-tile | 1.445 | 88.6k | 0.021% |

Two findings are the real point of the experiment:

1. **The B=1 ladder is a bandwidth story.** v0→v1 is pure access-pattern
   engineering (same bytes, 8× faster); v1→v8 is the quantization
   bytes-reduction ladder (each step trading measured accuracy for fewer
   HBM bytes) plus a second axis — v7 shows ILP alone does *not* beat v4,
   which is the tell that the kernel is instruction-bound rather than
   latency-bound, and v8 (vectorized loads, one shift-based group lookup)
   fixes exactly that.

2. **Batching has a wall, and it is not where you'd expect.** Throughput
   scales ~linearly with B only up to B≈4. From B=8 on, every output row has
   to touch the whole batched activation vector (`in_dim × B` floats), and
   that x-traffic plus SIMT instruction issue — not weight bandwidth or
   FLOPs — becomes the bottleneck. Three different kernel structures (v11,
   v12, v13) all stop at the same wall from different angles. What breaks
   through it is tensor-core GEMM (m16n8k16 MMA on the v13 tiling), which is
   what cuBLAS/vLLM actually run at large B. See the tutorial for the full
   derivation.

## Building & running

Requirements: CUDA toolkit 12+ (developed on CUDA 13.0), one NVIDIA GPU
(developed on an RTX 5090).

```sh
make                  # builds bin/decode_hotpath (ARCH defaults to sm_120)
./bin/decode_hotpath                        # 4096 x 4096
./bin/decode_hotpath 8192 16384             # 537 MB of fp32 weights
make ARCH=sm_89 run                         # e.g. RTX 4090
```

`in_dim` must be a multiple of 64. Raw logs for both shapes are in `results/`.

## Layout

```
src/decode_hotpath.cu   the whole experiment: kernels, quantizers, harness
Makefile                build (ARCH ?= sm_120)
results/                raw benchmark logs (RTX 5090)
docs/
  tutorial-decode-hotpath.md
                        the guided walkthrough: why decode is bandwidth-bound,
                        each kernel in detail, the measurement methodology,
                        and the batched-decode x-traffic wall
```

## Skills on display

- CUDA kernel design: memory coalescing, warp shuffles, shared memory,
  vectorized loads, register pressure, occupancy
- Inference quantization: fp16 / int8 (per-row) / int4 (per-group), DP4A,
  accuracy-vs-bandwidth trade-offs measured against an fp32 reference
- Performance analysis: roofline reasoning, L2-cache-defeating benchmark
  methodology, distinguishing bandwidth-bound vs instruction-bound vs
  register-spilling behavior from its timing fingerprint
- Batched (continuous-batching) decode: register blocking, lane-owned
  batches, GEMM-style smem tiling, and a self-tuning kernel selector
```
