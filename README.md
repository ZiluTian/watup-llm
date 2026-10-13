# WATuP-LLM

A collection of self-contained studies in **LLM inference engineering** — CUDA kernel design, the decode hot path, quantization, and the performance analysis that connects them. Each folder is a standalone experiment with its own README, build, source, and raw results.

## Contents

| Folder | Study |
|--------|-------|
| [`decode-hotpath/`](decode-hotpath/) | The decode linear layer (`y = W @ x`, batch-1 GEMV) — the single most important op in single-user LLM inference. Twelve CUDA kernels as an optimization ladder (coalescing → fp16/int8/int4 quantization → DP4A → vectorization), a batched-decode sweep (register blocking, lane-owned batches, smem tiling), a cache-cold measurement methodology, and a guided tutorial. Includes raw benchmark logs (RTX 5090). |
| [`python-profiling/`](python-profiling/) | Eight Python profiling methods (cProfile, line_profiler, tracemalloc, memory_profiler, py-spy, timeit, wall, resource) compared on accuracy / overhead / usability across nine common-pattern workloads, with ground truth known by construction. Includes raw benchmark logs (CPython 3.12.11). |
| [`cuda-profiling/`](cuda-profiling/) | Seven CUDA profiling tools (wall ± sync, cProfile, py-spy, torch.profiler, nsys, ncu) scored on the same three axes against a CUDA-event oracle on an async decode-style torch loop. Includes raw benchmark logs (RTX 5090). |

More folders are on the way.

## Conventions

Each study folder follows the same shape so results are easy to compare and reproduce:

```
<study>/
  README.md          what, why, headline results, how to build & run
  src/               the experiment (kernels, models, harness)
  docs/              guided walkthroughs and derivations
  results/           raw benchmark logs, tagged by GPU & shape
  Makefile           build (if compiled)
```

Every reported number names the device, shape, and (where relevant) the measurement methodology, so a result can be re-derived on different hardware.
