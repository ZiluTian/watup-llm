# CUDA Profiling Tools — Seven Tools Against an Event Oracle

A study of the tools you reach for when the hot code is a `torch` loop
launching CUDA work. A small **decode-style workload** (pinned H2D → 2
GEMMs → pointwise → reduce → D2H, ~251 µs/iter, GPU ~101% busy, genuinely
async end-to-end — nothing in the loop synchronizes) is profiled by
**seven tools**, each scored on the same three axes as the Python
benchmark — **accuracy**, **overhead**, **usability** — against a
**CUDA-event oracle**: `torch.cuda.Event` pairs around each region,
stream-ordered hardware timestamps with negligible overhead. That is what
makes the tools disagree in interesting ways, and it matches how real
inference loops are written.

| Tool | Mode | What it gives you |
|------|------|-------------------|
| `wall_nosync` | perf_counter, no sync | CPU launch time only (the classic trap) |
| `wall_sync` | perf_counter + `torch.cuda.synchronize` | correct total, zero attribution |
| `cProfile` | in-process C tracer | CPU side: the loop is *waiting on the GPU* |
| `torch.profiler` | in-process (kineto) | per-kernel GPU times + CPU ops + chrome trace |
| `py-spy` | external CPU sampler | where the *host* spends its time (launch side); GPU invisible |
| `nsys` | external full-system trace | whole-process CPU+GPU timeline, near-zero overhead, CSV stats |
| `ncu` | per-kernel replay | hardware throughput/occupancy; needs counter permission (degrades gracefully) |

## Headline results (RTX 5090, torch 2.10.0+cu128; full run in `results/`)

Oracle (event-bracketed, µs/iter): h2d 41.0 (16.4%) · gemm 174.3 (69.5%) ·
pointwise 11.1 (4.4%) · reduce 15.8 (6.3%) · d2h 8.7 (3.5%) — 250.9 µs
total, ~250 µs/iter.

| tool | overhead | accuracy | usability | verdict |
|------|--------:|--------:|----------:|---------|
| wall_nosync | 1.00× | 15.1 | 2.20 | misses the async GPU tail — never ship this |
| wall_sync | 1.00× | 98.7 | 2.65 | right total, no attribution; the baseline everything is checked against |
| cProfile | 1.00× | — | 3.00 | CPU side only: shows the loop is *waiting* on the GPU |
| py-spy | 1.00× | — | 3.15 | near-zero-cost view of the launch side |
| **torch.profiler** | **1.32×** | **65.7** | **4.70** | per-kernel GPU times + CPU ops, in-process |
| **nsys** | **1.01×** | **65.8** | **4.25** | whole-process CPU+GPU timeline at ~zero overhead |
| ncu | — | — | 4.10 | deepest view; needs counter permission (degrades gracefully) |

Accuracy = 100 × (1 − median per-region relative error vs the event
oracle). "—" = the tool measures a different quantity (CPU time), so it
isn't scored against the GPU oracle.

The findings that matter:

1. **The async wall-clock trap is the most common CUDA timing bug.**
   Timed without a sync, one iteration is 37.0 µs — CPU launch cost only —
   while the real iteration is 245.0 µs: a **6.6× under-report**. The
   sync is not overhead; it *is* the measurement.
2. **cProfile on a torch loop is a *GPU-wait* profiler in disguise**: its
   top CPU entry is `torch.cuda.synchronize` (25.3 ms of 37.1 ms). The
   most useful thing it tells you here — the host is blocked on the GPU —
   but nothing about *which* kernel.
3. **Kernel-busy vs event-bracketed time**: kernel profilers report the
   execution time of each kernel; the event oracle brackets the whole
   region, so it also counts inter-kernel gaps + ~4–5 µs per event-pair.
   The two agree well for one big kernel (gemm: 5.5% / 6.1% error for
   torch.profiler / nsys) and diverge for small multi-kernel regions
   (pointwise: 44–55%). A "40% error" on a 10 µs region is almost all
   *fixed event overhead*, not misattribution — the profiler is right
   about kernel time, the oracle is right about latency; both matter in a
   decode loop.
4. **kineto's 1.32× steady-state overhead is real but modest** (one-time
   CUPTI init excluded; with it included, a short capture looks ~14×).
   For repeated profiling of a long loop, use it; for one 50-iteration
   capture, the init cost dominates.
5. **nsys is the external tool to reach for**: 1.006× round-wall
   overhead, 99.3% region coverage, per-kernel totals matching
   torch.profiler within ~1 µs/iter, and it sees the whole process
   (CUDA + CPU + OS) at ~zero cost.
6. **Memory**: host RSS says 1.06 GiB (CUDA context + pinned ring); the
   151 MiB of model state lives on the device, invisible to every
   host-side memory tool. `torch.cuda.memory_summary` is the other half
   of the story.

## Building & running

Requirements: Python 3.10+ with a CUDA build of `torch` and one NVIDIA
GPU (developed on an RTX 5090, sm_120, torch 2.10.0+cu128); optional
`pip install py-spy`. `nsys` / `ncu` from the NVIDIA Nsight packages;
ncu needs GPU performance-counter permission (root / `CAP_SYS_ADMIN`) and
the bench degrades gracefully without it (`ERR_NVGPUCTRPERM`).

```sh
make run           # full, ~40 s
make run-fast      # reduced sizes, ~30 s
# or: python3 src/cuda_prof_bench.py [--fast]
```

Artifacts: `results/cuda_prof_bench_rtx5090.log`,
`results/cuda_prof_bench.json`, and the raw py-spy dump. The nsys binary
(`.nsys-rep`) is regenerable and git-ignored — keep the log.

## Layout

```
src/cuda_prof_bench.py   the whole experiment: 7 tools vs a CUDA-event oracle
docs/
  tutorial-cuda-profiling.md
                        the guided walkthrough: the async trap, event vs
                        kernel-busy time, tool-by-tool, and the debugging
                        notes (what actually went wrong, in order)
results/                raw benchmark logs (RTX 5090) + JSON + py-spy dump
Makefile                run targets (run / run-fast)
```

## Skills on display

- GPU performance analysis: CUDA-event oracles, the async wall-clock
  trap, kernel-busy vs event-bracketed time, GPU-busy-fraction
  measurement, device memory vs host RSS
- Profiler benchmarking: scoring in-process (cProfile, kineto) and
  external (py-spy, nsys, ncu) tools on accuracy / overhead / usability
  against a hardware-timestamp reference
- Tooling internals: kineto/CUPTI init cost, nsys stats CSV parsing,
  memcpy naming across profilers, ncu counter permissions, graceful
  degradation of the whole pipeline when a tool is missing or blocked
