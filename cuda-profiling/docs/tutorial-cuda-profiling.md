# Profiling CUDA: Seven Tools Against an Event Oracle

A guided walkthrough of `src/cuda_prof_bench.py`. All numbers below are from
an RTX 5090 (sm_120), torch 2.10.0+cu128; the script re-measures everything on
whatever GPU you run it on.

Companion to `tutorial-python-profiling.md`: same three axes —
**accuracy / overhead / usability** — applied to the tools you reach for when
the hot code is a `torch` loop that launches CUDA work.

## 1. The async trap (why wall-clock lies)

A typical inference loop is written *async*: copies and GEMMs are enqueued
with `non_blocking=True`, nothing inside the loop synchronizes, and the host
runs ahead. Now time one iteration with `perf_counter` and **no sync**:

```
nosync 1 iter  :  37.0 us     <- CPU launch cost only
sync   1 iter  : 245.0 us     <- the real iteration
trap           :  6.6x under-report
```

The 37.0 µs is just the CPU enqueueing ~10 CUDA ops; the GPU is still busy
for another ~208 µs after the host returns. Any "profiler" that is really a
wall-clock timer without a sync silently under-reports every async iteration
by the length of its un-finished tail. This is the single most common CUDA
timing bug, and the benchmark's first row exists to measure how badly it
bites (score: 15.1/100).

## 2. The workload

Per iteration (B=128, D=4096, fp32, ~250 µs of GPU work):

```
h2d       pinned CPU -> device copy of x          (async, 2 MB)
gemm      y = W1 @ x ; y = W2 @ y                 (2 cuBLAS/cutlass GEMMs)
pointwise y = (y * b).relu() + c                  (3 elementwise kernels)
reduce    y = y / y.norm(dim=0, keepdim=True)     (reduction + division)
d2h       async copy of a slice back to pinned    (async)
```

All copies use pinned memory and a rotating ring of 8 host batches, so the
loop is *genuinely* async end-to-end — nothing in the loop synchronizes.
N=150 iterations per pass. Measured GPU-busy fraction is ~101% (the CPU launch
side, 37 µs/iter, is fully hidden under the ~251 µs of GPU work), which is the
regime where the tools disagree in interesting ways.

## 3. The oracle: CUDA events, and why it's not the same as kernel time

`measure_oracle()` records a `torch.cuda.Event` pair around each region for 3
passes of N iterations and reports mean per-region device time. Events are
stream-ordered hardware timestamps with negligible overhead — the best
practical ground truth for **region latency**:

```
h2d 41.0 us (16.4%)   gemm 174.3 us (69.5%)   pointwise 11.1 us
reduce 15.8 us (6.3%)  d2h 8.7 us (3.5%)      total 250.9 us/iter
```

**Kernel-busy vs event-bracketed.** This is the subtlety that explains most of
the "error" column in the scorecard. Kernel profilers (torch.profiler, nsys,
ncu) report the *execution time of each kernel*; an event pair reports the
*elapsed stream time* across the whole region, which also includes
inter-kernel gaps and a ~4–5 µs per event-pair overhead. The two agree well
for one big kernel and diverge for small/multi-kernel regions:

| region | oracle | torch.profiler | nsys | why |
|--------|-------:|---------------:|-----:|-----|
| gemm | 174.3 | 164.8 (5.5%) | 166.0 (6.1%) | one big kernel: gap is small |
| h2d | 41.0 | 37.7 (8.1%) | 37.7 (9.6%) | one DMA: event overhead only |
| pointwise | 11.1 | 6.2 (44.4%) | 6.2 (54.5%) | 3 kernels: 2 gaps + overhead |
| reduce | 15.8 | 10.3 (34.3%) | 10.4 (34.2%) | reduction + division kernels |
| d2h | 8.7 | 5.1 (41.6%) | 5.1 (44.7%) | small DMA: overhead dominates |

So a "40% error" on a 10 µs region is almost all *fixed event overhead*, not
misattribution — the profiler is telling you the right thing about kernel
time, and the oracle is telling you the right thing about latency. Both
matters in a decode loop; the benchmark reports both.

## 4. The tools

| Tool | Mode | Overhead | What it gives you |
|------|------|--------:|-------------------|
| `wall_nosync` | perf_counter, no sync | 1.00× | CPU launch time only (the trap) |
| `wall_sync` | perf_counter + sync | 1.00× | correct total, zero attribution |
| `cProfile` | in-process C tracer | 1.00× | CPU side: `synchronize` 25.3 ms of 37.1 ms — the loop is *waiting on the GPU*, not computing |
| `torch.profiler` | in-process (kineto) | 1.32× | per-kernel GPU times + CPU ops + chrome trace; the only in-process GPU view |
| `py-spy` | external CPU sampler | 1.003× | where the *host* spends its time (launch side); GPU invisible |
| `nsys` | external full-system trace | 1.006× | whole-process CPU+GPU timeline, near-zero overhead, CSV stats |
| `ncu` | per-kernel replay | — | hardware throughput/occupancy; needs counter permission (blocked in this container: `ERR_NVGPUCTRPERM`) |

Plus a **device-memory** check: 151 MiB peak on the device while host RSS
reports 1081 MiB of *context + pinned buffers* — the GPU's working set is
invisible to every host-side memory tool.

## 5. The scorecard (RTX 5090)

| Tool | Overhead | Accuracy | Usability | Verdict |
|------|--------:|--------:|----------:|---------|
| wall_nosync | 1.00 | 15.1 | 2.20 | looks fast because it misses the async tail — never ship this |
| wall_sync | 1.00 | 98.7 | 2.65 | right total, no attribution; the baseline |
| cProfile | 1.00 | — | 3.00 | sees the CPU launch cost of aten ops; blind to kernels |
| py-spy | 1.00 | — | 3.15 | near-zero-cost CPU view of the launch side |
| **torch.profiler** | **1.32** | **65.7** | **4.70** | per-kernel GPU times + CPU ops; in-process |
| **nsys** | **1.01** | **65.8** | **4.25** | whole-process CPU+GPU timeline, near-zero overhead |
| ncu | — | — | 4.10 | deepest view; needs counter permission + replay |

(Accuracy = 100 × (1 − median per-region relative error vs the event oracle);
the kernel-busy bias of section 3 caps it for small regions. "—" = the tool
measures a different quantity (CPU time), so it isn't scored against the GPU
oracle.)

The findings that matter:

1. **For a torch loop, `wall_sync` is the cheapest correct total** (98.7
   accuracy, 1.00× overhead) — and `wall_nosync` is a 6.6× lie. The sync is
   not overhead; it's the measurement.
2. **cProfile on a torch loop is a *GPU-wait* profiler in disguise**: its top
   CPU entry is `torch.cuda.synchronize` (25.3 ms of 37.1 ms). That's the
   most useful thing it can tell you here — the host is blocked on the GPU —
   but it tells you nothing about *which* kernel.
3. **kineto's 1.32× steady-state overhead is real but modest** (one-time
   CUPTI init excluded; with it included, a short capture looks ~14×).
   For repeated profiling of a long loop, use it; for one 50-iteration
   capture, the init cost dominates.
4. **nsys is the external tool to reach for**: 1.006× round-wall overhead,
   99.3% region coverage, per-kernel totals that match torch.profiler within
   ~1 µs/iter. It sees the whole process (CUDA + CPU + OS) at ~zero cost,
   which is exactly the property you want from a sampler.
5. **py-spy on a CUDA loop sees the launch side**: gemm owns 10.3% of CPU
   samples — coincidentally the right hotspot, but for the *CPU* reason
   (matmul dispatch is the most expensive launch), not because it sees the
   GPU. Don't read a CPU sampler as a GPU profiler.
6. **Memory**: host RSS tools see 1.06 GiB (context + pinned ring); the
   151 MiB of model state lives on the device. `torch.cuda.memory_summary`
   is the other half of the story.

## 6. What each tool sees as "the hot thing"

Ground truth: **gemm owns 69% of GPU time.**

```
wall_nosync    : one iteration = 37.0 us (6.6x under-report of everything)
cProfile       : run_loop / synchronize  - CPU side only
py-spy         : gemm, 10% of CPU samples (launch side)
torch.profiler : cutlass_80_simt_sgemm_128x128  164.8 us/iter   <- correct
nsys           : cutlass_80_simt_sgemm_128x128  166.0 us/iter   <- correct
device mem     : 151 MiB on the device, invisible to host RSS
```

Both GPU-capable tools name the exact kernel (`cutlass_80_simt_sgemm_128x128_
8x4_nn`, the SIMT fp32 path cuBLAS picks for this shape on sm_120) at the
right time. The tools that can't see the GPU are at least honest about it.

## 7. Debugging notes (what actually went wrong, in order)

The first end-to-end run had every bug class a CUDA benchmark can have:

- **Memcpy naming**: torch's profiler calls copies `Memcpy HtoD (Pinned ->
  Device)` / `Memcpy DtoH ...`, nsys calls them `[CUDA memcpy
  Host-to-Device]`. A classifier written against `H2D`/`D2H` (digit two)
  matched neither, so H2D iteration boundaries were invisible and every
  region came out 0.0. The classifier now accepts `h2d|htod|host-to-device`
  and `d2h|dtoh|device-to-host`.
- **nsys stats CSV**: nsys 2025.x prints a `Generating SQLite ...` /
  `Processing ...` preamble with no `** Report` block markers, and the mem
  report's label column is `Operation`, not `Name`. The parser now anchors
  on the `Total Time (ns)` header line and accepts either label column.
- **Stale SQLite export**: re-running `nsys stats` on a fresh `.nsys-rep`
  silently no-ops if an older `.sqlite` exists; `--force-export=true` is
  required for reproducibility.
- **µs vs s**: `dur_us()` takes seconds; three call sites passed
  microseconds, printing the GEMM as "165226.9 ms". Unit-checked every
  time-value path.
- **Dividing by the wrong count**: nsys records the child's *entire*
  lifetime (warmup + oracle + measured rounds), so per-region totals must be
  divided by **total iterations recorded** (30 + 20 + 3·n + rounds·n), not by
  the number of measured rounds; and the wall passes had measured
  `N_ITERS=150` while the bench divided by `n` — a 3× inflation that made
  `wall_sync` score 0.0/100.
- **kineto init in the overhead**: the first `profile()` context pays a
  one-time CUPTI cost (~14× on a short loop); a throwaway 5-iteration
   profiled warm-up is done first so the reported 1.32× is steady-state.
- **`other` bucket**: nsys's `[CUDA memset]` total (allocation zeroing) was
  compared raw against per-iteration values, collapsing "coverage" to 2.3%;
  it's normalized per iteration now (coverage 99.3%).

## 8. What to try next

- **ncu with counter permission** (`CAP_SYS_ADMIN` / `--cap-add=SYS_ADMIN`
  for the container): per-kernel DRAM throughput %, occupancy, and whether
  the cutlass SIMT GEMM is bandwidth- or issue-bound. The bench already
  scopes it to 12 launches of a small child and degrades gracefully until
  then.
- **A compute-bound variant** (B=4096, tensor-core GEMM) to show the tools
  on the other side of the roofline.
- **nsys `--trace=cudnn` / NVTX ranges**: tag the five regions with
  `torch.cuda.nvtx.range_push` and let nsys attribute by *name* instead of by
  kernel-name heuristics — the robust version of what `_segment_iters` does
  from timing boundaries.
- **Compare the oracle to ncu's `gpu__time_duration`** once permission is
  available, to quantify the event-overhead term of section 3 directly.
