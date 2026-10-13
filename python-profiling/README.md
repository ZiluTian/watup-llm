# Python Profiling Methods — Eight Tools, Three Axes, Nine Patterns

A study of how well the Python profiling tools actually work: **eight
methods** compared on **accuracy** (against ground truth), **overhead**
(wall-clock slowdown), and **usability** (a rubric), across **nine
workloads** covering the common programming patterns where profilers
disagree — tight numeric loops, allocation churn, C-extension-heavy code,
dispatch-heavy call soup, recursion, I/O wait, generators, vectorized
numpy, and threads.

The key design choice is **ground truth by construction**. Every workload is
a sequence of *regions*, and each region does exactly `k` iterations of a
*calibrated* unit operation (unit cost measured with `perf_counter` under
the same conditions as the method passes), so
`expected(W, r) = k_r × unit_cost`. A profiler is checked against a number
computed before any profiler ran — never against another profiler. One
self-contained experiment (`src/python_prof_bench.py`) does it all:
calibrate the unit costs, run the bare baselines, run every method, and
score.

| Method | What it is | What it sees |
|--------|-----------|--------------|
| `wall` | `perf_counter` around the call | total time only — never "why" |
| `timeit` | `timeit.Timer`, gc off (its standard mode) | total time, tuned for micro-benchmarks |
| `cProfile` | stdlib C tracer, function-level | CPU time per function; blind to C-extension *internals* |
| `line_profiler` | per-line bytecode instrumentation | line-level CPU time |
| `tracemalloc` | stdlib allocation tracing | Python-level bytes per allocation site |
| `memory_profiler` | per-line process RSS (psutil) | page-level RSS deltas — includes C-extension memory, misses churn |
| `py-spy` | external 200 Hz sampler (subprocess mode) | CPU samples across *all* threads; zero in-process distortion |
| `resource` | POSIX `getrusage` | process CPU time, peak RSS — a one-liner sanity check |

The workloads: W1 tight numeric loop · W2 allocation churn + big allocs ·
W3 C-extension heavy (`sum(1M)`, `json`, `re`) · W4 dispatch-heavy (900k
tiny calls) · W5 recursion (tree build + walk) · W6 I/O wait (reads +
sleeps) · W7 generators vs plain loop · W8 vectorized numpy vs 500k-op
loop · W9 threads (informational only). W6 separates *wall* from *CPU*
time, W4 separates *call count* from *work done*, W8 separates
*Python-level* from *C-extension* time — the axes on which profilers
actually disagree.

## Headline results (CPython 3.12.11; full run in `results/`)

| method | overhead | accuracy | usability | verdict |
|--------|--------:|--------:|----------:|---------|
| wall | 1.00× | 96.4 | 2.35 | answers "how long", never "why" |
| timeit | 0.93× | 94.9 | 2.75 | micro-benchmark, not a profiler |
| **cProfile** | **1.55×** | **92.6** | **4.00** | first-stop function-level answer; taxes call count |
| line_profiler | 3.87× | 50.9 | 3.95 | targeted line drill-down, never whole-program |
| tracemalloc | 4.49× | 91.4 | 3.85 | the memory tool: who allocates what |
| memory_profiler | 107.49× | 0.0 | 3.10 | coarse RSS story incl. C-extension memory; blind to churn |
| py-spy | 1.01× | 74.0 | 3.85 | zero-distortion sampling across threads; attach to live processes |
| resource | — | 99.4 | 2.65 | one-liner sanity check: CPU vs wall, peak RSS |

Accuracy is share-based per-region fidelity against the constructed ground
truth (top-1 hot-region hit rate: cProfile 5/8, py-spy 6/7, line_profiler
4/8). The memory / sampling tools are the noisy ones by nature:
memory_profiler's RSS-delta and py-spy's sample-fraction scores vary a few
points per run — the log above is one full run.

The findings that matter:

1. **cProfile is the default, and it's good — but it taxes call count.**
   Geomean overhead 1.55×, yet per-workload it ranges 0.99–3.10×:
   dispatch-heavy W4 costs 2.48×, recursive W5 3.10×, tight-loop W1
   nothing (0.99×). The tracer's cost is per *event*, and every Python
   call is an event.
2. **line_profiler is a drill-down tool, not a program tool.** 3.87×
   geomean (up to 15.2× on W4, 13.5× on W7) and its absolute accuracy
   collapses (median |err| 500%): on a tight loop the per-line-event cost
   *is* most of the reported time. Use it on the 20 lines cProfile
   pointed at, not on the whole program.
3. **tracemalloc and memory_profiler measure different things and both
   are "right".** tracemalloc (91.4) sees the churn; memory_profiler
   (0.0 this run) sees `w2_big` at 15.9 MiB when 72 MiB was allocated,
   and the three small-churn regions at **0.0 MiB** even though each
   allocated ~10 MB — after churn the allocator reuses
   freed-but-mapped memory, so RSS never moves. That gap (it can range
   from "sees half" to "sees nothing" run to run) is the reason
   allocation tracing exists.
4. **py-spy is the only method that sees other threads** (W9: cProfile
   sees only the main thread) and the only one with ~1.0× overhead — at
   the price of statistical noise (region time = sample fraction ×
   window) and blindness to C-extension internals and the GPU.
5. **W6 is the wall-vs-CPU discriminator**: 331 ms wall vs 2 ms CPU
   (0.6% busy). Only `resource` (and a good profiler) shows the code is
   *waiting*, not computing.
6. **The instrumentation-overhead trap**: the same 600k `add3` ops as one
   function vs 200k tiny calls — the real call-overhead is 1.10×, but
   cProfile reports a 1.58× measured ratio (1.75× relative-ranking
   inflation). An instrumentation profiler distorts the *magnitude* of
   dispatch cost, not its direction — measured rather than hand-waved.

## Building & running

Requirements: Python 3.10+ with `numpy`; optional
`pip install py-spy line_profiler memory_profiler` (the bench degrades
gracefully without any of them; under `ptrace_scope=1` py-spy must launch
its child itself).

```sh
make run           # full, ~90 s
make run-fast      # reduced sizes, ~40 s
# or: python3 src/python_prof_bench.py [--fast]
```

Artifacts: `results/python_prof_bench_rtx5090.log` (text),
`results/python_prof_bench.json` (machine-readable), plus the raw py-spy
dump and flame SVG.

## Layout

```
src/python_prof_bench.py   the whole experiment: 8 methods x 9 workloads + ground truth
docs/
  tutorial-python-profiling.md
                        the guided walkthrough: ground-truth-by-construction,
                        the method ladder, the scorecard, the trap, and the
                        pitfalls that actually bit
results/                raw benchmark logs (RTX 5090 box) + JSON + py-spy artifacts
Makefile                run targets (run / run-fast)
```

## Skills on display

- Measurement methodology: ground truth by construction (no
  profiler-vs-profiler validation), absolute vs share-based accuracy,
  overhead measurement that survives sampling tools
- Python profiling: cProfile vs line_profiler vs tracemalloc vs
  memory_profiler vs py-spy vs timeit/resource — what each sees (and
  doesn't see: C-extension internals, other threads, allocation churn)
- Instrumentation-aware benchmarking: calibration under matched
  conditions, BLAS thread pinning, the ptrace_scope workaround, and the
  instrumentation-overhead trap
