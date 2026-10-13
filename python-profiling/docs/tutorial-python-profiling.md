# Profiling Python: A Benchmark of Eight Methods

A guided walkthrough of `src/python_prof_bench.py`. All numbers below are from
a CPython 3.12.11 run (RTX 5090 box, but the workload is CPU work); the script
re-measures everything on whatever machine you run it on.

The question the benchmark answers: **if you profile Python code, which tool
gives you the right answer, for what it costs, and how hard is it to use?**
Eight methods, three axes.

## 1. The three axes

- **Accuracy** — how close is the reported per-region time (or bytes) to the
  true value. This is the hard one: a profiler has no ground truth unless you
  build one (section 3).
- **Overhead** — the wall-clock slowdown factor of running the code under the
  method, per workload. Reported as a ratio (1.0 = invisible); the
  *geomean* across workloads is the headline.
- **Usability** — a 1–5 rubric per method: setup effort, granularity,
  actionability (does the output point at the fix?), scope, machine
  readability.

## 2. The workloads: nine common patterns

Each workload is a mix of *regions* with deliberately different "characters",
so no single tool wins everywhere:

| W | Pattern | Regions | Bare time |
|---|---------|---------|----------:|
| W1 | tight numeric loop | 5 loop variants (int math, bounds kept constant) | 47.0 ms |
| W2 | allocation churn | small alloc/free loops + 60 big 1 MiB allocs | 38.8 ms |
| W3 | C-extension heavy | `sum(1M)`, `json.loads(100KB)`, `re` on 100 KB | 626.6 ms |
| W4 | dispatch-heavy | 900k tiny calls + 800k ops (call overhead dominates) | 49.1 ms |
| W5 | recursion | build + walk an 8-level tree, ×300 | 19.7 ms |
| W6 | I/O wait | 500 file reads + 600 sleeps | 331.8 ms |
| W7 | generators | generator chain vs plain loop | 24.8 ms |
| W8 | vectorized | 200 × `numpy` 1024³ matmul vs 500k-op loop | 1552.6 ms |
| W9 | threads | 4 worker threads + main (informational, not scored) | 43 ms |

W6 separates *wall* from *CPU* time (sleeping); W4 separates *call count*
from *work done*; W8 separates *Python-level* from *C-extension* time. Those
are the axes on which profilers actually disagree.

## 3. Ground truth by construction

You can't validate a profiler against another profiler. So every region does
exactly `k` iterations of a canonical unit op whose cost is **calibrated**
with `perf_counter` (median of 5, gc on — same conditions as the method
passes): `expected(W, r) = k_r × unit_cost`. Example unit costs measured:
add3 iter 42.4 ns, small alloc 52.5 ns, `json(100KB)` 209 µs, `mm(1024³)`
7.8 ms.

Two views of accuracy are reported for region-level methods:

- **abs**: `|reported − expected| / expected`. Total fidelity — a method whose
  per-event overhead inflates reported times is penalized.
- **share**: `|f − e|/e` where `f = reported/Σreported`, `e = expected/Σexpected`.
  Hotspot-shape fidelity — does the method put the right *fraction* of the run
  in each region? (This one is the headline score: it's scale-invariant and
  answers "where is the time actually going?")

There's also a **top-1 hit rate**: does the method's #1 region match the
ground-truth #1 region? (cProfile 5/8, py-spy 6/7, line_profiler 4/8.)

## 4. The eight methods

| Method | What it is | What it sees |
|--------|-----------|--------------|
| `wall` | `perf_counter` around the call | total time only — never "why" |
| `timeit` | `timeit.Timer`, gc off (its standard mode) | total time, tuned for micro-benchmarks |
| `cProfile` | stdlib C tracer, per-function cumulative | CPU time per function; blind to C-extension *internals* (a `json.loads` call is one entry) |
| `line_profiler` | per-line bytecode instrumentation | line-level CPU time |
| `tracemalloc` | stdlib allocation tracing | Python-level bytes per allocation site |
| `memory_profiler` | per-line process RSS (psutil) | page-level RSS deltas — includes C-extension memory, misses churn |
| `py-spy` | external 200 Hz sampler (subprocess mode) | CPU samples across *all* threads; zero in-process distortion |
| `resource` | POSIX `getrusage` | process CPU time, peak RSS — a one-liner sanity check |

## 5. The scorecard (CPython 3.12.11)

| Method | Overhead | Accuracy | Usability | Verdict |
|--------|--------:|--------:|----------:|---------|
| wall | 1.00× | 96.4 | 2.35 | answers "how long", never "why" |
| timeit | 0.93× | 94.9 | 2.75 | micro-benchmark, not a profiler |
| **cProfile** | **1.55×** | **92.6** | **4.00** | first-stop function-level answer |
| line_profiler | 3.87× | 50.9 | 3.95 | targeted line drill-down only |
| tracemalloc | 4.49× | 91.4 | 3.85 | the memory tool: who allocates what |
| memory_profiler | 107.5× | 0.0 | 3.10 | coarse RSS story; blind to churn |
| py-spy | 1.01× | 74.0 | 3.85 | zero-distortion sampling, all threads |
| resource | — | 99.4 | 2.65 | CPU-vs-wall + peak RSS check |

(The RSS/sampling tools are noisy by nature: memory_profiler's score depends
on how much the allocator reuses freed pages in a given run, and py-spy's on
the sample mix. These two vary a few points per run; the rest are stable.)

The findings that matter:

1. **cProfile is the default, and it's good — but it taxes call count.**
   Its geomean overhead is 1.55×, yet per-workload it ranges 0.99–3.10×:
   dispatch-heavy W4 costs 2.48× and recursive W5 3.10×, while tight-loop W1
   costs nothing (0.99×). The tracer's cost is per *event*, and every Python
   call is an event. (See the trap in section 6.)
2. **line_profiler is a drill-down tool, not a program tool.** 3.87× geomean
   (up to 15.2× on W4 and 13.5× on W7) and its abs accuracy collapses (median
   |err| 500%): on a tight loop the per-line-event cost *is* most of the
   reported time. Use it on the 20 lines cProfile pointed at, not on the whole
   program.
3. **tracemalloc and memory_profiler measure different things and both are
   "right".** tracemalloc (Python-level, 91.4) sees the churn;
   memory_profiler (RSS deltas, 0.0 this run) sees `w2_big` at 15.9 MiB when
   72 MiB was allocated, and the three small-churn regions at **0.0 MiB** even
   though each allocated ~10 MB — because after churn the allocator reuses
   freed-but-mapped memory, so RSS never moves. That gap (it can range from
   "sees half" to "sees nothing" run to run) is the reason allocation tracing
   exists.
4. **py-spy is the only method that sees other threads** (W9: cProfile sees
   only the main thread) and the only one with ~1.0× overhead — at the price
   of statistical noise (it's a sampler: region time = sample fraction ×
   window) and blindness to C-extension internals and the GPU.
5. **W6 is the wall-vs-CPU discriminator**: 331 ms wall vs 2 ms CPU (0.6%
   busy). Only `resource` (and a good profiler) shows the code is *waiting*,
   not computing.
6. **W9 (GIL)**: 4 threads + main, 43 ms wall vs 42 ms process CPU — for
   CPU-bound Python, threads buy nothing; CPU ≈ wall is the fingerprint.

## 6. The instrumentation-overhead trap (fine vs coarse)

Same work, two shapes — 600k `add3` ops:

```
one function doing it all   : bare 24.7 ms | cProfile 24.6 ms (1.00x)
the same ops as 200k calls  : bare 27.2 ms | cProfile 43.0 ms (1.58x)
```

The *real* call-overhead is 1.10× (bare fine vs bare coarse). cProfile turns
that 1.10× into a **1.58×** measured ratio (a 1.75× relative ranking
inflation) — it amplifies the apparent cost of dispatch by ~1.6×. If you're comparing "one big function vs many small
calls", an instrumentation profiler is lying to you about *how much worse*
the small calls are (it won't lie about the *direction*, just the magnitude).
This is the standard "profiler overhead distorts relative rankings" trap,
measured rather than hand-waved.

## 7. Pitfalls that actually bit (and how the bench handles them)

- **Python 3.12 `pstats` layout**: stat keys are `(filename, lineno, name)`
  tuples and the value tuple is `(ncalls, cc, tt, ct, callers)` — cumulative
  time is index 3, not 2. Older tutorials index it wrong.
- **OpenBLAS spin-wait threads** keep burning process CPU long after the last
  matmul, polluting `resource` counters. The bench pins BLAS to 1 thread
  *before* importing numpy.
- **timeit's gc**: `timeit.Timer` unconditionally disables gc (there is no
  `gc=` kwarg in 3.12). The bench calibrates with gc *on* and notes the
  difference; comparing timeit numbers to gc-on numbers is a category error.
- **`bytes(n)` never touches pages**: a big `bytes` allocation shows no RSS
  growth until written. The big-alloc region uses `bytearray(n)` (zero-fill)
  so RSS is honest.
- **line_profiler at scale**: ~340 ns per line-event means it would dominate
  a 50 ms tight loop. The bench runs it at reduced sizes (×0.1–×0.25);
  accuracy is scale-invariant, overhead is reported at that size.
- **py-spy under `ptrace_scope=1`** can't attach to arbitrary PIDs: it must
  *launch* the child. The child prints one JSON line per completed round, so
  only complete rounds count toward the overhead ratio (py-spy kills the
  child at `--duration`).
- **memory_profiler is decorator-only** in current versions: `@mp.profile`
  with a `stream=` buffer; the "context manager" form raises `TypeError`.

## 8. What to try next

- Add `scalene` (async-aware, C+Python) and `vue-spy`/`austin` to the ladder.
- Score C-extension *internals*: profile a `json`/`re` workload with a tool
  that sees into C (e.g. `perf` + frame pointers) vs cProfile's one-entry view.
- Extend the ground truth to wall-vs-CPU (W6-style) for every method, not just
  `resource`.
- Sweep Python versions (3.10–3.13): PEP 669 inline caches and the new
  interpreter change per-event tracer costs.
