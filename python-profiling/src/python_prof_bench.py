#!/usr/bin/env python3
"""
Python profiling methods benchmark
===================================

Compares eight profiling methods for Python code on three axes:

  (a) accuracy   -- how close is the reported per-region time (or bytes) to a
                    ground-truth value that is KNOWN BY CONSTRUCTION. Two
                    views are reported for region-level methods:
                      abs   |reported - expected| / expected
                             (total fidelity; methods whose per-event
                             overhead inflates reported times are penalised)
                      share |f - e| / e with f = reported / sum(reported),
                             e = expected / sum(expected)
                             (hotspot-shape fidelity: does the method put the
                              right FRACTION of the run in each region)
                    The headline accuracy score is the share-based one.
  (b) overhead   -- wall-clock slowdown factor when running the code under the
                    method, per workload.
  (c) usability  -- qualitative rubric (setup, granularity, actionability,
                    scope, machine-readability), scored 1-5 per axis.

Ground truth by construction
----------------------------
Every workload is a sequence of *regions*. Each region function contains its
own copy of a canonical loop body and performs exactly k iterations of a
known unit operation (all bodies keep their ints bounded so per-iteration
cost is constant). The unit cost (seconds/iteration) of every operation is
calibrated with perf_counter (median of 5, gc left enabled -- the same
conditions as the method passes), so the expected cost of region (W, r) is
simply  k_r * unit_cost.  Allocation expectations are calibrated the same
way, with tracemalloc itself as the byte-level reference.

Methods under test
------------------
  wall             time.perf_counter around the call (median of 3)
  timeit           timeit.Timer, number=1, gc disabled (its standard mode)
  cProfile         stdlib C tracer, function-level cumulative times
  line_profiler    per-line bytecode instrumentation (pip: line_profiler)
  tracemalloc      stdlib allocation tracing, bytes per allocation site
  memory_profiler  per-line process RSS via psutil (pip: memory_profiler)
  py-spy           external sampling profiler, 200 Hz, subprocess mode
  resource         POSIX process counters (utime/stime/maxrss)

Workloads (common programming patterns)
----------------------------------------
  W1 tight numeric loop           W5 recursion (tree build + walk)
  W2 allocation churn + big alloc W6 I/O wait (file reads + sleeps)
  W3 C-extension heavy (sum/json/re) W7 generators vs plain loop
  W4 dispatch-heavy (many tiny calls)  W8 vectorized (numpy) vs loop
  W9 threads (informational only, not scored)

Notes
-----
* BLAS is pinned to 1 thread before importing numpy: OpenBLAS spin-wait
  threads otherwise pollute process-wide CPU counters (resource) long after
  the last matmul.
* line_profiler runs at reduced sizes (its per-line-event cost would
  dominate the runtime); accuracy is scale-invariant, overhead is measured
  at the reduced size.
* memory_profiler measures RSS deltas. After allocation churn the allocator
  reuses freed-but-mapped memory, so RSS deltas under-report gross bytes --
  exactly why allocation tracing (tracemalloc) exists. The benchmark keeps
  that behaviour visible instead of hiding it.
* py-spy (ptrace_scope=1 environments) must launch the child itself; the
  child prints per-round progress so only complete rounds count, and region
  times are sample-fraction x sampling-window.

Run:   python3 src/python_prof_bench.py            (~90 s, full)
       python3 src/python_prof_bench.py --fast      (reduced sizes, ~40 s)
       python3 src/python_prof_bench.py --skip-pyspy --skip-lineprof --skip-memprof
Artifacts: results/python_prof_bench.json
           results/python_prof_pyspy_raw.txt
           results/python_prof_pyspy.svg
"""

import argparse
import cProfile
import gc
import inspect
import io
import json as jsonlib
import math
import os
import pstats
import re as re_mod
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import timeit
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
HERE = Path(__file__).resolve()

# ---------------------------------------------------------------------------
# Canonical operations: each does exactly k iterations of one unit of work.
# Region functions below are byte-identical copies of these bodies (so every
# region is independently attributable) and the calibrated unit cost transfers.
# ---------------------------------------------------------------------------

FLT = [float(i % 7) for i in range(1_000_000)]
BIG_JSON = jsonlib.dumps({str(i): "x" * 20 for i in range(3000)})
RE_PAT = re_mod.compile(r"\bx\b")
BIG_TEXT = ("the quick brown fox jumps over the lazy dog. " * 2000).strip()

# Pin BLAS to one thread BEFORE importing numpy: (1) OpenBLAS spin-wait
# threads otherwise pollute process-wide CPU counters (resource.getrusage)
# for minutes after the last matmul, (2) timing becomes deterministic.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

try:
    import numpy as np
    MAT_A = np.random.default_rng(0).random((1024, 1024)).astype("float32")
    MAT_B = np.random.default_rng(1).random((1024, 1024)).astype("float32")
    HAVE_NUMPY = True
except ImportError:  # pragma: no cover
    HAVE_NUMPY = False


def cal_add3(k):
    x = 0
    for _ in range(k):
        x = (x * 3 + 1) & 0x7FFFFFFF
    return x


def cal_alloc(k):
    L = []
    ap = L.append
    for i in range(k):
        ap((i, i))
    return L


def cal_alloc_big(k):
    # bytearray: zero-filled -> pages are touched, so RSS (and thus
    # memory_profiler) can see it; bytes(n) leaves pages untouched.
    L = []
    for _ in range(k):
        L.append(bytearray(1_200_000))
    return L


def cal_csum(k):
    s = 0.0
    for _ in range(k):
        s += sum(FLT)
    return s


def cal_json(k):
    n = 0
    for _ in range(k):
        n += len(jsonlib.loads(BIG_JSON))
    return n


def cal_re(k):
    n = 0
    for _ in range(k):
        n += len(RE_PAT.findall(BIG_TEXT))
    return n


def cal_mm(k):
    out = None
    for _ in range(k):
        out = MAT_A @ MAT_B
    return out


def cal_gen(k):
    def g():
        for i in range(k):
            yield i

    s = 0
    for v in g():
        s += v
    return s


def cal_tiny():
    return 1 * 3 + 1


def cal_call(k):
    s = 0
    for _ in range(k):
        s += cal_tiny()
    return s


def cal_tree_build(d):
    if d == 0:
        return 1
    return [cal_tree_build(d - 1), cal_tree_build(d - 1)]


def cal_tree_walk(t):
    if isinstance(t, int):
        return 1
    return cal_tree_walk(t[0]) + cal_tree_walk(t[1])


def cal_tree(k):
    s = 0
    for _ in range(k):
        s += cal_tree_walk(cal_tree_build(8))
    return s


def cal_read(k, f):
    for _ in range(k):
        f.seek(0)
        f.read(65536)


def cal_sleep(k):
    for _ in range(k):
        time.sleep(0.0005)


# ---------------------------------------------------------------------------
# Region functions: one copy of each canonical body per region, so that
# cProfile / line_profiler / tracemalloc / py-spy can attribute time (or
# bytes) to the region by function name or source line.
# ---------------------------------------------------------------------------


def w1_a(k):
    x = 0
    for _ in range(k):
        x = (x * 3 + 1) & 0x7FFFFFFF
    return x


def w1_b(k):
    x = 0
    for _ in range(k):
        x = (x * 3 + 1) & 0x7FFFFFFF
    return x


def w1_c(k):
    x = 0
    for _ in range(k):
        x = (x * 3 + 1) & 0x7FFFFFFF
    return x


def w1_d(k):
    x = 0
    for _ in range(k):
        x = (x * 3 + 1) & 0x7FFFFFFF
    return x


def w1_e(k):
    x = 0
    for _ in range(k):
        x = (x * 3 + 1) & 0x7FFFFFFF
    return x


def w2_a(k):
    L = []
    ap = L.append
    for i in range(k):
        ap((i, i))
    return L


def w2_b(k):
    L = []
    ap = L.append
    for i in range(k):
        ap((i, i))
    return L


def w2_c(k):
    L = []
    ap = L.append
    for i in range(k):
        ap((i, i))
    return L


def w2_big(k):
    L = []
    for _ in range(k):
        L.append(bytearray(1_200_000))
    return L


def w3_sum(k):
    s = 0.0
    for _ in range(k):
        s += sum(FLT)
    return s


def w3_json(k):
    n = 0
    for _ in range(k):
        n += len(jsonlib.loads(BIG_JSON))
    return n


def w3_re(k):
    n = 0
    for _ in range(k):
        n += len(RE_PAT.findall(BIG_TEXT))
    return n


def w4_tiny():
    return 1 * 3 + 1


def w4_calls(k):
    s = 0
    for _ in range(k):
        s += w4_tiny()
    return s


def w4_add3(k):
    x = 0
    for _ in range(k):
        x = (x * 3 + 1) & 0x7FFFFFFF
    return x


def w5_build(d):
    if d == 0:
        return 1
    return [w5_build(d - 1), w5_build(d - 1)]


def w5_walk(t):
    if isinstance(t, int):
        return 1
    return w5_walk(t[0]) + w5_walk(t[1])


def w5_tree(k):
    s = 0
    for _ in range(k):
        s += w5_walk(w5_build(8))
    return s


def w5_add3(k):
    x = 0
    for _ in range(k):
        x = (x * 3 + 1) & 0x7FFFFFFF
    return x


def w6_read(k, f):
    for _ in range(k):
        f.seek(0)
        f.read(65536)


def w6_sleep(k):
    for _ in range(k):
        time.sleep(0.0005)


def w7_gen(k):
    def g():
        for i in range(k):
            yield i

    s = 0
    for v in g():
        s += v
    return s


def w7_add3(k):
    x = 0
    for _ in range(k):
        x = (x * 3 + 1) & 0x7FFFFFFF
    return x


def w8_mm(k):
    out = None
    for _ in range(k):
        out = MAT_A @ MAT_B
    return out


def w8_add3(k):
    x = 0
    for _ in range(k):
        x = (x * 3 + 1) & 0x7FFFFFFF
    return x


def w9_worker(k):
    x = 0
    for _ in range(k):
        x = (x * 3 + 1) & 0x7FFFFFFF
    return x


def w9_main(k):
    x = 0
    for _ in range(k):
        x = (x * 3 + 1) & 0x7FFFFFFF
    return x


# (workload, region, k, callable, helper functions sharing the region)
WORKLOADS = {
    "W1_tight_loop": [
        ("w1_a", 100_000, w1_a, []),
        ("w1_b", 300_000, w1_b, []),
        ("w1_c", 200_000, w1_c, []),
        ("w1_d", 400_000, w1_d, []),
        ("w1_e", 150_000, w1_e, []),
    ],
    "W2_alloc": [
        ("w2_a", 150_000, w2_a, []),
        ("w2_b", 250_000, w2_b, []),
        ("w2_c", 200_000, w2_c, []),
        ("w2_big", 60, w2_big, []),
    ],
    "W3_c_ext": [
        ("w3_sum", 120, w3_sum, []),
        ("w3_json", 250, w3_json, []),
        ("w3_re", 150, w3_re, []),
    ],
    "W4_dispatch": [
        ("w4_calls", 900_000, w4_calls, ["w4_tiny"]),
        ("w4_add3", 800_000, w4_add3, []),
    ],
    "W5_recursion": [
        ("w5_tree", 300, w5_tree, ["w5_build", "w5_walk"]),
        ("w5_add3", 300_000, w5_add3, []),
    ],
    "W6_io": [
        ("w6_read", 500, w6_read, []),
        ("w6_sleep", 600, w6_sleep, []),
    ],
    "W7_gen": [
        ("w7_gen", 400_000, w7_gen, []),
        ("w7_add3", 400_000, w7_add3, []),
    ],
    "W8_vec": [
        ("w8_mm", 200, w8_mm, []),
        ("w8_add3", 500_000, w8_add3, []),
    ],
    "W9_threads": [
        ("w9_main", 200_000, w9_main, []),
        ("w9_worker", 200_000, w9_worker, []),  # x4 threads, informational
    ],
}

# canonical function used to calibrate the unit cost of each region
CALIB = {
    "w1_a": ("cal_add3", 1), "w1_b": ("cal_add3", 1), "w1_c": ("cal_add3", 1),
    "w1_d": ("cal_add3", 1), "w1_e": ("cal_add3", 1),
    "w2_a": ("cal_alloc", 1), "w2_b": ("cal_alloc", 1), "w2_c": ("cal_alloc", 1),
    "w2_big": ("cal_alloc_big", 1),
    "w3_sum": ("cal_csum", 1), "w3_json": ("cal_json", 1), "w3_re": ("cal_re", 1),
    "w4_calls": ("cal_call", 1), "w4_add3": ("cal_add3", 1),
    "w5_tree": ("cal_tree", 1), "w5_add3": ("cal_add3", 1),
    "w6_read": ("cal_read", "FILE"), "w6_sleep": ("cal_sleep", 1),
    "w7_gen": ("cal_gen", 1), "w7_add3": ("cal_add3", 1),
    "w8_mm": ("cal_mm", 1), "w8_add3": ("cal_add3", 1),
    "w9_main": ("cal_add3", 1), "w9_worker": ("cal_add3", 1),
}

CALIB_K = {  # k used during calibration (seconds of work per calibration)
    "cal_add3": 200_000, "cal_alloc": 200_000, "cal_alloc_big": 20,
    "cal_csum": 40, "cal_json": 80, "cal_re": 60, "cal_mm": 80,
    "cal_gen": 200_000, "cal_call": 200_000, "cal_tree": 60,
    "cal_read": 200, "cal_sleep": 60,
}


def region_expected(w, r, k, unit_cost, io_file):
    """expected seconds for region r of workload w at iteration count k."""
    fn, arg = CALIB[r]
    if arg == "FILE":
        return k * unit_cost["cal_read"]
    return k * unit_cost[fn]


def run_workload(w, scale=1.0, io_file=None, quiet=True):
    for name, k, fn, _helpers in WORKLOADS[w]:
        kk = max(50, int(k * scale))
        if fn.__code__.co_argcount >= 2:  # (k, file) signature
            fn(kk, io_file)
        else:
            fn(kk)


def expected_total(w, unit_cost, scale=1.0):
    return sum(region_expected(w, r, max(50, int(k * scale)), unit_cost, None)
               for r, k, _f, _h in WORKLOADS[w])


# ---------------------------------------------------------------------------
# calibration (ground truth)
# ---------------------------------------------------------------------------

def calibrate(io_file):
    """unit cost (s/iter) per canonical op.

    Measured with perf_counter, gc left enabled, median of 5 full-k reps --
    i.e. under the exact conditions the method passes run in (timeit itself
    disables gc, which is one of its measured characteristics, so we do not
    use it for the ground truth).
    """
    unit = {}
    for fn_name, k in CALIB_K.items():
        fn = globals()[fn_name]
        if fn_name == "cal_read":
            def stmt():
                fn(k, io_file)
        else:
            def stmt():
                fn(k)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            stmt()
            ts.append(time.perf_counter() - t0)
        unit[fn_name] = statistics.median(ts) / k
    return unit


# ---------------------------------------------------------------------------
# line map: (script basename, line) -> (workload, region)
# ---------------------------------------------------------------------------

def build_line_map():
    m = {}
    for w, regions in WORKLOADS.items():
        for rname, _k, fn, helpers in regions:
            fns = [fn] + [globals()[h] for h in helpers]
            for f in fns:
                src, start = inspect.getsourcelines(f)
                for i, _line in enumerate(src):
                    m[(HERE.name, start + i)] = (w, rname)
    return m


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------

def method_wall(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def method_timeit(fn):
    # timeit disables the cyclic GC during timing (its standard behaviour)
    t = timeit.Timer(fn)
    return min(t.repeat(number=1, repeat=3))


def method_cprofile(fn):
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    fn()
    pr.disable()
    wall = time.perf_counter() - t0
    # keys are (filename, lineno, name); values (ncalls, tt, ct, callers)
    return wall, pstats.Stats(pr, stream=io.StringIO()).stats


def method_lineprof(fn, lp, funcs):
    for f in funcs:
        lp.add_function(f)
    t0 = time.perf_counter()
    lp.enable()
    fn()
    lp.disable()
    wall = time.perf_counter() - t0
    stats = lp.get_stats()
    per = {}
    for (_file, _ln, fname), lines in stats.timings.items():
        for _line, _hits, t in lines:
            per[fname] = per.get(fname, 0.0) + t * stats.unit
    return wall, per


def method_tracemalloc(w, scale, io_file=None):
    """per-region (current, peak) bytes for the whole workload."""
    out = {}
    for rname, k, fn, _h in WORKLOADS[w]:
        kk = max(50, int(k * scale))
        tracemalloc.start(1)
        if fn.__code__.co_argcount >= 2:
            ret = fn(kk, io_file)
        else:
            ret = fn(kk)
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del ret
        out[rname] = (cur, peak)
    return out


def alloc_bytes_per_item():
    """gross bytes allocated per item, measured with tracemalloc itself.

    Keep the batch alive so 'current' == total bytes of that allocation.
    """
    tracemalloc.start(1)
    L = cal_alloc(15_000)
    cur, _peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del L
    per_item = cur / 15_000
    tracemalloc.start(1)
    L = cal_alloc_big(10)
    cur, _peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del L
    return per_item, cur / 10


# ---------------------------------------------------------------------------
# py-spy (external sampler) - subprocess mode
# ---------------------------------------------------------------------------

def pyspy_bin():
    p = shutil.which("py-spy")
    if p:
        return p
    cand = Path.home() / ".local" / "bin" / "py-spy"
    return str(cand) if cand.exists() else None


def _child_lines(stdout):
    """JSON progress lines the child prints: one per completed round + final."""
    rounds_wall, final = [], None
    for line in stdout.strip().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rep = jsonlib.loads(line)
        except ValueError:
            continue
        if "round_wall" in rep:
            rounds_wall.append(rep["round_wall"])
        if rep.get("done"):
            final = rep
    return rounds_wall, final


def run_pyspy(rounds, dur, raw_out, svg_out, line_map, fast):
    """External sampling: py-spy launches the child itself (ptrace_scope=1
    only allows a parent to trace its own children).

    The child prints one JSON line per completed round; py-spy kills the
    child when --duration expires, so only complete rounds count. Overhead
    = mean completed-round wall under py-spy / mean baseline round wall.
    Region times = sample fraction x sampling window (dur seconds).
    """
    exe = pyspy_bin()
    if not exe:
        return None
    child_cmd = [sys.executable, str(HERE), "--child", str(rounds)]
    if fast:
        child_cmd.append("--fast")

    # baseline child, no sampler
    p = subprocess.run(child_cmd, capture_output=True, text=True)
    b_rounds, _ = _child_lines(p.stdout)

    # raw recording
    cmd = [exe, "record", "--rate", "200", "--duration", str(dur),
           "--format", "raw", "-o", str(raw_out), "--"] + child_cmd
    p = subprocess.run(cmd, capture_output=True, text=True)
    u_rounds, _ = _child_lines(p.stdout)
    overhead = (statistics.fmean(u_rounds) / statistics.fmean(b_rounds)
                if u_rounds and b_rounds else None)

    n_samples, region_samples, outside = 0, {}, 0
    for line in raw_out.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        try:
            frames_part, count = line.rsplit(" ", 1)
            count = int(count)
        except ValueError:
            continue
        n_samples += count
        hit = None
        for frame in reversed(frames_part.split(";")):
            m = re_mod.match(r"^(.*?)\s+\((.+?)\)$", frame.strip())
            if not m:
                continue
            _fn, loc = m.group(1), m.group(2)
            mm = re_mod.match(r"^(.+?):(\d+)$", loc)
            if mm and (mm.group(1), int(mm.group(2))) in line_map:
                hit = line_map[(mm.group(1), int(mm.group(2)))]
                break
        if hit:
            region_samples[hit] = region_samples.get(hit, 0) + count
        else:
            outside += count

    # flamegraph svg recording (separate run)
    cmd = [exe, "record", "--rate", "200", "--duration", str(dur),
           "--format", "flamegraph", "-o", str(svg_out), "--"] + child_cmd
    subprocess.run(cmd, capture_output=True, text=True)

    return {
        "baseline_round_walls": b_rounds,
        "under_pyspy_round_walls": u_rounds,
        "overhead_ratio": overhead,
        "total_samples": n_samples,
        "sampling_window_s": dur,
        "raw": {"region_samples": region_samples, "outside": outside},
        "svg": {"path": str(svg_out), "exists": svg_out.exists()},
    }


# ---------------------------------------------------------------------------
# main benchmark
# ---------------------------------------------------------------------------

def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def geomean(vals):
    if not vals:
        return float("nan")
    return math.exp(statistics.fmean(math.log(v) for v in vals))


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="halve all sizes")
    ap.add_argument("--skip-pyspy", action="store_true")
    ap.add_argument("--skip-lineprof", action="store_true")
    ap.add_argument("--skip-memprof", action="store_true")
    args, _unknown = ap.parse_known_args()

    if len(sys.argv) > 1 and "--child" in sys.argv:
        run_child(sys.argv[sys.argv.index("--child") + 1],
                  "--fast" in sys.argv)
        return

    scale = 0.5 if args.fast else 1.0
    RESULTS.mkdir(exist_ok=True)
    io_file = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    io_file.write(os.urandom(4 * 1024 * 1024))
    io_file.flush()
    io_file.seek(0)

    print("=" * 76)
    print("Python profiling methods benchmark")
    print(f"python {sys.version.split()[0]} | scale={scale} | "
          f"numpy={HAVE_NUMPY}")
    print("=" * 76)

    # ---- ground truth -------------------------------------------------
    print("\n[1/7] Calibrating unit costs (perf_counter, gc on, median of 5) ...")
    unit = calibrate(io_file)
    alloc_item, alloc_big_item = alloc_bytes_per_item()
    def dur(x):  # s -> best-unit string
        for scale_, name in ((1e3, "ms"), (1e6, "us"), (1e9, "ns")):
            if x * scale_ >= 1:
                return f"{x * scale_:8.1f} {name}"
        return f"{x * 1e9:8.1f} ns"

    for label, key in [("add3 iter", "cal_add3"), ("alloc iter", "cal_alloc"),
                       ("alloc_big item", "cal_alloc_big"),
                       ("csum(1M) call", "cal_csum"),
                       ("json(100KB) call", "cal_json"),
                       ("re(100KB) call", "cal_re"),
                       ("mm(1024^3) call", "cal_mm"), ("gen iter", "cal_gen"),
                       ("call+op iter", "cal_call"), ("tree(8) item", "cal_tree"),
                       ("read(64KB) call", "cal_read"),
                       ("sleep(0.5ms)", "cal_sleep")]:
        print(f"    {label:17s} {dur(unit[key])}")
    print(f"    alloc gross bytes/item (tracemalloc) {alloc_item:8.1f} B")

    line_map = build_line_map()

    # ---- bare baseline --------------------------------------------------
    print("\n[2/7] Bare baselines (perf_counter, median of 3) ...")
    bare = {}
    for w in WORKLOADS:
        if w == "W9_threads":
            continue
        bare[w] = method_wall(lambda: run_workload(w, scale, io_file))
        print(f"    {w:14s} {bare[w] * 1e3:9.1f} ms")

    # ---- overhead + attribution passes ----------------------------------
    print("\n[3/7] Per-method passes ...")
    overhead = {}   # method -> {workload: ratio}
    attributed = {}  # method -> {(w, r): seconds}

    # cProfile (full size, 1 rep)
    for w in list(WORKLOADS):
        if w == "W9_threads":
            continue
        wall_t, stats = method_cprofile(lambda: run_workload(w, scale, io_file))
        exp_tot = expected_total(w, unit, scale)
        overhead.setdefault("cProfile", {})[w] = wall_t / bare[w]
        for rname, k, fn, _h in WORKLOADS[w]:
            code = fn.__code__
            entry = next((v for k, v in stats.items()
                          if k[2] == fn.__name__), None)
            # 3.12 tuple: (ncalls, cc, tt, ct, callers) -> ct is index 3
            rep = entry[3] if entry else 0.0  # cumulative time
            attributed.setdefault("cProfile", {})[(w, rname)] = rep
    print("    cProfile done")

    # line_profiler (reduced size: overhead is tens of us per line event,
    # so full-size loops would dominate the benchmark runtime)
    if not args.skip_lineprof:
        try:
            from line_profiler import LineProfiler
        except ImportError:
            print("    line_profiler not installed; skipping")
            LineProfiler = None
        if LineProfiler:
            lp = LineProfiler()
            ls = {"W1_tight_loop": 0.1, "W2_alloc": 0.1, "W3_c_ext": 0.25,
                  "W4_dispatch": 0.1, "W5_recursion": 0.25, "W6_io": 0.25,
                  "W7_gen": 0.1, "W8_vec": 0.1, "W9_threads": 1.0}
            for w in list(WORKLOADS):
                if w == "W9_threads":
                    continue
                sc = ls[w]
                bare_red = method_wall(
                    lambda: run_workload(w, sc * scale, io_file))
                fns = [fn for _r, _k, fn, hs in WORKLOADS[w] for fn in [fn]]
                fns += [globals()[h] for _r, _k, fn, hs in WORKLOADS[w]
                        for h in hs]
                wall_t, per = method_lineprof(
                    lambda: run_workload(w, sc * scale, io_file),
                    lp, fns)
                overhead.setdefault("line_profiler", {})[w] = (
                    wall_t / bare_red if bare_red else float("nan"))
                for rname, _k, fn, hs in WORKLOADS[w]:
                    names = [fn.__name__] + hs
                    attributed.setdefault("line_profiler", {})[(w, rname)] = (
                        sum(per.get(n, 0.0) for n in names))
            print("    line_profiler done (reduced sizes: x0.1 python-loop, "
                  "x0.25 event-driven; accuracy is scale-invariant)")

    # tracemalloc (bytes per region)
    tm_bytes = {}
    tm_over = {}
    for w in list(WORKLOADS):
        if w == "W9_threads":
            continue
        t0 = time.perf_counter()
        tm_bytes[w] = method_tracemalloc(w, scale, io_file)
        tm_over[w] = time.perf_counter() - t0
        overhead.setdefault("tracemalloc", {})[w] = tm_over[w] / bare[w]
    print("    tracemalloc done")

    # timeit (full size, gc disabled = its standard mode)
    for w in list(WORKLOADS):
        if w == "W9_threads":
            continue
        t = method_timeit(lambda: run_workload(w, scale, io_file))
        overhead.setdefault("timeit", {})[w] = t / bare[w]
    print("    timeit done")

    # resource counters. W6 first (before any BLAS work): wall vs CPU is
    # the I/O-wait diagnostic. W8 is excluded from CPU-accuracy scoring:
    # OpenBLAS spin-wait threads make process CPU >> single-threaded wall.
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    run_workload("W6_io", scale, io_file)
    wall6 = time.perf_counter() - t0
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu6 = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    print(f"    resource: W6 wall {wall6 * 1e3:.0f} ms vs cpu {cpu6 * 1e3:.0f} ms"
          f" -> CPU busy fraction {pct(cpu6 / wall6)}")
    res_cpu = {}
    for w in ["W1_tight_loop", "W3_c_ext", "W4_dispatch", "W8_vec"]:
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        run_workload(w, scale, io_file)
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        res_cpu[w] = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)

    # memory_profiler on W2: decorate each region function itself (its own
    # lines are then what gets sampled). Churn regions at 5% (per-line
    # psutil polling is O(lines executed)); big-alloc at full size.
    memprof_peak = {}
    memprof_ov = {}
    if not args.skip_memprof:
        try:
            import memory_profiler as mp
        except ImportError:
            mp = None
            print("    memory_profiler not installed; skipping")
        if mp:
            for rname, k, fn, _h in WORKLOADS["W2_alloc"]:
                kk = 60 if rname == "w2_big" else max(50, int(k * 0.05))
                exp_time = kk * (unit["cal_alloc_big"] if rname == "w2_big"
                                 else unit["cal_alloc"])
                buf = io.StringIO()
                wrapped = mp.profile(stream=buf)(fn)
                t0 = time.perf_counter()
                ret = wrapped(kk)
                mp_wall = time.perf_counter() - t0
                del ret
                memprof_ov[rname] = mp_wall / exp_time
                baseline, peak = None, 0.0
                for line in buf.getvalue().splitlines():
                    m = re_mod.match(
                        r"^\s*(\d+)\s+(\d+\.\d+)\s+MiB\s+"
                        r"([+-]?\d+\.\d+)\s+MiB\s+(\d+)\s+(.*)$", line)
                    if not m:
                        continue
                    mem = float(m.group(2))
                    baseline = mem if baseline is None else baseline
                    peak = max(peak, mem)
                if baseline is not None:
                    memprof_peak[rname] = peak - baseline
            overhead.setdefault("memory_profiler", {})[
                "W2_alloc"] = geomean(list(memprof_ov.values()))
            print("    memory_profiler done (W2 at 5% churn + full big-alloc)")

    # py-spy
    pyspy_res = None
    if not args.skip_pyspy:
        print("    py-spy: baseline child + 2 x 6 s recordings (~25 s) ...")
        rounds = 5 if args.fast else 8
        dur = 4 if args.fast else 6
        raw_out = RESULTS / "python_prof_pyspy_raw.txt"
        svg_out = RESULTS / "python_prof_pyspy.svg"
        pyspy_res = run_pyspy(rounds, dur, raw_out, svg_out, line_map,
                              args.fast)
        if pyspy_res:
            window = pyspy_res["sampling_window_s"]
            tot = pyspy_res["total_samples"] or 1
            for (w, r), c in pyspy_res["raw"]["region_samples"].items():
                attributed.setdefault("py-spy", {})[(w, r)] = (
                    c / tot * window)
            outside_frac = pyspy_res["raw"]["outside"] / tot
            ov = pyspy_res["overhead_ratio"]
            print(f"    py-spy: {pyspy_res['total_samples']} samples in "
                  f"{window:.0f}s window, unattributed {pct(outside_frac)}, "
                  f"round-wall overhead {ov:.3f}x"
                  if ov else "    py-spy: no complete rounds under sampler")

    # ------------------------------------------------------------------
    # [4/7] accuracy
    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("[4/7] Accuracy vs ground truth  (per-region relative error)")
    print("=" * 76)

    def region_expected_map(method):
        """expected seconds per region, at the scale the method ran at."""
        ls = {"W1_tight_loop": 0.1, "W2_alloc": 0.1, "W3_c_ext": 0.25,
              "W4_dispatch": 0.1, "W5_recursion": 0.25, "W6_io": 0.25,
              "W7_gen": 0.1, "W8_vec": 0.1}
        exp = {}
        for (w, r), rep in attributed.get(method, {}).items():
            k = dict((x[0], x[1]) for x in WORKLOADS[w])[r]
            sc = ls[w] if method == "line_profiler" else scale
            exp[(w, r)] = region_expected(w, r, max(50, int(k * sc)),
                                          unit, None)
        return exp

    # Two accuracy views for region-level methods:
    #  abs  : |reported - expected| / expected  -> total fidelity; a method
    #         with per-event overhead is penalised for the inflation too
    #  share: |f - e| / e,  f = reported/sum(reported), e = expected/sum
    #         -> hotspot-shape fidelity (the question a profiler actually
    #         answers: which region owns what FRACTION of the run)
    timing_methods = ["cProfile", "line_profiler", "py-spy"]
    acc = {}
    for m in timing_methods:
        expm = region_expected_map(m)
        abs_e, sh_e = [], []
        by_w = {}
        for (w, r), rep in attributed.get(m, {}).items():
            e = expm[(w, r)]
            if e > 0:
                abs_e.append(abs(rep - e) / e)
            by_w.setdefault(w, {})[r] = (rep, e)
        for w, d in by_w.items():
            tot_r = sum(v[0] for v in d.values())
            tot_e = sum(v[1] for v in d.values())
            if tot_r > 0 and tot_e > 0:
                for r, (rep, e) in d.items():
                    if e > 0:
                        sh_e.append(abs(rep / tot_r - e / tot_e) / (e / tot_e))
        if abs_e:
            acc[m] = {
                "median_abs_relerr": statistics.median(abs_e),
                "worst_abs_relerr": max(abs_e),
                "median_share_relerr": statistics.median(sh_e),
                "score": max(0.0, 100 * (1 - statistics.median(sh_e))),
            }
            print(f"    {m:14s} abs median |err| "
                  f"{pct(acc[m]['median_abs_relerr'])} (worst "
                  f"{pct(acc[m]['worst_abs_relerr'])}) | share median |err| "
                  f"{pct(acc[m]['median_share_relerr'])}  "
                  f"score {acc[m]['score']:5.1f}/100  "
                  f"(n={len(by_w)} workloads)")

    # total-time-only methods
    for w in list(WORKLOADS):
        if w != "W9_threads":
            overhead.setdefault("wall", {})[w] = 1.0
    for m in ["wall", "timeit"]:
        errs = []
        for w in list(WORKLOADS):
            if w == "W9_threads":
                continue
            rep = overhead[m][w] * bare[w]
            exp = expected_total(w, unit, scale)
            errs.append(abs(rep - exp) / exp)
        acc[m] = {
            "median_relerr": statistics.median(errs),
            "mean_relerr": statistics.fmean(errs),
            "score": max(0.0, 100 * (1 - statistics.median(errs))),
        }
        print(f"    {m:14s} (total time only) median |err| "
              f"{pct(acc[m]['median_relerr'])}  score "
              f"{acc[m]['score']:5.1f}/100")

    # tracemalloc: bytes on W2 (gross allocation per region)
    tm_errs = []
    for rname, k, _f, _h in WORKLOADS["W2_alloc"]:
        per_item = alloc_big_item if rname == "w2_big" else alloc_item
        exp = max(50, int(k * scale)) * per_item
        _cur, peak = tm_bytes["W2_alloc"][rname]
        tm_errs.append(abs(peak - exp) / exp)
    acc["tracemalloc"] = {
        "median_relerr": statistics.median(tm_errs),
        "score": max(0.0, 100 * (1 - statistics.median(tm_errs))),
        "note": "gross bytes, W2 only (Python-level allocs)",
    }
    print(f"    {'tracemalloc':14s} (W2 alloc bytes) median |err| "
          f"{pct(acc['tracemalloc']['median_relerr'])}  score "
          f"{acc['tracemalloc']['score']:5.1f}/100")

    # memory_profiler: per-region peak RSS delta vs expected gross bytes
    mp_errs = []
    if memprof_peak:
        for rname, k, _f, _h in WORKLOADS["W2_alloc"]:
            kk = 60 if rname == "w2_big" else max(50, int(k * 0.05))
            per = alloc_big_item if rname == "w2_big" else alloc_item
            exp_mib = kk * per / 1e6
            rep = memprof_peak.get(rname, 0.0)
            mp_errs.append(abs(rep - exp_mib) / exp_mib)
        if mp_errs:
            acc["memory_profiler"] = {
                "median_relerr": statistics.median(mp_errs),
                "score": max(0.0, 100 * (1 - statistics.median(mp_errs))),
            }
            print(f"    {'memory_profiler':14s} (W2 peak RSS delta) "
                  f"median |err| {pct(acc['memory_profiler']['median_relerr'])}"
                  f"  score {acc['memory_profiler']['score']:5.1f}/100")
            print("        " + ", ".join(
                f"{r}={v:.1f} MiB" for r, v in sorted(memprof_peak.items())))
    else:
        acc["memory_profiler"] = {"median_relerr": float("nan"),
                                  "score": float("nan")}

    # resource: CPU time on CPU workloads vs expected (BLAS pinned to 1
    # thread so process CPU tracks the Python-loop wall time)
    rr_errs = []
    for w in ["W1_tight_loop", "W3_c_ext", "W4_dispatch", "W8_vec"]:
        exp = expected_total(w, unit, scale)
        rr_errs.append(abs(res_cpu[w] - exp) / exp)
    acc["resource"] = {
        "median_relerr": statistics.median(rr_errs),
        "score": max(0.0, 100 * (1 - statistics.median(rr_errs))),
        "note": "process utime+stime vs expected CPU time",
    }
    print(f"    {'resource':14s} (CPU time) median |err| "
          f"{pct(acc['resource']['median_relerr'])}  score "
          f"{acc['resource']['score']:5.1f}/100")

    # top-1 hot region hit rate
    print("\n    top-1 hot-region hit rate (does the method's #1 region match "
          "ground truth?):")
    for m in timing_methods:
        hits, tot = 0, 0
        for w in list(WORKLOADS):
            if w == "W9_threads":
                continue
            exp_r = max(WORKLOADS[w],
                        key=lambda x: region_expected(w, x[0], x[1], unit, None))
            got = max((r for r in
                       [x[0] for x in WORKLOADS[w]]
                       if (w, r) in attributed.get(m, {})),
                      key=lambda r: attributed[m][(w, r)], default=None)
            if got is not None:
                tot += 1
                hits += got == exp_r[0]
        print(f"    {m:14s} {hits}/{tot}")

    # ------------------------------------------------------------------
    # [5/7] overhead matrix
    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("[5/7] Overhead (wall-clock ratio vs bare; >1 = slowdown)")
    print("=" * 76)
    methods_ov = [m for m in ["wall", "cProfile", "timeit", "line_profiler",
                              "tracemalloc", "memory_profiler"]
                  if m in overhead]
    wls = [w for w in WORKLOADS if w != "W9_threads"]
    print(f"    {'method':16s}" + "".join(f"{w[:10]:>11s}" for w in wls))
    means = {}
    for m in methods_ov:
        row = []
        vals = []
        for w in wls:
            v = overhead.get(m, {}).get(w)
            if v is None:
                row.append(f"{'-':>11s}")
            else:
                row.append(f"{v:>11.2f}")
                vals.append(v)
        means[m] = geomean(vals)
        print(f"    {m:16s}" + "".join(row) +
              f"   geomean {means[m]:.2f}x")
    if pyspy_res:
        means["py-spy"] = pyspy_res["overhead_ratio"] or float("nan")
        print(f"    {'py-spy':16s}   (external sampler; whole-child wall "
              f"ratio {means['py-spy']:.2f}x, incl. sampling stops)")

    # ------------------------------------------------------------------
    # [6/7] cProfile fine/coarse trap
    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("[6/7] Instrumentation-overhead trap: same work, different call "
          "granularity")
    print("=" * 76)

    MASK = 0x7FFFFFFF

    def coarse():
        x = 0
        for _ in range(600_000):
            x = (x * 3 + 1) & MASK
        return x

    def fine_tiny(x):
        x = (x * 3 + 1) & MASK
        x = (x * 3 + 1) & MASK
        x = (x * 3 + 1) & MASK
        return x

    def fine():
        x = 0
        for _ in range(200_000):
            x = fine_tiny(x)
        return x

    c_bare = method_wall(coarse, reps=3)
    f_bare = method_wall(fine, reps=3)
    c_prof, _ = method_cprofile(coarse)
    f_prof, _ = method_cprofile(fine)
    print(f"    600k add3 ops, one function   : bare {c_bare * 1e3:6.1f} ms | "
          f"cProfile {c_prof * 1e3:6.1f} ms ({c_prof / c_bare:.2f}x)")
    print(f"    600k add3 ops, 200k tiny calls: bare {f_bare * 1e3:6.1f} ms | "
          f"cProfile {f_prof * 1e3:6.1f} ms ({f_prof / f_bare:.2f}x)")
    print(f"    real call-overhead (bare)      : "
          f"{(f_bare / c_bare):.2f}x ; cProfile amplifies to "
          f"{(f_prof / c_prof):.2f}x relative ranking inflation "
          f"{(f_prof / f_bare) / (c_prof / c_bare):.1f}x")

    # ------------------------------------------------------------------
    # [7/7] usability rubric + scorecard
    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("[7/7] Usability rubric (1-5 per axis; weights in parens)")
    print("=" * 76)
    RUBRIC = {  # setup, granularity, actionability, scope, machine-readable
        "wall":            (5, 1, 1, 2, 5, "total time of what you wrapped; "
                                            "no attribution"),
        "timeit":          (5, 2, 2, 1, 5, "micro-bench of your snippet only; "
                                            "gc off by default"),
        "cProfile":        (5, 4, 4, 3, 4, "function-level, stdlib, 1-liner; "
                                            "blind to lines/threads/GPU/C "
                                            "internals"),
        "line_profiler":   (3, 5, 5, 2, 3, "the exact line, once cProfile "
                                            "shortlists the function; "
                                            "10^2-10^3x overhead"),
        "tracemalloc":     (4, 4, 4, 3, 4, "alloc site + bytes (retained/"
                                            "gross); Python-level allocs only"),
        "memory_profiler": (3, 4, 3, 3, 2, "per-line RSS, OS page granularity; "
                                            "misses Python alloc churn"),
        "py-spy":          (3, 4, 4, 4, 4, "near-zero overhead, all threads, "
                                            "attach to live process; sampling "
                                            "(needs volume), no GPU/C detail"),
        "resource":        (5, 1, 2, 2, 5, "cheapest CPU-vs-wall + max-RSS "
                                            "sanity check; no attribution"),
    }
    WEIGHTS = (0.15, 0.25, 0.30, 0.15, 0.15)
    print(f"    {'method':16s} setup gran act scope mach  -> score/5")
    usab = {}
    for m, (s, g, a, sc, mr, _v) in RUBRIC.items():
        usab[m] = sum(w * v for w, v in zip(WEIGHTS, (s, g, a, sc, mr)))
        print(f"    {m:16s} {s}     {g}   {a}   {sc}    {mr}   -> "
              f"{usab[m]:.2f}")

    print("\n" + "=" * 76)
    print("FINAL SCORECARD")
    print("=" * 76)
    print(f"    {'method':16s} {'overhead':>9s} {'accuracy':>10s} "
          f"{'usability':>9s}   verdict")
    verdicts = {
        "wall": "answers 'how long', never 'why' - gating, not diagnosis",
        "timeit": "micro-benchmark, not a profiler",
        "cProfile": "first-stop function-level answer; watch call-count "
                    "overhead on dispatch-heavy code",
        "line_profiler": "targeted line-level drill-down after cProfile; "
                         "never whole-program",
        "tracemalloc": "the memory tool: who allocates what, retained vs "
                       "gross",
        "memory_profiler": "coarse RSS story incl. C-extension memory; "
                           "blind to churn",
        "py-spy": "zero-distortion sampling across threads; attach to live "
                  "processes; blind to GPU and C internals",
        "resource": "one-liner sanity check: CPU vs wall, peak RSS",
    }
    rows = []
    for m in ["wall", "timeit", "cProfile", "line_profiler", "tracemalloc",
              "memory_profiler", "py-spy", "resource"]:
        ov = means.get(m, float("nan"))
        ac = acc.get(m, {}).get("score", float("nan"))
        rows.append((m, ov, ac, usab[m]))
        ov_s = f"{ov:9.2f}" if ov == ov else f"{'-':>9s}"
        ac_s = f"{ac:10.1f}" if ac == ac else f"{'-':>10s}"
        print(f"    {m:16s} {ov_s} {ac_s} {usab[m]:>9.2f}   {verdicts[m]}")

    # ------------------------------------------------------------------
    # W9 threads informational
    # ------------------------------------------------------------------
    if not args.fast:
        print("\n    [informational] W9 threads: 4 threads x 200k iters + "
              "main 200k")
        import threading
        def worker():
            w9_worker(200_000)
        t0 = time.perf_counter()
        ths = [threading.Thread(target=worker) for _ in range(4)]
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        for th in ths:
            th.start()
        w9_main(200_000)
        for th in ths:
            th.join()
        wall9 = time.perf_counter() - t0
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        cpu9 = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
        print(f"        wall {wall9 * 1e3:.0f} ms, process CPU {cpu9 * 1e3:.0f} ms "
              f"(GIL: CPU ~ wall). cProfile: main thread only; py-spy: all "
              "threads.")

    # ------------------------------------------------------------------
    # JSON dump
    # ------------------------------------------------------------------
    out = {
        "python": sys.version.split()[0],
        "scale": scale,
        "unit_cost": unit,
        "alloc_bytes_per_item": [alloc_item, alloc_big_item],
        "bare_seconds": bare,
        "overhead": {m: ov for m, ov in overhead.items()},
        "overhead_geomean": means,
        "accuracy": acc,
        "attributed_seconds": {
            m: {f"{w}|{r}": v for (w, r), v in d.items()}
            for m, d in attributed.items()},
        "resource_cpu": res_cpu,
        "w6_wall_vs_cpu": {"wall": wall6, "cpu": cpu6},
        "tracemalloc_bytes": {
            w: {r: list(v) for r, v in d.items()} for w, d in tm_bytes.items()},
        "memory_profiler_peak_mib": memprof_peak,
        "memory_profiler_overhead_per_region": memprof_ov,
        "usability": usab,
        "pyspy": ({**pyspy_res,
                   "raw": {**pyspy_res["raw"],
                           "region_samples": {
                               f"{w}|{r}": c
                               for (w, r), c in
                               pyspy_res["raw"]["region_samples"].items()}}}
                  if pyspy_res else None),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "python_prof_bench.json").write_text(
        jsonlib.dumps(out, indent=2, default=str))
    print(f"\n    JSON: {RESULTS / 'python_prof_bench.json'}")
    io_file.close()
    os.unlink(io_file.name)
    print("done.")


def run_child(rounds, fast):
    """child process for py-spy: loop through workloads; one JSON progress
    line per completed round (py-spy kills the child at the recording
    deadline, so the parent only trusts complete rounds)."""
    scale = 0.5 if fast else 1.0
    io_file = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    io_file.write(os.urandom(4 * 1024 * 1024))
    io_file.flush()
    io_file.seek(0)
    t0 = time.perf_counter()
    for i in range(int(rounds)):
        rt0 = time.perf_counter()
        for w in list(WORKLOADS):
            if w == "W9_threads":
                continue
            run_workload(w, scale, io_file)
        print(jsonlib.dumps({"round": i + 1,
                             "round_wall": time.perf_counter() - rt0}),
              flush=True)
    print(jsonlib.dumps({"done": True,
                         "wall": time.perf_counter() - t0}), flush=True)
    io_file.close()
    os.unlink(io_file.name)


if __name__ == "__main__":
    main()
