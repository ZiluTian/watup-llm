#!/usr/bin/env python3
"""
CUDA profiling tools benchmark
==============================

Complements the Python profiling benchmark for the GPU side. A small
"decode-style" workload (a batched matmul chain with pointwise / reduction /
copy regions) is profiled by seven different tools and each tool is scored
on the same three axes as the Python bench:

  (a) accuracy   -- per-region GPU time vs a CUDA-event oracle (ground truth
                    is measured with torch.cuda.Event pairs around each
                    region; events are stream-ordered hardware timestamps,
                    negligible overhead).
  (b) overhead   -- wall-clock slowdown of the workload under the tool.
  (c) usability  -- qualitative rubric (setup, granularity, actionability,
                    scope, machine-readability), 1-5 per axis.

The workload
------------
Per iteration (batch B=128, width D=4096, fp32):
  h2d       pinned CPU -> device copy of x        (async)
  gemm      y = W1 @ x ; y = W2 @ y               (cuBLAS/cutlass, 2 GEMMs)
  pointwise y = (y * b).relu() + c                (3 elementwise kernels)
  reduce    y = y / y.norm(dim=0, keepdim=True)   (reduction + division)
  d2h       async copy of a slice back to pinned  (async)

All copies use pinned memory so the whole loop is genuinely async: nothing
inside the loop synchronizes. That is what makes the tools disagree in
interesting ways -- and it matches how real inference loops are written.

Tools under test
----------------
  wall_nosync  perf_counter around ONE iteration, no sync  (the classic trap)
  wall_sync    perf_counter + torch.cuda.synchronize       (the naive-but-right)
  cProfile     function-level CPU tracer on the torch loop
  torch.profiler  kineto: per-kernel GPU times + CPU ops
  py-spy       external CPU sampling (what the host side spends its time on)
  nsys         Nsight Systems: external full-system CUDA trace
  ncu          Nsight Compute: per-kernel hardware metrics (replay)
               NOTE: ncu needs GPU performance-counter permission
               (typically root); it is attempted and degrades gracefully.

Oracle
------
measure_oracle() records a CUDA event pair around each region for 3 passes
of N_ITERS iterations and reports the mean per-region device time in us.
This is the reference every GPU-time claim is checked against.

Run:   python3 src/cuda_prof_bench.py                (~60 s, full)
       python3 src/cuda_prof_bench.py --fast          (reduced, ~30 s)
       python3 src/cuda_prof_bench.py --skip-pyspy --skip-nsys --skip-ncu
Artifacts: results/cuda_prof_bench.json
           results/cuda_prof_nsys.nsys-rep           (when nsys runs)
"""

import argparse
import cProfile
import csv
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
import time
import warnings
from collections import defaultdict
from pathlib import Path

import torch

# kineto emits a one-time "Profiler clears events" UserWarning on the first
# profile() context; harmless, but it prints at the top of the log.
warnings.filterwarnings("ignore", message=".*Profiler clears events.*")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
HERE = Path(__file__).resolve()

# ---------------------------------------------------------------- config
B, D = 128, 4096            # batch, width
N_ITERS = 150               # iterations per pass
N_WARMUP = 30
N_SMALL = 15                # child-small (ncu scope)
NCU_LAUNCH_COUNT = 12
# ncu -k regex over demangled function names; cutlass GEMMs show up as
# "Kernel2" (cutlass::Kernel2<...>), split-K epilogues as splitKreduce_kernel
NCU_K_RE = r"cutlass|Kernel2|splitK|gemm|nvjet"
PYSPY_RATE, PYSPY_DUR = 200, 5
NSYS_ROUNDS = 120
RING_N = 8                  # pinned x batches rotated through the loop

REGIONS = ("h2d", "gemm", "pointwise", "reduce", "d2h")


# ---------------------------------------------------------------- state
def build_state():
    dev = torch.device("cuda")
    g = torch.Generator(device="cuda").manual_seed(0)
    W1 = torch.randn(D, D, device=dev, generator=g, dtype=torch.float32)
    W2 = torch.randn(D, D, device=dev, generator=g, dtype=torch.float32)
    b = torch.randn(D, 1, device=dev, generator=g, dtype=torch.float32)
    c = torch.randn(D, 1, device=dev, generator=g, dtype=torch.float32)
    ring = [torch.randn(D, B, device="cpu", pin_memory=True)
            for _ in range(RING_N)]
    x_gpu = torch.empty(D, B, device=dev, dtype=torch.float32)
    dst = torch.empty(512, B, device="cpu", pin_memory=True)
    return dict(W1=W1, W2=W2, b=b, c=c, ring=ring, x_gpu=x_gpu, dst=dst,
                device=dev)


# region functions (each is ONE source line inside run_loop -> py-spy can
# attribute CPU samples to a region by line)
def region_h2d(i, st):
    st["x_gpu"].copy_(st["ring"][i % RING_N], non_blocking=True)
    return st["x_gpu"]


def region_gemm(x, st):
    y = st["W1"] @ x
    y = st["W2"] @ y
    return y


def region_pointwise(y, st):
    return (y * st["b"]).relu() + st["c"]


def region_reduce(y, st):
    return y / y.norm(dim=0, keepdim=True)


def region_d2h(y, st):
    st["dst"].copy_(y[:512], non_blocking=True)
    return st["dst"]


def run_loop(n, st, sync_end=True, sync_per_iter=False):
    for i in range(n):
        x = region_h2d(i, st)
        y = region_gemm(x, st)
        y = region_pointwise(y, st)
        y = region_reduce(y, st)
        region_d2h(y, st)
        if sync_per_iter:
            torch.cuda.synchronize()
    if sync_end:
        torch.cuda.synchronize()


def measure_oracle(st, n):
    """per-region device time (us/iter) via CUDA event pairs; 3 passes."""
    run_loop(20, st)
    # one event pair per region per iteration (an event holds one timestamp)
    ev = {name: [(torch.cuda.Event(enable_timing=True),
                  torch.cuda.Event(enable_timing=True)) for _ in range(n)]
          for name in REGIONS}
    acc = {name: [] for name in REGIONS}
    for _ in range(3):
        for i in range(n):
            ev["h2d"][i][0].record();     x = region_h2d(i, st);     ev["h2d"][i][1].record()
            ev["gemm"][i][0].record();    y = region_gemm(x, st);    ev["gemm"][i][1].record()
            ev["pointwise"][i][0].record(); y = region_pointwise(y, st); ev["pointwise"][i][1].record()
            ev["reduce"][i][0].record();  y = region_reduce(y, st);  ev["reduce"][i][1].record()
            ev["d2h"][i][0].record();     region_d2h(y, st);         ev["d2h"][i][1].record()
        torch.cuda.synchronize()
        for name in REGIONS:
            tot_us = sum(ev[name][j][0].elapsed_time(ev[name][j][1])
                         for j in range(n)) * 1000.0
            acc[name].append(tot_us / n)
    return {name: statistics.fmean(acc[name]) for name in REGIONS}


# ----------------------------------------------------------------
# kernel name -> region.  Verified on sm_120 / torch 2.10+cu128:
#   gemm:      cutlass_80_simt_sgemm_* + cublasLt::splitKreduce_kernel
#   pointwise: vectorized_elementwise_kernel (mul/relu/add)
#   reduce:    reduce_kernel (norm) + one elementwise_kernel (division)
#   copies:    Memcpy HtoD / DtoH (Pinned -> Device / Device -> Pinned)
# The division is also an elementwise kernel, so within each iteration
# the LAST elementwise kernel (the one right before D2H) is counted as
# 'reduce'; earlier ones are 'pointwise'.
# ----------------------------------------------------------------
def classify_kernel(name):
    """name -> region.  Memcpy direction words vary by tool:
    torch.profiler: 'Memcpy HtoD (Pinned -> Device)' / 'Memcpy DtoH ...'
    nsys:          '[CUDA memcpy Host-to-Device]' / '[CUDA memcpy Device-to-Host]'
    so accept h2d/htod/host-to-device and d2h/dtoh/device-to-host."""
    n = name.lower()
    if "memcpy" in n or "mem cpy" in n:
        if "h2d" in n or "htod" in n or "host-to-device" in n:
            return "h2d"
        if "d2h" in n or "dtoh" in n or "device-to-host" in n:
            return "d2h"
        return "other"
    if "memset" in n:
        return "other"
    if "splitk" in n:
        return "gemm"
    if any(t in n for t in ("gemm", "gemv", "nvjet", "cutlass", "kernel2")):
        return "gemm"
    if "distribution" in n:      # randn init in build_state() - not a region
        return "other"
    if "reduce" in n:
        return "reduce"
    if "elementwise" in n:
        return "ew"
    return "other"


def _segment_iters(events):
    """sorted events -> per-iteration buckets [(dur, cls, name)] (split at
    H2D boundaries) + the set of kernel names occupying the last-elementwise
    slot (the division of the reduce region)."""
    iters, cur, div_names = [], None, set()
    for _start, dur, name in events:
        cls = classify_kernel(name)
        if cls == "h2d":
            if cur:
                iters.append(cur)
            cur = [(dur, cls, name)]
        elif cur is not None:
            cur.append((dur, cls, name))
    if cur:
        iters.append(cur)
    for it in iters:
        ew_idx = [i for i, (_d, c, _n) in enumerate(it) if c == "ew"]
        if ew_idx:
            div_names.add(it[ew_idx[-1]][2])
    return iters, div_names


def segment_region_us(events, n_iters):
    """events: sorted [(start_us, dur_us, name)] -> per-region us/iter,
    coverage, and the division kernel name(s)."""
    iters, div_names = _segment_iters(events)
    per = {r: 0.0 for r in REGIONS}
    per["other"] = 0.0
    for it in iters:
        for dur, cls, name in it:
            if cls == "ew":
                per["reduce" if name in div_names else "pointwise"] += dur
            else:
                per[cls] += dur
    for r in list(per):
        per[r] /= max(1, len(iters) or n_iters)
    total = sum(per.values())
    covered = sum(per[r] for r in REGIONS)
    return per, (covered / total if total else 0.0), div_names


# ---------------------------------------------------------------- helpers
def dur_us(x):
    for scale_, name in ((1e3, "ms"), (1e6, "us"), (1e9, "ns")):
        if x * scale_ >= 1:
            return f"{x * scale_:8.1f} {name}"
    return f"{x * 1e9:8.1f} ns"


def geomean(vals):
    if not vals:
        return float("nan")
    return math.exp(statistics.fmean(math.log(v) for v in vals))


def find_bin(name, cuda_root="/usr/local/cuda-13.0/bin"):
    p = shutil.which(name)
    if p:
        return p
    cand = os.path.join(cuda_root, name)
    return cand if os.path.exists(cand) else None


def _child_lines(stdout):
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


# ---------------------------------------------------------------- child
def run_child(rounds, n, fast):
    """py-spy/nsys child: print per-round wall + final oracle."""
    n = int(n)
    st = build_state()
    run_loop(N_WARMUP, st)
    oracle = measure_oracle(st, n)
    t0 = time.perf_counter()
    for i in range(int(rounds)):
        rt0 = time.perf_counter()
        run_loop(n, st)
        print(jsonlib.dumps({"round": i + 1,
                             "round_wall": time.perf_counter() - rt0}),
              flush=True)
    print(jsonlib.dumps({"done": True, "oracle": oracle,
                         "wall": time.perf_counter() - t0}), flush=True)


def run_child_small(n):
    """ncu child: warmup + a few iterations, print oracle."""
    st = build_state()
    run_loop(N_WARMUP, st)
    oracle = measure_oracle(st, n)
    run_loop(n, st)
    print(jsonlib.dumps({"done": True, "oracle": oracle}), flush=True)


# ---------------------------------------------------------------- passes
def wall_passes(st, n):
    out = {}
    # single iteration, no sync: measures CPU launch only (async tail escapes)
    ts = []
    for _ in range(7):
        t0 = time.perf_counter()
        run_loop(1, st, sync_end=False)
        ts.append(time.perf_counter() - t0)
        torch.cuda.synchronize()
    out["nosync_1iter"] = statistics.median(ts)
    # single iteration with sync
    ts = []
    for _ in range(7):
        t0 = time.perf_counter()
        run_loop(1, st)
        ts.append(time.perf_counter() - t0)
    out["sync_1iter"] = statistics.median(ts)
    # full loop, sync only at the end (pipelined)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        run_loop(n, st)
        ts.append(time.perf_counter() - t0)
    out["loop_final_sync"] = statistics.median(ts)
    # full loop, sync every iteration
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        run_loop(n, st, sync_per_iter=True)
        ts.append(time.perf_counter() - t0)
    out["loop_per_iter_sync"] = statistics.median(ts)
    return out


def cprofile_pass(st, n):
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    run_loop(n, st)
    pr.disable()
    wall = time.perf_counter() - t0
    stats = pstats.Stats(pr, stream=io.StringIO()).stats
    top = sorted(((k[2], v[3]) for k, v in stats.items()),
                 key=lambda t: -t[1])[:5]
    return {"wall": wall, "top_cpu": top}


def torch_profiler_pass(st, n):
    from torch.profiler import profile, ProfilerActivity
    acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    # one-time kineto init / CUPTI warm-up - not counted in the overhead
    with profile(activities=acts):
        run_loop(5, st)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=acts) as prof:
        run_loop(n, st)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    cuda = [e for e in prof.events()
            if str(e.device_type) == "DeviceType.CUDA"]
    events = sorted(((e.time_range.start,
                      e.time_range.end - e.time_range.start, e.name)
                     for e in cuda), key=lambda t: t[0])
    per, coverage, div_names = segment_region_us(events, n)
    # fraction of elementwise time that is the division (reduce region)
    div_ew_us = 0.0
    ew_all_us = 0.0
    for _s, d, name in events:
        if classify_kernel(name) == "ew":
            ew_all_us += d
            if name in div_names:
                div_ew_us += d
    ew_split_ratio = div_ew_us / ew_all_us if ew_all_us else 0.0
    agg = defaultdict(float)
    for _s, d, name in events:
        agg[name] += d
    top = sorted(agg.items(), key=lambda kv: -kv[1])[:5]
    cpu = defaultdict(float)
    for e in prof.events():
        if str(e.device_type) == "DeviceType.CPU":
            cpu[e.name] += e.time_range.end - e.time_range.start
    top_cpu = sorted(cpu.items(), key=lambda kv: -kv[1])[:5]
    return {"wall": wall, "per_region_us": per, "coverage": coverage,
            "top_kernels": top, "top_cpu": top_cpu, "div_names": div_names,
            "ew_split_ratio": ew_split_ratio}


def nsys_pass(fast, div_names, ew_split_ratio):
    nsys = find_bin("nsys")
    if not nsys:
        return {"error": "nsys not found"}
    rounds = 40 if fast else NSYS_ROUNDS
    child = [sys.executable, str(HERE), "--child", str(rounds),
             str(N_ITERS)]
    p0 = subprocess.run(child, capture_output=True, text=True)
    b_rounds, b_final = _child_lines(p0.stdout)
    base = RESULTS / "cuda_prof_nsys"
    cmd = [nsys, "profile", "--force-overwrite=true", "-o", str(base),
           "--"] + child
    p1 = subprocess.run(cmd, capture_output=True, text=True)
    u_rounds, u_final = _child_lines(p1.stdout)
    rep = f"{base}.nsys-rep"
    kern = subprocess.run([nsys, "stats", "--force-export=true",
                           "--report", "cuda_gpu_kern_sum",
                           "--format", "csv", rep],
                          capture_output=True, text=True)
    mem = subprocess.run([nsys, "stats", "--force-export=true",
                          "--report", "cuda_gpu_mem_time_sum",
                          "--format", "csv", rep],
                         capture_output=True, text=True)
    kern_us, mem_us = _parse_nsys_csv(kern.stdout), _parse_nsys_csv(mem.stdout)
    # the recording covers warmup + oracle + measured rounds; the child
    # always runs N_ITERS iters/round, so divide by the TOTAL iterations
    n_child = N_ITERS
    total_iters = N_WARMUP + (20 + 3 * n_child) + rounds * n_child
    per, coverage = _nsys_regions(kern_us, mem_us, total_iters, div_names,
                                  ew_split_ratio)
    return {
        "report": str(rep),
        "rounds": rounds,
        "total_iters": total_iters,
        "baseline_round_walls": b_rounds,
        "under_nsys_round_walls": u_rounds,
        "overhead_ratio": (statistics.fmean(u_rounds) /
                           statistics.fmean(b_rounds)
                           if u_rounds and b_rounds else None),
        "oracle_child": (u_final or b_final or {}).get("oracle"),
        "per_region_us": per,
        "coverage": coverage,
        "top_kernels": sorted(kern_us.items(), key=lambda kv: -kv[1])[:5],
    }


def _parse_nsys_csv(out):
    """nsys stats csv -> {label: total_us}.

    Robust to the nsys 2025.x output layout: a 'Generating SQLite ...' /
    'Processing ...' preamble followed by a bare CSV table.  The header row
    is the line containing 'Total Time (ns)'; its label column is 'Name'
    (cuda_gpu_kern_sum) or 'Operation' (cuda_gpu_mem_time_sum)."""
    lines = out.splitlines()
    hdr_idx = next((i for i, l in enumerate(lines)
                    if "Total Time (ns)" in l), None)
    if hdr_idx is None:
        return {}
    block = []
    for line in lines[hdr_idx:]:
        if not line.strip():
            break
        block.append(line)
    rows = list(csv.reader(block))
    if len(rows) < 2:
        return {}
    hdr = [h.strip() for h in rows[0]]
    label = next((c for c in ("Name", "Operation") if c in hdr), None)
    if label is None or "Total Time (ns)" not in hdr:
        return {}
    li, ti = hdr.index(label), hdr.index("Total Time (ns)")
    res = {}
    for row in rows[1:]:
        if len(row) <= max(li, ti):
            continue
        name = row[li].strip()
        if not name:
            continue
        try:
            res[name] = float(row[ti]) / 1000.0
        except ValueError:
            continue
    return res


def _nsys_regions(kern_us, mem_us, iters, div_names, ew_split_ratio):
    """kern_us/mem_us: {label: total_us over the WHOLE recording}.
    Divide by the total number of iterations recorded (warmup + oracle +
    measured rounds) to get us/iter."""
    per = {r: 0.0 for r in REGIONS}
    other = 0.0
    ew_total = 0.0
    ew_named = 0.0
    for name, us in {**mem_us, **kern_us}.items():
        cls = classify_kernel(name)
        if cls == "ew":
            ew_total += us
            if name in div_names:
                per["reduce"] += us
                ew_named += us
            else:
                per["pointwise"] += us
        elif cls in per:
            per[cls] += us
        else:
            other += us
    if ew_total and ew_named < ew_total * 0.5:
        # name matching failed (truncated/merged demangled names): split the
        # elementwise total in the ratio measured by torch.profiler
        per["pointwise"] = ew_total * (1.0 - ew_split_ratio)
        per["reduce"] = per["reduce"] - ew_named + ew_total * ew_split_ratio
    for r in list(per):
        per[r] /= max(1, iters)
    other /= max(1, iters)
    total = sum(per.values()) + other
    covered = sum(per.values())
    return per, (covered / total if total else 0.0)


def ncu_pass():
    ncu = find_bin("ncu")
    if not ncu:
        return {"error": "ncu not found"}
    child = [sys.executable, str(HERE), "--child-small", str(N_SMALL)]
    t0 = time.perf_counter()
    p0 = subprocess.run(child, capture_output=True, text=True)
    bare_wall = time.perf_counter() - t0
    _, fin0 = _child_lines(p0.stdout)
    cmd = [ncu, "--launch-count", str(NCU_LAUNCH_COUNT),
           "-k", "regex:" + NCU_K_RE,
           "--target-processes", "application-only",
           "--", *child]
    t0 = time.perf_counter()
    p1 = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    under_wall = time.perf_counter() - t0
    blob = p1.stdout + p1.stderr
    if "ERR_NVGPUCTRPERM" in blob:
        return {"error": "GPU performance-counter permission denied "
                         "(ERR_NVGPUCTRPERM) - ncu needs root/CAP_SYS_ADMIN",
                "raw_tail": blob.strip().splitlines()[-5:]}
    durs = re_mod.findall(r"^\s*Duration\s*:\s*([\d.]+)\s*(us|ms|ns)",
                          p1.stdout, re_mod.M)
    durs_us = [(float(v) * (1.0 if u == "us" else 1e3 if u == "ms" else 1e-3))
               for v, u in durs]
    oracle = (fin0 or {}).get("oracle", {})
    per_launch_us = oracle.get("gemm", 0.0) / 2.0  # 2 gemm launches/iter
    return {
        "baseline_wall": bare_wall,
        "under_ncu_wall": under_wall,
        "scoped_overhead": under_wall / bare_wall if bare_wall else None,
        "profiled_launches": len(durs_us),
        "mean_kernel_duration_us": statistics.fmean(durs_us) if durs_us
                                   else None,
        "oracle_per_launch_us": per_launch_us,
        "oracle_child": oracle,
    }


def device_memory_pass():
    dev_alloc = torch.cuda.max_memory_allocated() / 1e6
    dev_reserved = torch.cuda.memory_reserved() / 1e6
    host_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {"device_peak_mib": dev_alloc, "device_reserved_mib": dev_reserved,
            "host_maxrss_mib": host_maxrss}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--skip-pyspy", action="store_true")
    ap.add_argument("--skip-nsys", action="store_true")
    ap.add_argument("--skip-ncu", action="store_true")
    args, _unknown = ap.parse_known_args()

    if len(sys.argv) > 1 and "--child" in sys.argv:
        i = sys.argv.index("--child")
        run_child(sys.argv[i + 1], sys.argv[i + 2],
                  "--fast" in sys.argv)
        return
    if len(sys.argv) > 1 and "--child-small" in sys.argv:
        i = sys.argv.index("--child-small")
        run_child_small(int(sys.argv[i + 1]))
        return

    fast = args.fast
    n = 50 if fast else N_ITERS
    RESULTS.mkdir(exist_ok=True)
    print("=" * 76)
    print("CUDA profiling tools benchmark")
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} "
          f"(sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}) "
          f"| B={B} D={D} N={n}")
    print("=" * 76)

    print("\n[1/8] Building state + CUDA-event oracle ...")
    st = build_state()
    run_loop(N_WARMUP, st)
    oracle = measure_oracle(st, n)
    oracle_total = sum(oracle.values())
    print("    per-region device time (us/iter):")
    for r in REGIONS:
        print(f"      {r:10s} {dur_us(oracle[r] / 1e6)}  "
              f"({100 * oracle[r] / oracle_total:4.1f}%)")
    print(f"      {'TOTAL':10s} {dur_us(oracle_total / 1e6)}")

    print("\n[2/8] Wall-clock passes ...")
    wall = wall_passes(st, n)
    per_iter_wall = wall["loop_final_sync"] / n
    gpu_busy = oracle_total / 1e6 / per_iter_wall
    print(f"    nosync 1 iter   : {dur_us(wall['nosync_1iter'])} "
          f"(measures CPU launch only; async tail escapes)")
    print(f"    sync   1 iter   : {dur_us(wall['sync_1iter'])}")
    print(f"    loop, final sync: {dur_us(wall['loop_final_sync'])} "
          f"({per_iter_wall * 1e6:.0f} us/iter)")
    print(f"    loop, sync/iter : {dur_us(wall['loop_per_iter_sync'])} "
          f"({wall['loop_per_iter_sync'] / n * 1e6:.0f} us/iter; "
          f"sync cost "
          f"{wall['loop_per_iter_sync'] / wall['loop_final_sync']:.2f}x)")
    print(f"    trap: nosync under-reports a full iteration by "
          f"{wall['sync_1iter'] / wall['nosync_1iter']:.1f}x")
    print(f"    GPU busy fraction (oracle wall / loop wall): "
          f"{100 * gpu_busy:.1f}%")

    print("\n[3/8] cProfile on the torch loop ...")
    cp = cprofile_pass(st, n)
    cp_over = cp["wall"] / wall["loop_final_sync"]
    print(f"    overhead {cp_over:.2f}x ; top CPU entries (cumulative):")
    for name, t in cp["top_cpu"]:
        print(f"      {name[:50]:50s} {dur_us(t)}")

    print("\n[4/8] torch.profiler (kineto) ...")
    tp = torch_profiler_pass(st, n)
    tp_over = tp["wall"] / wall["loop_final_sync"]
    per = tp["per_region_us"]
    errs = [abs(per[r] - oracle[r]) / oracle[r] for r in REGIONS]
    print(f"    overhead {tp_over:.2f}x ; region attribution vs oracle "
          f"(coverage {100 * tp['coverage']:.1f}%):")
    for r in REGIONS:
        e = abs(per[r] - oracle[r]) / oracle[r]
        print(f"      {r:10s} oracle {oracle[r]:9.1f} us  torchprof "
              f"{per[r]:9.1f} us  err {100 * e:5.1f}%")
    print("    top kernels (us/iter):")
    for name, t in tp["top_kernels"]:
        print(f"      {name[:52]:52s} {dur_us(t / n / 1e6)}")
    print("    note: kernel-profiler times are pure kernel-busy; the event")
    print("          oracle brackets each region, so it also counts inter-")
    print("          kernel gaps + ~4-5us/event-pair overhead. The 'err' is")
    print("          smallest for one big kernel (gemm) and largest for the")
    print("          small multi-kernel regions (pointwise/reduce/d2h).")

    # ------------------------------------------------------------------
    print("\n[5/8] py-spy (external CPU sampling of the launch side) ...")
    pyspy_res = None
    if not args.skip_pyspy:
        pyspy_res = run_pyspy_pass(n, fast)
        if pyspy_res.get("error"):
            print(f"    py-spy: {pyspy_res['error']}")
        else:
            ov = pyspy_res["overhead_ratio"]
            print(f"    samples {pyspy_res['total_samples']} in "
                  f"{PYSPY_DUR}s window; round-wall overhead "
                  f"{ov:.3f}x" if ov else "    py-spy: no complete rounds")
            for r in REGIONS:
                f = pyspy_res["region_share"].get(r, 0.0)
                print(f"      {r:10s} {100 * f:5.1f}% of CPU samples")

    print("\n[6/8] nsys (Nsight Systems, external) ...")
    nsys_res = None
    if not args.skip_nsys:
        nsys_res = nsys_pass(fast, tp["div_names"], tp["ew_split_ratio"])
        if nsys_res.get("error"):
            print(f"    nsys: {nsys_res['error']}")
        else:
            ov = nsys_res["overhead_ratio"]
            orc = nsys_res["oracle_child"] or {}
            print(f"    round-wall overhead {ov:.3f}x ; "
                  f"region attribution vs child oracle "
                  f"(coverage {100 * nsys_res['coverage']:.1f}%):")
            for r in REGIONS:
                if orc:
                    e = (abs(nsys_res["per_region_us"][r] - orc[r]) / orc[r]
                         if orc[r] else 0.0)
                    print(f"      {r:10s} oracle {orc[r]:9.1f} us  nsys "
                          f"{nsys_res['per_region_us'][r]:9.1f} us  "
                          f"err {100 * e:5.1f}%")
            print("    top kernels (total over recording):")
            for name, t in nsys_res["top_kernels"]:
                print(f"      {name[:52]:52s} {t:10.1f} us")

    print("\n[7/8] ncu (Nsight Compute, kernel metrics, replay) ...")
    ncu_res = None
    if not args.skip_ncu:
        ncu_res = ncu_pass()
        if ncu_res.get("error"):
            print(f"    ncu: {ncu_res['error']}")
            if ncu_res.get("raw_tail"):
                for l in ncu_res["raw_tail"][-2:]:
                    print(f"           {l[:70]}")
        elif ncu_res.get("profiled_launches", 0) == 0:
            print("    ncu: no Duration lines parsed (check output format)")
        else:
            ov = ncu_res["scoped_overhead"]
            d = ncu_res["mean_kernel_duration_us"]
            exp = ncu_res["oracle_per_launch_us"]
            err = abs(d - exp) / exp if exp else float("nan")
            print(f"    scoped (12 launches of a {N_SMALL}-iter child): "
                  f"child wall {ncu_res['baseline_wall']:.1f}s -> "
                  f"{ncu_res['under_ncu_wall']:.1f}s ({ov:.1f}x)")
            print(f"    mean profiled-kernel duration {d:.2f} us vs oracle "
                  f"per-launch {exp:.2f} us (err {100 * err:.1f}%)")

    print("\n[8/8] Device memory (the story RSS never tells) ...")
    dm = device_memory_pass()
    print(f"    device peak allocated  {dm['device_peak_mib']:8.1f} MiB "
          f"(reserved {dm['device_reserved_mib']:.1f} MiB)")
    print(f"    host max RSS           {dm['host_maxrss_mib']:8.1f} MiB "
          f"(CUDA context + pinned buffers only)")

    # ------------------------------------------------------------------
    # trap table: what each tool THINKS is hot, vs ground truth
    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("WHAT EACH TOOL SEES AS 'THE HOT THING'  (oracle: "
          f"{max(REGIONS, key=lambda r: oracle[r])} owns "
          f"{100 * max(oracle.values()) / oracle_total:.0f}% of GPU time)")
    print("=" * 76)
    hot_region = max(REGIONS, key=lambda r: oracle[r])
    print(f"  wall_nosync : a single iteration = "
          f"{wall['nosync_1iter'] * 1e6:.1f} us  "
          f"({wall['sync_1iter'] / wall['nosync_1iter']:.0f}x under-report)")
    name, t = cp["top_cpu"][0]
    print(f"  cProfile    : {name[:40]} (CPU-side launch, "
          f"{t * 1e6:.0f} us over {n} iters) - says nothing about the GPU")
    if pyspy_res and not pyspy_res.get("error"):
        top_r = max(pyspy_res["region_share"],
                    key=pyspy_res["region_share"].get)
        print(f"  py-spy      : {top_r} owns "
              f"{100 * pyspy_res['region_share'][top_r]:.0f}% of CPU samples "
              f"(launch side; GPU invisible)")
    name, t = tp["top_kernels"][0]
    print(f"  torch.profiler: {name[:44]} ({t / n:.1f} us/iter) - "
          f"right kernel, right time")
    if nsys_res and not nsys_res.get("error") and nsys_res["top_kernels"]:
        name, t = nsys_res["top_kernels"][0]
        ti = max(1, nsys_res.get("total_iters", 1))
        print(f"  nsys        : {name[:44]} "
              f"({t / ti:.1f} us/iter averaged over the recording)")
    if ncu_res and not ncu_res.get("error") and ncu_res.get("profiled_launches"):
        print(f"  ncu         : per-kernel hardware throughput (deepest view; "
              f"scoped, replayed)")
    print(f"  device mem  : {dm['device_peak_mib']:.0f} MiB on the device, "
          f"invisible to host RSS tools")

    # ------------------------------------------------------------------
    # scorecard
    # ------------------------------------------------------------------
    acc = {}
    acc["torch.profiler"] = {
        "median_relerr": statistics.median(errs),
        "score": max(0.0, 100 * (1 - statistics.median(errs))),
    }
    if nsys_res and not nsys_res.get("error") and nsys_res.get("oracle_child"):
        orc = nsys_res["oracle_child"]
        ne = [abs(nsys_res["per_region_us"][r] - orc[r]) / orc[r]
              for r in REGIONS if orc.get(r)]
        if ne:
            acc["nsys"] = {"median_relerr": statistics.median(ne),
                           "score": max(0.0, 100 * (1 - statistics.median(ne)))}
    if ncu_res and not ncu_res.get("error") and ncu_res.get("profiled_launches"):
        d = ncu_res["mean_kernel_duration_us"]
        exp = ncu_res["oracle_per_launch_us"]
        if exp:
            e = abs(d - exp) / exp
            acc["ncu"] = {"median_relerr": e, "score": max(0.0, 100 * (1 - e))}
    e_wall = abs(wall["loop_final_sync"] / n - oracle_total / 1e6) / (oracle_total / 1e6)
    acc["wall_sync"] = {"median_relerr": e_wall,
                        "score": max(0.0, 100 * (1 - e_wall))}
    e_ns = (wall["sync_1iter"] - wall["nosync_1iter"]) / wall["sync_1iter"]
    acc["wall_nosync"] = {"median_relerr": e_ns,
                          "score": max(0.0, 100 * (1 - e_ns))}

    overhead = {
        "wall_nosync": 1.0,
        "wall_sync": 1.0,
        "cProfile": cp_over,
        "torch.profiler": tp_over,
        "py-spy": (pyspy_res or {}).get("overhead_ratio") or float("nan"),
        "nsys": (nsys_res or {}).get("overhead_ratio") or float("nan"),
        "ncu (scoped)": (ncu_res or {}).get("scoped_overhead") or float("nan"),
    }

    RUBRIC = {
        # setup, granularity, actionability, scope, machine-readable
        "wall_nosync":  (5, 1, 1, 1, 5, "looks fast because it misses the "
                                        "async GPU tail - never ship this"),
        "wall_sync":    (5, 1, 2, 2, 5, "right total, no attribution; the "
                                        "baseline everything else is checked "
                                        "against"),
        "cProfile":     (5, 3, 2, 2, 4, "sees the CPU launch cost of aten ops; "
                                        "blind to kernels"),
        "py-spy":       (3, 3, 3, 3, 4, "near-zero-cost CPU view of the launch "
                                        "side; attach to live processes"),
        "torch.profiler": (4, 5, 5, 5, 4, "per-kernel GPU times + CPU ops + "
                                        "chrome trace; in-process"),
        "nsys":         (3, 5, 4, 5, 4, "whole-process CPU+GPU timeline, "
                                        "near-zero overhead, CSV stats"),
        "ncu":          (3, 5, 5, 2, 4, "per-kernel hardware metrics "
                                        "(throughput %, occupancy); needs "
                                        "counter permission + replay"),
    }
    WEIGHTS = (0.15, 0.25, 0.30, 0.15, 0.15)
    print("\n" + "=" * 76)
    print("Usability rubric (1-5 per axis)")
    print("=" * 76)
    usab = {}
    print(f"    {'tool':16s} setup gran act scope mach  -> score/5")
    for m, (s, g, a, sc, mr, _v) in RUBRIC.items():
        usab[m] = sum(w * v for w, v in zip(WEIGHTS, (s, g, a, sc, mr)))
        print(f"    {m:16s} {s}     {g}   {a}   {sc}    {mr}   -> {usab[m]:.2f}")

    print("\n" + "=" * 76)
    print("FINAL SCORECARD")
    print("=" * 76)
    print(f"    {'tool':16s} {'overhead':>9s} {'accuracy':>10s} {'usability':>9s}   verdict")
    for m, (_s, _g, _a, _sc, _mr, verdict) in RUBRIC.items():
        ov = overhead.get(m, float("nan"))
        ac = acc.get(m, {}).get("score", float("nan"))
        ov_s = f"{ov:9.2f}" if ov == ov else f"{'-':>9s}"
        ac_s = f"{ac:10.1f}" if ac == ac else f"{'-':>10s}"
        print(f"    {m:16s} {ov_s} {ac_s} {usab[m]:>9.2f}   {verdict}")

    out = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "shape": {"B": B, "D": D, "N": n},
        "oracle_us_per_iter": oracle,
        "oracle_total_us_per_iter": oracle_total,
        "gpu_busy_fraction": gpu_busy,
        "wall": wall,
        "cprofile": {"overhead": cp_over,
                     "top_cpu": [[a, b] for a, b in cp["top_cpu"]]},
        "torch_profiler": {"overhead": tp_over,
                           "per_region_us": per, "coverage": tp["coverage"],
                           "top_kernels": [[a, b] for a, b in
                                           tp["top_kernels"]]},
        "pyspy": pyspy_res,
        "nsys": nsys_res,
        "ncu": ncu_res,
        "device_memory": dm,
        "accuracy": acc,
        "overhead": overhead,
        "usability": usab,
    }
    (RESULTS / "cuda_prof_bench.json").write_text(
        jsonlib.dumps(out, indent=2, default=str))
    print(f"\n    JSON: {RESULTS / 'cuda_prof_bench.json'}")
    print("done.")


def run_pyspy_pass(n, fast):
    exe = shutil.which("py-spy") or str(Path.home() / ".local" / "bin" / "py-spy")
    if not os.path.exists(exe):
        return {"error": "py-spy not found (pip install py-spy)"}
    import inspect
    # line map for run_loop: which region each call line belongs to
    src, start = inspect.getsourcelines(run_loop)
    line_map = {}
    for i, line in enumerate(src):
        s = start + i
        if "region_h2d(" in line:
            line_map[s] = "h2d"
        elif "region_gemm(" in line:
            line_map[s] = "gemm"
        elif "region_pointwise(" in line:
            line_map[s] = "pointwise"
        elif "region_reduce(" in line:
            line_map[s] = "reduce"
        elif "region_d2h(" in line:
            line_map[s] = "d2h"

    rounds = 100 if fast else 150
    child = [sys.executable, str(HERE), "--child", str(rounds), str(n)]

    def child_wall(cmd):
        p = subprocess.run(cmd, capture_output=True, text=True)
        rw, _fin = _child_lines(p.stdout)
        return rw, p

    b_rounds, _ = child_wall(child)
    raw_out = RESULTS / "cuda_prof_pyspy_raw.txt"
    cmd = [exe, "record", "--rate", str(PYSPY_RATE), "--duration",
           str(PYSPY_DUR), "--format", "raw", "-o", str(raw_out),
           "--"] + child
    u_rounds, p = child_wall(cmd)
    region_samples, outside, total = {}, 0, 0
    for line in raw_out.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        try:
            frames_part, count = line.rsplit(" ", 1)
            count = int(count)
        except ValueError:
            continue
        total += count
        hit = None
        for frame in reversed(frames_part.split(";")):
            m = re_mod.match(r"^(.*?)\s+\((.+?)\)$", frame.strip())
            if not m:
                continue
            loc = m.group(2)
            mm = re_mod.match(r"^(.+?):(\d+)$", loc)
            if mm and HERE.name == mm.group(1) \
                    and int(mm.group(2)) in line_map:
                hit = line_map[int(mm.group(2))]
                break
        if hit:
            region_samples[hit] = region_samples.get(hit, 0) + count
        else:
            outside += count
    share = {r: (region_samples.get(r, 0) / total if total else 0.0)
             for r in REGIONS}
    return {
        "baseline_round_walls": b_rounds,
        "under_pyspy_round_walls": u_rounds,
        "overhead_ratio": (statistics.fmean(u_rounds) /
                           statistics.fmean(b_rounds)
                           if u_rounds and b_rounds else None),
        "total_samples": total,
        "sampling_window_s": PYSPY_DUR,
        "region_share": share,
        "outside": outside,
        "raw": str(raw_out),
    }


if __name__ == "__main__":
    main()
