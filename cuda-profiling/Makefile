# CUDA profiling tools benchmark -- seven tools vs a CUDA-event oracle.
#
# Requirements: Python 3.10+ with a CUDA build of torch, one NVIDIA GPU
# (developed on RTX 5090 / sm_120, torch 2.10.0+cu128).
# Optional: pip install py-spy.  nsys/ncu from the NVIDIA Nsight packages;
# ncu needs GPU performance-counter permission (root/CAP_SYS_ADMIN) and the
# bench degrades gracefully without it (ERR_NVGPUCTRPERM).
#
# Run:    make run         (full, ~40 s)
#         make run-fast    (reduced sizes, ~30 s)

PY ?= python3

run:
	$(PY) src/cuda_prof_bench.py

run-fast:
	$(PY) src/cuda_prof_bench.py --fast

clean:
	rm -rf bin

.PHONY: run run-fast clean
