# Python profiling methods benchmark -- eight methods x nine workloads.
#
# Requirements: Python 3.10+ with numpy.
# Optional (the bench degrades gracefully without them):
#   pip install py-spy line_profiler memory_profiler
#
# Run:    make run         (full, ~90 s)
#         make run-fast    (reduced sizes, ~40 s)

PY ?= python3

run:
	$(PY) src/python_prof_bench.py

run-fast:
	$(PY) src/python_prof_bench.py --fast

clean:
	rm -rf bin

.PHONY: run run-fast clean
