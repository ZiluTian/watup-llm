# The LLM Prefill Hot Path -- CUDA GEMM optimization experiments.
#
# Build:  make
# Run:    make run             (defaults to 4096 x 4096 x 4096)
#         make run-large       (M = 8192: long-prompt prefill)
#
# ARCH must match your GPU's compute capability:
#   sm_80  A100            sm_86  RTX 30xx        sm_89  RTX 40xx
#   sm_90  H100/H800       sm_100 B100/B200       sm_120 RTX 50xx (5090)

ARCH   ?= sm_120
NVCC   ?= nvcc
CFLAGS := -O3 -arch=$(ARCH) -std=c++17

SRC := src/prefill_hotpath.cu
BIN := bin/prefill_hotpath

$(BIN): $(SRC)
	@mkdir -p bin
	$(NVCC) $(CFLAGS) -lcublas -o $@ $<

run: $(BIN)
	./$(BIN)

run-large: $(BIN)
	./$(BIN) 8192 4096 4096

clean:
	rm -rf bin

.PHONY: run run-large clean
