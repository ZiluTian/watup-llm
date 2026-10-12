# The LLM Decode Hot Path -- CUDA GEMV optimization experiments.
#
# Build:  make
# Run:    make run             (defaults to 4096 x 4096)
#         make run-large       (8192 x 16384, ~537 MB of fp32 weights)
#
# ARCH must match your GPU's compute capability:
#   sm_80  A100            sm_86  RTX 30xx        sm_89  RTX 40xx
#   sm_90  H100/H800       sm_100 B100/B200       sm_120 RTX 50xx (5090)

ARCH   ?= sm_120
NVCC   ?= nvcc
CFLAGS := -O3 -arch=$(ARCH) -std=c++17

SRC := src/decode_hotpath.cu
BIN := bin/decode_hotpath

$(BIN): $(SRC)
	@mkdir -p bin
	$(NVCC) $(CFLAGS) -o $@ $<

run: $(BIN)
	./$(BIN)

run-large: $(BIN)
	./$(BIN) 8192 16384

clean:
	rm -rf bin

.PHONY: run run-large clean
