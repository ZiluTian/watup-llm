# WATuP-LLM

LLM fine-tuning and low-latency inference pipeline.

> **Work in progress** — the full pipeline, models, and evaluation scripts are being added.

## What this covers

- Fine-tuning open-weight LLMs with parameter-efficient methods (LoRA / QLoRA)
- A lightweight RAG inference service tuned for production-like latency constraints
- Model quantization and operator-level optimizations to reduce memory footprint on constrained hardware
- Precision / speed / resource-utilization trade-off evaluation across inference workloads
