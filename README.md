# Forge

Unified model merging, quantization, and evaluation tool for Apple Silicon. Built in Rust.

> Bare-metal optimized for M4/M5/M6. Leverages Metal GPU, Neural Engine, CPU, and unified memory.

## Features

- **16+ Merge Strategies**: linear, slerp, ties, dare, della, passthrough, frankenmerge, model_stock, breadcrumbs, nearswap, ram, souper + Darwin V6 evolutionary
- **Cross-Architecture**: Dense, sparse (MoE), Transformer, SSM/Mamba, hybrids — different heads/dims/hidden sizes, heterogeneous & polygenous families
- **Darwin V6 Engine**: 14-dimensional genome, MRI-Trust Fusion, CMA-ES evolution, multi-generation recursive merging
- **Quantization**: JangQ (MLX + GGUF), Unsloth Dynamic 3.0, Apex MoE-aware, BTL4 Compact, generic mixed-precision
- **Evaluation**: 5 benchmarks (HellaSwag → GPQA Diamond) + 5 evals (ACE → HLE) with before/after comparison
- **Training Pipeline**: LoRA/adapter extraction, data ripping (weight diff → activation probe → knowledge distill), fusing pipeline
- **Apple Silicon**: Metal kernels, ANE dispatch, M4/M5/M6 detection, Thunderbolt distributed merging
- **HuggingFace Hub**: Search & download built-in
- **Interfaces**: CLI + TUI (ratatui) + GUI (Tauri + Vue)

## Install

```bash
brew tap forge-ml/forge
brew install forge
```

Or build from source:

```bash
cargo build --release
```

## Quick Start

```bash
# Model info
forge info ./model
forge info meta-llama/Llama-3.1-8B-Instruct

# Search & download
forge search "qwen 27b coding"
forge download meta-llama/Llama-3.1-70B-Instruct --output ./models

# Merge
forge merge --config config.yaml --output ./merged
forge merge --models modelA modelB --method slerp --t 0.5 --output ./merged
forge merge --models modelA modelB --method darwin --generations 30 --population 40 --output ./merged

# Quantize
forge quant ./model --method jang --profile JANG_2L --output ./quantized
forge quant ./model --method dynamic3 --density 0.5 --output ./quantized
forge quant ./model --method apex --profile balanced --output ./quantized

# Evaluate
forge eval ./model --benchmarks hella,mmlu,arc,gsm8k,gpqa
forge eval ./model --benchmarks mmlu,gsm8k --evals ace,swe --original ./base

# Fuse adapters
forge fuse --base ./base --adapters ./lora_dir --output ./fused
forge extract --model ./finetuned --base ./base --output ./adapter.safetensors --rank 16

# TUI / GUI
forge tui
forge gui
```

## Benchmarks & Evals

| # | Benchmark | Difficulty |
|---|-----------|------------|
| 1 | HellaSwag | Easy |
| 2 | MMLU | Medium |
| 3 | ARC-Challenge | Medium-Hard |
| 4 | GSM8K | Hard |
| 5 | GPQA Diamond | Very Hard |

| # | Eval | Difficulty |
|---|------|------------|
| 1 | ACE | Easy |
| 2 | SWE-bench | Medium |
| 3 | TerminalBench | Medium-Hard |
| 4 | GAIA | Hard |
| 5 | HLE | Very Hard |

## Architecture

```
crates/
  forge-core    # Types, config (YAML+TOML+JSON), errors
  forge-io      # mmap TensorStore (from alloy), StreamingWriter
  forge-merge   # 16+ strategies + FrankenMerge + ArchitectureMapper
  forge-darwin  # Darwin V6: 14-dim genome, MRI-Trust, CMA-ES
  forge-quant   # JangQ, Dynamic3, Apex, BTL4, mixed-prec
  forge-eval    # Benchmark/eval runners + comparison
  forge-train   # LoRA extraction, data ripping, fusing
  forge-hub     # HuggingFace search/download
  forge-metal   # Metal GPU, ANE, hardware detection
  forge-cli     # CLI binary
  forge-tui     # Terminal UI
```

## License

MIT OR Apache-2.0

Forked from [PMetal](https://github.com/Epistates/pmetal) + ported from [Alloy](https://github.com/srijitiyer/alloy) + [Darwin](https://arxiv.org/abs/2605.14386) + [JangQ](https://github.com/jjang-ai/jangq) + [Apex](https://github.com/localai-org/apex-quant).
