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
brew tap shadyuwugurl/forge
brew install forge
# or: brew install shadyuwugurl/forge/forge
```

Or build from source:

```bash
cargo build --release
```

## Quick Start (Modern Models)

### Latest Model Families (2024-2025)
- **Qwen 2.5 / Qwen 3** — Strong coding/reasoning, MoE variants (Qwen 2.5-32B, Qwen 3-235B-A22B)
- **Nemotron 3 Ultra / Nemotron 4** — NVIDIA's reasoning models, strong on GPQA
- **Llama 3.1 / 3.2** — Meta's latest, 8B/70B/405B, tool calling native
- **Nemotron 3 Ultra** — 120B reasoning, strong on MMLU/GPQA
- **Gemma 2 / 3** — Google's lightweight, great on-device

```bash
# Model info (auto-detects architecture)
forge info qwen/Qwen2.5-32B-Instruct
forge info nvidia/Nemotron-3-Ultra
forge info meta-llama/Llama-3.2-70B-Instruct

# Search & download modern models
forge search "qwen 2.5 coding"
forge search "nemotron reasoning"
forge search "llama 3.2 tool use"

forge download qwen/Qwen2.5-32B-Instruct --output ./models
forge download nvidia/Nemotron-3-Ultra --output ./models

# Cross-architecture merge (e.g., Qwen 2.5 2.6B + Qwen 3.8 27B → 3B fused)
forge merge --models ./Qwen2.5-2.6B ./Qwen3.8-27B --method frankenmerge --output ./fused

# Modern quant for Apple Silicon
forge quant ./model --method jang --profile JANG_2L --output ./quantized    # MLX-native mixed-prec
forge quant ./model --method dynamic3 --density 0.5 --output ./quantized   # Unsloth Dynamic 3.0
forge quant ./model --method apex --profile i_quality --output ./quantized  # Apex MoE-aware
forge quant ./model --method btl4 --output ./quantized                      # BTL4 compact

# Evaluate with modern benchmarks
forge eval ./model --benchmarks hella,mmlu,arc,gsm8k,gpqa
forge eval ./model --evals ace,swe,terminal,gaia,hle
forge eval --original ./base --model ./merged --benchmarks gpqa --evals hle  # before/after delta

# LoRA/QLoRA/DoRA/GRPO training
forge train --model qwen/Qwen2.5-7B --dataset ./data.jsonl --output ./lora --method lora --rank 32 --alpha 64 --lr 2e-4 --epochs 3
forge train --model qwen/Qwen2.5-7B --dataset ./reasoning.jsonl --output ./grpo --method grpo --epochs 5

# Extract LoRA from fine-tuned model
forge extract --model ./finetuned --base ./base --output ./adapter --rank 16

# Fuse + quant + eval pipeline
forge fuse --base ./base --adapters ./lora_dir --output ./fused
forge quant ./fused --method dynamic3 --density 0.4 --output ./quantized
forge eval ./quantized --benchmarks gpqa --evals hle --original ./base
```

## Model-Specific Quick Recipes

### Qwen 2.5/3 Merge (Cross-Architecture)
```bash
# FrankenMerge: layer-stack Qwen 2.5 2.6B + Qwen 3.8 27B → grab 3B params from big
forge merge --models ./Qwen2.5-2.6B ./Qwen3.8-27B --method frankenmerge --output ./qwen-fused

# Darwin V6 evolutionary (best for reasoning)
forge merge --models ./Qwen2.5-7B ./Nemotron-3-8B --method darwin --generations 50 --population 60 --output ./qwen-nemotron
```

### Nemotron 3 Ultra Quantization (Apple Silicon)
```bash
# Apex i_quality: 21.3GB, beats Q8_0 at half size
forge quant ./Nemotron-3-Ultra --method apex --profile i_quality --output ./nemotron-iq

# Dynamic 3.0: model-specific, Apple Silicon formats (Q4_NL, Q5.1, Q5.0, Q4.1, Q4.0)
forge quant ./Nemotron-3-Ultra --method dynamic3 --density 0.5 --output ./nemotron-dyn
```

### Llama 3.1/3.2 + Tool Use
```bash
# Merge Llama 3.1 70B with Nemotron for reasoning + tools
forge merge --models ./Llama-3.1-70B-Instruct ./Nemotron-3-8B --method darwin --generations 30 --output ./llama-nemotron

# Quantize with tool-use preservation (Apex preserves attention)
forge quant ./llama-nemotron --method apex --profile balanced --output ./llama-nemotron-quant
```

### Data Ripping / Training Data Extraction
```bash
# Level 1: Weight diff (base vs finetuned)
forge extract --model ./finetuned --base ./base --output ./adapter --rank 16

# Level 2: Activation probing (needs calibration set)
forge extract --model ./finetuned --base ./base --output ./data --method probe --calib ./calib.jsonl

# Level 3: Knowledge distillation reconstruction
forge extract --model ./finetuned --base ./base --output ./reconstructed --method distill --teacher ./teacher

# Full pipeline: extract → merge adapters → fuse → quant → eval
forge fuse --base ./base --adapters ./adapters --output ./fused
```

### Distributed Merge (Multi-Mac Thunderbolt)
```bash
# On each Mac: forge cluster status
# On coordinator: forge merge --models A B C --method darwin --distributed --output ./distributed-merged
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