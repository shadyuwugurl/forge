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

---

## 🔬 Forge R&D Roadmap — Advanced Model Surgery & Compression

This section outlines the implementation plan for integrating cutting-edge research into Forge's merging, quantization, training, and extraction capabilities.

### 📚 Research Papers & Implementations

| Technique | Paper | Repository | Target Crate | Status |
|-----------|-------|------------|--------------|--------|
| **LoPT** (Low-Rank Parameter Tuning) | [arXiv:2605.04913](https://arxiv.org/abs/2605.04913) | [HumyuShi/LoPT](https://github.com/HumyuShi/LoPT) | `forge-train`, `forge-merge` | 📋 Planned |
| **QuEPT** (Quantization-Aware PEFT) | [arXiv:2602.12609](https://arxiv.org/abs/2602.12609) | [xuke225/QuEPT](https://github.com/xuke225/QuEPT) | `forge-quant`, `forge-train` | 📋 Planned |
| **OneComp** (One-Shot Compression) | [arXiv:2603.28845](https://arxiv.org/abs/2603.28845) | [FujitsuResearch/OneCompression](https://github.com/FujitsuResearch/OneCompression) | `forge-quant` | 📋 Planned |
| **E-PMQ** (Efficient Post-Training Mixed Quant) | [arXiv:2605.16882](https://arxiv.org/abs/2605.16882) | [wwjzhy/E-PMQ](https://github.com/wwjzhy/E-PMQ) | `forge-quant` | 📋 Planned |
| **Blockwise SFT** | [arXiv:2508.19529](https://arxiv.org/abs/2508.19529) | [Bowen-Sun-0728/Blockwise-SFT](https://github.com/Bowen-Sun-0728/Blockwise-SFT) | `forge-train` | 📋 Planned |
| **CT-SFT** (Curriculum Training for SFT) | [arXiv:2601.08146](https://arxiv.org/abs/2601.08146) | — | `forge-train` | 📋 Planned |

### 🍔 FrankenMoE Burger-Style Merge Support

| Feature | Description | Target Crate | Status |
|---------|-------------|--------------|--------|
| **Burger Architecture** | Bottom (embed) → Middle (expert layers) → Top (output/norm) stacking | `forge-merge` | 📋 Planned |
| **Expert-Aware Slicing** | Route expert layers independently per model family | `forge-merge` | 📋 Planned |
| **Router Fusion** | Merge/router adaptation for cross-family MoE | `forge-merge` | 📋 Planned |
| **Top-K Expert Selection** | Select most compatible experts across models | `forge-merge` | 📋 Planned |
| **Layer-Group Stacking** | Stack MLP/Attention blocks as "patties" with shared norms as "buns" | `forge-merge` | 📋 Planned |

### 🧬 Model Merging & Architecture Fusion

| Feature | Description | Target Crate | Status |
|---------|-------------|--------------|--------|
| **Darwin Merge V6+** | Evolutionary merging with 14-dim genome, MRI-Trust, CMA-ES | `forge-darwin` | ✅ Partial |
| **FrankenMerge++** | Layer-wise with dimension adapters, cross-arch support | `forge-merge` | ✅ Partial |
| **Cross-Architecture Layer Swapping** | Swap layers between architectures (Mamba2 → Mamba3, Transformer ↔ SSM) | `forge-merge` | 📋 Planned |
| **Fusion Layers & Weights** | Fuse layers/weights from multiple models into target | `forge-merge` | 📋 Planned |
| **Multi-Model & Multi-Adapter Merging** | Merge N models, LoRAs, adapters, different architectures | `forge-merge` | 📋 Planned |
| **MergeKit Feature Parity** | Port mergekit strategies: linear, slerp, ties, dare, delta, passthrough, breadcrumbs, ram, nearswap, model_stock, souper | `forge-merge` | ✅ Partial |

### ⚡ Active-Aware Quantization

| Feature | Description | Target Crate | Status |
|---------|-------------|--------------|--------|
| **Active Layer Quantization** | Runtime profiling to identify quantization-sensitive layers | `forge-quant` | 📋 Planned |
| **Mixed Precision Per-Layer** | User/auto-assigned bits per tensor (2-8 bits) | `forge-quant` | ✅ Partial (`mixed.rs`) |
| **Binary/Ternary Quantization** | 1-bit (binary) and 2-bit (ternary) weight quantization | `forge-quant` | 📋 Planned |
| **Activation-Aware Quant** | Quantize based on activation statistics, not just weights | `forge-quant` | 📋 Planned |
| **Hessian/Fisher-Aware Quant** | Second-order sensitivity for layer-wise bit allocation | `forge-quant` | 📋 Planned |
| **QuEPT Integration** | Quantization-aware PEFT during training | `forge-quant`, `forge-train` | 📋 Planned |

### 🏗️ Training from Scratch & Extraction

| Feature | Description | Target Crate | Status |
|---------|-------------|--------------|--------|
| **Train from Scratch Guide** | Complete walkthrough: data → tokenizer → pretraining → eval | `forge-train`, docs | 📋 Planned |
| **LoRA/Adapter Extraction** | Extract LoRA from fine-tuned model (L1 weight diff) | `forge-train` | ✅ Done |
| **Activation Probing (L2)** | Calibration-based activation analysis | `forge-train` | ✅ Done |
| **Knowledge Distillation (L3)** | Reconstruct training data via distillation | `forge-train` | ✅ Partial |
| **Full Training Pipeline** | Extract → Merge Adapters → Fuse → Quant → Eval | `forge-train`, `forge-merge`, `forge-quant` | 📋 Planned |
| **Dataset Reconstruction** | Reverse-engineer dataset from model behavior | `forge-train` | 📋 Planned |
| **Custom Training Reverse Engineering** | Recover hyperparams, LR schedule, optimizer state from weights | `forge-train` | 📋 Planned |

### 🔄 Dense ↔ Sparse Conversion

| Feature | Description | Target Crate | Status |
|---------|-------------|--------------|--------|
| **Dense → MoE (Sparse)** | Convert dense model to Mixture of Experts | `forge-merge` | 📋 Planned |
| **MoE → Dense (Merge Experts)** | Merge experts into single dense FFN | `forge-merge` | 📋 Planned |
| **Expert Splitting/Clustering** | Split FFN into experts via weight clustering | `forge-merge` | 📋 Planned |
| **Router Initialization** | Initialize router from dense attention patterns | `forge-merge` | 📋 Planned |

### 📁 Implementation Structure (New/Extended Crates)

```
crates/
├── forge-core/          # Extended: LoPT/QuEPT types, extraction types, burger config
├── forge-io/            # Extended: streaming for large model surgery
├── forge-merge/         # Extended: burger_merge.rs, layer_swap.rs, fusion.rs, sparse_dense.rs
├── forge-darwin/        # Extended: LoPT/QuEPT genome encoding
├── forge-quant/         # Extended: active_aware.rs, binary_ternary.rs, epmq.rs, onecomp.rs, quept.rs
├── forge-eval/          # Extended: paper-specific eval suites
├── forge-train/         # Extended: lopt.rs, blockwise.rs, ctsft.rs, scratch.rs, extraction_v2.rs
├── forge-hub/           # Extended: paper/model search
├── forge-metal/         # Extended: kernels for new quant formats
├── forge-cli/           # Extended: new subcommands
├── forge-tui/           # Extended: interactive burger/darwin builders
├── forge-gui/           # Extended: visual merge/quant builder
└── forge-surgery/       # NEW: cross-arch layer swap, fusion, dense↔sparse
```

### 🎯 Implementation Priority (Phased)

#### Phase 1: Core Infrastructure (Weeks 1-3)
- [ ] `forge-surgery` crate: ArchitectureMapper, LayerSwapper, FusionEngine
- [ ] Extend `forge-merge` with burger MoE merge strategy
- [ ] Extend `forge-quant` with active-aware quantization framework
- [ ] Add binary/ternary quantization kernels in `forge-metal`

#### Phase 2: Paper Implementations (Weeks 4-8)
- [ ] **LoPT**: Low-rank parameter tuning integration in `forge-train` + `forge-merge`
- [ ] **QuEPT**: Quantization-aware PEFT in `forge-quant` + `forge-train`
- [ ] **OneComp**: One-shot compression in `forge-quant`
- [ ] **E-PMQ**: Mixed precision post-training quant in `forge-quant`
- [ ] **Blockwise SFT**: Block-wise training in `forge-train`
- [ ] **CT-SFT**: Curriculum training in `forge-train`

#### Phase 3: Advanced Features (Weeks 9-14)
- [ ] Dense ↔ Sparse conversion pipeline
- [ ] Cross-architecture layer swapping (Mamba ↔ Transformer)
- [ ] Training extraction v2: full pipeline (deltas → LoRA → dataset → config)
- [ ] MergeKit feature parity audit & completion
- [ ] Train-from-scratch guide & scripts

#### Phase 4: Integration & Polish (Weeks 15-18)
- [ ] CLI/TUI/GUI integration for all new features
- [ ] Benchmarks & eval suites for each paper
- [ ] Documentation: guides, API docs, recipes
- [ ] Performance optimization on M4/M5/M6
- [ ] Release v1.0 with all features

### 🛠️ New CLI Commands (Target)

```bash
# FrankenMoE Burger Merge
forge merge --models A B C --method burger --bottom-layers 4 --middle-experts 8 --top-layers 2 --output ./burger

# Layer Swapping (Cross-Architecture)
forge surgery swap --model ./mamba2 --target-arch mamba3 --layers 12-24 --output ./mamba3-swapped

# Fusion Layers
forge surgery fuse --models A B C --layers attention+mlp --weights 0.5,0.3,0.2 --output ./fused

# Dense ↔ Sparse
forge surgery densify --model ./moe --output ./dense
forge surgery sparsify --model ./dense --num-experts 8 --router-type topk --output ./moe

# Active-Aware Quant
forge quant ./model --method active --profile sensitive_layers.json --output ./quant

# Binary/Ternary Quant
forge quant ./model --method binary --output ./bin
forge quant ./model --method ternary --output ./ternary

# Paper-Specific Training
forge train --model ./base --method lopt --rank 8 --alpha 16 --dataset ./data
forge train --model ./base --method quept --bits 4 --dataset ./data
forge train --model ./base --method blockwise --blocks 4 --dataset ./data
forge train --model ./base --method ctsft --curriculum ./curriculum.yaml --dataset ./data

# Train from Scratch
forge train-scratch --config ./pretrain_config.yaml --data ./corpus --output ./model

# Full Extraction Pipeline
forge extract --model ./finetuned --base ./base --full-pipeline --output ./extracted
# → outputs: deltas, lora, dataset, config, fused_model, quantized_model
```

### 📖 Documentation Plan

| Document | Location | Status |
|----------|----------|--------|
| FrankenMoE Burger Merge Guide | `docs/burger_merge.md` | 📋 Planned |
| Layer Swapping / Cross-Arch Surgery | `docs/surgery.md` | 📋 Planned |
| Active-Aware Quantization | `docs/active_quant.md` | 📋 Planned |
| Binary/Ternary Quant Guide | `docs/binary_ternary.md` | 📋 Planned |
| Paper Implementation Guides (6 papers) | `docs/papers/` | 📋 Planned |
| Train from Scratch Walkthrough | `docs/scratch_training.md` | 📋 Planned |
| Extraction Pipeline Deep Dive | `docs/extraction.md` | 📋 Planned |
| Dense ↔ Sparse Conversion | `docs/dense_sparse.md` | 📋 Planned |
| MergeKit Parity Matrix | `docs/mergekit_parity.md` | 📋 Planned |

### 🤝 Contributing

This is a research-focused fork. For upstream contributions, see the original repositories. For Forge-specific features, PRs welcome!

### 📄 License

MIT OR Apache-2.0 — same as upstream.