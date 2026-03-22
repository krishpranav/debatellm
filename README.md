---
license: mit
language:
- en
base_model:
- mistralai/Mistral-7B-Instruct-v0.3
- microsoft/Phi-3-mini-4k-instruct
tags:
- legal
---

# DebateLM

**The steel-manning language model. Built in Rust. Runs everywhere.**

> *The only language model trained to argue against you better than you can argue against yourself.*

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange)](https://rustup.rs)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-debatelm-yellow)](https://huggingface.co/krishpranav34/debatelm)

---

## What is Steel-Manning?

Steel-manning is the intellectual opposite of strawmanning. Instead of weakening an
opponent's argument to defeat it easily, you strengthen it to its absolute best form
before engaging with it.

This is how great lawyers, debaters, philosophers, and policy analysts think.

**No existing model is trained specifically for this.** GPT-4, Claude, Gemini — they
all do it sometimes, poorly, when asked. DebateLM does it always, deeply, as its
primary function.

---

## Quick Start

### Install

```bash
# From crates.io (once published)
cargo install debatelm

# Or build from source
git clone https://github.com/krishpranav/debatelm
cd debatelm
cargo build --release --features cuda   # GPU (recommended)
cargo build --release                   # CPU
```

### Run

```bash
# Single argument
debatelm --model-id your-username/debatelm \
         --argument "Remote work permanently damages company culture"

# Interactive REPL
debatelm --model-id your-username/debatelm --interactive

# From local model directory (no network needed)
debatelm --local-path ./debatelm-merged/ \
         --argument "Universal healthcare reduces medical innovation"

# High-quality output (top-p + temperature tuned)
debatelm --model-id your-username/debatelm \
         --argument "AI will eliminate more jobs than it creates" \
         --temperature 0.8 \
         --top-p 0.95 \
         --max-tokens 600
```

### Example Output

```
🎯 Steel-manning: "Universal Basic Income reduces the incentive to work"

────────────────────────────────────────────────────────────

The strongest case for this position rests on robust empirical and theoretical
foundations that mainstream UBI advocates consistently underweight.

Labour supply elasticity studies consistently document 5–8% reductions in work
hours among unconditional cash transfer recipients. At the scale of a national
UBI program, this aggregate effect could suppress labour force participation by
millions of workers in sectors where marginal productivity already barely exceeds
the transfer amount — particularly entry-level service work and gig economy
participation.

The theoretical mechanism is straightforward: UBI shifts the reservation wage
upward. Workers who would previously have accepted low-wage, high-disutility
employment rather than face income insecurity now have a credible outside option.
The sectors providing these positions — elder care, food production, logistics —
cannot easily automate and cannot raise wages sufficiently to compensate without
price increases that erode the real value of the very UBI intended to help.

────────────────────────────────────────────────────────────
```

---

## Architecture

```
Training (Python)                    Inference (Rust)
─────────────────                    ────────────────
Raw Datasets                         User Input
    │                                    │
    ▼                                    ▼
Data Pipeline                        CLI (clap)
(download → clean → format)              │
    │                                    ▼
    ▼                               Candle Engine
LoRA Fine-tuning                   (loads .safetensors
(Mistral-7B base)                   via mmap)
    │                                    │
    ▼                                    ▼
Merge Weights                       Tokenize → Forward Pass
(.safetensors)                           │
    │                                    ▼
    ▼                               Sample (top-p)
HuggingFace Hub ──────────────────►     │
                                         ▼
                                     Stream Output
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Base Model | Mistral-7B-Instruct-v0.3 | Pretrained transformer backbone |
| Fine-tuning | Python + PEFT/LoRA | Domain-specific argumentation training |
| Inference | Rust + HuggingFace Candle | Fast, zero-dependency inference binary |
| Weights | SafeTensors | Memory-mapped weight loading |
| Tokenizer | HuggingFace tokenizers (Rust) | Zero-copy BPE tokenization |
| CLI | clap derive macros | Type-safe argument parsing |
| Sampling | Greedy + Temperature + Top-p | Production-quality generation |

---

## CLI Reference

```
USAGE:
    debatelm [OPTIONS]

OPTIONS:
    -a, --argument <ARGUMENT>        The argument to steel-man
    -i, --interactive                Run in interactive REPL mode
        --model-id <MODEL_ID>        HuggingFace Hub model ID [env: DEBATELM_MODEL_ID]
        --local-path <PATH>          Load from local directory
        --max-tokens <N>             Maximum tokens to generate [default: 512]
        --temperature <TEMP>         Sampling temperature [default: 0.7]
        --top-p <P>                  Top-p nucleus sampling threshold [default: 0.9]
        --seed <SEED>                Random seed for reproducible output
        --dtype <DTYPE>              Weight dtype: auto|f32|f16|bf16 [default: auto]
        --no-stream                  Print full response when complete (no streaming)
    -v, --verbose                    Enable debug logging
    -h, --help                       Print help
    -V, --version                    Print version
```

---

## Building from Source

### Prerequisites

- Rust 1.75+ (`rustup.rs`)
- Python 3.10+ (training only)
- CUDA 12.x + cuDNN 8.x (GPU inference)
- ~50GB disk space (datasets + model weights)

### Build

```bash
# CPU (development)
cargo build --release

# CUDA GPU (production — recommended)
cargo build --release --features cuda

# Apple Silicon
cargo build --release --features metal

# Flash Attention 2 (A100/H100 — max throughput)
cargo build --release --features "cuda flash-attn"
```

### Test

```bash
cargo test
```

---

## Training Your Own DebateLM

### 1. Set up Python environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch transformers peft trl datasets safetensors \
            accelerate bitsandbytes huggingface_hub
```

### 2. Download and prepare data

```bash
export HUGGINGFACE_TOKEN=hf_xxxx

python data_pipeline/download.py
python data_pipeline/clean.py
python data_pipeline/format.py --output data/debatelm_train.jsonl
python data_pipeline/split.py
python data_pipeline/verify.py
```

### 3. Train

```bash
# Fast iteration (Phi-3-mini, ~8GB VRAM)
python train.py --model phi3-mini --epochs 3

# Production quality (Mistral-7B, ~24GB VRAM)
python train.py --model mistral --epochs 3
```

### 4. Export to SafeTensors

```bash
python export.py --adapter ./debatelm-checkpoints/final_adapter \
                 --output   debatelm-merged/ \
                 --push     \
                 --hub-id   your-username/debatelm
```

### 5. Run inference

```bash
./target/release/debatelm --local-path debatelm-merged/ \
    --argument "Open source software is less secure than proprietary software"
```

---

## Environment Variables

```bash
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxx    # HuggingFace API token
DEBATELM_MODEL_ID=your-user/debatelm # Default model ID for CLI
CUDA_VISIBLE_DEVICES=0               # GPU device selection
RUST_LOG=debug                       # Enable debug logging
```

---

## Project Structure

```
debatelm/
├── Cargo.toml                  # Rust dependencies and features
├── src/
│   ├── main.rs                 # CLI entry point (clap)
│   ├── error.rs                # Typed error hierarchy (thiserror)
│   ├── config.rs               # config.json deserialization + shard index
│   ├── tokenizer.rs            # tokenizers crate wrapper
│   ├── model.rs                # Candle model variant (Mistral/Phi-3)
│   ├── prompt.rs               # Instruction template formatting
│   └── inference.rs            # Engine: load + prefill + decode + sample
├── data_pipeline/
│   ├── download.py             # Dataset acquisition
│   ├── clean.py                # Quality filtering + deduplication
│   ├── format.py               # Unified JSONL instruction format
│   ├── split.py                # Stratified train/val/test split
│   └── verify.py               # Pre-training sanity checks
├── train.py                    # LoRA fine-tuning (SFTTrainer)
├── export.py                   # Merge LoRA + SafeTensors export
└── README.md
```

---

## Who Uses This

| User | Use Case |
|------|----------|
| **Lawyers & Legal Teams** | Stress-test case arguments before trial; anticipate opposing counsel |
| **Policy Analysts** | Find weaknesses in proposed legislation |
| **Journalists** | Devil's advocate generation for balanced reporting |
| **Academics** | Peer review simulation; find gaps in paper arguments |
| **Debate Coaches** | High-quality opposition material for practice |
| **Developers** | Embed into writing tools, research platforms, decision software |

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

*Built with Rust + HuggingFace Candle. No Python at inference time.*