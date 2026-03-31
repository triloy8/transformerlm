<div align="center">
<h1>🤖 Transformer Language Model</h1>
</div>

## 🚀 Runs: 
- [trixyL/transformerlm-ar-8k-simplestories](https://hf.co/trixyL/transformerlm-ar-8k-simplestories) — AR SimpleStories run (512 ctx, 8K vocab) 🤖📚 ([demo](https://huggingface.co/spaces/trixyL/simplestories-ar-demo))
- [trixyL/transformerlm-diff-32-mnist](https://hf.co/trixyL/transformerlm-diff-32-mnist) — MNIST image diffusion run (32-bin quantized pixels, 28x28) 🤖🌫️ ([demo](https://huggingface.co/spaces/trixyL/mnist-diff-demo)) 
- [trixyL/transformerlm-flow-mnist](https://hf.co/trixyL/transformerlm-flow-mnist) — MNIST image Flow-matching run (lerp, euler) 🤖🌊 ([demo](https://huggingface.co/spaces/trixyL/mnist-flow-demo)) 

## ✨ What Is This?

A from‑scratch Transformer LM repo that now acts as a thin experiment layer on top of `leash`.

The intended split is:
- `leash`: reusable runtime for training and basic inference across this repo and future sibling repos.
- `transformerlm`: concrete Transformer LM models, tokenizer implementation, config schemas, experiment wiring, and repo-specific CLIs.

See `leash/README.md` for training‑specific details.

## 🧭 Vision

This repo is moving toward a small ecosystem shape:

- `leash` owns the reusable generative runtime:
  - training loop
  - DDP/runtime concerns
  - checkpointing
  - logging
  - common objectives
  - batching/data utilities
  - basic generation utilities
- `transformerlm` owns the concrete LM implementation:
  - model architecture
  - tokenizer training and IO
  - config schemas and example TOMLs
  - experiment presets and benchmarking
  - repo-specific model loading and prediction adapters

The goal is for future experiment repos to look more like `transformerlm`: thin consumers of `leash`, not copies of training/runtime code.

## 🧩 Ownership

As a rule of thumb:

- If code depends on optimization, schedules, batches, distributed execution, checkpoints, logging, or basic token/image generation mechanics, it probably belongs in `leash`.
- If code depends on `TransformerLM` model classes, tokenizer assets, this repo's config tree, or experiment taxonomy, it belongs in `transformerlm`.
- If something is reusable across your own model repos, it is a good `leash` candidate even if it is not universally generic.

## 🧭 Overview

- From‑scratch model: decoder‑only Transformer LM (RMSNorm, SwiGLU, RoPE, SDPA/MHA), implemented directly with PyTorch modules.
- From‑scratch tokenizer: byte‑level BPE training and IO, producing `vocab.json` and `merges.txt`.
- Objectives: diffusion (bidirectional decoding) or autoregressive (causal), selected via config.
- Training and basic generation runtime live in `leash` (see `leash/README.md`).
- CLI + TOML configs: consistent entry points built around config schemas.
- Benchmarking + profiling: tokenizer/inference throughput checks and memory/runtime inspection.

## ▶️ Usage

This project expects [`uv`](https://github.com/astral-sh/uv) for running entry points and scripts.

Entry points live in `cli/` and are driven by TOML configs in `config/resources/`.

- Train the tokenizer (BPE):

```bash
uv run transformerlm-train-tokenizer --config config/resources/train_tokenizer.toml
```

- Start model training (DDP works for single GPU too):

```bash
uv run transformerlm-train --config config/resources/train.toml
```

> See `leash/README.md` for training/runtime behavior (DDP backend, logging, optimizer options, validation, accumulation).

- Generate text:

```bash
uv run transformerlm-infer --config config/resources/infer.toml
```

> `inference.total_length` should be the final sequence length (prompt + generated tokens) and must not exceed `model.context_length`; the generated span is computed automatically as `total_length - prompt_tokens`.

- Inspect effective configuration without running:

```bash
uv run transformerlm-train --config config/resources/train.toml --print-config
```
> Config loaders use Pydantic validation, so missing/extra keys or bad values raise `pydantic.ValidationError` with detailed paths.

## 📦 Datasets & Tokenizer Assets

To prepare your environment:

1. **Download tokenizer + (optional) Megatron bins** with `scripts/fetch_data.sh <output_dir>`. It pulls:
   - `vocab_simplestories_8k.json`, `merges_simplestories_8k.txt`, `special_tokens_simplestories_8k.json`
   - `simplestories_*_text_document.{bin,idx}` (for Megatron pipeline)
2. **Select a pipeline** via `[data].pipeline_mode`:
   - `"megatron"`: uses `megatron_train_prefix` / `megatron_val_prefix` (bin/idx).
   - `"packed"` or `"rows"`: streams from `datasets.load_dataset` using `dataset_name`, `train_split`, `val_split`, `text_field`.
3. **Offline caching (streaming)**: set `HF_DATASETS_CACHE=/path/to/cache`, run once with network access, then set `HF_DATASETS_OFFLINE=1`. Private datasets need `huggingface-cli login` or `HF_TOKEN`.

## 🛰️ Remote Orchestration

The `Justfile` plus helper scripts under `scripts/` provide a thin remote control plane focused on Prime Intellect hosts; set `PRIME_HOST`/`REMOTE_ROOT` to point at the target machine and path. All available recipes:
- `just bootstrap-remote` runs `scripts/bootstrap_remote.sh` over SSH to install uv/just/tmux, clone the repo and prepare `data/`, `runs/` and `env/` directories on the remote.
- `just data-remote` executes `scripts/fetch_data.sh` remotely to download the GPT‑2 tokenizer vocab/merges into the remote `data/` directory (streaming datasets are fetched via Hugging Face at runtime).
- `just build-remote` syncs dependencies by running `uv sync` (with a frozen lockfile fallback) inside the remote checkout.
- `just train config=<toml> extra="<args>"` launches `scripts/run_train_remote.sh` inside a tmux session (`transformerlm-train`) after validating Weights & Biases credentials.
- `just infer command="<cmd>" args="<extra>"` calls `scripts/run_infer_remote.sh` to execute arbitrary inference or benchmarking commands; defaults derive from `CMD_INFER`.
- `just nvitop` opens `nvitop` on the remote box for lightweight GPU monitoring (assumes it is installed by the bootstrap step).
- `just attach-train` attaches your terminal to the `transformerlm-train` tmux session, while `just kill-train` tears it down if you need to restart.
- `just fetch any_file=<path>` pulls any file or directory from `${REMOTE_ROOT}/<path>` into the current working directory; `just list-runs` prints the remote run directory names.
- `just sync-env` uploads your local `env/wandb.env` (copy `env/wandb.env.example` and populate `WANDB_API_KEY`) so the remote training session can authenticate with W&B.
- `just auto-train` runs the full bootstrap + data + env sync + training workflow (`bootstrap-remote`, `data-remote`, `sync-env`, `train`) in one shot.

## 🧩 Modules

- `transformerlm.models`: core Transformer layers + attention (`transformerlm/models/transformer.py`).
- `transformerlm.inference`: repo-specific prediction/model-loading adapters over `leash` generation (`transformerlm/inference/predictor.py`).
- `transformerlm.tokenizer`: BPE trainer + tokenizer IO (`transformerlm/tokenizer/tokenizer.py`).
- `leash`: shared runtime package for training, objectives, data, checkpointing, logging, and basic inference (`leash/`).
- `cli`: training/inference entry points (`cli/train.py`).
- `benchmarking`: quick perf checks (`benchmarking/bench_infer_latency.py`).
- `config`: schemas + example TOMLs (`config/resources/*.toml`).
- `profiling`: memory/runtime helpers (`profiling/nvtx.py`).
- `utils`: shared small helpers (`transformerlm/utils/dtypes.py`).

## 🧪 Benchmarking

- Benchmarks live under `benchmarking/` and use TOML configs in `config/resources/`.
- Run (latency): `uv run transformerlm-bench-infer --config config/resources/bench_infer.toml`
- Output: stdout only (latency, tokens/sec, diffusion stats).


## ✅ Tests

- Run tests: `uv run pytest`
- Markers:
  - `slow`: long‑running tests. Deselect with `-m "not slow"`.
  - `gpu`: requires CUDA/GPU. Deselect with `-m "not gpu"`.
- Examples:
  - Quick CPU suite: `uv run pytest -m "not slow and not gpu"`
  - Select a file/test: `uv run pytest tests/tokenizer/test_tokenizer.py -q`
  - Filter by name: `uv run pytest -k tokenizer`

## ⏱️ Profiling

- Tools:
  - NVTX ranges for GPU timeline annotation (via `profiling/nvtx.py`).
  - CUDA memory helpers for summaries and allocator history (via `profiling/memory.py`).

- NVTX usage:
  - Enable globally with env vars:
    - `TRANSFORMERLM_NVTX=1` to turn on annotations
    - `TRANSFORMERLM_NVTX_LEVEL=coarse|fine|verbose` to control detail
  - Example (collect with Nsight Systems):

```bash
TRANSFORMERLM_NVTX=1 TRANSFORMERLM_NVTX_LEVEL=fine \
  nsys profile -o result \
  uv run transformerlm-train --config config/resources/train.toml
```

- Memory helpers (Python):

```python
from profiling import memory

# Optionally record allocator history (CUDA only)
memory.start_history(True)

# ... run training or inference ...

# Snapshot/summary (safe no-ops on CPU)
ok = memory.dump_snapshot("alloc_history.json")
print(memory.summary())
print(memory.peaks())
memory.reset_peaks()
```
