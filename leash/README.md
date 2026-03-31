# Leash

Reusable training and basic inference infrastructure extracted from `transformerlm`. It is model‑agnostic and expects
callers to supply model/tokenizer/objective builders.

## What It Provides

- Training loop with gradient accumulation, AMP, and validation hooks.
- DDP wrapper, optimizer state sharding, and rank‑zero logging.
- Checkpointing with resume support and optional S3 upload hooks.
- Streaming data pipeline over Hugging Face `datasets`.
- Objective adapters for diffusion and autoregressive training.

## Repo Wiring

The CLI entry points live in `cli/` and pass builders from `transformerlm` into
`leash.trainer.train_ddp`:

- Model and tokenizer builders: `transformerlm/builders.py`
- Objective selection: `leash.objectives.build_objective`

This keeps `leash` reusable while `transformerlm` stays focused on modeling and tokenization.

## Runtime Notes

- CPU uses `gloo`; CUDA uses `nccl`. Set `[model].device` and `[ddp].backend` accordingly.
- Prefer `[logging].backend = "console"` for local runs.
- Optimizer state sharding is enabled by default in the training entry point to reduce per-rank optimizer memory.
- Choose the optimizer via `[optimizer].optimizer_name` (`"adamw"` default, `"muon"` experimental) while keeping the rest of the `[optimizer]` schedule knobs unchanged.
- When using Muon, configure per-group hyperparameters under `[optimizer.muon.*]` (hidden, head, embed, scalar) so each param group carries its own learning-rate range and optimizer settings; AdamW ignores those subtables.
- To fully skip validation (e.g., when no separate split exists), set `[training].skip_validation = true`; otherwise the loop will run `max_val_iteration` batches every `val_freq_iteration` steps.
- Gradient accumulation is controlled by `[training].grad_accum_steps`, `max_train_iteration` is still counted in micro-steps (each accumulation step), so scale it by `grad_accum_steps` if you want to keep the same number of optimizer steps.

## Quick Usage (in this repo)

```bash
uv run transformerlm-train --config config/resources/train.toml
```

## API Surface

- `leash.trainer.train_ddp`: main DDP entry point, takes builder callables.
- `leash.training.loop.train_loop`: core training loop (model, optimizer, objective).
- `leash.objectives`: diffusion/AR batching + losses and `build_objective`.
- `leash.data.streaming`: HF streaming iterator + batchers.
- `leash.checkpointing`: manifest‑based checkpoint saving/resume.
- `leash.logger`: console/W&B logging and rank‑zero wrappers.

## Logging

- Backends:
  - `console` (default): prints structured JSON lines with metrics like `metrics.loss`, `metrics.lr`, `metrics.grad_l2_norm`, plus optional activation/weight norms.
  - `wandb`: logs to Weights & Biases and uploads artifacts (checkpoints, tokenizer files, optional inference outputs).
- Configure in `config/resources/train.toml` under `[logging]`:

```toml
[logging]
backend = "console"   # or "wandb"
run_name = ""         # optional; defaults to timestamp
architecture = "TransformerLM"
dataset = "TinyStoriesV2-GPT4"

# Optional if using Wandb
[wandb]
entity = "your-entity"
project = "your-project"
```

- Training logs can optionally include activation/weight norms via `[logging]`:
  - Keys: `log_activation_norms`, `log_weight_norms`.
- Training logs include per-parameter-group learning rates (in addition to `metrics.lr`):
  - Keys: `metrics.lr/head`, `metrics.lr/embed`, `metrics.lr/scalar`, `metrics.lr/hidden`.
- Tip (console backend): Pipe to `jq` for readability:
  - `uv run transformerlm-train --config config/resources/train.toml | jq -r "."`

## DDP Policy

- Rank-zero logging: only rank 0 constructs a real logger and emits logs; other ranks are no-ops via `RankZeroLogger`.
- Aggregated metrics: scalar train/val metrics are all-reduced (mean) across ranks before logging.
- Synchronized run name: rank 0 generates the `run_name` and broadcasts it so all ranks agree on the checkpoint directory.
- Checkpoints: only rank 0 writes checkpoints and logs artifacts.
- Optimizer sharding: `OptimizerStateSharding` keeps optimizer state on the owning rank and re-broadcasts parameters after each step.

## Notes

- `leash` assumes configuration is validated by the caller (see `config/schemas.py`).
- For single‑process experiments, you can call `train_loop` directly with a concrete objective.
