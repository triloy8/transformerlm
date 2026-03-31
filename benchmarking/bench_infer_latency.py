from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

import math

import torch
from safetensors.torch import load_file

from cli.utils import add_config_args, load_config_or_print
from config import load_bench_infer_config
from transformerlm.tokenizer.tokenizer import Tokenizer
from transformerlm.models import TransformerLM
from transformerlm.models.attention import set_sdp_backend
from transformerlm.utils.dtypes import DTYPES
from leash.inference.generate import diffusion_generate
from leash.objectives.loss import cross_entropy, diffusion_cross_entropy
from leash.training.optim import AdamW
from leash.training.grad import gradient_clipping
from leash.objectives.data import get_batch as diffusion_get_batch
from leash.data.streaming import HFTokenIteratorFactory, StreamingBatcher, RowBatcher
from leash.logger import ConsoleLogger
from profiling import nvtx

from .common import measure, mean, stddev


def _build_streaming_batcher(
    data_cfg,
    tokenizer: Tokenizer,
    device: str,
    *,
    shuffle_buffer_size: int,
    shuffle_seed: Optional[int],
) -> StreamingBatcher | RowBatcher:
    iterator = HFTokenIteratorFactory(
        dataset_name=str(data_cfg.dataset_name),
        dataset_config=(str(data_cfg.dataset_config) if data_cfg.dataset_config is not None else None),
        split=str(data_cfg.split),
        text_field=str(data_cfg.text_field),
        tokenizer=tokenizer,
        shuffle_buffer_size=max(0, shuffle_buffer_size),
        shuffle_seed=shuffle_seed,
        world_size=1,
        rank=0,
    )
    pipeline_mode = str(getattr(data_cfg, "pipeline_mode", "packed")).lower()
    if pipeline_mode == "rows":
        pad_token_id = getattr(data_cfg, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError("data.pad_token_id must be set when pipeline_mode='rows'")
        return RowBatcher(iterator, device=device, pad_token_id=int(pad_token_id))
    return StreamingBatcher(iterator, device=device)


def _sample_eval_batch_streaming(
    batcher: StreamingBatcher | RowBatcher,
    batch_size: int,
    context_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    drawn = batcher.draw(batch_size=batch_size, context_length=context_length + 1)
    attention_mask = None
    if isinstance(drawn, tuple):
        seqs, attention_mask = drawn
    else:
        seqs = drawn
    inputs = seqs[:, :-1]
    targets = seqs[:, 1:]
    input_mask = attention_mask[:, :-1] if attention_mask is not None else None
    target_mask = attention_mask[:, 1:] if attention_mask is not None else None
    return inputs, targets, input_mask, target_mask


def _parse_only_config():
    parser = argparse.ArgumentParser(description="Benchmark: inference latency via config.", allow_abbrev=False)
    add_config_args(parser, type_=str)
    return parser.parse_args()


def main():
    args_cfg = _parse_only_config()
    cfg = load_config_or_print(load_bench_infer_config, args_cfg.config, args_cfg.print_config)
    if cfg is None:
        return

    logger = ConsoleLogger()

    # Prepare tokenizer and inputs
    with nvtx.range("bench/setup/tokenizer"):
        tokenizer = Tokenizer.from_files(
            vocab_filepath=str(cfg.tokenizer.vocab_path),
            merges_filepath=str(cfg.tokenizer.merges_path),
            special_tokens_path=str(cfg.tokenizer.special_tokens_path),
        )
        prompts: List[str] = [cfg.inference.prompt]
        ids: List[List[int]] = [tokenizer.encode(prompt) for prompt in prompts]
    prompt_lens = [len(x) for x in ids]
    batch_size = len(ids)
    total_length = int(cfg.inference.total_length)
    if total_length > cfg.model.context_length:
        raise ValueError("inference.total_length must be <= model.context_length")
    gen_length = max(total_length - prompt_lens[0], 0)

    # Model
    with nvtx.range("bench/setup/model_load"):
        set_sdp_backend(cfg.model.attention_sdp_backend)
        model = TransformerLM(
            vocab_size=cfg.model.vocab_size,
            context_length=cfg.model.context_length,
            d_model=cfg.model.d_model,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            d_ff=cfg.model.d_ff,
            rope_theta=cfg.model.rope_theta,
            attention_backend=cfg.model.attention_backend,
            device=cfg.model.device,
            dtype=DTYPES[cfg.model.dtype],
        )
        model_state = load_file(str(cfg.checkpoint.ckpt_path))
        model.load_state_dict(model_state)
        # Snapshot initial model state on CPU for per-repeat resets
        initial_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    # Snapshot initial model state on CPU for per-repeat resets
    initial_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    in_indices = torch.tensor(ids, device=cfg.model.device)

    train_batcher: Optional[StreamingBatcher] = None
    if cfg.data is not None:
        with nvtx.range("bench/setup/data"):
            train_batcher = _build_streaming_batcher(
                cfg.data,
                tokenizer,
                cfg.model.device,
                shuffle_buffer_size=int(getattr(cfg.data, "shuffle_buffer_size", 0)),
                shuffle_seed=getattr(cfg.data, "shuffle_seed", None),
            )

    # Start run logging
    run_config = {
        "benchmark": "infer_latency",
        "device": cfg.model.device,
        "dtype": cfg.model.dtype,
        "batch_size": batch_size,
        "prompt_len_avg": float(mean([float(x) for x in prompt_lens])),
        "prompt_count": batch_size,
        "backward": bool(cfg.benchmark.backward),
        # model
        "vocab_size": cfg.model.vocab_size,
        "context_length": cfg.model.context_length,
        "d_model": cfg.model.d_model,
        "num_layers": cfg.model.num_layers,
        "num_heads": cfg.model.num_heads,
        "d_ff": cfg.model.d_ff,
        "rope_theta": cfg.model.rope_theta,
        # sampling
        "temperature": cfg.inference.temperature,
        "top_p": cfg.inference.top_p,
        "steps": cfg.inference.steps,
        "total_length": total_length,
        "gen_length": gen_length,
        "block_length": cfg.inference.block_length,
        "mask_id": cfg.inference.mask_id,
        "seed": cfg.inference.seed,
        "eos_token_id": cfg.inference.eos_token_id,
        "cfg_scale": cfg.inference.cfg_scale,
        "remasking": cfg.inference.remasking,
        "logits_eos_inf": cfg.inference.logits_eos_inf,
        "confidence_eos_eot_inf": cfg.inference.confidence_eos_eot_inf,
        # bench
        "warmup": cfg.benchmark.warmup,
        "repeats": cfg.benchmark.repeats,
        "steps": cfg.benchmark.steps,
        "synchronize": cfg.benchmark.synchronize,
    }
    if cfg.data is not None:
        run_config["data.dataset_name"] = cfg.data.dataset_name
        run_config["data.split"] = cfg.data.split
    # Optimizer (optional)
    optimizer = None
    if cfg.benchmark.optimizer_step:
        # If no optimizer config provided, use safe defaults per loader (lr=0.0, etc.)
        opt_cfg = cfg.optimizer
        if opt_cfg is None:
            raise ValueError("optimizer_step enabled but no optimizer config provided")
        with nvtx.range("bench/setup/optimizer"):
            optimizer = AdamW(
                model.parameters(),
                lr=opt_cfg.lr,
                betas=opt_cfg.betas,
                eps=opt_cfg.eps,
                weight_decay=opt_cfg.weight_decay,
            )

    # Annotate run with optimizer config if present
    if cfg.benchmark.optimizer_step and cfg.optimizer is not None:
        run_config.update(
            {
                "optimizer_step": True,
                "optimizer.lr": float(cfg.optimizer.lr),
                "optimizer.weight_decay": float(cfg.optimizer.weight_decay),
                "optimizer.betas": tuple(cfg.optimizer.betas),
                "optimizer.eps": float(cfg.optimizer.eps),
                "optimizer.grad_clip_max_l2_norm": float(cfg.optimizer.grad_clip_max_l2_norm),
            }
        )
    else:
        run_config.update({"optimizer_step": False})

    with nvtx.range("bench/setup/run_config"):
        info = logger.start_run(run_config)

    clip_enabled = bool(cfg.optimizer is not None and cfg.optimizer.grad_clip_max_l2_norm > 0.0)
    train_dataset = train_batcher if cfg.benchmark.backward else None
    if cfg.benchmark.backward and train_dataset is None:
        raise ValueError("benchmark.backward requires a [data] section with a streaming dataset")
    train_batch_size = int(getattr(cfg.benchmark, "train_batch_size", max(batch_size, 1)))
    train_batch_size = max(train_batch_size, 1)

    processed_tokens_last: int = 0
    mask_ratio_last: Optional[float] = None
    generator = None
    if cfg.inference.seed is not None:
        generator = torch.Generator(device=cfg.model.device)
        generator.manual_seed(int(cfg.inference.seed))

    def _run_workload():
        nonlocal processed_tokens_last, mask_ratio_last, optimizer
        processed_tokens = 0
        mask_ratio_accum = 0.0

        if not cfg.benchmark.backward:
            model.eval()
            with torch.no_grad():
                output = None
                for _ in range(cfg.benchmark.steps):
                    if nvtx.enabled("fine"):
                        with nvtx.range("bench/iter/generate"):
                            output = diffusion_generate(
                                model,
                                in_indices,
                                mask_id=int(cfg.inference.mask_id),
                                eos_token_id=cfg.inference.eos_token_id,
                                steps=int(cfg.inference.steps),
                                gen_length=int(gen_length),
                                block_length=int(cfg.inference.block_length),
                                temperature=float(cfg.inference.temperature),
                                top_p=(None if cfg.inference.top_p is None else float(cfg.inference.top_p)),
                                cfg_scale=float(cfg.inference.cfg_scale),
                                remasking=str(cfg.inference.remasking),
                                logits_eos_inf=bool(cfg.inference.logits_eos_inf),
                                confidence_eos_eot_inf=bool(cfg.inference.confidence_eos_eot_inf),
                                generator=generator,
                            )
                    else:
                        output = diffusion_generate(
                            model,
                            in_indices,
                            mask_id=int(cfg.inference.mask_id),
                            eos_token_id=cfg.inference.eos_token_id,
                            steps=int(cfg.inference.steps),
                            gen_length=int(gen_length),
                            block_length=int(cfg.inference.block_length),
                            temperature=float(cfg.inference.temperature),
                            top_p=(None if cfg.inference.top_p is None else float(cfg.inference.top_p)),
                            cfg_scale=float(cfg.inference.cfg_scale),
                            remasking=str(cfg.inference.remasking),
                            logits_eos_inf=bool(cfg.inference.logits_eos_inf),
                            confidence_eos_eot_inf=bool(cfg.inference.confidence_eos_eot_inf),
                            generator=generator,
                        )
            processed_tokens = int(gen_length) * batch_size * int(cfg.benchmark.steps)
            processed_tokens_last = processed_tokens
            mask_ratio_last = None
            return output

        if gen_length <= 0:
            processed_tokens_last = 0
            mask_ratio_last = None
            return None

        model.train()
        last_loss = None
        for _ in range(cfg.benchmark.steps):
            model.zero_grad(set_to_none=True)
            if nvtx.enabled("fine"):
                with nvtx.range("bench/iter/batch"):
                    batch = diffusion_get_batch(
                        train_dataset,
                        batch_size=train_batch_size,
                        context_length=cfg.model.context_length,
                        device=cfg.model.device,
                        mask_token_id=cfg.model.mask_token_id,
                        noise_epsilon=cfg.model.noise_epsilon,
                        random_trunc_prob=cfg.model.random_trunc_prob,
                    )
            else:
                batch = diffusion_get_batch(
                    train_dataset,
                    batch_size=train_batch_size,
                    context_length=cfg.model.context_length,
                    device=cfg.model.device,
                    mask_token_id=cfg.model.mask_token_id,
                    noise_epsilon=cfg.model.noise_epsilon,
                    random_trunc_prob=cfg.model.random_trunc_prob,
                )
            attn_mask = getattr(batch, "attention_mask", None)
            if nvtx.enabled("fine"):
                with nvtx.range("bench/iter/forward"):
                    if attn_mask is not None:
                        logits = model(batch.noisy_inputs, attention_mask=attn_mask)
                    else:
                        logits = model(batch.noisy_inputs)
                with nvtx.range("bench/iter/loss"):
                    loss = diffusion_cross_entropy(
                        logits,
                        batch.clean_targets,
                        batch.mask,
                        batch.p_mask,
                        loss_mask=getattr(batch, "loss_mask", None),
                    )
                with nvtx.range("bench/iter/backward"):
                    loss.backward()
            else:
                if attn_mask is not None:
                    logits = model(batch.noisy_inputs, attention_mask=attn_mask)
                else:
                    logits = model(batch.noisy_inputs)
                loss = diffusion_cross_entropy(
                    logits,
                    batch.clean_targets,
                    batch.mask,
                    batch.p_mask,
                    loss_mask=getattr(batch, "loss_mask", None),
                )
                loss.backward()
            last_loss = loss
            processed_tokens += int(batch.metadata.get("token_count", batch.clean_targets.numel()))
            mask_ratio_accum += float(batch.metadata.get("mask_ratio", 0.0))

            if cfg.benchmark.optimizer_step and optimizer is not None:
                if clip_enabled:
                    if nvtx.enabled("fine"):
                        with nvtx.range("bench/iter/clip"):
                            gradient_clipping(model.parameters(), cfg.optimizer.grad_clip_max_l2_norm)  # type: ignore[arg-type]
                    else:
                        gradient_clipping(model.parameters(), cfg.optimizer.grad_clip_max_l2_norm)  # type: ignore[arg-type]
                if nvtx.enabled("fine"):
                    with nvtx.range("bench/iter/opt_step"):
                        optimizer.step()
                else:
                    optimizer.step()

        processed_tokens_last = processed_tokens
        mask_ratio_last = (mask_ratio_accum / cfg.benchmark.steps) if cfg.benchmark.steps > 0 else None
        return last_loss

    with nvtx.range("bench/warmup"):
        for _ in range(cfg.benchmark.warmup):
            _ = _run_workload()

    # Timed repeats
    latencies_ms: List[float] = []
    tokens_per_sec: List[float] = []
    mask_ratios: List[float] = []

    for r in range(cfg.benchmark.repeats):
        # Always reset model (and optimizer state) before each timed repeat
        with nvtx.range(f"bench/repeat[{r}]/reset"):
            model.load_state_dict(initial_model_state, strict=True)
            model.zero_grad(set_to_none=True)
            if cfg.benchmark.optimizer_step and cfg.optimizer is not None:
                optimizer = AdamW(
                    model.parameters(),
                    lr=cfg.optimizer.lr,
                    betas=cfg.optimizer.betas,
                    eps=cfg.optimizer.eps,
                    weight_decay=cfg.optimizer.weight_decay,
                )

        processed_tokens_last = 0
        mask_ratio_last = None
        nvtx.mark("bench/measure_start")
        with nvtx.range(f"bench/repeat[{r}]/timed"):
            _, dt = measure(cfg.model.device, _run_workload, synchronize=cfg.benchmark.synchronize)
        nvtx.mark("bench/measure_end")
        lat_ms = dt * 1000.0
        tps = (float(processed_tokens_last) / dt) if dt > 0 else 0.0
        latencies_ms.append(lat_ms)
        tokens_per_sec.append(tps)

        with nvtx.range(f"bench/repeat[{r}]/log"):
            logger.log(
                {
                    "phase": "bench_infer",
                    "metrics.latency_ms": lat_ms,
                    "metrics.tokens_sec": tps,
                    "metrics.processed_tokens": int(processed_tokens_last),
                    "metrics.batch_size": int(batch_size),
                    "metrics.backward": bool(cfg.benchmark.backward),
                },
                step=r,
            )
            if mask_ratio_last is not None:
                mask_ratios.append(float(mask_ratio_last))
                logger.log(
                    {
                        "phase": "bench_infer",
                        "metrics.mask_ratio": float(mask_ratio_last),
                    },
                    step=r,
                )

    eval_summary: dict[str, float | int] = {}
    if cfg.data is not None:
        eval_batches_cfg = getattr(cfg.benchmark, "perplexity_max_batches", None)
        eval_batch_size_cfg = getattr(cfg.benchmark, "perplexity_batch_size", None)
        eval_seed = getattr(cfg.benchmark, "perplexity_seed", None)

        eval_batches = int(eval_batches_cfg) if eval_batches_cfg is not None else 32
        eval_batch_size = int(eval_batch_size_cfg) if eval_batch_size_cfg is not None else (batch_size or 1)
        eval_batch_size = max(eval_batch_size, 1)

        total_loss = 0.0
        total_tokens = 0
        actual_batches = 0

        eval_batcher = _build_streaming_batcher(
            cfg.data,
            tokenizer,
            cfg.model.device,
            shuffle_buffer_size=0,
            shuffle_seed=(int(eval_seed) if eval_seed is not None else None),
        )

        try:
            model.eval()
            with torch.no_grad():
                for _ in range(eval_batches):
                    seqs, tgts, input_mask, target_mask = _sample_eval_batch_streaming(
                        eval_batcher,
                        eval_batch_size,
                        cfg.model.context_length,
                    )
                    if input_mask is not None:
                        logits = model(seqs, attention_mask=input_mask)
                    else:
                        logits = model(seqs)
                    loss_per_position = cross_entropy(logits, tgts, reduction="none")
                    if target_mask is not None:
                        loss_per_position = loss_per_position * target_mask.to(loss_per_position.dtype)
                        batch_loss_sum = float(loss_per_position.sum().item())
                        batch_tokens = int(target_mask.sum().item())
                    else:
                        batch_loss_sum = float(loss_per_position.sum().item())
                        batch_tokens = int(loss_per_position.numel())
                    total_loss += batch_loss_sum
                    total_tokens += batch_tokens
                    actual_batches += 1
        except ValueError as exc:
            logger.log(
                {
                    "phase": "bench_infer",
                    "event": "perplexity_skip",
                    "reason": str(exc),
                }
            )

        if total_tokens > 0:
            mean_loss = total_loss / float(total_tokens)
            if not math.isfinite(mean_loss):
                ppl = float("inf")
            else:
                safe_loss = min(max(mean_loss, -80.0), 80.0)
                ppl = float(math.exp(safe_loss))
            eval_summary = {
                "metrics.perplexity": ppl,
                "metrics.perplexity.loss": mean_loss,
                "metrics.perplexity.tokens": int(total_tokens),
                "metrics.perplexity.batches": int(actual_batches),
                "metrics.perplexity.batch_size": int(eval_batch_size),
            }

    # Summary
    with nvtx.range("bench/summary"):
        summary_payload = {
            "phase": "bench_infer",
            "event": "summary",
            "metrics.latency_ms.mean": mean(latencies_ms),
            "metrics.tokens_sec.mean": mean(tokens_per_sec),
            "metrics.latency_ms.stddev": stddev(latencies_ms),
            "metrics.tokens_sec.stddev": stddev(tokens_per_sec),
            "metrics.iters": int(cfg.benchmark.repeats),
        }
        if mask_ratios:
            summary_payload["metrics.mask_ratio.mean"] = mean(mask_ratios)
            summary_payload["metrics.mask_ratio.stddev"] = stddev(mask_ratios)
        summary_payload.update(eval_summary)
        logger.log(summary_payload)

    logger.finish()


if __name__ == "__main__":
    main()
