from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import os
import random
import time
import torch
from contextlib import nullcontext

from leash.checkpointing import CheckpointManager
from leash.data import (
    HFTokenIteratorFactory,
    StreamingBatcher,
    RowBatcher,
    TokenizerLike,
    MegatronPackedBatcher,
)
from leash.data.hf_image import build_mnist_batcher
from leash.ddp import DDP, OptimizerStateSharding
from leash.ddp.utils import broadcast_string, setup_process_group, cleanup_process_group, allreduce_mean
from leash.logger import Logger, ConsoleLogger, WandbLogger, RankZeroLogger
from leash.objectives import Objective
from leash.training.grad import gradient_clipping
from leash.training.loop import train_loop
from leash.training.optim import build_optimizer_param_groups, resolve_optimizer_cls
from leash.training.schedule import lr_cosine_schedule, lr_constant_schedule, lr_constant_with_warmup_schedule


def _seed_everything(seed: int, device: str | torch.device, *, rank: int = 0) -> torch.Generator:
    """Seed python, numpy, torch, and return a generator for reproducible sampling."""

    effective_seed = int(seed) + int(rank)

    if isinstance(device, torch.device):
        device_type = device.type
    else:
        device_type = str(device)

    generator_device = "cuda" if device_type == "cuda" and torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(effective_seed)

    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)

    return generator


def _prepare_optimizer_setup(cfg, model):
    optimizer_name = str(getattr(cfg, "optimizer_name", "adamw")).lower()
    setattr(cfg, "optimizer_name", optimizer_name)
    optimizer_cls = resolve_optimizer_cls(optimizer_name)
    muon_cfg = getattr(cfg, "muon_cfg", None)
    param_groups = build_optimizer_param_groups(model, optimizer_name, muon_cfg)
    kwargs = {
        "lr": float(cfg.initial_learning_rate),
        "weight_decay": float(cfg.weight_decay),
    }
    betas = getattr(cfg, "betas", (0.9, 0.95))
    beta_tuple = (float(betas[0]), float(betas[1]))
    eps = float(getattr(cfg, "eps", 1e-8))
    if optimizer_name == "adamw":
        kwargs["betas"] = beta_tuple
        kwargs["eps"] = eps
    elif optimizer_name == "muon":
        kwargs["momentum"] = float(getattr(cfg, "muon_momentum", 0.95))
        kwargs["betas"] = beta_tuple
        kwargs["eps"] = eps
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'")
    for group in param_groups:
        group.setdefault("initial_lr", float(cfg.initial_learning_rate))
        group.setdefault("max_lr", float(cfg.max_learning_rate))
        group.setdefault("min_lr", float(cfg.min_learning_rate))
        group.setdefault("warmup_iters", int(cfg.warmup_iters))
        group.setdefault("cosine_cycle_iters", int(cfg.cosine_cycle_iters))
        group.setdefault("lr", float(group.get("initial_lr")))
    return optimizer_cls, param_groups, kwargs


def train_ddp(
    local_rank: int,
    args,
    cfg_dc,
    config_snapshot: dict,
    model_builder: Callable[[object], torch.nn.Module],
    tokenizer_builder: Callable[[object], TokenizerLike],
    objective_builder: Callable[[object, TokenizerLike], Objective],
    activation_module_filter: Optional[Callable[[torch.nn.Module], bool]] = None,
) -> None:
    cfg = args

    num_nodes = int(getattr(cfg, "num_nodes", 1))
    num_gpus_per_node = int(getattr(cfg, "num_gpus_per_node", 1))
    local_rank_int = int(local_rank)
    node_rank = int(getattr(cfg, "node_rank", 0))
    master_addr = getattr(cfg, "master_addr", "localhost")
    master_port = getattr(cfg, "master_port", "29500")

    if getattr(cfg, "nccl_p2p_disable", None) is not None:
        os.environ["NCCL_P2P_DISABLE"] = "1" if cfg.nccl_p2p_disable else "0"

    world_size = max(1, num_nodes * num_gpus_per_node)
    global_rank = node_rank * num_gpus_per_node + local_rank_int
    setattr(cfg, "world_size", world_size)
    setattr(cfg, "global_rank", global_rank)

    setup_process_group(
        backend=cfg.backend,
        local_rank=local_rank_int,
        num_gpus_per_node=num_gpus_per_node,
        num_nodes=num_nodes,
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=master_port,
    )

    seed = getattr(cfg, "rng_seed", getattr(cfg, "seed", None))
    torch_generator = None
    if seed is not None:
        torch_generator = _seed_everything(int(seed), cfg.device, rank=global_rank)
    setattr(cfg, "torch_generator", torch_generator)

    logger, run_name, ckpting_save_folder = init_logging(global_rank, cfg, cfg_dc)
    config_path = Path(getattr(cfg, "config_path", ""))
    checkpoint_manager = CheckpointManager(
        checkpointing_cfg=getattr(cfg_dc, "checkpointing", None),
        runs_path=cfg.runs_path,
        run_name=run_name,
        rank=global_rank,
        world_size=cfg.world_size,
        config_path=config_path,
        config_snapshot=config_snapshot,
    )
    checkpoint_manager.prepare_run(torch_generator)
    ckpting_save_folder = checkpoint_manager.run_dir

    model = model_builder(cfg)

    if bool(getattr(cfg, "compile_enabled", False)):
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build")
        compile_kwargs = {
            "backend": getattr(cfg, "compile_backend", "inductor"),
            "mode": getattr(cfg, "compile_mode", "default"),
            "fullgraph": bool(getattr(cfg, "compile_fullgraph", False)),
            "dynamic": bool(getattr(cfg, "compile_dynamic", False)),
        }
        compile_options = getattr(cfg, "compile_options", None)
        if compile_options:
            compile_kwargs["options"] = compile_options
        model = torch.compile(model, **compile_kwargs)

    ddp_model = DDP(model, cfg.world_size, cfg.bucket_size_mb)

    optimizer_cls, param_groups, optimizer_kwargs = _prepare_optimizer_setup(cfg, model)
    optimizer = OptimizerStateSharding(
        param_groups,
        optimizer_cls,
        **optimizer_kwargs,
    )

    tokenizer = tokenizer_builder(args)
    objective = objective_builder(cfg, tokenizer)
    shuffle_seed = getattr(args, "shuffle_seed", None)
    if shuffle_seed is None:
        shuffle_seed = getattr(args, "rng_seed", getattr(cfg, "seed", None))
    val_log_every = int(getattr(cfg, "val_log_every", 0))
    val_log_samples = int(getattr(cfg, "val_log_samples", 0))
    per_rank_seed = (int(shuffle_seed) if shuffle_seed is not None else 0) + global_rank
    pipeline_mode = str(getattr(args, "pipeline_mode", "packed")).lower()
    eot_token_id = getattr(cfg, "eot_token_id", None)
    if pipeline_mode in {"packed", "rows"} and eot_token_id is None:
        raise ValueError("eot_token_id must be set for packed/rows datasets")
    pad_token_id = getattr(args, "pad_token_id", None)
    pad_random_shift = bool(getattr(args, "pad_random_shift", False))
    if pipeline_mode not in {"packed", "rows", "megatron", "mnist"}:
        raise ValueError("pipeline_mode must be one of: packed, rows, megatron, mnist")
    if pipeline_mode == "megatron":
        train_prefix = getattr(args, "megatron_train_prefix", None)
        val_prefix = getattr(args, "megatron_val_prefix", None)
        if train_prefix is None or val_prefix is None:
            raise ValueError("megatron_train_prefix and megatron_val_prefix must be set when pipeline_mode='megatron'")
        train_batcher = MegatronPackedBatcher(train_prefix, device=str(cfg.device), logger=logger)
        val_batcher = MegatronPackedBatcher(val_prefix, device=str(cfg.device), logger=logger)
    elif pipeline_mode == "rows":
        if pad_token_id is None:
            raise ValueError("pad_token_id must be set when pipeline_mode='rows'")
        train_iterator_factory = HFTokenIteratorFactory(
            dataset_name=str(args.dataset_name),
            dataset_config=(str(args.dataset_config) if args.dataset_config is not None else None),
            split=str(args.train_split),
            text_field=str(args.text_field),
            tokenizer=tokenizer,
            context_length=int(cfg.context_length),
            eot_token_id=int(eot_token_id),
            shuffle_buffer_size=int(getattr(args, "shuffle_buffer_size", 0)),
            shuffle_seed=per_rank_seed,
            cache_all=bool(getattr(args, "cache_all", False)),
            world_size=cfg.world_size,
            rank=global_rank,
            logger=logger,
            hf_debug_logging=True,
        )
        val_iterator_factory = HFTokenIteratorFactory(
            dataset_name=str(args.dataset_name),
            dataset_config=(str(args.dataset_config) if args.dataset_config is not None else None),
            split=str(args.val_split),
            text_field=str(args.text_field),
            tokenizer=tokenizer,
            context_length=int(cfg.context_length),
            eot_token_id=int(eot_token_id),
            shuffle_buffer_size=0,
            shuffle_seed=None,
            cache_all=bool(getattr(args, "cache_all", False)),
            world_size=cfg.world_size,
            rank=global_rank,
            logger=logger,
            hf_debug_logging=True,
        )
        train_batcher = RowBatcher(
            train_iterator_factory,
            device=str(cfg.device),
            pad_token_id=int(pad_token_id),
            pad_random_shift=pad_random_shift,
            logger=logger,
        )
        val_batcher = RowBatcher(
            val_iterator_factory,
            device=str(cfg.device),
            pad_token_id=int(pad_token_id),
            pad_random_shift=pad_random_shift,
            logger=logger,
        )
    elif pipeline_mode == "mnist":
        pixel_bins = int(getattr(cfg, "pixel_bins", 256))
        train_batcher = build_mnist_batcher(
            dataset_name=str(args.dataset_name),
            dataset_config=(str(args.dataset_config) if args.dataset_config is not None else None),
            split=str(args.train_split),
            device=str(cfg.device),
            pixel_bins=pixel_bins,
            shuffle=True,
            shuffle_seed=per_rank_seed,
            world_size=cfg.world_size,
            rank=global_rank,
        )
        val_batcher = build_mnist_batcher(
            dataset_name=str(args.dataset_name),
            dataset_config=(str(args.dataset_config) if args.dataset_config is not None else None),
            split=str(args.val_split),
            device=str(cfg.device),
            pixel_bins=pixel_bins,
            shuffle=False,
            shuffle_seed=per_rank_seed,
            world_size=cfg.world_size,
            rank=global_rank,
        )
    else:
        train_iterator_factory = HFTokenIteratorFactory(
            dataset_name=str(args.dataset_name),
            dataset_config=(str(args.dataset_config) if args.dataset_config is not None else None),
            split=str(args.train_split),
            text_field=str(args.text_field),
            tokenizer=tokenizer,
            context_length=int(cfg.context_length),
            eot_token_id=int(eot_token_id),
            shuffle_buffer_size=int(getattr(args, "shuffle_buffer_size", 0)),
            shuffle_seed=per_rank_seed,
            cache_all=bool(getattr(args, "cache_all", False)),
            world_size=cfg.world_size,
            rank=global_rank,
            logger=logger,
            hf_debug_logging=True,
        )
        val_iterator_factory = HFTokenIteratorFactory(
            dataset_name=str(args.dataset_name),
            dataset_config=(str(args.dataset_config) if args.dataset_config is not None else None),
            split=str(args.val_split),
            text_field=str(args.text_field),
            tokenizer=tokenizer,
            context_length=int(cfg.context_length),
            eot_token_id=int(eot_token_id),
            shuffle_buffer_size=0,
            shuffle_seed=None,
            cache_all=bool(getattr(args, "cache_all", False)),
            world_size=cfg.world_size,
            rank=global_rank,
            logger=logger,
            hf_debug_logging=True,
        )
        train_batcher = StreamingBatcher(train_iterator_factory, device=str(cfg.device), logger=logger)
        val_batcher = StreamingBatcher(val_iterator_factory, device=str(cfg.device), logger=logger)
    checkpoint_manager.attach_batchers(
        generator=torch_generator,
        train_batcher=train_batcher,
        val_batcher=val_batcher,
    )

    activation_norms = {}
    log_activation_norms_enabled = bool(getattr(cfg, "log_activation_norms", False))

    if log_activation_norms_enabled and activation_module_filter is not None:
        def get_activation_norm_hook(name):
            def hook(module, input, output):
                activation_norms[name] = output.norm().item()

            return hook

        for name, module in model.named_modules():
            if activation_module_filter(module):
                module.register_forward_hook(get_activation_norm_hook(name))

    amp_enabled = bool(getattr(cfg, "amp_enabled", False))
    amp_dtype = str(getattr(cfg, "amp_dtype", "float16")).lower()
    use_amp = amp_enabled and str(cfg.device).startswith("cuda") and torch.cuda.is_available()
    amp_torch_dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp and amp_torch_dtype == torch.float16)

    start_iteration = checkpoint_manager.maybe_resume(
        ddp_model=ddp_model,
        optimizer=optimizer,
        scaler=scaler,
        train_batcher=train_batcher,
        val_batcher=val_batcher,
        generator=torch_generator,
        device=str(cfg.device),
    )

    def _sync():
        ddp_model.finish_gradient_synchronization()

    checkpoint_callback = checkpoint_manager.make_checkpoint_callback()

    is_rank_zero = (global_rank == 0)
    train_infer_cfg = getattr(cfg_dc, "train_infer", None)
    infer_every = int(getattr(train_infer_cfg, "infer_every", 0)) if train_infer_cfg else 0
    prompts = [str(p) for p in getattr(train_infer_cfg, "prompts", [])] if train_infer_cfg else []
    infer_total_length = getattr(train_infer_cfg, "total_length", None) if train_infer_cfg else None
    if infer_total_length is None:
        infer_total_length = int(cfg.context_length)
    else:
        infer_total_length = int(infer_total_length)
    if infer_total_length > int(cfg.context_length):
        raise ValueError("train_infer.total_length must be <= model.context_length")

    def _step_callback(train_iteration, model, _optimizer, _train_loss, _lr):
        if infer_every <= 0 or not prompts or not is_rank_zero or logger is None:
            return
        if train_iteration % infer_every != 0:
            return
        was_training = model.training
        model.eval()
        rows = []
        generation_mode = str(getattr(train_infer_cfg, "generation_mode", "diffusion")) if train_infer_cfg else "diffusion"
        top_p = getattr(train_infer_cfg, "top_p", None) if train_infer_cfg else None
        cfg_scale = float(getattr(train_infer_cfg, "cfg_scale", 0.0)) if train_infer_cfg else 0.0
        remasking = str(getattr(train_infer_cfg, "remasking", "random")) if train_infer_cfg else "random"
        logits_eos_inf = bool(getattr(train_infer_cfg, "logits_eos_inf", False)) if train_infer_cfg else False
        confidence_eos_eot_inf = bool(
            getattr(train_infer_cfg, "confidence_eos_eot_inf", False)
        ) if train_infer_cfg else False
        steps = int(getattr(train_infer_cfg, "steps", 0)) if train_infer_cfg else 0
        block_length = int(getattr(train_infer_cfg, "block_length", 0)) if train_infer_cfg else 0
        temperature = float(getattr(train_infer_cfg, "temperature", 1.0)) if train_infer_cfg else 1.0
        base_seed = getattr(train_infer_cfg, "seed", None) if train_infer_cfg else None
        eos_token_id = getattr(cfg, "eot_token_id", None)
        mask_id = int(getattr(cfg, "mask_token_id", cfg.vocab_size - 1))

        autocast_ctx = torch.autocast("cuda", dtype=amp_torch_dtype) if use_amp else nullcontext()
        with torch.no_grad(), autocast_ctx:
            for idx, prompt in enumerate(prompts):
                try:
                    prompt_ids = objective.encode(prompt)
                except NotImplementedError:
                    return
                prompt_len = len(prompt_ids)
                if infer_total_length < prompt_len:
                    rows.append({"prompt": prompt, "output": "", "latency_ms": 0.0, "error": "prompt_too_long"})
                    continue
                gen_length = infer_total_length - prompt_len
                in_indices = torch.tensor([prompt_ids], device=str(cfg.device))
                generator = None
                if base_seed is not None:
                    generator = torch.Generator(device=str(cfg.device))
                    generator.manual_seed(int(base_seed) + int(train_iteration) + int(idx))
                t0 = time.perf_counter()
                if gen_length > 0:
                    try:
                        out_indices = objective.generate(
                            model,
                            in_indices,
                            mask_id=int(mask_id),
                            eos_token_id=(None if eos_token_id is None else int(eos_token_id)),
                            steps=int(steps),
                            gen_length=int(gen_length),
                            block_length=int(block_length),
                            temperature=float(temperature),
                            top_p=(None if top_p is None else float(top_p)),
                            cfg_scale=float(cfg_scale),
                            remasking=str(remasking),
                            logits_eos_inf=bool(logits_eos_inf),
                            confidence_eos_eot_inf=bool(confidence_eos_eot_inf),
                            generator=generator,
                            generation_mode=generation_mode,
                        )
                    except NotImplementedError:
                        return
                else:
                    out_indices = in_indices
                elapsed_ms = float((time.perf_counter() - t0) * 1000.0)
                try:
                    output = objective.decode(out_indices[0].tolist())
                except NotImplementedError:
                    return
                rows.append(
                    {
                        "prompt": prompt,
                        "output": output,
                        "latency_ms": elapsed_ms,
                        "error": "",
                    }
                )
        logger.log_table("train_infer/samples", rows, step=train_iteration)
        if was_training:
            model.train()

    lr_schedule_name = str(getattr(cfg, "lr_schedule", "cosine")).lower()
    if lr_schedule_name == "constant":
        lr_schedule = lr_constant_schedule
    elif lr_schedule_name == "constant_with_warmup":
        lr_schedule = lr_constant_with_warmup_schedule
    else:
        lr_schedule = lr_cosine_schedule

    val_sample_decode = None
    try:
        objective.decode([])
        val_sample_decode = objective.decode
    except NotImplementedError:
        val_sample_decode = None

    repeat_masking_seed = getattr(cfg, "repeat_masking_seed", None)
    if repeat_masking_seed is not None:
        repeat_masking_seed = int(repeat_masking_seed) + global_rank

    train_loop(
        ddp_model,
        optimizer,
        train_data=train_batcher,
        val_data=val_batcher,
        batch_size=cfg.batch_size,
        context_length=cfg.context_length,
        device=str(cfg.device),
        max_learning_rate=cfg.max_learning_rate,
        min_learning_rate=cfg.min_learning_rate,
        warmup_iters=cfg.warmup_iters,
        cosine_cycle_iters=cfg.cosine_cycle_iters,
        max_train_iteration=cfg.max_train_iteration,
        max_val_iteration=cfg.max_val_iteration,
        val_freq_iteration=cfg.val_freq_iteration,
        grad_accum_steps=int(getattr(cfg, "grad_accum_steps", 1)),
        amp_enabled=bool(getattr(cfg, "amp_enabled", False)),
        amp_dtype=str(getattr(cfg, "amp_dtype", "float16")),
        grad_clip_max_l2_norm=cfg.grad_clip_max_l2_norm,
        ckpting_save_iter=cfg.ckpting_save_iter,
        ckpting_save_folder=ckpting_save_folder,
        lr_cosine_schedule=lr_schedule,
        gradient_clipping=gradient_clipping,
        checkpoint_callback=checkpoint_callback,
        objective=objective,
        batch_generator=torch_generator,
        repeat_masking_seed=repeat_masking_seed,
        logger=logger,
        train_loss_ema_decay=float(getattr(cfg, "train_loss_ema_decay", 0.0)),
        scaler=scaler,
        activation_norms=activation_norms,
        log_activation_norms=bool(getattr(cfg, "log_activation_norms", False)),
        log_weight_norms=bool(getattr(cfg, "log_weight_norms", False)),
        log_p_mask_bucket_loss=bool(getattr(cfg, "log_p_mask_bucket_loss", False)),
        log_grad_norms=bool(getattr(cfg, "log_grad_norms", False)),
        val_log_every=val_log_every,
        val_log_samples=val_log_samples,
        val_sample_decode=val_sample_decode,
        sync_gradients=_sync,
        reduce_metric=allreduce_mean,
        is_rank_zero=is_rank_zero,
        skip_validation=bool(getattr(cfg, "skip_validation", False)),
        step_callback=_step_callback,
        start_iteration=start_iteration,
    )

    cleanup_process_group()


def build_run_config(cfg, cfg_dc):
    """Pure builder for run configuration payload.

    No rank checks or I/O.
    """
    run_config = {
        "vocab_size": cfg.vocab_size,
        "pixel_bins": getattr(cfg, "pixel_bins", None),
        "context_length": cfg.context_length,
        "d_model": cfg.d_model,
        "num_layers": cfg.num_layers,
        "num_heads": cfg.num_heads,
        "d_ff": cfg.d_ff,
        "rope_theta": cfg.rope_theta,
        "model_type": getattr(cfg, "model_type", "lm"),
        "label_vocab_size": getattr(cfg, "label_vocab_size", None),
        "null_label_id": getattr(cfg, "null_label_id", None),
        "attention_backend": cfg.attention_backend,
        "attention_sdp_backend": cfg.attention_sdp_backend,
        "mask_token_id": cfg.mask_token_id,
        "noise_epsilon": getattr(cfg, "noise_epsilon", None),
        "random_trunc_prob": getattr(cfg, "random_trunc_prob", None),
        "betas": cfg.betas,
        "eps": cfg.eps,
        "weight_decay": cfg.weight_decay,
        "initial_learning_rate": cfg.initial_learning_rate,
        "grad_clip_max_l2_norm": cfg.grad_clip_max_l2_norm,
        "max_learning_rate": cfg.max_learning_rate,
        "min_learning_rate": cfg.min_learning_rate,
        "warmup_iters": cfg.warmup_iters,
        "cosine_cycle_iters": cfg.cosine_cycle_iters,
        "lr_schedule": str(getattr(cfg, "lr_schedule", "cosine")),
        "max_train_iteration": cfg.max_train_iteration,
        "max_val_iteration": cfg.max_val_iteration,
        "val_freq_iteration": cfg.val_freq_iteration,
        "batch_size": cfg.batch_size,
        "device": cfg.device,
        "dtype": cfg.dtype,
        "ckpting_save_iter": cfg.ckpting_save_iter,
        "grad_accum_steps": int(getattr(cfg, "grad_accum_steps", 1)),
        "amp_enabled": bool(getattr(cfg, "amp_enabled", False)),
        "amp_dtype": str(getattr(cfg, "amp_dtype", "float16")),
        "training_objective": str(getattr(cfg, "training_objective", "diffusion")),
        "uncond_label_dropout_prob": float(getattr(cfg, "uncond_label_dropout_prob", 0.0)),
    }

    seed_value = getattr(cfg, "rng_seed", getattr(cfg, "seed", None))
    if seed_value is not None:
        run_config["rng_seed"] = seed_value

    # Optional metadata from config
    if getattr(cfg_dc, "wandb", None):
        if getattr(cfg_dc.wandb, "architecture", None):
            run_config["architecture"] = cfg_dc.wandb.architecture
        if getattr(cfg_dc.wandb, "dataset", None):
            run_config["dataset"] = cfg_dc.wandb.dataset

    if getattr(cfg_dc, "logging", None):
        if getattr(cfg_dc.logging, "architecture", None):
            run_config["architecture"] = cfg_dc.logging.architecture
        if getattr(cfg_dc.logging, "dataset", None):
            run_config["dataset"] = cfg_dc.logging.dataset

    return run_config


def init_logging(rank: int, cfg, cfg_dc):
    """Initialize logging on rank 0 and broadcast run_name.

    Returns (logger, run_name, ckpt_dir). Only rank 0 will create directories
    and own a real logger; other ranks get a RankZeroLogger(NoOp).
    """
    run_config = build_run_config(cfg, cfg_dc)

    real_logger: Logger | None = None
    run_name: str | None = None

    if rank == 0:
        # Select logger backend (prefer [logging], default to console)
        real_logger = ConsoleLogger()
        if getattr(cfg_dc, "logging", None) and cfg_dc.logging and cfg_dc.logging.backend:
            backend = (cfg_dc.logging.backend or "console").lower()
            if backend == "console":
                real_logger = ConsoleLogger()
            elif backend == "wandb":
                entity = getattr(cfg_dc.wandb, "entity", None) if getattr(cfg_dc, "wandb", None) else None
                project = getattr(cfg_dc.wandb, "project", None) if getattr(cfg_dc, "wandb", None) else None
                default_name = (
                    getattr(cfg_dc.logging, "run_name", None)
                    or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                )
                real_logger = WandbLogger(entity=entity, project=project, name=default_name)

        info = real_logger.start_run(run_config)
        run_name = info.get("run_name") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ensure all ranks agree on run_name
    run_name = broadcast_string(run_name, src=0)

    # proxy so only rank 0 logs
    rz_logger: Logger = RankZeroLogger(rank, real_logger)

    # compute ckpt dir, create only on rank 0
    ckpt_dir = cfg.runs_path / run_name
    if rank == 0 and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    return rz_logger, run_name, ckpt_dir
