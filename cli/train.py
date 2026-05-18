import argparse
from pathlib import Path

import torch.multiprocessing as mp

from config import asdict_pretty, load_train_config
from transformerlm.builders import build_model, build_tokenizer, build_activation_filter
from leash.objectives import build_objective
from leash.trainer import train_ddp
from cli.utils import add_config_args, load_config_or_print


def build_train_namespace(cfg_dc, config_path: str) -> argparse.Namespace:
    return argparse.Namespace(
        config_path=config_path,
        # optimizer
        optimizer_name=cfg_dc.optimizer.optimizer_name,
        betas=(cfg_dc.optimizer.betas[0], cfg_dc.optimizer.betas[1]),
        eps=cfg_dc.optimizer.eps,
        weight_decay=cfg_dc.optimizer.weight_decay,
        initial_learning_rate=cfg_dc.optimizer.initial_learning_rate,
        max_learning_rate=cfg_dc.optimizer.max_learning_rate,
        min_learning_rate=cfg_dc.optimizer.min_learning_rate,
        warmup_iters=cfg_dc.optimizer.warmup_iters,
        cosine_cycle_iters=cfg_dc.optimizer.cosine_cycle_iters,
        lr_decay_iters=cfg_dc.optimizer.lr_decay_iters,
        wsd_decay_iters=cfg_dc.optimizer.wsd_decay_iters,
        wsd_decay_style=cfg_dc.optimizer.wsd_decay_style,
        lr_schedule=cfg_dc.optimizer.lr_schedule,
        grad_clip_max_l2_norm=cfg_dc.optimizer.grad_clip_max_l2_norm,
        # model
        vocab_size=cfg_dc.model.vocab_size,
        pixel_bins=cfg_dc.model.pixel_bins,
        context_length=cfg_dc.model.context_length,
        d_model=cfg_dc.model.d_model,
        num_layers=cfg_dc.model.num_layers,
        num_heads=cfg_dc.model.num_heads,
        d_ff=cfg_dc.model.d_ff,
        rope_theta=cfg_dc.model.rope_theta,
        model_type=getattr(cfg_dc.model, "model_type", "lm"),
        label_vocab_size=getattr(cfg_dc.model, "label_vocab_size", None),
        null_label_id=getattr(cfg_dc.model, "null_label_id", None),
        attention_backend=cfg_dc.model.attention_backend,
        attention_sdp_backend=cfg_dc.model.attention_sdp_backend,
        mask_token_id=cfg_dc.model.mask_token_id,
        eot_token_id=cfg_dc.model.eot_token_id,
        noise_epsilon=cfg_dc.model.noise_epsilon,
        random_trunc_prob=cfg_dc.model.random_trunc_prob,
        use_rope_2d=getattr(cfg_dc.model, "use_rope_2d", False),
        image_height=getattr(cfg_dc.model, "image_height", None),
        image_width=getattr(cfg_dc.model, "image_width", None),
        log_activation_norms=bool(getattr(cfg_dc.logging, "log_activation_norms", False)) if cfg_dc.logging else False,
        log_weight_norms=bool(getattr(cfg_dc.logging, "log_weight_norms", False)) if cfg_dc.logging else False,
        log_grad_norms=bool(getattr(cfg_dc.logging, "log_grad_norms", False)) if cfg_dc.logging else False,
        log_p_mask_bucket_loss=bool(getattr(cfg_dc.logging, "log_p_mask_bucket_loss", False))
        if cfg_dc.logging
        else False,
        p_mask_bucket_edges=getattr(cfg_dc.logging, "p_mask_bucket_edges", None) if cfg_dc.logging else None,
        val_log_every=int(getattr(cfg_dc.logging, "val_log_every", 0)) if cfg_dc.logging else 0,
        val_log_samples=int(getattr(cfg_dc.logging, "val_log_samples", 0)) if cfg_dc.logging else 0,
        # global
        device=cfg_dc.model.device,
        dtype=cfg_dc.model.dtype,
        max_iteration=None,
        ckpting_save_iter=cfg_dc.checkpointing.ckpting_save_iter,
        batch_size=cfg_dc.training.batch_size,
        max_train_iteration=cfg_dc.training.max_train_iteration,
        max_val_iteration=cfg_dc.training.max_val_iteration,
        val_freq_iteration=cfg_dc.training.val_freq_iteration,
        skip_validation=cfg_dc.training.skip_validation,
        grad_accum_steps=cfg_dc.training.grad_accum_steps,
        train_loss_ema_decay=cfg_dc.training.train_loss_ema_decay,
        amp_enabled=bool(getattr(cfg_dc.training, "amp_enabled", False)),
        amp_dtype=str(getattr(cfg_dc.training, "amp_dtype", "float16")),
        training_objective=str(getattr(cfg_dc.training, "objective", "diffusion")),
        joint_diffusion_alpha=float(getattr(cfg_dc.training, "joint_diffusion_alpha", 0.3)),
        joint_diffusion_alpha_end=getattr(cfg_dc.training, "joint_diffusion_alpha_end", None),
        joint_alpha_schedule=str(getattr(cfg_dc.training, "joint_alpha_schedule", "constant")),
        joint_alpha_schedule_start=float(getattr(cfg_dc.training, "joint_alpha_schedule_start", 0.0)),
        joint_alpha_schedule_end=float(getattr(cfg_dc.training, "joint_alpha_schedule_end", 1.0)),
        p_mask_schedule=str(getattr(cfg_dc.training, "p_mask_schedule", "none")),
        p_mask_start=getattr(cfg_dc.training, "p_mask_start", None),
        p_mask_end=getattr(cfg_dc.training, "p_mask_end", None),
        p_mask_schedule_start=float(getattr(cfg_dc.training, "p_mask_schedule_start", 0.0)),
        p_mask_schedule_end=float(getattr(cfg_dc.training, "p_mask_schedule_end", 1.0)),
        eot_mask_loss=bool(getattr(cfg_dc.training, "eot_mask_loss", False)),
        repeat_masking_seed=getattr(cfg_dc.training, "repeat_masking_seed", None),
        p_mask_override=getattr(cfg_dc.training, "p_mask_override", None),
        deterministic_mask=bool(getattr(cfg_dc.training, "deterministic_mask", False)),
        uncond_label_dropout_prob=float(getattr(cfg_dc.training, "uncond_label_dropout_prob", 0.0)),
        # compile
        compile_enabled=bool(getattr(cfg_dc.compile, "enabled", False)) if cfg_dc.compile else False,
        compile_backend=getattr(cfg_dc.compile, "backend", "inductor") if cfg_dc.compile else "inductor",
        compile_mode=getattr(cfg_dc.compile, "mode", "default") if cfg_dc.compile else "default",
        compile_fullgraph=bool(getattr(cfg_dc.compile, "fullgraph", False)) if cfg_dc.compile else False,
        compile_dynamic=bool(getattr(cfg_dc.compile, "dynamic", False)) if cfg_dc.compile else False,
        compile_options=getattr(cfg_dc.compile, "options", None) if cfg_dc.compile else None,
        # data/paths
        runs_path=cfg_dc.data.runs_path,
        dataset_name=cfg_dc.data.dataset_name,
        dataset_config=cfg_dc.data.dataset_config,
        train_split=cfg_dc.data.train_split,
        val_split=cfg_dc.data.val_split,
        text_field=cfg_dc.data.text_field,
        tokenizer_vocab_path=(
            cfg_dc.data.tokenizer.vocab_path if cfg_dc.data.tokenizer is not None else None
        ),
        tokenizer_merges_path=(
            cfg_dc.data.tokenizer.merges_path if cfg_dc.data.tokenizer is not None else None
        ),
        tokenizer_special_tokens_path=(
            str(cfg_dc.data.tokenizer.special_tokens_path)
            if cfg_dc.data.tokenizer is not None
            else None
        ),
        pipeline_mode=cfg_dc.data.pipeline_mode,
        pad_token_id=cfg_dc.data.pad_token_id,
        pad_random_shift=bool(getattr(cfg_dc.data, "pad_random_shift", False)),
        shuffle_buffer_size=cfg_dc.data.shuffle_buffer_size,
        shuffle_seed=cfg_dc.data.shuffle_seed,
        cache_all=bool(getattr(cfg_dc.data, "cache_all", False)),
        megatron_train_prefix=(
            str(cfg_dc.data.megatron_train_prefix) if cfg_dc.data.megatron_train_prefix is not None else None
        ),
        megatron_val_prefix=(
            str(cfg_dc.data.megatron_val_prefix) if cfg_dc.data.megatron_val_prefix is not None else None
        ),
        rng_seed=cfg_dc.training.seed,
        muon_cfg=cfg_dc.optimizer.muon,
        # ddp
        backend=cfg_dc.ddp.backend,
        num_nodes=cfg_dc.ddp.num_nodes,
        num_gpus_per_node=cfg_dc.ddp.num_gpus_per_node,
        node_rank=cfg_dc.ddp.node_rank,
        master_addr=cfg_dc.ddp.master_addr,
        master_port=cfg_dc.ddp.master_port,
        bucket_size_mb=cfg_dc.ddp.bucket_size_mb,
        nccl_p2p_disable=cfg_dc.ddp.nccl_p2p_disable,
    )


def _parse_only_config():
    parser = argparse.ArgumentParser(description="Training via config file only.", allow_abbrev=False)
    add_config_args(parser, type_=Path)
    return parser.parse_args()


def main():
    # Config-only entry point
    args_cfg = _parse_only_config()
    cfg_dc = load_config_or_print(load_train_config, args_cfg.config, args_cfg.print_config)
    if cfg_dc is None:
        return
    if cfg_dc.ddp is None:
        raise ValueError("DDP config is required when using transformerlm-train")

    # Build an argparse-like namespace expected by existing code
    ns = build_train_namespace(cfg_dc, str(args_cfg.config))
    config_snapshot = asdict_pretty(cfg_dc)

    nprocs = max(1, int(cfg_dc.ddp.num_gpus_per_node))
    mp.spawn(
        train_ddp,
        args=(
            ns,
            cfg_dc,
            config_snapshot,
            build_model,
            build_tokenizer,
            build_objective,
            build_activation_filter(),
        ),
        nprocs=nprocs,
        join=True,
    )


if __name__ == "__main__":
    main()
