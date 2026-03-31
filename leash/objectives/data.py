from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch
import random


@dataclass
class DiffusionBatch:
    noisy_inputs: torch.Tensor
    clean_targets: torch.Tensor
    mask: torch.Tensor
    p_mask: torch.Tensor
    attention_mask: torch.Tensor | None
    loss_mask: torch.Tensor | None
    metadata: Dict[str, Any]
    labels: torch.Tensor | None = None


@dataclass
class AutoregressiveBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    attention_mask: torch.Tensor | None
    loss_mask: torch.Tensor | None
    metadata: Dict[str, Any]


@dataclass
class JointBatch:
    diffusion: DiffusionBatch
    autoregressive: AutoregressiveBatch
    metadata: Dict[str, Any]


@dataclass
class FlowMatchingBatch:
    noisy_inputs: torch.Tensor
    clean_targets: torch.Tensor
    target_velocity: torch.Tensor
    timesteps: torch.Tensor
    attention_mask: torch.Tensor | None
    loss_mask: torch.Tensor | None
    metadata: Dict[str, Any]
    labels: torch.Tensor | None = None


@dataclass
class CategoricalFlowBatch:
    x0_prior: torch.Tensor
    clean_targets: torch.Tensor
    s_timesteps: torch.Tensor
    t_timesteps: torch.Tensor
    attention_mask: torch.Tensor | None
    loss_mask: torch.Tensor | None
    metadata: Dict[str, Any]
    labels: torch.Tensor | None = None


def _rand_uniform(shape, *, device: torch.device, generator: torch.Generator | None = None):
    if generator is not None:
        return torch.rand(shape, generator=generator, device=device)
    return torch.rand(shape, device=device)


def _rand_int(high: int, *, device: torch.device, generator: torch.Generator | None = None) -> int:
    if generator is not None:
        return int(torch.randint(1, high + 1, (1,), generator=generator, device=device).item())
    return random.randint(1, high)


def _draw_clean_targets(
    dataset,
    batch_size: int,
    context_length: int,
    device: str,
    *,
    random_trunc_prob: float = 0.01,
    min_length: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, bool, torch.Tensor | None]:
    device_obj = torch.device(device)
    rng_device = generator.device if generator is not None and hasattr(generator, "device") else torch.device("cpu")

    attention_mask = None
    labels = None
    drawn = dataset.draw(batch_size=batch_size, context_length=context_length)
    if hasattr(drawn, "tokens") and hasattr(drawn, "labels"):
        clean_targets = drawn.tokens
        labels = drawn.labels
    elif isinstance(drawn, tuple):
        clean_targets, attention_mask = drawn
    else:
        clean_targets = drawn
    clean_targets = clean_targets.to(device_obj, dtype=torch.long)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device_obj, dtype=torch.bool)
    if labels is not None:
        labels = labels.to(device_obj, dtype=torch.long)

    random_trunc_applied = False
    if random_trunc_prob > 0:
        trunc_flag = _rand_uniform((1,), device=rng_device, generator=generator)
        if bool((trunc_flag < random_trunc_prob).item()):
            max_len = clean_targets.shape[1]
            trunc_len = _rand_int(max_len, device=rng_device, generator=generator)
            if min_length is not None and max_len >= min_length and trunc_len < min_length:
                trunc_len = min_length
            clean_targets = clean_targets[:, :trunc_len]
            random_trunc_applied = True

    if attention_mask is not None and random_trunc_applied:
        attention_mask = attention_mask[:, : clean_targets.shape[1]]

    return clean_targets, attention_mask, random_trunc_applied, labels


def get_batch(
    dataset,
    batch_size: int,
    context_length: int,
    device: str,
    *,
    mask_token_id: int,
    noise_epsilon: float = 1e-3,
    random_trunc_prob: float = 0.01,
    p_mask_override: Optional[float] = None,
    deterministic_mask: bool = False,
    generator: torch.Generator | None = None,
) -> DiffusionBatch:
    clean_targets, attention_mask, random_trunc_applied, labels = _draw_clean_targets(
        dataset,
        batch_size,
        context_length,
        device,
        random_trunc_prob=random_trunc_prob,
        min_length=2,
        generator=generator,
    )
    device_obj = clean_targets.device

    batch_size, seq_len = clean_targets.shape

    if p_mask_override is not None:
        p_mask = torch.full((batch_size, 1), float(p_mask_override), device=device_obj)
    else:
        t = _rand_uniform((batch_size,), device=device_obj, generator=generator)
        p_mask = (1.0 - noise_epsilon) * t[:, None] + noise_epsilon

    if deterministic_mask:
        mask_len = (p_mask.view(-1) * seq_len).floor().to(torch.long)
        positions = torch.arange(seq_len, device=device_obj)
        mask = positions[None, :] < mask_len[:, None]
    else:
        mask_rand = _rand_uniform((batch_size, seq_len), device=device_obj, generator=generator)
        mask = mask_rand < p_mask
    if attention_mask is not None:
        mask = mask & attention_mask

    mask_token_tensor = torch.full_like(clean_targets, fill_value=mask_token_id)
    noisy_inputs = torch.where(mask, mask_token_tensor, clean_targets)

    loss_mask = attention_mask
    token_count = int(loss_mask.sum().item()) if loss_mask is not None else int(clean_targets.numel())
    metadata: Dict[str, Any] = {
        "random_truncation_applied": random_trunc_applied,
        "sequence_length": seq_len,
        "mask_ratio": float(mask.float().mean().detach().cpu().item()),
        "token_count": token_count,
        "p_mask_stats": {
            "mean": float(p_mask.mean().detach().cpu().item()),
            "min": float(p_mask.min().detach().cpu().item()),
            "max": float(p_mask.max().detach().cpu().item()),
            "std": float(p_mask.std(unbiased=False).detach().cpu().item()),
            "inv_mean": float((1.0 / p_mask).mean().detach().cpu().item()),
            "inv_max": float((1.0 / p_mask).max().detach().cpu().item()),
        },
    }

    return DiffusionBatch(
        noisy_inputs=noisy_inputs,
        clean_targets=clean_targets,
        mask=mask,
        p_mask=p_mask,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        labels=labels,
        metadata=metadata,
    )


def get_megadlm_diffusion_batch(
    dataset,
    batch_size: int,
    context_length: int,
    device: str,
    *,
    mask_token_id: int,
    eot_token_id: int | None = None,
    eot_mask_loss: bool = False,
    random_trunc_prob: float = 0.01,
    p_mask_override: Optional[float] = None,
    generator: torch.Generator | None = None,
) -> DiffusionBatch:
    clean_targets, attention_mask, random_trunc_applied, labels = _draw_clean_targets(
        dataset,
        batch_size,
        context_length,
        device,
        random_trunc_prob=random_trunc_prob,
        min_length=2,
        generator=generator,
    )
    device_obj = clean_targets.device
    batch_size, seq_len = clean_targets.shape

    if p_mask_override is not None:
        p_mask = torch.full((batch_size, 1), float(p_mask_override), device=device_obj)
    else:
        t = _rand_uniform((batch_size,), device=device_obj, generator=generator)
        p_mask = t[:, None]
    mask = _rand_uniform((batch_size, seq_len), device=device_obj, generator=generator) < p_mask
    if attention_mask is not None:
        mask = mask & attention_mask

    # Ensure at least one masked position per row.
    zero_masked = mask.sum(dim=1) == 0
    if bool(zero_masked.any().item()):
        if attention_mask is not None:
            valid_positions = attention_mask.to(torch.bool)
        else:
            valid_positions = torch.ones_like(mask, dtype=torch.bool)
        for row in zero_masked.nonzero(as_tuple=False).view(-1):
            valid_idx = valid_positions[row].nonzero(as_tuple=False).view(-1)
            if valid_idx.numel() == 0:
                continue
            if generator is not None:
                choice = torch.randint(
                    0,
                    valid_idx.numel(),
                    (1,),
                    device=valid_idx.device,
                    generator=generator,
                ).item()
            else:
                choice = int(torch.randint(0, valid_idx.numel(), (1,), device=valid_idx.device).item())
            mask[row, valid_idx[choice]] = True

    if attention_mask is not None:
        denom = attention_mask.to(torch.float32).sum(dim=1).clamp_min(1.0)
    else:
        denom = torch.full((batch_size,), float(seq_len), device=device_obj)
    p_mask = (mask.to(torch.float32).sum(dim=1) / denom).clamp_min(1.0 / max(seq_len, 1))
    p_mask = p_mask[:, None]

    mask_token_tensor = torch.full_like(clean_targets, fill_value=mask_token_id)
    noisy_inputs = torch.where(mask, mask_token_tensor, clean_targets)

    loss_mask = attention_mask
    if eot_mask_loss and eot_token_id is not None:
        eot_mask = clean_targets != int(eot_token_id)
        if loss_mask is None:
            loss_mask = eot_mask
        else:
            loss_mask = loss_mask & eot_mask
    token_count = int(loss_mask.sum().item()) if loss_mask is not None else int(clean_targets.numel())
    metadata: Dict[str, Any] = {
        "random_truncation_applied": random_trunc_applied,
        "sequence_length": seq_len,
        "mask_ratio": float(mask.float().mean().detach().cpu().item()),
        "token_count": token_count,
        "p_mask_stats": {
            "mean": float(p_mask.mean().detach().cpu().item()),
            "min": float(p_mask.min().detach().cpu().item()),
            "max": float(p_mask.max().detach().cpu().item()),
            "std": float(p_mask.std(unbiased=False).detach().cpu().item()),
            "inv_mean": float((1.0 / p_mask).mean().detach().cpu().item()),
            "inv_max": float((1.0 / p_mask).max().detach().cpu().item()),
        },
    }

    return DiffusionBatch(
        noisy_inputs=noisy_inputs,
        clean_targets=clean_targets,
        mask=mask,
        p_mask=p_mask,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        labels=labels,
        metadata=metadata,
    )


def get_autoregressive_batch(
    dataset,
    batch_size: int,
    context_length: int,
    device: str,
    *,
    random_trunc_prob: float = 0.01,
    generator: torch.Generator | None = None,
) -> AutoregressiveBatch:
    clean_targets, attention_mask, random_trunc_applied, _labels = _draw_clean_targets(
        dataset,
        batch_size,
        context_length,
        device,
        random_trunc_prob=random_trunc_prob,
        generator=generator,
    )
    if clean_targets.shape[1] < 2:
        raise ValueError("context_length must be >= 2 for autoregressive training")

    inputs = clean_targets[:, :-1]
    targets = clean_targets[:, 1:]
    loss_mask = attention_mask[:, 1:] if attention_mask is not None else None

    seq_len = inputs.shape[1]
    causal_mask = torch.ones((seq_len, seq_len), device=inputs.device, dtype=torch.bool).tril()
    if attention_mask is not None:
        key_mask = attention_mask[:, :-1]
        query_mask = attention_mask[:, :-1]
        causal_mask = causal_mask[None, :, :] & key_mask[:, None, :] & query_mask[:, :, None]
    else:
        causal_mask = causal_mask[None, :, :].expand(inputs.shape[0], -1, -1)

    token_count = int(loss_mask.sum().item()) if loss_mask is not None else int(targets.numel())
    metadata: Dict[str, Any] = {
        "random_truncation_applied": random_trunc_applied,
        "sequence_length": seq_len,
        "token_count": token_count,
    }

    return AutoregressiveBatch(
        inputs=inputs,
        targets=targets,
        attention_mask=causal_mask,
        loss_mask=loss_mask,
        metadata=metadata,
    )


def get_joint_batch(
    dataset,
    batch_size: int,
    context_length: int,
    device: str,
    *,
    mask_token_id: int,
    noise_epsilon: float = 1e-3,
    random_trunc_prob: float = 0.01,
    p_mask_override: Optional[float] = None,
    deterministic_mask: bool = False,
    generator: torch.Generator | None = None,
) -> JointBatch:
    clean_targets, attention_mask, random_trunc_applied, labels = _draw_clean_targets(
        dataset,
        batch_size,
        context_length,
        device,
        random_trunc_prob=random_trunc_prob,
        min_length=2,
        generator=generator,
    )
    device_obj = clean_targets.device
    batch_size, seq_len = clean_targets.shape

    if p_mask_override is not None:
        p_mask = torch.full((batch_size, 1), float(p_mask_override), device=device_obj)
    else:
        t = _rand_uniform((batch_size,), device=device_obj, generator=generator)
        p_mask = (1.0 - noise_epsilon) * t[:, None] + noise_epsilon

    if deterministic_mask:
        mask_len = (p_mask.view(-1) * seq_len).floor().to(torch.long)
        positions = torch.arange(seq_len, device=device_obj)
        mask = positions[None, :] < mask_len[:, None]
    else:
        mask_rand = _rand_uniform((batch_size, seq_len), device=device_obj, generator=generator)
        mask = mask_rand < p_mask
    if attention_mask is not None:
        mask = mask & attention_mask

    mask_token_tensor = torch.full_like(clean_targets, fill_value=mask_token_id)
    noisy_inputs = torch.where(mask, mask_token_tensor, clean_targets)
    diff_loss_mask = attention_mask
    diff_token_count = int(diff_loss_mask.sum().item()) if diff_loss_mask is not None else int(clean_targets.numel())
    diffusion_batch = DiffusionBatch(
        noisy_inputs=noisy_inputs,
        clean_targets=clean_targets,
        mask=mask,
        p_mask=p_mask,
        attention_mask=attention_mask,
        loss_mask=diff_loss_mask,
        labels=labels,
        metadata={
            "random_truncation_applied": random_trunc_applied,
            "sequence_length": seq_len,
            "mask_ratio": float(mask.float().mean().detach().cpu().item()),
            "token_count": diff_token_count,
            "p_mask_stats": {
                "mean": float(p_mask.mean().detach().cpu().item()),
                "min": float(p_mask.min().detach().cpu().item()),
                "max": float(p_mask.max().detach().cpu().item()),
                "std": float(p_mask.std(unbiased=False).detach().cpu().item()),
                "inv_mean": float((1.0 / p_mask).mean().detach().cpu().item()),
                "inv_max": float((1.0 / p_mask).max().detach().cpu().item()),
            },
        },
    )

    ar_inputs = clean_targets[:, :-1]
    ar_targets = clean_targets[:, 1:]
    ar_loss_mask = attention_mask[:, 1:] if attention_mask is not None else None
    ar_seq_len = ar_inputs.shape[1]
    ar_causal_mask = torch.ones((ar_seq_len, ar_seq_len), device=device_obj, dtype=torch.bool).tril()
    if attention_mask is not None:
        key_mask = attention_mask[:, :-1]
        query_mask = attention_mask[:, :-1]
        ar_causal_mask = ar_causal_mask[None, :, :] & key_mask[:, None, :] & query_mask[:, :, None]
    else:
        ar_causal_mask = ar_causal_mask[None, :, :].expand(ar_inputs.shape[0], -1, -1)
    ar_token_count = int(ar_loss_mask.sum().item()) if ar_loss_mask is not None else int(ar_targets.numel())
    autoregressive_batch = AutoregressiveBatch(
        inputs=ar_inputs,
        targets=ar_targets,
        attention_mask=ar_causal_mask,
        loss_mask=ar_loss_mask,
        metadata={
            "random_truncation_applied": random_trunc_applied,
            "sequence_length": ar_seq_len,
            "token_count": ar_token_count,
        },
    )

    metadata: Dict[str, Any] = {
        "random_truncation_applied": random_trunc_applied,
        "sequence_length": seq_len,
        "token_count": diff_token_count + ar_token_count,
        "token_count_diffusion": diff_token_count,
        "token_count_ar": ar_token_count,
        "mask_ratio": diffusion_batch.metadata["mask_ratio"],
        "p_mask_stats": diffusion_batch.metadata["p_mask_stats"],
    }

    return JointBatch(diffusion=diffusion_batch, autoregressive=autoregressive_batch, metadata=metadata)


def get_flow_matching_batch(
    dataset,
    batch_size: int,
    context_length: int,
    device: str,
    *,
    pixel_bins: int,
    random_trunc_prob: float = 0.0,
    generator: torch.Generator | None = None,
) -> FlowMatchingBatch:
    if pixel_bins <= 1:
        raise ValueError("pixel_bins must be > 1 for flow matching")
    clean_targets, attention_mask, random_trunc_applied, labels = _draw_clean_targets(
        dataset,
        batch_size,
        context_length,
        device,
        random_trunc_prob=random_trunc_prob,
        min_length=1,
        generator=generator,
    )
    x1 = clean_targets.to(torch.float32)
    x1 = 2.0 * (x1 / float(pixel_bins - 1)) - 1.0
    x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
    t = _rand_uniform((x1.shape[0], 1), device=x1.device, generator=generator)
    xt = (1.0 - t) * x0 + t * x1
    u = x1 - x0

    loss_mask = attention_mask
    token_count = int(loss_mask.sum().item()) if loss_mask is not None else int(x1.numel())
    metadata: Dict[str, Any] = {
        "random_truncation_applied": random_trunc_applied,
        "sequence_length": int(x1.shape[1]),
        "token_count": token_count,
        "t_stats": {
            "mean": float(t.mean().detach().cpu().item()),
            "min": float(t.min().detach().cpu().item()),
            "max": float(t.max().detach().cpu().item()),
        },
    }
    return FlowMatchingBatch(
        noisy_inputs=xt,
        clean_targets=x1,
        target_velocity=u,
        timesteps=t,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        labels=labels,
        metadata=metadata,
    )


def get_categorical_flow_batch(
    dataset,
    batch_size: int,
    context_length: int,
    device: str,
    *,
    vocab_size: int,
    prior_type: str = "discunif",
    random_trunc_prob: float = 0.0,
    generator: torch.Generator | None = None,
) -> CategoricalFlowBatch:
    if vocab_size <= 1:
        raise ValueError("vocab_size must be > 1 for categorical flow")
    prior_type = str(prior_type).lower()
    if prior_type not in {"discunif", "uniform"}:
        raise ValueError("prior_type must be one of: discunif, uniform")
    clean_targets, attention_mask, random_trunc_applied, labels = _draw_clean_targets(
        dataset,
        batch_size,
        context_length,
        device,
        random_trunc_prob=random_trunc_prob,
        min_length=1,
        generator=generator,
    )
    if clean_targets.min().item() < 0 or clean_targets.max().item() >= vocab_size:
        raise ValueError("clean_targets must be in [0, vocab_size)")

    if prior_type == "discunif":
        x0_tokens = torch.randint(
            low=0,
            high=vocab_size,
            size=clean_targets.shape,
            device=clean_targets.device,
            generator=generator,
        )
        x0 = torch.nn.functional.one_hot(x0_tokens, num_classes=vocab_size).to(torch.float32)
    else:
        x0 = torch.full(
            (*clean_targets.shape, vocab_size),
            fill_value=1.0 / float(vocab_size),
            device=clean_targets.device,
            dtype=torch.float32,
        )
    t = _rand_uniform((clean_targets.shape[0], 1), device=clean_targets.device, generator=generator)
    s = _rand_uniform((clean_targets.shape[0], 1), device=clean_targets.device, generator=generator) * t

    loss_mask = attention_mask
    token_count = int(loss_mask.sum().item()) if loss_mask is not None else int(clean_targets.numel())
    metadata: Dict[str, Any] = {
        "random_truncation_applied": random_trunc_applied,
        "sequence_length": int(clean_targets.shape[1]),
        "token_count": token_count,
        "s_stats": {
            "mean": float(s.mean().detach().cpu().item()),
            "min": float(s.min().detach().cpu().item()),
            "max": float(s.max().detach().cpu().item()),
        },
        "t_stats": {
            "mean": float(t.mean().detach().cpu().item()),
            "min": float(t.min().detach().cpu().item()),
            "max": float(t.max().detach().cpu().item()),
        },
    }
    return CategoricalFlowBatch(
        x0_prior=x0,
        clean_targets=clean_targets,
        s_timesteps=s,
        t_timesteps=t,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        labels=labels,
        metadata=metadata,
    )


__all__ = [
    "DiffusionBatch",
    "AutoregressiveBatch",
    "JointBatch",
    "FlowMatchingBatch",
    "CategoricalFlowBatch",
    "get_batch",
    "get_autoregressive_batch",
    "get_joint_batch",
    "get_flow_matching_batch",
    "get_categorical_flow_batch",
]
