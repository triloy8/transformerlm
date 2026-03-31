from __future__ import annotations

import math

import torch
from leash.inference.sampling import add_gumbel_noise, compute_transfer_schedule, softmax, top_p_filter


@torch.no_grad()
def diffusion_generate(
    model,
    prompt_indices: torch.Tensor,
    *,
    mask_id: int,
    eos_token_id: int | None = None,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float = 0.0,
    top_p: float | None = None,
    cfg_scale: float = 0.0,
    remasking: str = "random",
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    context: torch.Tensor | None = None,
    guide_model=None,
    guidance_mode: str = "none",
    guide_fallback_strategy: str = "single_token",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate sequences via the diffusion reverse process."""

    if prompt_indices.dim() != 2:
        raise ValueError("prompt_indices must be 2D (batch, seq)")
    if block_length <= 0:
        raise ValueError("block_length must be > 0")
    if steps <= 0:
        raise ValueError("steps must be > 0")
    guidance_mode = str(guidance_mode).lower()
    if guidance_mode not in {"none", "ar_agreement"}:
        raise ValueError("guidance_mode must be one of: none, ar_agreement")
    if guidance_mode == "ar_agreement" and guide_model is None:
        raise ValueError("guide_model must be set when guidance_mode='ar_agreement'")
    guide_fallback_strategy = str(guide_fallback_strategy).lower()
    if guide_fallback_strategy not in {"single_token"}:
        raise ValueError("guide_fallback_strategy must be one of: single_token")

    if gen_length <= 0:
        return prompt_indices

    blocks = max(1, math.ceil(gen_length / block_length))
    if steps < blocks:
        raise ValueError("steps must be >= number of blocks")
    base_steps = steps // blocks
    extra_steps = steps % blocks

    device = prompt_indices.device
    batch_size, prompt_len = prompt_indices.shape
    total_len = prompt_len + gen_length

    context_limit = getattr(model, "context_length", None)
    if context_limit is not None and total_len > int(context_limit):
        raise ValueError("prompt length + gen_length exceeds model context_length")

    x = torch.full(
        (batch_size, total_len),
        fill_value=mask_id,
        device=device,
        dtype=prompt_indices.dtype,
    )
    x[:, :prompt_len] = prompt_indices
    prompt_index = (x != mask_id)

    try:
        from tqdm import tqdm
    except Exception as exc:  # pragma: no cover - import-only path
        raise RuntimeError("tqdm is required for diffusion_generate") from exc
    pbar = tqdm(total=steps, desc="diffusion_generate", leave=False)

    for block_idx in range(blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = min(block_start + block_length, total_len)
        block_steps = base_steps + (1 if block_idx < extra_steps else 0)
        if block_steps <= 0:
            block_steps = 1
        block_mask = (x[:, block_start:block_end] == mask_id)
        transfer_counts = compute_transfer_schedule(block_mask, block_steps)

        for step_idx in range(block_steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                if context is not None:
                    ctx = torch.cat([context, context], dim=0)
                    logits = model(torch.cat([x, un_x], dim=0), context=ctx)
                else:
                    logits = model(torch.cat([x, un_x], dim=0))
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1.0) * (logits - un_logits)
            else:
                if context is not None:
                    logits = model(x, context=context)
                else:
                    logits = model(x)

            if logits_eos_inf and eos_token_id is not None:
                logits[:, :, eos_token_id] = float("-inf")

            if top_p is not None:
                probs = softmax(logits, dim=-1)
                probs = top_p_filter(probs, float(top_p))
                logits = torch.where(
                    probs > 0,
                    logits,
                    torch.full_like(logits, float("-inf")),
                )

            logits_with_noise = add_gumbel_noise(logits, temperature, generator=generator)
            predictions = torch.argmax(logits_with_noise, dim=-1)
            predictions = torch.where(mask_index, predictions, x)
            agreement_mask = None
            if guidance_mode == "ar_agreement":
                if context is not None:
                    guide_logits = guide_model(x, context=context)
                else:
                    guide_logits = guide_model(x)
                guide_predictions = torch.argmax(guide_logits, dim=-1)
                agreement_mask = torch.logical_and(mask_index, predictions == guide_predictions)

            if remasking == "low_confidence":
                probs = softmax(logits, dim=-1)
                confidence = torch.squeeze(
                    torch.gather(probs, dim=-1, index=torch.unsqueeze(predictions, -1)),
                    -1,
                )
            elif remasking == "random":
                confidence = torch.rand(
                    (batch_size, total_len),
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )
            else:
                raise ValueError(f"Unsupported remasking strategy: {remasking}")

            if confidence_eos_eot_inf and eos_token_id is not None:
                confidence = torch.where(
                    predictions == eos_token_id,
                    torch.full_like(confidence, float("-inf")),
                    confidence,
                )

            confidence[:, block_end:] = float("-inf")
            confidence = torch.where(mask_index, confidence, torch.full_like(confidence, float("-inf")))
            guided_confidence = confidence
            if agreement_mask is not None:
                guided_confidence = torch.where(
                    agreement_mask,
                    confidence,
                    torch.full_like(confidence, float("-inf")),
                )

            transfer_mask = torch.zeros_like(mask_index)
            for b in range(batch_size):
                k = int(transfer_counts[b, step_idx].item())
                if k <= 0:
                    continue
                selected = False
                active_confidence = guided_confidence[b] if agreement_mask is not None else confidence[b]
                available = active_confidence > float("-inf")
                available_count = int(available.sum().item())
                if available_count > 0:
                    if available_count < k:
                        k = available_count
                    topk_indices = torch.topk(active_confidence, k=k, dim=-1).indices
                    transfer_mask[b, topk_indices] = True
                    selected = True
                if (
                    not selected
                    and agreement_mask is not None
                    and guide_fallback_strategy == "single_token"
                ):
                    fallback_available = confidence[b] > float("-inf")
                    if int(fallback_available.sum().item()) > 0:
                        fallback_idx = torch.topk(confidence[b], k=1, dim=-1).indices
                        transfer_mask[b, fallback_idx] = True

            x = torch.where(transfer_mask, predictions, x)
            pbar.update(1)

    pbar.close()
    return x


@torch.no_grad()
def image_diffusion_generate(
    model,
    prompt_indices: torch.Tensor,
    *,
    context: torch.Tensor,
    mask_id: int,
    eos_token_id: int | None = None,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float = 0.0,
    top_p: float | None = None,
    cfg_scale: float = 0.0,
    uncond_context: torch.Tensor | None = None,
    remasking: str = "random",
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Diffusion generation specialized for image models with label context CFG."""

    if prompt_indices.dim() != 2:
        raise ValueError("prompt_indices must be 2D (batch, seq)")
    if context.dim() != 1:
        raise ValueError("context must be 1D (batch,)")
    if prompt_indices.shape[0] != context.shape[0]:
        raise ValueError("context batch size must match prompt batch size")
    if block_length <= 0:
        raise ValueError("block_length must be > 0")
    if steps <= 0:
        raise ValueError("steps must be > 0")

    if gen_length <= 0:
        return prompt_indices

    blocks = max(1, math.ceil(gen_length / block_length))
    if steps < blocks:
        raise ValueError("steps must be >= number of blocks")
    base_steps = steps // blocks
    extra_steps = steps % blocks

    device = prompt_indices.device
    batch_size, prompt_len = prompt_indices.shape
    total_len = prompt_len + gen_length

    context_limit = getattr(model, "context_length", None)
    if context_limit is not None and total_len > int(context_limit):
        raise ValueError("prompt length + gen_length exceeds model context_length")

    x = torch.full(
        (batch_size, total_len),
        fill_value=mask_id,
        device=device,
        dtype=prompt_indices.dtype,
    )
    x[:, :prompt_len] = prompt_indices

    if uncond_context is not None:
        if uncond_context.dim() != 1:
            raise ValueError("uncond_context must be 1D (batch,)")
        if uncond_context.shape[0] != batch_size:
            raise ValueError("uncond_context batch size must match prompt batch size")
        uncond_context = uncond_context.to(device=context.device, dtype=context.dtype)

    try:
        from tqdm import tqdm
    except Exception as exc:  # pragma: no cover - import-only path
        raise RuntimeError("tqdm is required for image_diffusion_generate") from exc
    pbar = tqdm(total=steps, desc="image_diffusion_generate", leave=False)

    for block_idx in range(blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = min(block_start + block_length, total_len)
        block_steps = base_steps + (1 if block_idx < extra_steps else 0)
        if block_steps <= 0:
            block_steps = 1
        block_mask = (x[:, block_start:block_end] == mask_id)
        transfer_counts = compute_transfer_schedule(block_mask, block_steps)

        for step_idx in range(block_steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.0:
                if uncond_context is None:
                    raise ValueError("uncond_context must be set when cfg_scale > 0 for image_diffusion_generate")
                cond_logits = model(x, context=context)
                uncond_logits = model(x, context=uncond_context)
                logits = uncond_logits + (cfg_scale + 1.0) * (cond_logits - uncond_logits)
            else:
                logits = model(x, context=context)

            if logits_eos_inf and eos_token_id is not None:
                logits[:, :, eos_token_id] = float("-inf")

            if top_p is not None:
                probs = softmax(logits, dim=-1)
                probs = top_p_filter(probs, float(top_p))
                logits = torch.where(
                    probs > 0,
                    logits,
                    torch.full_like(logits, float("-inf")),
                )

            logits_with_noise = add_gumbel_noise(logits, temperature, generator=generator)
            predictions = torch.argmax(logits_with_noise, dim=-1)
            predictions = torch.where(mask_index, predictions, x)

            if remasking == "low_confidence":
                probs = softmax(logits, dim=-1)
                confidence = torch.squeeze(
                    torch.gather(probs, dim=-1, index=torch.unsqueeze(predictions, -1)),
                    -1,
                )
            elif remasking == "random":
                confidence = torch.rand(
                    (batch_size, total_len),
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )
            else:
                raise ValueError(f"Unsupported remasking strategy: {remasking}")

            if confidence_eos_eot_inf and eos_token_id is not None:
                confidence = torch.where(
                    predictions == eos_token_id,
                    torch.full_like(confidence, float("-inf")),
                    confidence,
                )

            confidence[:, block_end:] = float("-inf")
            confidence = torch.where(mask_index, confidence, torch.full_like(confidence, float("-inf")))

            transfer_mask = torch.zeros_like(mask_index)
            for b in range(batch_size):
                k = int(transfer_counts[b, step_idx].item())
                if k <= 0:
                    continue
                available = confidence[b] > float("-inf")
                available_count = int(available.sum().item())
                if available_count == 0:
                    continue
                if available_count < k:
                    k = available_count
                topk_indices = torch.topk(confidence[b], k=k, dim=-1).indices
                transfer_mask[b, topk_indices] = True

            x = torch.where(transfer_mask, predictions, x)
            pbar.update(1)

    pbar.close()
    return x


@torch.no_grad()
def autoregressive_generate(
    model,
    prompt_indices: torch.Tensor,
    *,
    gen_length: int,
    temperature: float = 0.0,
    top_p: float | None = None,
    eos_token_id: int | None = None,
    logits_eos_inf: bool = False,
    context: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate sequences token-by-token with a causal attention mask."""

    if prompt_indices.dim() != 2:
        raise ValueError("prompt_indices must be 2D (batch, seq)")
    if gen_length <= 0:
        return prompt_indices

    try:
        from tqdm import tqdm
    except Exception as exc:  # pragma: no cover - import-only path
        raise RuntimeError("tqdm is required for autoregressive_generate") from exc

    x = prompt_indices
    pbar = tqdm(total=gen_length, desc="autoregressive_generate", leave=False)
    for _ in range(gen_length):
        seq_len = x.shape[1]
        causal = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))
        attention_mask = causal.unsqueeze(0).expand(x.shape[0], -1, -1)

        if context is not None:
            logits = model(x, attention_mask=attention_mask, context=context)
        else:
            logits = model(x, attention_mask=attention_mask)
        next_logits = logits[:, -1, :]

        if logits_eos_inf and eos_token_id is not None:
            next_logits[:, eos_token_id] = float("-inf")

        if temperature <= 0:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            scaled_logits = next_logits / float(temperature)
            probs = softmax(scaled_logits, dim=-1)
            if top_p is not None:
                probs = top_p_filter(probs, float(top_p))
            next_token = torch.multinomial(probs, 1)
        x = torch.cat([x, next_token], dim=1)
        pbar.update(1)

    pbar.close()
    return x


@torch.no_grad()
def categorical_flow_image_generate(
    model,
    prompt_indices: torch.Tensor,
    *,
    context: torch.Tensor,
    steps: int,
    temperature: float = 0.0,
    top_p: float | None = None,
    cfg_scale: float = 0.0,
    uncond_context: torch.Tensor | None = None,
    prior_type: str = "discunif",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if prompt_indices.dim() != 2:
        raise ValueError("prompt_indices must be 2D (batch, seq)")
    if context.dim() != 1:
        raise ValueError("context must be 1D (batch,)")
    if prompt_indices.shape[0] != context.shape[0]:
        raise ValueError("context batch size must match prompt batch size")
    if prompt_indices.shape[1] != 0:
        raise ValueError("categorical_flow_image_generate expects empty prompt_indices for full-image generation")
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    prior_type = str(prior_type).lower()
    if prior_type not in {"discunif", "uniform"}:
        raise ValueError("prior_type must be one of: discunif, uniform")

    batch_size = context.shape[0]
    gen_length = int(getattr(model, "context_length"))
    vocab_size = int(getattr(model, "vocab_size"))
    if prior_type == "discunif":
        x_tokens = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, gen_length),
            device=prompt_indices.device,
            generator=generator,
        )
        x = torch.nn.functional.one_hot(x_tokens, num_classes=vocab_size).to(torch.float32)
    else:
        x = torch.full(
            (batch_size, gen_length, vocab_size),
            fill_value=1.0 / float(vocab_size),
            device=prompt_indices.device,
            dtype=torch.float32,
        )

    if uncond_context is not None:
        if uncond_context.dim() != 1:
            raise ValueError("uncond_context must be 1D (batch,)")
        if uncond_context.shape[0] != batch_size:
            raise ValueError("uncond_context batch size must match prompt batch size")
        uncond_context = uncond_context.to(device=context.device, dtype=context.dtype)

    for step_idx in range(steps):
        s_val = float(step_idx) / float(steps)
        t_val = float(step_idx + 1) / float(steps)
        s = torch.full((batch_size,), s_val, device=x.device, dtype=x.dtype)
        t = torch.full((batch_size,), t_val, device=x.device, dtype=x.dtype)
        if cfg_scale > 0.0:
            if uncond_context is None:
                raise ValueError("uncond_context must be set when cfg_scale > 0 for categorical_flow_image_generate")
            logits_cond = model(x, s, t, context=context)
            logits_uncond = model(x, s, t, context=uncond_context)
            logits = logits_uncond + (cfg_scale + 1.0) * (logits_cond - logits_uncond)
        else:
            logits = model(x, s, t, context=context)
        probs = softmax(logits, dim=-1)
        gamma = (t_val - s_val) / max(1.0 - s_val, 1e-6)
        x = x + gamma * (probs - x)
        x = x.clamp_min(0.0)
        x = x / x.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    logits = torch.log(x.clamp_min(1e-6))
    if top_p is not None:
        probs = top_p_filter(softmax(logits, dim=-1), float(top_p))
        logits = torch.where(probs > 0, logits, torch.full_like(logits, float("-inf")))
    logits = add_gumbel_noise(logits, temperature, generator=generator)
    return torch.argmax(logits, dim=-1)


@torch.no_grad()
def flow_image_generate(
    model,
    prompt_indices: torch.Tensor,
    *,
    context: torch.Tensor,
    steps: int,
    cfg_scale: float = 0.0,
    uncond_context: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if prompt_indices.dim() != 2:
        raise ValueError("prompt_indices must be 2D (batch, seq)")
    if context.dim() != 1:
        raise ValueError("context must be 1D (batch,)")
    if prompt_indices.shape[0] != context.shape[0]:
        raise ValueError("context batch size must match prompt batch size")
    if prompt_indices.shape[1] != 0:
        raise ValueError("flow_image_generate expects empty prompt_indices for full-image generation")
    if steps <= 0:
        raise ValueError("steps must be > 0")

    batch_size = context.shape[0]
    gen_length = int(getattr(model, "context_length"))
    x = torch.randn(
        (batch_size, gen_length),
        device=prompt_indices.device,
        dtype=torch.float32,
        generator=generator,
    )
    dt = 1.0 / float(steps)

    if uncond_context is not None:
        if uncond_context.dim() != 1:
            raise ValueError("uncond_context must be 1D (batch,)")
        if uncond_context.shape[0] != batch_size:
            raise ValueError("uncond_context batch size must match prompt batch size")
        uncond_context = uncond_context.to(device=context.device, dtype=context.dtype)

    for k in range(steps):
        t = torch.full((batch_size,), float(k) / float(steps), device=x.device, dtype=x.dtype)
        if cfg_scale > 0.0:
            if uncond_context is None:
                raise ValueError("uncond_context must be set when cfg_scale > 0 for flow_image_generate")
            v_cond = model(x, t, context=context)
            v_uncond = model(x, t, context=uncond_context)
            v = v_uncond + (cfg_scale + 1.0) * (v_cond - v_uncond)
        else:
            v = model(x, t, context=context)
        x = x + dt * v
    return x


# Backwards-compatible alias
generate = diffusion_generate


__all__ = [
    "diffusion_generate",
    "image_diffusion_generate",
    "autoregressive_generate",
    "categorical_flow_image_generate",
    "flow_image_generate",
    "generate",
]
