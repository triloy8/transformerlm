import torch
from torch import Tensor


def cross_entropy(inputs: Tensor, targets: Tensor, *, reduction: str = "mean_batch") -> Tensor:
    inputs_max = inputs.max(dim=-1, keepdim=True).values
    inputs_stable = inputs - inputs_max
    exp_inputs_stable = torch.exp(inputs_stable)
    log_sum_exp_inputs_stable = torch.log(exp_inputs_stable.sum(dim=-1, keepdim=True))

    indices = targets.long().unsqueeze(-1)
    gathered_inputs_stable = torch.gather(inputs_stable, dim=-1, index=indices)

    l = (-gathered_inputs_stable + log_sum_exp_inputs_stable).squeeze(-1)

    if reduction == "none":
        return l
    if reduction == "mean_batch":
        return l.mean(dim=0)
    if reduction == "mean":
        return l.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def diffusion_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    p_mask: Tensor,
    *,
    loss_mask: Tensor | None = None,
) -> Tensor:
    per_token = cross_entropy(logits, targets, reduction="none")
    mask_f = mask.to(per_token.dtype)
    if loss_mask is not None:
        loss_mask_f = loss_mask.to(per_token.dtype)
        mask_f = mask_f * loss_mask_f
    weighted = (per_token * mask_f) / p_mask
    if loss_mask is not None:
        denom = loss_mask_f.sum().item()
    else:
        denom = targets.shape[0] * targets.shape[1]
    return weighted.sum() / max(denom, 1)


def unweighted_diffusion_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    *,
    loss_mask: Tensor | None = None,
) -> Tensor:
    per_token = cross_entropy(logits, targets, reduction="none")
    mask_f = mask.to(per_token.dtype)
    if loss_mask is not None:
        mask_f = mask_f * loss_mask.to(per_token.dtype)
    denom = mask_f.sum().item()
    return (per_token * mask_f).sum() / max(denom, 1)


def uniform_gidd_loss(
    logits: Tensor,
    z_t: Tensor,
    targets: Tensor,
    t: Tensor,
    *,
    vocab_size: int | None = None,
    beta_is: float = 1.0,
    z_loss_strength: float | None = 1e-5,
    loss_mask: Tensor | None = None,
    reduction_mask: Tensor | None = None,
    eps: float = 1e-12,
    reduction: str = "mean",
) -> Tensor:
    """Uniform discrete-diffusion GIDD/NELBO loss.

    The forward process is q(z_t=v|x)=alpha*1[v=x]+(1-alpha)/V with alpha=1-t.
    The per-token loss is w*KL[q(.|x)||q(.|x_hat)] + beta_is*w*D_IS(z_t),
    optionally plus z_loss_strength*logsumexp(logits)^2.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must have shape [B, S, V], got {tuple(logits.shape)}")
    if z_t.shape != targets.shape:
        raise ValueError(f"z_t and targets must have the same shape, got {tuple(z_t.shape)} and {tuple(targets.shape)}")
    if logits.shape[:2] != targets.shape:
        raise ValueError(f"logits prefix shape must match targets, got {tuple(logits.shape)} and {tuple(targets.shape)}")
    if vocab_size is None:
        vocab_size = int(logits.size(-1))
    if vocab_size <= 0 or vocab_size > logits.size(-1):
        raise ValueError(f"vocab_size must be in (0, {logits.size(-1)}], got {vocab_size}")
    if beta_is < 0:
        raise ValueError("beta_is must be >= 0")
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of: none, mean, sum")

    logits = logits[..., :vocab_size].float()
    z_t = z_t.long()
    targets = targets.long()

    t = t.to(device=logits.device, dtype=torch.float32)
    if t.dim() == 1:
        t = t[:, None]
    if t.dim() != 2 or t.size(0) != targets.size(0) or (t.size(1) != 1 and t.size(1) != targets.size(1)):
        raise ValueError(
            f"t must have shape [B], [B, 1], or [B, S] for targets shape {tuple(targets.shape)}, got {tuple(t.shape)}"
        )
    alpha = (1.0 - t).clamp(min=eps, max=1.0 - eps).expand_as(targets)
    u = (1.0 - alpha) / float(vocab_size)

    x_hat = torch.softmax(logits, dim=-1)
    log_q_v = torch.log(alpha.unsqueeze(-1) * x_hat + u.unsqueeze(-1))

    log_q_at_x = log_q_v.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    sum_log_q = log_q_v.sum(dim=-1)
    x_hat_at_zt = x_hat.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)

    h_ce = -alpha * log_q_at_x - u * sum_log_q
    alpha_plus_u = alpha + u
    h_q = -(
        alpha_plus_u * torch.log(alpha_plus_u.clamp(min=eps))
        + (vocab_size - 1) * u * torch.log(u.clamp(min=eps))
    )
    kl = h_ce - h_q

    is_zt_eq_x = (z_t == targets).to(alpha.dtype)
    q_zt_x = alpha * is_zt_eq_x + u
    q_zt_x_hat = alpha * x_hat_at_zt + u
    log_ratio = torch.log(q_zt_x.clamp(min=eps)) - torch.log(q_zt_x_hat.clamp(min=eps))
    is_div = torch.exp(log_ratio) - log_ratio - 1.0

    w_kept = (1.0 - alpha) / (1.0 + (vocab_size - 1) * alpha)
    w = torch.where(z_t == targets, w_kept, torch.ones_like(w_kept))

    per_token = w * kl + float(beta_is) * w * is_div
    if z_loss_strength is not None and z_loss_strength > 0.0:
        log_z = torch.logsumexp(logits, dim=-1)
        per_token = per_token + float(z_loss_strength) * log_z * log_z

    if reduction == "none":
        return per_token

    reduce_mask = torch.ones_like(per_token, dtype=per_token.dtype)
    if loss_mask is not None:
        reduce_mask = reduce_mask * loss_mask.to(device=per_token.device, dtype=per_token.dtype)
    if reduction_mask is not None:
        reduce_mask = reduce_mask * reduction_mask.to(device=per_token.device, dtype=per_token.dtype)

    weighted = per_token * reduce_mask
    if reduction == "sum":
        return weighted.sum()
    return weighted.sum() / reduce_mask.sum().clamp_min(1.0)


def mntp_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    p_mask: Tensor,
    *,
    loss_mask: Tensor | None = None,
) -> Tensor:
    # MNTP uses next-token targets while keeping masked-token weighting.
    if logits.shape[1] < 2:
        return logits.new_tensor(0.0)
    shifted_logits = logits[:, :-1, :]
    shifted_targets = targets[:, 1:]
    shifted_mask = mask[:, :-1]
    shifted_loss_mask = loss_mask[:, 1:] if loss_mask is not None else None
    return diffusion_cross_entropy(
        shifted_logits,
        shifted_targets,
        shifted_mask,
        p_mask,
        loss_mask=shifted_loss_mask,
    )


def autoregressive_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    *,
    loss_mask: Tensor | None = None,
) -> Tensor:
    per_token = cross_entropy(logits, targets, reduction="none")
    if loss_mask is not None:
        loss_mask_f = loss_mask.to(per_token.dtype)
        per_token = per_token * loss_mask_f
        denom = loss_mask_f.sum().item()
    else:
        denom = targets.shape[0] * targets.shape[1]
    return per_token.sum() / max(denom, 1)


__all__ = [
    "cross_entropy",
    "diffusion_cross_entropy",
    "unweighted_diffusion_cross_entropy",
    "uniform_gidd_loss",
    "mntp_cross_entropy",
    "autoregressive_cross_entropy",
]
