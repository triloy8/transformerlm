import torch


def softmax(x: torch.Tensor, dim: int):
    x_max = x.max(dim=dim, keepdim=True).values
    x_stable = x - x_max
    exp_x = torch.exp(x_stable)
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp_x


def top_p_filter(probs: torch.Tensor, p: float) -> torch.Tensor:
    if probs.dim() < 2:
        raise ValueError("probs must have at least 2 dimensions")
    orig_shape = probs.shape
    vocab = orig_shape[-1]
    probs = probs.reshape(-1, vocab)
    if p <= 0:
        argmax = probs.argmax(dim=-1)
        out = torch.zeros_like(probs)
        out.scatter_(-1, argmax.unsqueeze(-1), 1.0)
        return out.reshape(orig_shape)
    if p >= 1:
        return (probs / probs.sum(dim=-1, keepdim=True)).reshape(orig_shape)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    keep = cumulative <= p
    keep[..., 0] = True
    first_ge = (cumulative >= p).float().argmax(dim=-1)
    rows = torch.arange(keep.shape[0], device=keep.device)
    keep[rows, first_ge] = True

    filtered_sorted = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    norm = filtered_sorted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    filtered_sorted = filtered_sorted / norm

    filtered = torch.zeros_like(probs)
    filtered.scatter_(dim=-1, index=sorted_indices, src=filtered_sorted)
    return filtered.reshape(orig_shape)


def add_gumbel_noise(logits: torch.Tensor, temperature: float, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """Apply Gumbel noise to logits when temperature > 0."""

    if temperature <= 0:
        return logits

    noise = torch.rand(logits.shape, device=logits.device, dtype=torch.float64, generator=generator)
    gumbel_noise = (-torch.log(noise)) ** temperature
    logits64 = logits.to(torch.float64)
    perturbed = logits64.exp() / gumbel_noise
    return perturbed.to(logits.dtype)


def compute_transfer_schedule(mask: torch.Tensor, steps: int) -> torch.Tensor:
    """Compute how many tokens to reveal per step for each batch item."""

    if steps <= 0:
        raise ValueError("steps must be > 0")
    if mask.dim() != 2:
        raise ValueError("mask must be 2D (batch, block_length)")

    counts = mask.sum(dim=1, keepdim=True).to(torch.int64)
    base = counts // steps
    remainder = counts % steps

    schedule = base.expand(-1, steps).clone()
    for idx in range(schedule.size(0)):
        r = remainder[idx, 0].item()
        if r > 0:
            schedule[idx, :r] += 1
    return schedule


__all__ = [
    "softmax",
    "top_p_filter",
    "add_gumbel_noise",
    "compute_transfer_schedule",
]
