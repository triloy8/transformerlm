import torch
from collections.abc import Iterable


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    eps = 1e-6
    grads = [p.grad for p in parameters if p.grad is not None]
    l2_norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]))
    if max_l2_norm < l2_norm:
        for g in grads:
            g.mul_(max_l2_norm / (l2_norm + eps))

__all__ = ["gradient_clipping"]

