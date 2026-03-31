from collections.abc import Callable
from typing import Any, Dict, List, Optional
import math
import torch
from torch import nn
from torch.nn import Parameter


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                t = state.get("t", 0)
                grad = p.grad.data

                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad ** 2

                lr_t = lr * (math.sqrt(1 - betas[1] ** (t + 1)) / (1 - betas[0] ** (t + 1)))

                p.data = (1 - lr * weight_decay) * p.data  - lr_t * (m / (torch.sqrt(v) + eps))

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss
    

# The Muon code is adapted from https://github.com/KellerJordan/Muon, 
# the rewrite is just mirroring the above AdamW notations
def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """Muon with AdamW fallback"""

    def __init__(
        self,
        param_groups,
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        betas=(0.9, 0.95),
        eps: float = 1e-10,
    ):
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Muon param group must set 'use_muon'")
            extra_keys = {"initial_lr", "max_lr", "min_lr", "warmup_iters", "cosine_cycle_iters"}
            if group["use_muon"]:
                group.setdefault("lr", lr)
                group.setdefault("momentum", momentum)
                group.setdefault("weight_decay", weight_decay)
                allowed = {"params", "lr", "momentum", "weight_decay", "use_muon", "name", *extra_keys}
            else:
                group.setdefault("lr", lr)
                group.setdefault("betas", betas)
                group.setdefault("eps", eps)
                group.setdefault("weight_decay", weight_decay)
                allowed = {"params", "lr", "betas", "eps", "weight_decay", "use_muon", "name", *extra_keys}
            unexpected = set(group.keys()) - allowed
            if unexpected:
                raise ValueError(f"Unexpected keys {unexpected} in Muon param group")
        super().__init__(param_groups, dict())

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization

                    state = self.state[p]
                    m = state.get("m", torch.zeros_like(p))
                    grad = p.grad

                    m = group["momentum"] * m + (1 - group["momentum"]) * grad
                    # nesterov by default
                    grad = (1 - group["momentum"]) * grad + group["momentum"] * m
                    b = grad 
                    if b.ndim == 4: # for the case of conv filters
                        b = b.view(len(b), -1)
                    o = zeropower_via_newtonschulz5(b, steps=5)
                    o *= max(1, grad.size(-2) / grad.size(-1))**0.5

                    p.data *= 1 - group["lr"] * group["weight_decay"]
                    p.data -= group["lr"] * o.reshape(p.shape)

                    # momentum buffer update
                    state["m"] = m
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # Force synchronization

                    state = self.state[p]
                    m = state.get("m", torch.zeros_like(p))
                    v = state.get("v", torch.zeros_like(p))
                    t = state.get("t", 0)
                    grad = p.grad

                    m = group["betas"][0] * m + (1 - group["betas"][0]) * grad
                    v = group["betas"][1] * v + (1 - group["betas"][1]) * grad ** 2

                    lr_t = group["lr"] * (math.sqrt(1 - group["betas"][1] ** (t + 1)) / (1 - group["betas"][0] ** (t + 1)))

                    p.data = (1 - group["lr"] * group["weight_decay"]) * p.data  - lr_t * (m / (torch.sqrt(v) + group["eps"]))

                    state["t"] = t + 1
                    state["m"] = m
                    state["v"] = v

        return loss


OPTIMIZER_REGISTRY: Dict[str, type[torch.optim.Optimizer]] = {
    "adamw": AdamW,
    "muon": Muon,
}


def resolve_optimizer_cls(name: str) -> type[torch.optim.Optimizer]:
    key = name.lower()
    if key not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unsupported optimizer '{name}'. Available: {sorted(OPTIMIZER_REGISTRY)}")
    return OPTIMIZER_REGISTRY[key]


def build_optimizer_param_groups(
    model: nn.Module,
    optimizer_name: str,
    muon_cfg: Any | None = None,
) -> List[Dict[str, object]]:
    optimizer_key = optimizer_name.lower()
    if optimizer_key != "muon":
        return [{"params": list(model.parameters())}]

    if muon_cfg is None:
        raise ValueError("Muon optimizer requires muon_cfg")

    hidden_matrix_params: List[Parameter] = []
    embed_params: List[Parameter] = []
    scalar_params: List[Parameter] = []
    head_params: List[Parameter] = []

    for name, param in model.named_parameters():
        if not isinstance(param, Parameter):
            continue
        lower_name = name.lower()
        if lower_name.endswith("lm_head.weight"):
            head_params.append(param)
        elif "embed" in lower_name:
            embed_params.append(param)
        elif param.ndim < 2:
            scalar_params.append(param)
        elif lower_name.startswith("layers."):
            if param.ndim >= 2:
                hidden_matrix_params.append(param)

    if not hidden_matrix_params:
        hidden_matrix_params = [p for p in model.parameters() if isinstance(p, Parameter) and p.ndim >= 2]

    def _deduplicate(params: List[Parameter]) -> List[Parameter]:
        seen: set[int] = set()
        deduped: List[Parameter] = []
        for tensor in params:
            pid = id(tensor)
            if pid in seen:
                continue
            seen.add(pid)
            deduped.append(tensor)
        return deduped

    head_params = _deduplicate(head_params)
    embed_params = _deduplicate(embed_params)
    scalar_params = _deduplicate(scalar_params)
    hidden_matrix_params = _deduplicate(hidden_matrix_params)

    def _adam_group(params: List[Parameter], group_cfg: Any, name: str) -> Dict[str, object]:
        if not params:
            return {}
        betas = getattr(group_cfg, "betas")
        return {
            "params": params,
            "name": name,
            "lr": float(group_cfg.initial_learning_rate),
            "initial_lr": float(group_cfg.initial_learning_rate),
            "max_lr": float(group_cfg.max_learning_rate),
            "min_lr": float(group_cfg.min_learning_rate),
            "betas": (float(betas[0]), float(betas[1])),
            "eps": float(group_cfg.eps),
            "weight_decay": float(group_cfg.weight_decay),
            "use_muon": False,
        }

    head_cfg = getattr(muon_cfg, "head")
    embed_cfg = getattr(muon_cfg, "embed")
    scalar_cfg = getattr(muon_cfg, "scalar")

    adam_groups = [
        _adam_group(head_params, head_cfg, "head"),
        _adam_group(embed_params, embed_cfg, "embed"),
        _adam_group(scalar_params, scalar_cfg, "scalar"),
    ]
    adam_groups = [group for group in adam_groups if group]

    hidden_cfg = getattr(muon_cfg, "hidden")
    muon_group = {
        "params": hidden_matrix_params,
        "name": "hidden",
        "lr": float(hidden_cfg.initial_learning_rate),
        "initial_lr": float(hidden_cfg.initial_learning_rate),
        "max_lr": float(hidden_cfg.max_learning_rate),
        "min_lr": float(hidden_cfg.min_learning_rate),
        "momentum": float(hidden_cfg.momentum),
        "weight_decay": float(hidden_cfg.weight_decay),
        "use_muon": True,
    }

    return [*adam_groups, muon_group]


__all__ = [
    "AdamW",
    "Muon",
    "OPTIMIZER_REGISTRY",
    "resolve_optimizer_cls",
    "build_optimizer_param_groups",
]
