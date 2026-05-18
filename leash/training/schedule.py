import math


def lr_cosine_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int):
    if it < warmup_iters:
        lr = (it / warmup_iters) * max_learning_rate if warmup_iters > 0 else max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        lr = min_learning_rate + 0.5 * (1 + math.cos(((it - warmup_iters) / (cosine_cycle_iters - warmup_iters)) * math.pi)) * (max_learning_rate - min_learning_rate)
    elif cosine_cycle_iters < it:
        lr = min_learning_rate
    else:
        raise ValueError(f"Invalid learning rate schedule state at it={it}")
    return lr


def lr_constant_schedule(
    _it: int,
    max_learning_rate: float,
    _min_learning_rate: float,
    _warmup_iters: int,
    _cosine_cycle_iters: int,
) -> float:
    return float(max_learning_rate)


def lr_constant_with_warmup_schedule(
    it: int,
    max_learning_rate: float,
    _min_learning_rate: float,
    warmup_iters: int,
    _cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate if warmup_iters > 0 else float(max_learning_rate)
    return float(max_learning_rate)


def lr_wsd_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    lr_decay_iters: int,
    *,
    wsd_decay_iters: int,
    wsd_decay_style: str = "exponential",
) -> float:
    if lr_decay_iters <= 0:
        raise ValueError("lr_decay_iters must be > 0 for WSD schedule")
    if warmup_iters >= lr_decay_iters:
        raise ValueError("warmup_iters must be < lr_decay_iters for WSD schedule")
    if wsd_decay_iters <= 0:
        raise ValueError("wsd_decay_iters must be > 0 for WSD schedule")
    if wsd_decay_iters > lr_decay_iters:
        raise ValueError("wsd_decay_iters must be <= lr_decay_iters for WSD schedule")

    if warmup_iters > 0 and it <= warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    if it > lr_decay_iters:
        return float(min_learning_rate)

    anneal_start = lr_decay_iters - wsd_decay_iters
    if it <= anneal_start:
        coeff = 1.0
    else:
        ratio = float(it - anneal_start) / float(wsd_decay_iters)
        style = str(wsd_decay_style).lower()
        if style == "linear":
            coeff = 1.0 - ratio
        elif style == "cosine":
            coeff = 0.5 * (math.cos(math.pi * ratio) + 1.0)
        elif style == "exponential":
            coeff = (2.0 * math.pow(0.5, ratio)) - 1.0
        else:
            raise ValueError("wsd_decay_style must be one of: exponential, linear, cosine")

    return float(min_learning_rate + coeff * (max_learning_rate - min_learning_rate))

__all__ = [
    "lr_cosine_schedule",
    "lr_constant_schedule",
    "lr_constant_with_warmup_schedule",
    "lr_wsd_schedule",
]
