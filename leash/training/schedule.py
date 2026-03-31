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

__all__ = ["lr_cosine_schedule", "lr_constant_schedule", "lr_constant_with_warmup_schedule"]
