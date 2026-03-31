from __future__ import annotations

import math


def resolve_scheduled_p_mask(
    *,
    step: int,
    total_steps: int,
    override: float | None,
    schedule: str,
    start_value: float | None,
    end_value: float | None,
    schedule_start: float,
    schedule_end: float,
) -> float | None:
    if override is not None:
        return float(override)
    if start_value is None or end_value is None:
        return None

    mode = str(schedule).lower()
    if mode in {"none", ""}:
        return None
    if mode not in {"linear", "cosine"}:
        raise ValueError("p_mask_schedule must be one of: none, linear, cosine")

    if total_steps <= 1:
        return float(start_value)

    progress = float(step) / float(max(1, total_steps - 1))
    if progress <= float(schedule_start):
        return float(start_value)
    if progress >= float(schedule_end):
        return float(end_value)

    interp = (progress - float(schedule_start)) / max(float(schedule_end) - float(schedule_start), 1e-8)
    if mode == "cosine":
        interp = 0.5 * (1.0 - math.cos(interp * math.pi))
    return float(start_value + (end_value - start_value) * interp)


__all__ = ["resolve_scheduled_p_mask"]
