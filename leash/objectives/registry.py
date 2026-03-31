from __future__ import annotations

from collections.abc import Callable

from leash.objectives.autoregressive import AutoregressiveObjective
from leash.objectives.base import Objective
from leash.objectives.categorical_flow import CategoricalFlowObjective
from leash.objectives.diffusion import DiffusionObjective, FlowMatchingObjective, MegaDlmDiffusionObjective
from leash.objectives.joint import JointDiffusionAutoregressiveObjective, JointMntpAutoregressiveObjective

ObjectiveFactory = Callable[[object, object], Objective]

_OBJECTIVE_FACTORIES: dict[str, ObjectiveFactory] = {
    "diffusion": DiffusionObjective,
    "ar": AutoregressiveObjective,
    "megadlm-diffusion": MegaDlmDiffusionObjective,
    "flow": FlowMatchingObjective,
    "categorical-flow": CategoricalFlowObjective,
    "joint-diffusion-ar": JointDiffusionAutoregressiveObjective,
    "joint-mntp-ar": JointMntpAutoregressiveObjective,
}


def register_objective(name: str, factory: ObjectiveFactory) -> None:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("objective name must be non-empty")
    _OBJECTIVE_FACTORIES[key] = factory


def get_objective_factory(name: str) -> ObjectiveFactory:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("objective name must be non-empty")
    try:
        return _OBJECTIVE_FACTORIES[key]
    except KeyError as exc:
        available = ", ".join(sorted(_OBJECTIVE_FACTORIES))
        raise ValueError(f"Unknown training objective '{name}'. Available: {available}") from exc


def list_objectives() -> tuple[str, ...]:
    return tuple(sorted(_OBJECTIVE_FACTORIES))


def build_objective(cfg, tokenizer) -> Objective:
    name = str(getattr(cfg, "training_objective", "diffusion")).lower()
    factory = get_objective_factory(name)
    return factory(cfg, tokenizer)


__all__ = [
    "ObjectiveFactory",
    "build_objective",
    "get_objective_factory",
    "list_objectives",
    "register_objective",
]
