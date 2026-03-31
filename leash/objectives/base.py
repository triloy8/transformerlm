from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import torch


class Objective(ABC):
    """Objective interface for leash."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def get_batch(
        self,
        *,
        dataset: Any,
        batch_size: int,
        context_length: int,
        device: str,
        generator: torch.Generator | None = None,
    ) -> Any:
        ...

    @abstractmethod
    def model_inputs(self, batch: Any):
        ...

    @abstractmethod
    def attention_mask(self, batch: Any) -> Optional[torch.Tensor]:
        ...

    @abstractmethod
    def compute_loss(self, logits: torch.Tensor, batch: Any) -> torch.Tensor:
        ...

    def forward_with_model(self, model: torch.nn.Module, batch: Any) -> Optional[dict]:
        return None

    def on_step(self, step: int, max_steps: int | None, is_train: bool) -> None:
        return None

    def extra_metrics(
        self,
        logits: torch.Tensor,
        batch: Any,
        reduce_metric: Optional[Callable[[float], float]],
    ) -> Optional[dict]:
        return None

    def val_samples(
        self,
        inputs: torch.Tensor,
        logits: torch.Tensor,
        batch: Any,
        max_samples: int,
    ) -> Optional[list[dict]]:
        return None

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, tokens: list[int]) -> str:
        raise NotImplementedError

    def generate(self, model, prompt_indices: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError


__all__ = ["Objective"]
