from __future__ import annotations

import torch

from leash.inference.generate import autoregressive_generate
from leash.objectives.base import Objective
from leash.objectives.data import AutoregressiveBatch, get_autoregressive_batch
from leash.objectives.loss import autoregressive_cross_entropy


class AutoregressiveObjective(Objective):
    def __init__(self, cfg, tokenizer) -> None:
        super().__init__("ar")
        self._tokenizer = tokenizer
        self.random_trunc_prob = float(getattr(cfg, "random_trunc_prob", 0.01))

    def get_batch(self, *, dataset, batch_size: int, context_length: int, device: str, generator=None):
        return get_autoregressive_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
            random_trunc_prob=self.random_trunc_prob,
            generator=generator,
        )

    def model_inputs(self, batch: AutoregressiveBatch) -> torch.Tensor:
        return batch.inputs

    def attention_mask(self, batch: AutoregressiveBatch):
        return batch.attention_mask

    def compute_loss(self, logits: torch.Tensor, batch: AutoregressiveBatch) -> torch.Tensor:
        return autoregressive_cross_entropy(logits, batch.targets, loss_mask=batch.loss_mask)

    def val_samples(self, inputs: torch.Tensor, logits: torch.Tensor, batch: AutoregressiveBatch, max_samples: int):
        if max_samples <= 0:
            return None
        count = min(int(max_samples), int(inputs.shape[0]))
        if count <= 0:
            return None
        targets = batch.targets
        inputs_list = inputs[:count].detach().cpu().tolist()
        preds_list = logits[:count].argmax(dim=-1).detach().cpu().tolist()
        targets_list = targets[:count].detach().cpu().tolist()
        return [
            {"inputs": inputs_list[i], "predictions": preds_list[i], "targets": targets_list[i]}
            for i in range(count)
        ]

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)

    def generate(self, model, prompt_indices: torch.Tensor, **kwargs) -> torch.Tensor:
        return autoregressive_generate(
            model,
            prompt_indices,
            gen_length=int(kwargs.get("gen_length", 0)),
            temperature=float(kwargs.get("temperature", 0.0)),
            top_p=kwargs.get("top_p"),
            eos_token_id=kwargs.get("eos_token_id"),
            logits_eos_inf=bool(kwargs.get("logits_eos_inf", False)),
            generator=kwargs.get("generator"),
        )


__all__ = ["AutoregressiveObjective"]
