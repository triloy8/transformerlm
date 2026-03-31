from __future__ import annotations

from dataclasses import replace
import math

import torch

from leash.inference.generate import autoregressive_generate, diffusion_generate
from leash.objectives.base import Objective
from leash.objectives.data import JointBatch, get_joint_batch
from leash.objectives.loss import autoregressive_cross_entropy, diffusion_cross_entropy, mntp_cross_entropy
from leash.objectives.schedule import resolve_scheduled_p_mask


class JointDiffusionAutoregressiveObjective(Objective):
    def __init__(self, cfg, tokenizer) -> None:
        super().__init__("joint-diffusion-ar")
        self._tokenizer = tokenizer
        self.mask_token_id = int(getattr(cfg, "mask_token_id", cfg.vocab_size - 1))
        self.noise_epsilon = float(getattr(cfg, "noise_epsilon", 1e-3))
        self.random_trunc_prob = float(getattr(cfg, "random_trunc_prob", 0.01))
        self.p_mask_override = getattr(cfg, "p_mask_override", None)
        self.deterministic_mask = bool(getattr(cfg, "deterministic_mask", False))

        self.alpha_start = float(getattr(cfg, "joint_diffusion_alpha", 0.3))
        alpha_end_cfg = getattr(cfg, "joint_diffusion_alpha_end", None)
        self.alpha_end = self.alpha_start if alpha_end_cfg is None else float(alpha_end_cfg)
        self.alpha_schedule = str(getattr(cfg, "joint_alpha_schedule", "constant")).lower()
        self.alpha_schedule_start = float(getattr(cfg, "joint_alpha_schedule_start", 0.0))
        self.alpha_schedule_end = float(getattr(cfg, "joint_alpha_schedule_end", 1.0))
        self.total_steps = int(getattr(cfg, "max_train_iteration", 0))
        self.p_mask_schedule = str(getattr(cfg, "p_mask_schedule", "none")).lower()
        self.p_mask_start = getattr(cfg, "p_mask_start", None)
        self.p_mask_end = getattr(cfg, "p_mask_end", None)
        self.p_mask_schedule_start = float(getattr(cfg, "p_mask_schedule_start", 0.0))
        self.p_mask_schedule_end = float(getattr(cfg, "p_mask_schedule_end", 1.0))
        self._step = 0

    def on_step(self, step: int, max_steps: int | None, is_train: bool) -> None:
        if is_train:
            self._step = int(step)

    def _current_alpha(self) -> float:
        if self.alpha_schedule == "constant" or self.total_steps <= 1:
            return float(self.alpha_start)
        if self.alpha_schedule not in {"linear", "cosine"}:
            raise ValueError("joint_alpha_schedule must be one of: constant, linear, cosine")
        progress = float(self._step) / float(max(1, self.total_steps - 1))
        start = self.alpha_schedule_start
        end = self.alpha_schedule_end
        if progress <= start:
            return float(self.alpha_start)
        if progress >= end:
            return float(self.alpha_end)
        interp = (progress - start) / max(end - start, 1e-8)
        if self.alpha_schedule == "cosine":
            interp = 0.5 * (1.0 - math.cos(interp * math.pi))
        return float(self.alpha_start + (self.alpha_end - self.alpha_start) * interp)

    def get_batch(self, *, dataset, batch_size: int, context_length: int, device: str, generator=None):
        scheduled_p_mask = resolve_scheduled_p_mask(
            step=self._step,
            total_steps=self.total_steps,
            override=self.p_mask_override,
            schedule=self.p_mask_schedule,
            start_value=self.p_mask_start,
            end_value=self.p_mask_end,
            schedule_start=self.p_mask_schedule_start,
            schedule_end=self.p_mask_schedule_end,
        )
        return get_joint_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
            mask_token_id=self.mask_token_id,
            noise_epsilon=self.noise_epsilon,
            random_trunc_prob=self.random_trunc_prob,
            p_mask_override=scheduled_p_mask,
            deterministic_mask=self.deterministic_mask,
            generator=generator,
        )

    def model_inputs(self, batch: JointBatch):
        labels = getattr(batch.diffusion, "labels", None)
        if labels is None:
            return batch.diffusion.noisy_inputs
        return batch.diffusion.noisy_inputs, labels

    def attention_mask(self, batch: JointBatch):
        return batch.diffusion.attention_mask

    def compute_loss(self, logits: torch.Tensor, batch: JointBatch) -> torch.Tensor:
        # Fallback path only; train_loop should call forward_with_model for this objective.
        alpha = self._current_alpha()
        diff_loss = diffusion_cross_entropy(
            logits,
            batch.diffusion.clean_targets,
            batch.diffusion.mask,
            batch.diffusion.p_mask,
            loss_mask=batch.diffusion.loss_mask,
        )
        return alpha * diff_loss

    def forward_with_model(self, model: torch.nn.Module, batch: JointBatch) -> dict:
        alpha = self._current_alpha()

        diff_inputs = batch.diffusion.noisy_inputs
        diff_attn = batch.diffusion.attention_mask
        diff_labels = getattr(batch.diffusion, "labels", None)
        if diff_attn is not None:
            if diff_labels is not None:
                diff_logits = model(diff_inputs, attention_mask=diff_attn, context=diff_labels)
            else:
                diff_logits = model(diff_inputs, attention_mask=diff_attn)
        else:
            if diff_labels is not None:
                diff_logits = model(diff_inputs, context=diff_labels)
            else:
                diff_logits = model(diff_inputs)
        diff_loss = diffusion_cross_entropy(
            diff_logits,
            batch.diffusion.clean_targets,
            batch.diffusion.mask,
            batch.diffusion.p_mask,
            loss_mask=batch.diffusion.loss_mask,
        )

        ar_inputs = batch.autoregressive.inputs
        ar_attn = batch.autoregressive.attention_mask
        if ar_attn is not None:
            ar_logits = model(ar_inputs, attention_mask=ar_attn)
        else:
            ar_logits = model(ar_inputs)
        ar_loss = autoregressive_cross_entropy(
            ar_logits,
            batch.autoregressive.targets,
            loss_mask=batch.autoregressive.loss_mask,
        )

        total_loss = alpha * diff_loss + (1.0 - alpha) * ar_loss

        return {
            "loss": total_loss,
            "logits": diff_logits,
            "inputs": diff_inputs,
            "batch": replace(batch, metadata=batch.diffusion.metadata),
            "metrics": {
                "metrics.train_loss_diffusion": float(diff_loss.detach().item()),
                "metrics.train_loss_ar": float(ar_loss.detach().item()),
                "metrics.train_joint_alpha": float(alpha),
            },
        }

    def val_samples(self, inputs: torch.Tensor, logits: torch.Tensor, batch: JointBatch, max_samples: int):
        if max_samples <= 0:
            return None
        count = min(int(max_samples), int(inputs.shape[0]))
        if count <= 0:
            return None
        targets = batch.diffusion.clean_targets
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
        generation_mode = str(kwargs.get("generation_mode", "diffusion")).lower()
        if generation_mode == "ar":
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
        return diffusion_generate(
            model,
            prompt_indices,
            mask_id=int(kwargs.get("mask_id")),
            eos_token_id=kwargs.get("eos_token_id"),
            steps=int(kwargs.get("steps", 0)),
            gen_length=int(kwargs.get("gen_length", 0)),
            block_length=int(kwargs.get("block_length", 0)),
            temperature=float(kwargs.get("temperature", 0.0)),
            top_p=kwargs.get("top_p"),
            cfg_scale=float(kwargs.get("cfg_scale", 0.0)),
            remasking=str(kwargs.get("remasking", "random")),
            logits_eos_inf=bool(kwargs.get("logits_eos_inf", False)),
            confidence_eos_eot_inf=bool(kwargs.get("confidence_eos_eot_inf", False)),
            generator=kwargs.get("generator"),
        )


class JointMntpAutoregressiveObjective(JointDiffusionAutoregressiveObjective):
    def __init__(self, cfg, tokenizer) -> None:
        super().__init__(cfg, tokenizer)
        self.name = "joint-mntp-ar"

    def compute_loss(self, logits: torch.Tensor, batch: JointBatch) -> torch.Tensor:
        # Fallback path only; train_loop should call forward_with_model for this objective.
        alpha = self._current_alpha()
        mntp_loss = mntp_cross_entropy(
            logits,
            batch.diffusion.clean_targets,
            batch.diffusion.mask,
            batch.diffusion.p_mask,
            loss_mask=batch.diffusion.loss_mask,
        )
        return alpha * mntp_loss

    def forward_with_model(self, model: torch.nn.Module, batch: JointBatch) -> dict:
        alpha = self._current_alpha()

        diff_inputs = batch.diffusion.noisy_inputs
        diff_attn = batch.diffusion.attention_mask
        diff_labels = getattr(batch.diffusion, "labels", None)
        if diff_attn is not None:
            if diff_labels is not None:
                diff_logits = model(diff_inputs, attention_mask=diff_attn, context=diff_labels)
            else:
                diff_logits = model(diff_inputs, attention_mask=diff_attn)
        else:
            if diff_labels is not None:
                diff_logits = model(diff_inputs, context=diff_labels)
            else:
                diff_logits = model(diff_inputs)
        mntp_loss = mntp_cross_entropy(
            diff_logits,
            batch.diffusion.clean_targets,
            batch.diffusion.mask,
            batch.diffusion.p_mask,
            loss_mask=batch.diffusion.loss_mask,
        )

        ar_inputs = batch.autoregressive.inputs
        ar_attn = batch.autoregressive.attention_mask
        if ar_attn is not None:
            ar_logits = model(ar_inputs, attention_mask=ar_attn)
        else:
            ar_logits = model(ar_inputs)
        ar_loss = autoregressive_cross_entropy(
            ar_logits,
            batch.autoregressive.targets,
            loss_mask=batch.autoregressive.loss_mask,
        )

        total_loss = alpha * mntp_loss + (1.0 - alpha) * ar_loss

        return {
            "loss": total_loss,
            "logits": diff_logits,
            "inputs": diff_inputs,
            "batch": replace(batch, metadata=batch.diffusion.metadata),
            "metrics": {
                "metrics.train_loss_mntp": float(mntp_loss.detach().item()),
                "metrics.train_loss_ar": float(ar_loss.detach().item()),
                "metrics.train_joint_alpha": float(alpha),
            },
        }


__all__ = ["JointDiffusionAutoregressiveObjective", "JointMntpAutoregressiveObjective"]
