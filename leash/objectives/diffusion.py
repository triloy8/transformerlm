from __future__ import annotations

from typing import Optional
import torch

from leash.inference.generate import autoregressive_generate, diffusion_generate, flow_image_generate
from leash.objectives.base import Objective
from leash.objectives.data import (
    DiffusionBatch,
    FlowMatchingBatch,
    get_batch,
    get_flow_matching_batch,
    get_megadlm_diffusion_batch,
)
from leash.objectives.loss import cross_entropy, diffusion_cross_entropy
from leash.objectives.schedule import resolve_scheduled_p_mask


class DiffusionObjective(Objective):
    def __init__(self, cfg, tokenizer) -> None:
        super().__init__("diffusion")
        self._tokenizer = tokenizer
        self.mask_token_id = int(getattr(cfg, "mask_token_id", cfg.vocab_size - 1))
        self.noise_epsilon = float(getattr(cfg, "noise_epsilon", 1e-3))
        self.random_trunc_prob = float(getattr(cfg, "random_trunc_prob", 0.01))
        self.p_mask_override = getattr(cfg, "p_mask_override", None)
        self.deterministic_mask = bool(getattr(cfg, "deterministic_mask", False))
        self.p_mask_bucket_edges = getattr(cfg, "p_mask_bucket_edges", None)
        self.null_label_id = getattr(cfg, "null_label_id", None)
        self.uncond_label_dropout_prob = float(getattr(cfg, "uncond_label_dropout_prob", 0.0))
        if self.uncond_label_dropout_prob > 0 and self.null_label_id is None:
            raise ValueError("uncond_label_dropout_prob > 0 requires null_label_id")
        self.p_mask_schedule = str(getattr(cfg, "p_mask_schedule", "none")).lower()
        self.p_mask_start = getattr(cfg, "p_mask_start", None)
        self.p_mask_end = getattr(cfg, "p_mask_end", None)
        self.p_mask_schedule_start = float(getattr(cfg, "p_mask_schedule_start", 0.0))
        self.p_mask_schedule_end = float(getattr(cfg, "p_mask_schedule_end", 1.0))
        self.total_steps = int(getattr(cfg, "max_train_iteration", 0))
        self._step = 0

    def on_step(self, step: int, max_steps: int | None, is_train: bool) -> None:
        if is_train:
            self._step = int(step)

    def _scheduled_p_mask(self):
        return resolve_scheduled_p_mask(
            step=self._step,
            total_steps=self.total_steps,
            override=self.p_mask_override,
            schedule=self.p_mask_schedule,
            start_value=self.p_mask_start,
            end_value=self.p_mask_end,
            schedule_start=self.p_mask_schedule_start,
            schedule_end=self.p_mask_schedule_end,
        )

    def get_batch(self, *, dataset, batch_size: int, context_length: int, device: str, generator=None):
        batch = get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
            mask_token_id=self.mask_token_id,
            noise_epsilon=self.noise_epsilon,
            random_trunc_prob=self.random_trunc_prob,
            p_mask_override=self._scheduled_p_mask(),
            deterministic_mask=self.deterministic_mask,
            generator=generator,
        )
        labels = getattr(batch, "labels", None)
        if labels is not None and self.uncond_label_dropout_prob > 0:
            keep = torch.rand(
                labels.shape,
                device=labels.device,
                generator=generator,
            ) >= self.uncond_label_dropout_prob
            null_labels = torch.full_like(labels, int(self.null_label_id))
            batch.labels = torch.where(keep, labels, null_labels)
        return batch

    def model_inputs(self, batch: DiffusionBatch):
        labels = getattr(batch, "labels", None)
        if labels is None:
            return batch.noisy_inputs
        return batch.noisy_inputs, labels

    def attention_mask(self, batch: DiffusionBatch):
        return batch.attention_mask

    def compute_loss(self, logits: torch.Tensor, batch: DiffusionBatch) -> torch.Tensor:
        return diffusion_cross_entropy(
            logits,
            batch.clean_targets,
            batch.mask,
            batch.p_mask,
            loss_mask=batch.loss_mask,
        )

    def extra_metrics(self, logits: torch.Tensor, batch: DiffusionBatch, reduce_metric):
        edges = self.p_mask_bucket_edges or [i / 10.0 for i in range(11)]
        p_mask = batch.p_mask
        mask = batch.mask
        targets = batch.clean_targets
        loss_mask = getattr(batch, "loss_mask", None)
        cleaned = sorted({float(e) for e in edges})
        if len(cleaned) < 2:
            cleaned = [0.0, 1.0]
        with torch.no_grad():
            per_token = cross_entropy(logits, targets, reduction="none")
            mask_f = mask.to(per_token.dtype)
            if loss_mask is not None:
                loss_mask_f = loss_mask.to(per_token.dtype)
                mask_f = mask_f * loss_mask_f
            else:
                loss_mask_f = None
            weighted = (per_token * mask_f) / p_mask
            if loss_mask_f is not None:
                denom = loss_mask_f.sum(dim=1)
            else:
                denom = torch.full(
                    (targets.shape[0],),
                    targets.shape[1],
                    device=per_token.device,
                    dtype=per_token.dtype,
                )
            per_example_loss = weighted.sum(dim=1) / denom.clamp_min(1)
            p_mask_vals = p_mask.view(-1)
            if len(cleaned) > 2:
                boundaries = torch.tensor(cleaned[1:-1], device=p_mask_vals.device, dtype=p_mask_vals.dtype)
                bucket_ids = torch.bucketize(p_mask_vals, boundaries)
            else:
                bucket_ids = torch.zeros_like(p_mask_vals, dtype=torch.long)
            payload = {}
            for i in range(len(cleaned) - 1):
                in_bucket = bucket_ids == i
                count = int(in_bucket.sum().item())
                if count == 0:
                    continue
                mean_val = float(per_example_loss[in_bucket].mean().item())
                if reduce_metric is not None:
                    mean_val = float(reduce_metric(mean_val))
                label = f"{cleaned[i]:.2f}-{cleaned[i + 1]:.2f}"
                payload[f"metrics.p_mask_bucket_loss/{label}"] = mean_val
                payload[f"metrics.p_mask_bucket_count/{label}"] = count
            return payload if payload else None

    def val_samples(self, inputs: torch.Tensor, logits: torch.Tensor, batch: DiffusionBatch, max_samples: int):
        if max_samples <= 0:
            return None
        count = min(int(max_samples), int(inputs.shape[0]))
        if count <= 0:
            return None
        targets = batch.clean_targets
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


class FlowMatchingObjective(Objective):
    def __init__(self, cfg, tokenizer) -> None:
        super().__init__("flow")
        self._tokenizer = tokenizer
        self.pixel_bins = int(getattr(cfg, "pixel_bins", 256))
        self.random_trunc_prob = float(getattr(cfg, "random_trunc_prob", 0.0))
        self.null_label_id = getattr(cfg, "null_label_id", None)
        self.uncond_label_dropout_prob = float(getattr(cfg, "uncond_label_dropout_prob", 0.0))
        if self.uncond_label_dropout_prob > 0 and self.null_label_id is None:
            raise ValueError("uncond_label_dropout_prob > 0 requires null_label_id")

    def get_batch(self, *, dataset, batch_size: int, context_length: int, device: str, generator=None):
        batch = get_flow_matching_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
            pixel_bins=self.pixel_bins,
            random_trunc_prob=self.random_trunc_prob,
            generator=generator,
        )
        labels = getattr(batch, "labels", None)
        if labels is not None and self.uncond_label_dropout_prob > 0:
            keep = torch.rand(
                labels.shape,
                device=labels.device,
                generator=generator,
            ) >= self.uncond_label_dropout_prob
            null_labels = torch.full_like(labels, int(self.null_label_id))
            batch.labels = torch.where(keep, labels, null_labels)
        return batch

    def model_inputs(self, batch: FlowMatchingBatch):
        return batch.noisy_inputs

    def attention_mask(self, batch: FlowMatchingBatch):
        return None

    def forward_with_model(self, model: torch.nn.Module, batch: FlowMatchingBatch) -> Optional[dict]:
        labels = getattr(batch, "labels", None)
        if labels is None:
            raise ValueError("flow objective requires labels for class conditioning")
        preds = model(batch.noisy_inputs, batch.timesteps, context=labels)
        loss = self.compute_loss(preds, batch)
        return {"logits": preds, "loss": loss, "inputs": batch.noisy_inputs}

    def compute_loss(self, logits: torch.Tensor, batch: FlowMatchingBatch) -> torch.Tensor:
        target = batch.target_velocity
        sq = (logits - target).pow(2)
        loss_mask = getattr(batch, "loss_mask", None)
        if loss_mask is None:
            return sq.mean()
        mask = loss_mask.to(sq.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (sq * mask).sum() / denom

    def val_samples(self, inputs: torch.Tensor, logits: torch.Tensor, batch: FlowMatchingBatch, max_samples: int):
        return None

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)

    def generate(self, model, prompt_indices: torch.Tensor, **kwargs) -> torch.Tensor:
        context = kwargs.get("context")
        if context is None:
            raise ValueError("flow generation requires context labels")
        return flow_image_generate(
            model,
            prompt_indices,
            context=context,
            steps=int(kwargs.get("steps", 0)),
            cfg_scale=float(kwargs.get("cfg_scale", 0.0)),
            uncond_context=kwargs.get("uncond_context"),
            generator=kwargs.get("generator"),
        )


class MegaDlmDiffusionObjective(Objective):
    def __init__(self, cfg, tokenizer) -> None:
        super().__init__("megadlm-diffusion")
        self._tokenizer = tokenizer
        self.mask_token_id = int(getattr(cfg, "mask_token_id", cfg.vocab_size - 1))
        self.eot_token_id = getattr(cfg, "eot_token_id", None)
        if self.eot_token_id is not None:
            self.eot_token_id = int(self.eot_token_id)
        self.eot_mask_loss = bool(getattr(cfg, "eot_mask_loss", False))
        self.random_trunc_prob = float(getattr(cfg, "random_trunc_prob", 0.01))
        self.p_mask_bucket_edges = getattr(cfg, "p_mask_bucket_edges", None)
        self.null_label_id = getattr(cfg, "null_label_id", None)
        self.uncond_label_dropout_prob = float(getattr(cfg, "uncond_label_dropout_prob", 0.0))
        if self.uncond_label_dropout_prob > 0 and self.null_label_id is None:
            raise ValueError("uncond_label_dropout_prob > 0 requires null_label_id")
        self.p_mask_override = getattr(cfg, "p_mask_override", None)
        self.p_mask_schedule = str(getattr(cfg, "p_mask_schedule", "none")).lower()
        self.p_mask_start = getattr(cfg, "p_mask_start", None)
        self.p_mask_end = getattr(cfg, "p_mask_end", None)
        self.p_mask_schedule_start = float(getattr(cfg, "p_mask_schedule_start", 0.0))
        self.p_mask_schedule_end = float(getattr(cfg, "p_mask_schedule_end", 1.0))
        self.total_steps = int(getattr(cfg, "max_train_iteration", 0))
        self._step = 0

    def on_step(self, step: int, max_steps: int | None, is_train: bool) -> None:
        if is_train:
            self._step = int(step)

    def _scheduled_p_mask(self):
        return resolve_scheduled_p_mask(
            step=self._step,
            total_steps=self.total_steps,
            override=self.p_mask_override,
            schedule=self.p_mask_schedule,
            start_value=self.p_mask_start,
            end_value=self.p_mask_end,
            schedule_start=self.p_mask_schedule_start,
            schedule_end=self.p_mask_schedule_end,
        )

    def get_batch(self, *, dataset, batch_size: int, context_length: int, device: str, generator=None):
        batch = get_megadlm_diffusion_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
            mask_token_id=self.mask_token_id,
            eot_token_id=self.eot_token_id,
            eot_mask_loss=self.eot_mask_loss,
            random_trunc_prob=self.random_trunc_prob,
            p_mask_override=self._scheduled_p_mask(),
            generator=generator,
        )
        labels = getattr(batch, "labels", None)
        if labels is not None and self.uncond_label_dropout_prob > 0:
            keep = torch.rand(
                labels.shape,
                device=labels.device,
                generator=generator,
            ) >= self.uncond_label_dropout_prob
            null_labels = torch.full_like(labels, int(self.null_label_id))
            batch.labels = torch.where(keep, labels, null_labels)
        return batch

    def model_inputs(self, batch: DiffusionBatch) -> torch.Tensor:
        labels = getattr(batch, "labels", None)
        if labels is None:
            return batch.noisy_inputs
        return batch.noisy_inputs, labels

    def attention_mask(self, batch: DiffusionBatch):
        return batch.attention_mask

    def compute_loss(self, logits: torch.Tensor, batch: DiffusionBatch) -> torch.Tensor:
        return diffusion_cross_entropy(
            logits,
            batch.clean_targets,
            batch.mask,
            batch.p_mask,
            loss_mask=batch.loss_mask,
        )

    def extra_metrics(self, logits: torch.Tensor, batch: DiffusionBatch, reduce_metric):
        edges = self.p_mask_bucket_edges or [i / 10.0 for i in range(11)]
        p_mask = batch.p_mask
        mask = batch.mask
        targets = batch.clean_targets
        loss_mask = getattr(batch, "loss_mask", None)
        cleaned = sorted({float(e) for e in edges})
        if len(cleaned) < 2:
            cleaned = [0.0, 1.0]
        with torch.no_grad():
            per_token = cross_entropy(logits, targets, reduction="none")
            mask_f = mask.to(per_token.dtype)
            if loss_mask is not None:
                loss_mask_f = loss_mask.to(per_token.dtype)
                mask_f = mask_f * loss_mask_f
            else:
                loss_mask_f = None
            weighted = (per_token * mask_f) / p_mask
            if loss_mask_f is not None:
                denom = loss_mask_f.sum(dim=1)
            else:
                denom = torch.full(
                    (targets.shape[0],),
                    targets.shape[1],
                    device=per_token.device,
                    dtype=per_token.dtype,
                )
            per_example_loss = weighted.sum(dim=1) / denom.clamp_min(1)
            p_mask_vals = p_mask.view(-1)
            if len(cleaned) > 2:
                boundaries = torch.tensor(cleaned[1:-1], device=p_mask_vals.device, dtype=p_mask_vals.dtype)
                bucket_ids = torch.bucketize(p_mask_vals, boundaries)
            else:
                bucket_ids = torch.zeros_like(p_mask_vals, dtype=torch.long)
            payload = {}
            for i in range(len(cleaned) - 1):
                in_bucket = bucket_ids == i
                count = int(in_bucket.sum().item())
                if count == 0:
                    continue
                mean_val = float(per_example_loss[in_bucket].mean().item())
                if reduce_metric is not None:
                    mean_val = float(reduce_metric(mean_val))
                label = f"{cleaned[i]:.2f}-{cleaned[i + 1]:.2f}"
                payload[f"metrics.p_mask_bucket_loss/{label}"] = mean_val
                payload[f"metrics.p_mask_bucket_count/{label}"] = count
            return payload if payload else None

    def val_samples(self, inputs: torch.Tensor, logits: torch.Tensor, batch: DiffusionBatch, max_samples: int):
        if max_samples <= 0:
            return None
        count = min(int(max_samples), int(inputs.shape[0]))
        if count <= 0:
            return None
        targets = batch.clean_targets
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


__all__ = ["DiffusionObjective", "MegaDlmDiffusionObjective"]
