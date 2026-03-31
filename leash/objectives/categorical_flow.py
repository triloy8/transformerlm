from __future__ import annotations

from typing import Optional

import torch

from leash.inference.generate import categorical_flow_image_generate
from leash.objectives.base import Objective
from leash.objectives.data import CategoricalFlowBatch, get_categorical_flow_batch
from leash.objectives.loss import cross_entropy


def _masked_mean(values: torch.Tensor, loss_mask: torch.Tensor | None) -> torch.Tensor:
    if loss_mask is None:
        return values.mean()
    weights = loss_mask.to(values.dtype)
    denom = weights.sum().clamp_min(1.0)
    return (values * weights).sum() / denom


def _soft_cross_entropy(student_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    return -(teacher_probs * student_log_probs).sum(dim=-1)


def _subset(value, index):
    if value is None:
        return None
    return value[index]


def _finite_difference_time_derivative(fn, t: torch.Tensor, *, eps: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    t_plus = (t + eps).clamp_max(1.0)
    t_minus = (t - eps).clamp_min(0.0)
    y_plus = fn(t_plus)
    y_minus = fn(t_minus)
    denom = (t_plus - t_minus).clamp_min(1e-6).unsqueeze(-1)
    y_mid = fn(t)
    return y_mid, (y_plus - y_minus) / denom


def _interpolate(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if x0.ndim != x1.ndim + 1:
        raise ValueError("x0 must have exactly one more dimension than x1")
    if t.ndim != x0.ndim:
        raise ValueError("t must be broadcast-shaped like x0")
    xt = x0 * (1.0 - t)
    xt.scatter_add_(-1, x1[..., None], t.expand_as(xt[..., :1]))
    return xt


class CategoricalFlowObjective(Objective):
    def __init__(self, cfg, tokenizer) -> None:
        super().__init__("categorical-flow")
        self._tokenizer = tokenizer
        self.vocab_size = int(getattr(cfg, "vocab_size"))
        self.random_trunc_prob = float(getattr(cfg, "random_trunc_prob", 0.0))
        self.null_label_id = getattr(cfg, "null_label_id", None)
        self.uncond_label_dropout_prob = float(getattr(cfg, "uncond_label_dropout_prob", 0.0))
        self.prior_type = str(getattr(cfg, "categorical_flow_prior_type", "discunif")).lower()
        self.diag_fraction = float(getattr(cfg, "categorical_flow_diag_fraction", 0.5))
        self.inf_weight = float(getattr(cfg, "categorical_flow_inf_weight", 1.0))
        self.ec_weight = float(getattr(cfg, "categorical_flow_ec_weight", 1.0))
        self.td_weight = float(getattr(cfg, "categorical_flow_td_weight", 1.0))
        if self.uncond_label_dropout_prob > 0 and self.null_label_id is None:
            raise ValueError("uncond_label_dropout_prob > 0 requires null_label_id")
        if self.prior_type not in {"discunif", "uniform"}:
            raise ValueError("categorical_flow_prior_type must be one of: discunif, uniform")
        if not (0.0 <= self.diag_fraction <= 1.0):
            raise ValueError("categorical_flow_diag_fraction must be in [0, 1]")
        if self.inf_weight < 0:
            raise ValueError("categorical_flow_inf_weight must be >= 0")
        if self.ec_weight < 0:
            raise ValueError("categorical_flow_ec_weight must be >= 0")
        if self.td_weight < 0:
            raise ValueError("categorical_flow_td_weight must be >= 0")

    def get_batch(self, *, dataset, batch_size: int, context_length: int, device: str, generator=None):
        batch = get_categorical_flow_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
            vocab_size=self.vocab_size,
            prior_type=self.prior_type,
            random_trunc_prob=self.random_trunc_prob,
            generator=generator,
        )
        labels = getattr(batch, "labels", None)
        if labels is not None and self.uncond_label_dropout_prob > 0:
            keep = torch.rand(labels.shape, device=labels.device, generator=generator) >= self.uncond_label_dropout_prob
            null_labels = torch.full_like(labels, int(self.null_label_id))
            batch.labels = torch.where(keep, labels, null_labels)
        return batch

    def model_inputs(self, batch: CategoricalFlowBatch):
        return batch.x0_prior

    def attention_mask(self, batch: CategoricalFlowBatch):
        return None

    def compute_loss(self, logits: torch.Tensor, batch: CategoricalFlowBatch) -> torch.Tensor:
        del logits
        raise RuntimeError("CategoricalFlowObjective requires forward_with_model")

    def forward_with_model(self, model: torch.nn.Module, batch: CategoricalFlowBatch) -> Optional[dict]:
        labels = getattr(batch, "labels", None)
        if labels is None:
            raise ValueError("categorical flow objective requires labels for class conditioning")

        batch_size = batch.clean_targets.shape[0]
        diag_count = int(self.diag_fraction * batch_size)
        diag_count = max(0, min(diag_count, batch_size))
        if diag_count == 0 and self.inf_weight > 0:
            diag_count = min(1, batch_size)
        if diag_count == batch_size and (self.ec_weight > 0 or self.td_weight > 0):
            diag_count = max(0, batch_size - 1)

        diag_index = slice(0, diag_count)
        off_index = slice(diag_count, batch_size)

        inf_logits = None
        inf_loss = batch.x0_prior.new_zeros(())
        if diag_count > 0:
            x0_diag = _subset(batch.x0_prior, diag_index)
            t_diag = _subset(batch.t_timesteps, diag_index)
            targets_diag = _subset(batch.clean_targets, diag_index)
            labels_diag = _subset(labels, diag_index)
            loss_mask_diag = _subset(batch.loss_mask, diag_index)
            x_t = _interpolate(x0_diag, targets_diag, t_diag.unsqueeze(-1))
            inf_logits = model(x_t, t_diag, t_diag, context=labels_diag)
            inf_loss = _masked_mean(cross_entropy(inf_logits, targets_diag, reduction="none"), loss_mask_diag)

        ec_loss = batch.x0_prior.new_zeros(())
        td_loss = batch.x0_prior.new_zeros(())
        logged_inputs = x_t if diag_count > 0 else None
        if diag_count < batch_size:
            x0_off = _subset(batch.x0_prior, off_index)
            s = _subset(batch.s_timesteps, off_index)
            t = _subset(batch.t_timesteps, off_index)
            targets_off = _subset(batch.clean_targets, off_index)
            labels_off = _subset(labels, off_index)
            loss_mask_off = _subset(batch.loss_mask, off_index)
            x_s = _interpolate(x0_off, targets_off, s.unsqueeze(-1))

            def _pi_probs(t_in: torch.Tensor) -> torch.Tensor:
                logits = model(x_s, s, t_in, context=labels_off, jvp_attention=True)
                return torch.softmax(logits, dim=-1)

            tangent = torch.ones_like(t)
            try:
                if hasattr(torch, "func") and hasattr(torch.func, "jvp"):
                    student_probs, d_pi_dt = torch.func.jvp(_pi_probs, (t,), (tangent,))
                else:  # pragma: no cover
                    student_probs, d_pi_dt = torch.autograd.functional.jvp(
                        _pi_probs, (t,), (tangent,), create_graph=False
                    )
            except NotImplementedError:
                student_probs, d_pi_dt = _finite_difference_time_derivative(_pi_probs, t)

            student_logits = torch.log(student_probs.clamp_min(1e-8))
            gamma = ((t - s) / (1.0 - s).clamp_min(1e-6)).unsqueeze(-1)
            transported = x_s + gamma * (student_probs - x_s)
            transported = transported.clamp_min(0.0)
            transported = transported / transported.sum(dim=-1, keepdim=True).clamp_min(1e-6)

            with torch.no_grad():
                teacher_logits = model(transported, t, t, context=labels_off)
                teacher_probs = torch.softmax(teacher_logits, dim=-1)

            ec_per_token = _soft_cross_entropy(student_logits, teacher_probs)
            ec_loss = _masked_mean(ec_per_token, loss_mask_off)

            td_per_token = gamma.squeeze(-1) * d_pi_dt.pow(2).sum(dim=-1)
            td_loss = _masked_mean(td_per_token, loss_mask_off)

            if inf_logits is None:
                inf_logits = teacher_logits
            if logged_inputs is None:
                logged_inputs = x_s

        total_loss = self.inf_weight * inf_loss + self.ec_weight * ec_loss + self.td_weight * td_loss

        return {
            "loss": total_loss,
            "logits": inf_logits,
            "inputs": logged_inputs,
            "metrics": {
                "metrics.train_loss_categorical_flow_inf": float(inf_loss.detach().item()),
                "metrics.train_loss_categorical_flow_ec": float(ec_loss.detach().item()),
                "metrics.train_loss_categorical_flow_td": float(td_loss.detach().item()),
                "metrics.train_loss_categorical_flow_total": float(total_loss.detach().item()),
                "metrics.train_categorical_flow_diag_fraction": float(diag_count / max(batch_size, 1)),
            },
        }

    def val_samples(self, inputs: torch.Tensor, logits: torch.Tensor, batch: CategoricalFlowBatch, max_samples: int):
        if max_samples <= 0:
            return None
        count = min(int(max_samples), int(inputs.shape[0]))
        if count <= 0:
            return None
        targets = batch.clean_targets
        inputs_list = inputs[:count].argmax(dim=-1).detach().cpu().tolist()
        preds_list = logits[:count].argmax(dim=-1).detach().cpu().tolist()
        targets_list = targets[:count].detach().cpu().tolist()
        return [{"inputs": inputs_list[i], "predictions": preds_list[i], "targets": targets_list[i]} for i in range(count)]

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)

    def generate(self, model, prompt_indices: torch.Tensor, **kwargs) -> torch.Tensor:
        context = kwargs.get("context")
        if context is None:
            raise ValueError("categorical flow generation requires context labels")
        return categorical_flow_image_generate(
            model,
            prompt_indices,
            context=context,
            steps=int(kwargs.get("steps", 0)),
            temperature=float(kwargs.get("temperature", 0.0)),
            top_p=kwargs.get("top_p"),
            cfg_scale=float(kwargs.get("cfg_scale", 0.0)),
            uncond_context=kwargs.get("uncond_context"),
            generator=kwargs.get("generator"),
        )


__all__ = ["CategoricalFlowObjective"]
