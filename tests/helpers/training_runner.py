from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import tempfile

import numpy as np
import random
import torch
import torch.distributed as dist

from leash.objectives import Objective
from leash.training.loop import train_loop
from leash.objectives.data import get_batch, DiffusionBatch
from leash.training.grad import gradient_clipping
from leash.objectives.loss import diffusion_cross_entropy
from leash.training.schedule import lr_cosine_schedule
from leash.training.optim import AdamW
from leash.checkpointing import (
    CheckpointCoordinator,
    load_manifest,
    load_model_from_manifest,
    load_optimizer_shard,
    load_rng_state,
)
from leash.checkpointing.state import restore_rng_state

from leash.ddp import DDP, OptimizerStateSharding
from leash.ddp.utils import setup_process_group, cleanup_process_group, allreduce_mean

from tests.fixtures import TrainingBundle


@dataclass(frozen=True)
class TrainingStepSnapshot:
    step: int
    loss: float
    parameter_tensors: List[Tuple[str, torch.Tensor]]
    gradient_tensors: List[Tuple[str, torch.Tensor]]
    gradient_norms: Dict[str, float]
    optimizer_state: Dict[str, Any]
    learning_rate: float


def _clone_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    cloned: Dict[str, Any] = {"state": {}, "param_groups": []}
    for idx, state in state_dict.get("state", {}).items():
        cloned_state: Dict[str, Any] = {}
        for key, value in state.items():
            if torch.is_tensor(value):
                cloned_state[key] = value.detach().cpu().clone()
            else:
                cloned_state[key] = value
        cloned["state"][idx] = cloned_state

    for group in state_dict.get("param_groups", []):
        cloned_group = dict(group)
        cloned_group["params"] = tuple(group.get("params", ()))
        cloned["param_groups"].append(cloned_group)

    return cloned


def _canonical_name(name: str) -> str:
    return name.split(".", 1)[-1] if name.startswith("model.") else name


def _set_all_seeds(seed: int, device: torch.device) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _build_checkpoint_coordinator(run_dir: Path, *, rank: int, world_size: int) -> CheckpointCoordinator:
    run_id = run_dir.name or "run"
    coordinator = CheckpointCoordinator(
        run_dir=run_dir,
        runs_root_parent=run_dir.parent,
        run_id=run_id,
        config_src_path=Path(""),
        config_snapshot={"test": True},
        best_metric_name="val_loss",
        best_mode="min",
        s3_cfg=None,
        rank=rank,
        world_size=world_size,
    )
    coordinator.prepare_run()
    return coordinator


def _checkpoint_manifest_path(run_dir: Path, step_idx: int) -> Path:
    version_id = f"v{int(step_idx):06d}"
    return run_dir / "versions" / version_id / "manifest.json"


class InMemoryStreamingBatcher:
    """Simple in-memory token stream used for tests."""

    def __init__(self, tokens: torch.Tensor, device: torch.device) -> None:
        flat = tokens.detach().view(-1).cpu().to(torch.long)
        if flat.numel() == 0:
            raise ValueError("Token buffer must be non-empty for streaming batcher tests.")
        self.tokens = flat
        self.device = device
        self._idx = 0

    def _next_token(self) -> int:
        token = int(self.tokens[self._idx].item())
        self._idx = (self._idx + 1) % self.tokens.numel()
        return token

    def draw(self, batch_size: int, context_length: int) -> torch.Tensor:
        sequences = []
        for _ in range(batch_size):
            seq = [self._next_token() for _ in range(context_length)]
            sequences.append(seq)
        return torch.tensor(sequences, dtype=torch.long, device=self.device)

    def get_state(self) -> int:
        return int(self._idx)

    def set_state(self, state: int) -> None:
        self._idx = int(state) % self.tokens.numel()


def _prepare_streaming_batchers(bundle: TrainingBundle, device: torch.device) -> Tuple[InMemoryStreamingBatcher, InMemoryStreamingBatcher]:
    train_batcher = InMemoryStreamingBatcher(bundle.dataset.train_tokens, device)
    valid_batcher = InMemoryStreamingBatcher(bundle.dataset.valid_tokens, device)
    return train_batcher, valid_batcher


def _make_step_callback(
    snapshots: List[TrainingStepSnapshot],
    hook: Optional[Callable[[int, torch.nn.Module, torch.optim.Optimizer], None]] = None,
) -> Callable[[int, torch.nn.Module, torch.optim.Optimizer, float, float], None]:
    def _callback(
        iteration: int,
        model_ref: torch.nn.Module,
        optimizer_ref: torch.optim.Optimizer,
        loss_value: float,
        lr_value: float,
    ) -> None:
        step_index = iteration

        param_tensors = [(_canonical_name(name), param.detach().cpu().clone()) for name, param in model_ref.named_parameters()]

        grad_tensors: List[Tuple[str, torch.Tensor]] = []
        grad_norms: Dict[str, float] = {}
        for name, param in model_ref.named_parameters():
            if param.grad is None:
                continue
            canonical = _canonical_name(name)
            grad_cpu = param.grad.detach().cpu().clone()
            grad_tensors.append((canonical, grad_cpu))
            grad_norms[canonical] = float(torch.linalg.vector_norm(param.grad.detach()).item())

        snapshots.append(TrainingStepSnapshot(
            step=step_index,
            loss=float(loss_value),
            parameter_tensors=param_tensors,
            gradient_tensors=grad_tensors,
            gradient_norms=grad_norms,
            optimizer_state=_clone_state_dict(optimizer_ref.state_dict()),
            learning_rate=float(lr_value),
        ))

        if hook is not None:
            hook(step_index, model_ref, optimizer_ref)

    return _callback


def _run_loop(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    bundle: TrainingBundle,
    generator: torch.Generator,
    train_batcher: InMemoryStreamingBatcher,
    valid_batcher: InMemoryStreamingBatcher,
    num_steps: int,
    base_iteration: int,
    snapshots: List[TrainingStepSnapshot],
    hook: Optional[Callable[[int, torch.nn.Module, torch.optim.Optimizer], None]] = None,
    sync_gradients: Optional[Callable[[], None]] = None,
    reduce_metric: Optional[Callable[[float], float]] = None,
    is_rank_zero: bool = True,
) -> None:
    if num_steps <= 0:
        return

    model_cfg = bundle.train_config.model
    optimizer_cfg = bundle.train_config.optimizer
    training_cfg = bundle.train_config.training

    step_callback = _make_step_callback(snapshots, hook)

    mask_token_id = getattr(model_cfg, "mask_token_id", model_cfg.vocab_size - 1)
    noise_epsilon = getattr(training_cfg, "noise_epsilon", 1e-3)
    random_trunc_prob = getattr(training_cfg, "random_trunc_prob", 0.01)

    class _DiffusionObjective(Objective):
        def __init__(self) -> None:
            super().__init__("diffusion")

        def get_batch(self, *, dataset, batch_size, context_length, device, generator=None):
            return get_batch(
                dataset=dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
                mask_token_id=mask_token_id,
                noise_epsilon=noise_epsilon,
                random_trunc_prob=random_trunc_prob,
                generator=generator,
            )

        def model_inputs(self, batch: DiffusionBatch) -> torch.Tensor:
            return batch.noisy_inputs

        def attention_mask(self, batch: DiffusionBatch):
            return batch.attention_mask

        def compute_loss(self, logits: torch.Tensor, batch: DiffusionBatch) -> torch.Tensor:
            return diffusion_cross_entropy(logits, batch.clean_targets, batch.mask, batch.p_mask)

    objective = _DiffusionObjective()

    train_loop(
        module,
        optimizer,
        train_data=train_batcher,
        val_data=valid_batcher,
        batch_size=training_cfg.batch_size,
        context_length=model_cfg.context_length,
        device=str(model_cfg.device),
        max_learning_rate=optimizer_cfg.max_learning_rate,
        min_learning_rate=optimizer_cfg.min_learning_rate,
        warmup_iters=optimizer_cfg.warmup_iters,
        cosine_cycle_iters=optimizer_cfg.cosine_cycle_iters,
        max_train_iteration=base_iteration + num_steps - 1,
        max_val_iteration=training_cfg.max_val_iteration,
        val_freq_iteration=training_cfg.val_freq_iteration,
        grad_clip_max_l2_norm=optimizer_cfg.grad_clip_max_l2_norm,
        ckpting_save_iter=bundle.train_config.checkpointing.ckpting_save_iter,
        ckpting_save_folder=None,
        lr_cosine_schedule=lr_cosine_schedule,
        gradient_clipping=gradient_clipping,
        objective=objective,
        batch_generator=generator,
        logger=None,
        activation_norms=None,
        log_activation_norms=False,
        log_weight_norms=False,
        sync_gradients=sync_gradients,
        reduce_metric=reduce_metric,
        is_rank_zero=is_rank_zero,
        step_callback=step_callback,
        start_iteration=base_iteration,
    )


def run_training_steps(bundle: TrainingBundle, *, num_steps: int, seed: int | None = None) -> List[TrainingStepSnapshot]:
    if num_steps <= 0:
        raise ValueError("num_steps must be > 0")

    model_cfg = bundle.train_config.model
    training_cfg = bundle.train_config.training

    device = torch.device(model_cfg.device)
    chosen_seed = seed if seed is not None else (training_cfg.seed or 0)
    generator = _set_all_seeds(chosen_seed, device)

    model = bundle.model_factory()
    optimizer = bundle.optimizer_factory(model.parameters())

    train_batcher, valid_batcher = _prepare_streaming_batchers(bundle, device)
    snapshots: List[TrainingStepSnapshot] = []

    _run_loop(
        module=model,
        optimizer=optimizer,
        bundle=bundle,
        generator=generator,
        train_batcher=train_batcher,
        valid_batcher=valid_batcher,
        num_steps=num_steps,
        base_iteration=0,
        snapshots=snapshots,
    )

    return snapshots


def run_training_steps_ddp(bundle: TrainingBundle, *, num_steps: int, seed: int | None = None) -> List[TrainingStepSnapshot]:
    if num_steps <= 0:
        raise ValueError("num_steps must be > 0")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    model_cfg = bundle.train_config.model
    training_cfg = bundle.train_config.training
    optimizer_cfg = bundle.train_config.optimizer

    device = torch.device(model_cfg.device)
    chosen_seed = seed if seed is not None else (training_cfg.seed or 0)
    generator = _set_all_seeds(chosen_seed, device)

    need_cleanup = False
    if not dist.is_initialized():
        setup_process_group(
            backend="gloo",
            local_rank=0,
            num_gpus_per_node=1,
            num_nodes=1,
            node_rank=0,
        )
        need_cleanup = True
    elif dist.get_world_size() != 1:
        raise RuntimeError("Existing process group world_size is not 1")

    try:
        model = bundle.model_factory()
        ddp_model = DDP(model, world_size=1, bucket_size_mb=0)
        ddp_model.broadcast_parameters(src=0)

        optimizer = OptimizerStateSharding(
            model.parameters(),
            AdamW,
            lr=optimizer_cfg.initial_learning_rate,
            betas=optimizer_cfg.betas,
            eps=float(optimizer_cfg.eps),
            weight_decay=optimizer_cfg.weight_decay,
        )

        train_batcher, valid_batcher = _prepare_streaming_batchers(bundle, device)
        snapshots: List[TrainingStepSnapshot] = []

        def _sync():
            ddp_model.finish_gradient_synchronization()

        _run_loop(
            module=ddp_model,
            optimizer=optimizer,
            bundle=bundle,
            generator=generator,
            train_batcher=train_batcher,
            valid_batcher=valid_batcher,
            num_steps=num_steps,
            base_iteration=0,
            snapshots=snapshots,
            sync_gradients=_sync,
            reduce_metric=allreduce_mean,
            is_rank_zero=True,
        )

        return snapshots
    finally:
        if need_cleanup and dist.is_initialized():
            cleanup_process_group()


def run_training_with_checkpoint(
    bundle: TrainingBundle,
    *,
    total_steps: int,
    checkpoint_step: int,
    seed: int | None = None,
) -> Tuple[List[TrainingStepSnapshot], List[TrainingStepSnapshot]]:
    if not (0 < checkpoint_step < total_steps):
        raise ValueError("checkpoint_step must be between 0 and total_steps")

    baseline = run_training_steps(bundle, num_steps=total_steps, seed=seed)

    model_cfg = bundle.train_config.model
    training_cfg = bundle.train_config.training

    device = torch.device(model_cfg.device)
    chosen_seed = seed if seed is not None else (training_cfg.seed or 0)

    generator = _set_all_seeds(chosen_seed, device)
    model = bundle.model_factory()
    optimizer = bundle.optimizer_factory(model.parameters())
    train_batcher, valid_batcher = _prepare_streaming_batchers(bundle, device)

    snapshots_resumed: List[TrainingStepSnapshot] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "runs" / "test-run"
        coordinator = _build_checkpoint_coordinator(run_dir, rank=0, world_size=1)
        coordinator.attach_state_sources(
            generator=generator,
            train_batcher=train_batcher,
            val_batcher=valid_batcher,
        )
        state_holder: Dict[str, Any] = {}

        def _hook(step_idx: int, model_ref: torch.nn.Module, optimizer_ref: torch.optim.Optimizer) -> None:
            if step_idx == checkpoint_step - 1 and "manifest_path" not in state_holder:
                coordinator.save_version(
                    step_idx,
                    model=model_ref,
                    optimizer=optimizer_ref,
                    metrics=None,
                    all_gather=None,
                )
                state_holder["manifest_path"] = _checkpoint_manifest_path(run_dir, step_idx)
                state_holder["iteration"] = step_idx

        _run_loop(
            module=model,
            optimizer=optimizer,
            bundle=bundle,
            generator=generator,
            train_batcher=train_batcher,
            valid_batcher=valid_batcher,
            num_steps=checkpoint_step,
            base_iteration=0,
            snapshots=snapshots_resumed,
            hook=_hook,
        )

        if "manifest_path" not in state_holder:
            raise RuntimeError("Checkpoint hook did not trigger")

        remaining = total_steps - checkpoint_step
        if remaining > 0:
            resume_base_iteration = int(state_holder["iteration"]) + 1
            generator_resume = torch.Generator(device="cpu")

            model_resume = bundle.model_factory()
            optimizer_resume = bundle.optimizer_factory(model_resume.parameters())
            manifest = load_manifest(state_holder["manifest_path"], root_parent=run_dir.parent)
            model_state = load_model_from_manifest(manifest, run_dir, root_parent=run_dir.parent)
            model_resume.load_state_dict(model_state)
            optimizer_state = load_optimizer_shard(
                manifest,
                run_dir,
                rank=0,
                map_location=str(device),
                root_parent=run_dir.parent,
            )
            optimizer_resume.load_state_dict(optimizer_state)
            rng_state = load_rng_state(manifest, run_dir, rank=0, root_parent=run_dir.parent)
            _ = restore_rng_state(rng_state, generator_resume)

            train_batcher_resume, valid_batcher_resume = _prepare_streaming_batchers(bundle, device)
            batchers = rng_state.get("batchers", {})
            if "train" in batchers:
                train_batcher_resume.set_state(batchers["train"])
            if "val" in batchers:
                valid_batcher_resume.set_state(batchers["val"])

            _run_loop(
                module=model_resume,
                optimizer=optimizer_resume,
                bundle=bundle,
                generator=generator_resume,
                train_batcher=train_batcher_resume,
                valid_batcher=valid_batcher_resume,
                num_steps=remaining,
                base_iteration=resume_base_iteration,
                snapshots=snapshots_resumed,
            )

    return baseline, snapshots_resumed


def run_training_with_checkpoint_ddp(
    bundle: TrainingBundle,
    *,
    total_steps: int,
    checkpoint_step: int,
    seed: int | None = None,
) -> Tuple[List[TrainingStepSnapshot], List[TrainingStepSnapshot]]:
    if not (0 < checkpoint_step < total_steps):
        raise ValueError("checkpoint_step must be between 0 and total_steps")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    baseline = run_training_steps_ddp(bundle, num_steps=total_steps, seed=seed)

    model_cfg = bundle.train_config.model
    training_cfg = bundle.train_config.training
    optimizer_cfg = bundle.train_config.optimizer

    device = torch.device(model_cfg.device)
    chosen_seed = seed if seed is not None else (training_cfg.seed or 0)

    need_cleanup = False
    if not dist.is_initialized():
        setup_process_group(
            backend="gloo",
            local_rank=0,
            num_gpus_per_node=1,
            num_nodes=1,
            node_rank=0,
        )
        need_cleanup = True
    elif dist.get_world_size() != 1:
        raise RuntimeError("Existing process group world_size is not 1")

    try:
        generator = _set_all_seeds(chosen_seed, device)
        model = bundle.model_factory()
        ddp_model = DDP(model, world_size=1, bucket_size_mb=0)
        ddp_model.broadcast_parameters(src=0)

        optimizer = OptimizerStateSharding(
            model.parameters(),
            AdamW,
            lr=optimizer_cfg.initial_learning_rate,
            betas=optimizer_cfg.betas,
            eps=float(optimizer_cfg.eps),
            weight_decay=optimizer_cfg.weight_decay,
        )

        train_batcher, valid_batcher = _prepare_streaming_batchers(bundle, device)
        snapshots_resumed: List[TrainingStepSnapshot] = []

        def _sync():
            ddp_model.finish_gradient_synchronization()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "test-run"
            coordinator = _build_checkpoint_coordinator(run_dir, rank=0, world_size=1)
            coordinator.attach_state_sources(
                generator=generator,
                train_batcher=train_batcher,
                val_batcher=valid_batcher,
            )
            state_holder: Dict[str, Any] = {}

            def _hook(step_idx: int, model_ref: torch.nn.Module, optimizer_ref: torch.optim.Optimizer) -> None:
                if step_idx == checkpoint_step - 1 and "manifest_path" not in state_holder:
                    coordinator.save_version(
                        step_idx,
                        model=model_ref,
                        optimizer=optimizer_ref,
                        metrics=None,
                        all_gather=None,
                    )
                    state_holder["manifest_path"] = _checkpoint_manifest_path(run_dir, step_idx)
                    state_holder["iteration"] = step_idx

            _run_loop(
                module=ddp_model,
                optimizer=optimizer,
                bundle=bundle,
                generator=generator,
                train_batcher=train_batcher,
                valid_batcher=valid_batcher,
                num_steps=checkpoint_step,
                base_iteration=0,
                snapshots=snapshots_resumed,
                hook=_hook,
                sync_gradients=_sync,
                reduce_metric=allreduce_mean,
                is_rank_zero=True,
            )

            if "manifest_path" not in state_holder:
                raise RuntimeError("Checkpoint hook did not trigger")

            remaining = total_steps - checkpoint_step
            if remaining > 0:
                resume_base_iteration = int(state_holder["iteration"]) + 1
                generator_resume = torch.Generator(device="cpu")

                model_resume = bundle.model_factory()
                ddp_model_resume = DDP(model_resume, world_size=1, bucket_size_mb=0)
                ddp_model_resume.broadcast_parameters(src=0)

                optimizer_resume = OptimizerStateSharding(
                    model_resume.parameters(),
                    AdamW,
                    lr=optimizer_cfg.initial_learning_rate,
                    betas=optimizer_cfg.betas,
                    eps=float(optimizer_cfg.eps),
                    weight_decay=optimizer_cfg.weight_decay,
                )
                manifest = load_manifest(state_holder["manifest_path"], root_parent=run_dir.parent)
                model_state = load_model_from_manifest(manifest, run_dir, root_parent=run_dir.parent)
                ddp_model_resume.load_state_dict(model_state)
                optimizer_state = load_optimizer_shard(
                    manifest,
                    run_dir,
                    rank=0,
                    map_location=str(device),
                    root_parent=run_dir.parent,
                )
                optimizer_resume.load_state_dict(optimizer_state)
                rng_state = load_rng_state(manifest, run_dir, rank=0, root_parent=run_dir.parent)
                _ = restore_rng_state(rng_state, generator_resume)

                train_batcher_resume, valid_batcher_resume = _prepare_streaming_batchers(bundle, device)
                batchers = rng_state.get("batchers", {})
                if "train" in batchers:
                    train_batcher_resume.set_state(batchers["train"])
                if "val" in batchers:
                    valid_batcher_resume.set_state(batchers["val"])

                def _sync_resume():
                    ddp_model_resume.finish_gradient_synchronization()

                _run_loop(
                    module=ddp_model_resume,
                    optimizer=optimizer_resume,
                    bundle=bundle,
                    generator=generator_resume,
                    train_batcher=train_batcher_resume,
                    valid_batcher=valid_batcher_resume,
                    num_steps=remaining,
                    base_iteration=resume_base_iteration,
                    snapshots=snapshots_resumed,
                    sync_gradients=_sync_resume,
                    reduce_metric=allreduce_mean,
                    is_rank_zero=True,
                )

        return baseline, snapshots_resumed
    finally:
        if need_cleanup and dist.is_initialized():
            cleanup_process_group()


def run_single_step(bundle: TrainingBundle, *, seed: int | None = None) -> TrainingStepSnapshot:
    return run_training_steps(bundle, num_steps=1, seed=seed)[0]
