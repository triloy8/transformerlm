from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Optional

import torch

from .manifest import (
    CheckpointCoordinator,
    resolve_checkpoint_reference,
    load_manifest,
    load_model_from_manifest,
    load_optimizer_shard,
    load_scaler_shard,
    load_rng_state,
)
from .state import restore_rng_state
from .storage import S3ConfigData, S3Uploader


class CheckpointManager:
    def __init__(
        self,
        *,
        checkpointing_cfg: Any,
        runs_path: Path,
        run_name: str,
        rank: int,
        world_size: int,
        config_path: Path,
        config_snapshot: dict,
    ) -> None:
        self._cfg = checkpointing_cfg
        self._runs_path = runs_path
        self._run_name = run_name
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._config_path = config_path
        self._config_snapshot = config_snapshot

        self.enabled = bool(getattr(checkpointing_cfg, "enabled", True)) if checkpointing_cfg else False
        self.resume_from = getattr(checkpointing_cfg, "resume_from", None) if checkpointing_cfg else None
        self.resume_run_id = getattr(checkpointing_cfg, "run_id", None) if checkpointing_cfg else None
        self.resume_optimizer = bool(getattr(checkpointing_cfg, "resume_optimizer", True)) if checkpointing_cfg else True
        self.best_metric_name = getattr(checkpointing_cfg, "best_metric_name", "val_loss") if checkpointing_cfg else "val_loss"
        self.best_mode = getattr(checkpointing_cfg, "best_mode", "min") if checkpointing_cfg else "min"

        self.run_id = self._run_name
        self.run_dir = self._runs_path / self.run_id
        self.resume_run_dir = self._resolve_resume_run_dir()

        self._s3_cfg = self._build_s3_config()
        self._s3_uploader = S3Uploader(self._s3_cfg) if self._s3_cfg is not None else None

        self.coordinator = CheckpointCoordinator(
            run_dir=self.run_dir,
            runs_root_parent=self._runs_path.parent,
            run_id=self.run_id,
            config_src_path=self._config_path,
            config_snapshot=self._config_snapshot,
            best_metric_name=self.best_metric_name,
            best_mode=self.best_mode,
            s3_cfg=self._s3_cfg,
            rank=self._rank,
            world_size=self._world_size,
        )

    def _build_s3_config(self) -> Optional[S3ConfigData]:
        def _env(name: str) -> Optional[str]:
            value = os.environ.get(name)
            if value is None:
                return None
            value = value.strip()
            return value or None

        env_bucket = _env("CHECKPOINTING_S3_BUCKET")
        env_prefix = _env("CHECKPOINTING_S3_PREFIX")
        env_endpoint = _env("CHECKPOINTING_S3_ENDPOINT_URL")
        env_region = _env("CHECKPOINTING_S3_REGION")
        env_access = _env("CHECKPOINTING_S3_ACCESS_KEY_ID")
        env_secret = _env("CHECKPOINTING_S3_SECRET_ACCESS_KEY")
        env_token = _env("CHECKPOINTING_S3_SESSION_TOKEN")

        bucket = env_bucket
        if bucket is None:
            return None

        return S3ConfigData(
            bucket=bucket,
            prefix=env_prefix if env_prefix is not None else "",
            endpoint_url=env_endpoint,
            region_name=env_region,
            access_key_id=env_access,
            secret_access_key=env_secret,
            session_token=env_token,
        )

    def _resolve_resume_run_dir(self) -> Path:
        resume_from = self.resume_from
        if not resume_from:
            return self.run_dir
        if resume_from in {"latest", "best"}:
            if self.resume_run_id is None:
                raise ValueError("checkpointing.run_id must be set when resume_from is an alias")
            return self._runs_path / self.resume_run_id
        ref_path = Path(resume_from)
        if ref_path.exists():
            if ref_path.is_dir():
                ref_path = ref_path / "manifest.json"
            if ref_path.name == "manifest.json" and ref_path.parent.name == "versions":
                return ref_path.parent.parent
            if "runs" in ref_path.parts:
                idx = ref_path.parts.index("runs")
                if idx + 1 < len(ref_path.parts):
                    return Path(*ref_path.parts[: idx + 2])
        return self.run_dir

    def prepare_run(self, generator: Optional[torch.Generator]) -> None:
        self.coordinator.attach_state_sources(generator=generator)
        self.coordinator.prepare_run()

    def attach_batchers(
        self,
        *,
        generator: Optional[torch.Generator],
        train_batcher: Optional[Any],
        val_batcher: Optional[Any],
    ) -> None:
        self.coordinator.attach_state_sources(
            generator=generator,
            train_batcher=train_batcher,
            val_batcher=val_batcher,
        )

    def maybe_resume(
        self,
        *,
        ddp_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler] = None,
        train_batcher: Optional[Any],
        val_batcher: Optional[Any],
        generator: Optional[torch.Generator],
        device: str,
    ) -> int:
        if not (self.enabled and self.resume_from):
            ddp_model.broadcast_parameters(src=0)
            return 0

        manifest_path = resolve_checkpoint_reference(
            self.resume_run_dir,
            self.resume_from,
            s3=self._s3_uploader,
            root_parent=self._runs_path.parent,
        )
        manifest = load_manifest(manifest_path, root_parent=self._runs_path.parent, s3=self._s3_uploader)

        if self._rank == 0:
            model_state = load_model_from_manifest(
                manifest,
                self.resume_run_dir,
                root_parent=self._runs_path.parent,
                s3=self._s3_uploader,
            )
            ddp_model.load_state_dict(model_state)
        ddp_model.broadcast_parameters(src=0)

        if self.resume_optimizer:
            optimizer_state = load_optimizer_shard(
                manifest,
                self.resume_run_dir,
                self._rank,
                map_location=device,
                root_parent=self._runs_path.parent,
                s3=self._s3_uploader,
            )
            optimizer.load_state_dict(optimizer_state)
            if scaler is not None:
                scaler_state = load_scaler_shard(
                    manifest,
                    self.resume_run_dir,
                    self._rank,
                    map_location=device,
                    root_parent=self._runs_path.parent,
                    s3=self._s3_uploader,
                )
                if scaler_state is not None:
                    scaler.load_state_dict(scaler_state)

        rng_state = load_rng_state(
            manifest,
            self.resume_run_dir,
            self._rank,
            root_parent=self._runs_path.parent,
            s3=self._s3_uploader,
        )
        _ = restore_rng_state(rng_state, generator)

        batchers = rng_state.get("batchers", {})
        if train_batcher is not None and "train" in batchers and hasattr(train_batcher, "set_state"):
            train_batcher.set_state(batchers["train"])
        if val_batcher is not None and "val" in batchers and hasattr(val_batcher, "set_state"):
            val_batcher.set_state(batchers["val"])

        return int(manifest.get("resume", {}).get("base_step", manifest.get("step", 0) + 1))

    def make_checkpoint_callback(self) -> Optional[callable]:
        if not self.enabled:
            return None

        def _gather_objects(payload):
            if self._world_size <= 1:
                return [payload]
            gathered = [None for _ in range(self._world_size)]
            torch.distributed.all_gather_object(gathered, payload)
            return gathered

        def _checkpoint_callback(step_idx, module, opt, metrics, amp_scaler):
            self.coordinator.save_version(
                step_idx,
                model=module,
                optimizer=opt,
                scaler=amp_scaler,
                metrics=metrics,
                all_gather=_gather_objects,
            )

        return _checkpoint_callback
