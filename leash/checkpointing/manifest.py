from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import json
from datetime import datetime, timezone

import torch

from .storage import (
    S3ConfigData,
    S3Uploader,
    ensure_dir,
    ensure_local,
    file_info,
    path_to_key,
)
from .state import capture_rng_state, jsonable


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


class CheckpointCoordinator:
    def __init__(
        self,
        *,
        run_dir: Path,
        runs_root_parent: Path,
        run_id: str,
        config_src_path: Path,
        config_snapshot: Dict[str, Any],
        best_metric_name: str,
        best_mode: str,
        s3_cfg: Optional[S3ConfigData],
        rank: int,
        world_size: int,
    ) -> None:
        self.run_dir = run_dir
        self.runs_root_parent = runs_root_parent
        self.run_id = run_id
        self.best_metric_name = best_metric_name
        self.best_mode = best_mode
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._s3 = S3Uploader(s3_cfg) if s3_cfg is not None else None
        self._config_src_path = config_src_path
        self._config_snapshot = config_snapshot
        self._run_manifest_path = self.run_dir / "manifest.json"
        self._aliases_dir = self.run_dir / "aliases"
        self._versions_dir = self.run_dir / "versions"
        self._config_dir = self.run_dir / "config"
        self._config_info: Optional[Dict[str, Any]] = None
        self._best_alias: Optional[Dict[str, Any]] = None
        self._state_sources: Dict[str, Any] = {}

    def attach_state_sources(
        self,
        *,
        generator: Optional[torch.Generator] = None,
        train_batcher: Optional[Any] = None,
        val_batcher: Optional[Any] = None,
    ) -> None:
        self._state_sources["generator"] = generator
        self._state_sources["train_batcher"] = train_batcher
        self._state_sources["val_batcher"] = val_batcher

    def prepare_run(self) -> None:
        if self.rank != 0:
            return
        ensure_dir(self._config_dir)
        ensure_dir(self._aliases_dir)
        ensure_dir(self._versions_dir)

        config_dst = self._config_dir / "train.toml"
        if self._config_src_path.is_file():
            config_dst.write_text(self._config_src_path.read_text(encoding="utf-8"), encoding="utf-8")
        config_json_path = self._config_dir / "config.json"
        save_json(config_json_path, self._config_snapshot)

        if config_dst.exists():
            self._config_info = file_info(config_dst, self.runs_root_parent)
        else:
            self._config_info = {
                "key": path_to_key(config_dst, self.runs_root_parent),
                "sha256": "",
                "bytes": 0,
            }

        if self._s3 is not None:
            self._s3.upload(config_dst, self._config_info["key"])
            self._s3.upload(config_json_path, path_to_key(config_json_path, self.runs_root_parent))

        if not self._run_manifest_path.exists():
            run_manifest = {
                "schema_version": 1,
                "run_id": self.run_id,
                "created_at": _utc_now_iso(),
                "paths": self._paths_payload(),
                "config": dict(self._config_info),
                "aliases": {},
                "versions": [],
            }
            save_json(self._run_manifest_path, run_manifest)
            if self._s3 is not None:
                self._s3.upload(self._run_manifest_path, path_to_key(self._run_manifest_path, self.runs_root_parent))

    def _paths_payload(self) -> Dict[str, Any]:
        payload = {
            "layout_version": 1,
            "root_local": path_to_key(self.run_dir, self.runs_root_parent),
        }
        if self._s3 is not None:
            prefix = self._s3._cfg.prefix.strip("/")
            root_remote = f"s3://{self._s3._cfg.bucket}/{prefix}/{self.run_id}" if prefix else f"s3://{self._s3._cfg.bucket}/{self.run_id}"
            payload["root_remote"] = root_remote
        return payload

    def _load_run_manifest(self) -> Dict[str, Any]:
        if self._run_manifest_path.exists():
            return json.loads(self._run_manifest_path.read_text(encoding="utf-8"))
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "created_at": _utc_now_iso(),
            "paths": self._paths_payload(),
            "config": dict(self._config_info) if self._config_info else {},
            "aliases": {},
            "versions": [],
        }

    def _maybe_upload(self, path: Path) -> None:
        if self._s3 is None:
            return
        self._s3.upload(path, path_to_key(path, self.runs_root_parent))

    def _save_rng_payload(self) -> Dict[str, Any]:
        generator = self._state_sources.get("generator")
        train_batcher = self._state_sources.get("train_batcher")
        val_batcher = self._state_sources.get("val_batcher")

        payload: Dict[str, Any] = capture_rng_state(generator)
        payload["rank"] = self.rank
        payload["world_size"] = self.world_size

        batchers: Dict[str, Any] = {}
        exact = True
        if train_batcher is not None and hasattr(train_batcher, "get_state"):
            train_state = train_batcher.get_state()
            batchers["train"] = jsonable(train_state)
            exact = exact and bool(getattr(train_state, "get", lambda _k, _d=None: False)("exact", False))
        else:
            exact = False
        if val_batcher is not None and hasattr(val_batcher, "get_state"):
            val_state = val_batcher.get_state()
            batchers["val"] = jsonable(val_state)
            exact = exact and bool(getattr(val_state, "get", lambda _k, _d=None: False)("exact", False))
        else:
            exact = False
        if batchers:
            payload["batchers"] = batchers
        payload["exact"] = bool(exact)
        return payload

    def _best_status(self, metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if metrics is None:
            return None
        if self.best_metric_name not in metrics:
            return None
        value = metrics[self.best_metric_name]
        return {
            "metric_name": self.best_metric_name,
            "mode": self.best_mode,
            "value": float(value),
        }

    def save_version(
        self,
        step: int,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler] = None,
        metrics: Optional[Dict[str, Any]],
        all_gather: Optional[Any] = None,
    ) -> None:
        version_id = f"v{int(step):06d}"
        version_dir = self._versions_dir / version_id
        ensure_dir(version_dir)

        model_path = version_dir / "model.safetensors"
        opt_path = version_dir / f"opt_shard_rank{self.rank:04d}.bin"
        rng_path = version_dir / f"rng_rank{self.rank:04d}.json"
        scaler_path = version_dir / f"scaler_rank{self.rank:04d}.bin"

        if self.rank == 0:
            from safetensors.torch import save_file

            state_dict = model.state_dict()
            save_file(state_dict, str(model_path))
            self._maybe_upload(model_path)

        torch.save(optimizer.state_dict(), opt_path)
        self._maybe_upload(opt_path)
        if scaler is not None:
            torch.save(scaler.state_dict(), scaler_path)
            self._maybe_upload(scaler_path)

        rng_payload = self._save_rng_payload()
        save_json(rng_path, rng_payload)
        self._maybe_upload(rng_path)

        optimizer_entry = file_info(opt_path, self.runs_root_parent)
        optimizer_entry["rank"] = self.rank
        scaler_entry = None
        if scaler is not None:
            scaler_entry = file_info(scaler_path, self.runs_root_parent)
            scaler_entry["rank"] = self.rank
        shard_info = {
            "rank": self.rank,
            "optimizer": optimizer_entry,
            "rng": {
                "rank": self.rank,
                "key": path_to_key(rng_path, self.runs_root_parent),
            },
            "amp_scaler": scaler_entry,
            "exact": bool(rng_payload.get("exact", False)),
        }

        all_shards = [shard_info]
        if all_gather is not None:
            all_shards = all_gather(shard_info)

        if self.rank != 0:
            return

        if self._config_info is None:
            config_dst = self._config_dir / "train.toml"
            self._config_info = file_info(config_dst, self.runs_root_parent)

        optimizer_shards = [s["optimizer"] for s in all_shards]
        optimizer_shards.sort(key=lambda s: int(s.get("rank", 0)))
        scaler_shards = [s.get("amp_scaler") for s in all_shards if s.get("amp_scaler") is not None]
        scaler_shards.sort(key=lambda s: int(s.get("rank", 0)))
        rng_keys = [s["rng"] for s in all_shards]
        rng_keys.sort(key=lambda s: int(s.get("rank", 0)))
        exact_resume = all(bool(s.get("exact", False)) for s in all_shards)

        version_manifest = {
            "schema_version": 1,
            "run_id": self.run_id,
            "version_id": version_id,
            "created_at": _utc_now_iso(),
            "step": int(step),
            "paths": self._paths_payload(),
            "config": dict(self._config_info),
            "model": file_info(model_path, self.runs_root_parent),
            "optimizer": {
                "sharding": "custom",
                "shards": optimizer_shards,
            },
            "amp_scaler": (
                {
                    "sharding": "custom",
                    "shards": scaler_shards,
                }
                if scaler_shards
                else None
            ),
            "rng": {
                "per_rank": True,
                "keys": rng_keys,
            },
            "resume": {
                "base_step": int(step) + 1,
                "exact": bool(exact_resume),
            },
            "code": {},
            "metrics": metrics or {},
        }

        manifest_path = version_dir / "manifest.json"
        save_json(manifest_path, version_manifest)
        self._maybe_upload(manifest_path)

        run_manifest = self._load_run_manifest()
        run_manifest["paths"] = self._paths_payload()
        run_manifest["config"] = dict(self._config_info)
        run_manifest.setdefault("versions", [])
        run_manifest["versions"].append({
            "version_id": version_id,
            "step": int(step),
            "created_at": version_manifest["created_at"],
            "model_key": version_manifest["model"]["key"],
            "metrics": metrics or {},
        })

        latest_alias = {
            "schema_version": 1,
            "run_id": self.run_id,
            "alias": "latest",
            "version_id": version_id,
            "step": int(step),
            "manifest_key": path_to_key(manifest_path, self.runs_root_parent),
            "status": "active",
        }

        best_update = self._best_status(metrics)
        best_alias = self._best_alias or run_manifest.get("aliases", {}).get("best")
        if best_update is None:
            if best_alias is None:
                best_alias = {
                    "schema_version": 1,
                    "run_id": self.run_id,
                    "alias": "best",
                    "version_id": version_id,
                    "step": int(step),
                    "manifest_key": path_to_key(manifest_path, self.runs_root_parent),
                    "status": "pending",
                }
        else:
            if best_alias is None or best_alias.get("status") == "pending":
                best_alias = {
                    "schema_version": 1,
                    "run_id": self.run_id,
                    "alias": "best",
                    "version_id": version_id,
                    "step": int(step),
                    "manifest_key": path_to_key(manifest_path, self.runs_root_parent),
                    "status": "active",
                    **best_update,
                }
            else:
                current = float(best_alias.get("value", float("inf")))
                candidate = float(best_update["value"])
                is_better = candidate < current if self.best_mode == "min" else candidate > current
                if is_better:
                    best_alias = {
                        "schema_version": 1,
                        "run_id": self.run_id,
                        "alias": "best",
                        "version_id": version_id,
                        "step": int(step),
                        "manifest_key": path_to_key(manifest_path, self.runs_root_parent),
                        "status": "active",
                        **best_update,
                    }

        if best_alias is not None and "manifest_key" not in best_alias:
            if "version_id" in best_alias:
                best_alias["manifest_key"] = path_to_key(
                    self._versions_dir / best_alias["version_id"] / "manifest.json",
                    self.runs_root_parent,
                )

        run_manifest["aliases"] = {
            "latest": {"version_id": version_id, "step": int(step)},
            "best": {
                "version_id": best_alias["version_id"],
                "step": best_alias["step"],
                "metric_name": best_alias.get("metric_name"),
                "mode": best_alias.get("mode"),
                "value": best_alias.get("value"),
                "status": best_alias.get("status"),
            },
        }

        save_json(self._run_manifest_path, run_manifest)
        self._maybe_upload(self._run_manifest_path)

        save_json(self._aliases_dir / "latest.json", latest_alias)
        save_json(self._aliases_dir / "best.json", best_alias)
        self._maybe_upload(self._aliases_dir / "latest.json")
        self._maybe_upload(self._aliases_dir / "best.json")

        self._best_alias = best_alias


def resolve_checkpoint_reference(
    run_dir: Path,
    ref: str,
    *,
    s3: Optional[S3Uploader] = None,
    root_parent: Optional[Path] = None,
) -> Path:
    if ref in {"latest", "best"}:
        alias_path = run_dir / "aliases" / f"{ref}.json"
        if not alias_path.exists() and s3 is not None:
            s3.download(alias_path, path_to_key(alias_path, run_dir.parent))
        if not alias_path.exists():
            raise FileNotFoundError(f"alias not found: {alias_path}")
        data = json.loads(alias_path.read_text(encoding="utf-8"))
        manifest_key = data.get("manifest_key")
        if manifest_key:
            base = root_parent if root_parent is not None else run_dir.parent
            manifest_path = base / manifest_key
            if not manifest_path.exists() and s3 is not None:
                s3.download(manifest_path, manifest_key)
            return manifest_path
        version_id = data.get("version_id")
        if version_id:
            manifest_path = run_dir / "versions" / version_id / "manifest.json"
            if not manifest_path.exists() and s3 is not None:
                s3.download(manifest_path, path_to_key(manifest_path, run_dir.parent))
            return manifest_path
        raise ValueError(f"alias missing manifest reference: {alias_path}")

    ref_path = Path(ref)
    if ref_path.is_dir():
        candidate = ref_path / "manifest.json"
        if candidate.exists():
            return candidate
    if ref_path.exists():
        return ref_path
    raise FileNotFoundError(f"checkpoint reference not found: {ref}")


def load_manifest(path: Path, *, root_parent: Optional[Path] = None, s3: Optional[S3Uploader] = None) -> Dict[str, Any]:
    if not path.exists() and root_parent is not None:
        ensure_local(path, root_parent, s3)
    return json.loads(path.read_text(encoding="utf-8"))


def load_model_from_manifest(
    manifest: Dict[str, Any],
    run_dir: Path,
    *,
    root_parent: Optional[Path] = None,
    s3: Optional[S3Uploader] = None,
) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    model_key = manifest["model"]["key"]
    base = root_parent if root_parent is not None else run_dir.parent
    model_path = base / model_key
    if root_parent is not None:
        ensure_local(model_path, root_parent, s3)
    return load_file(str(model_path))


def load_optimizer_shard(
    manifest: Dict[str, Any],
    run_dir: Path,
    rank: int,
    map_location: Optional[str] = None,
    *,
    root_parent: Optional[Path] = None,
    s3: Optional[S3Uploader] = None,
) -> Dict[str, Any]:
    shards = manifest["optimizer"]["shards"]
    match = None
    for shard in shards:
        if int(shard.get("rank", -1)) == int(rank):
            match = shard
            break
    if match is None:
        raise FileNotFoundError(f"optimizer shard for rank {rank} not found in manifest")
    base = root_parent if root_parent is not None else run_dir.parent
    shard_path = base / match["key"]
    if root_parent is not None:
        ensure_local(shard_path, root_parent, s3)
    return torch.load(shard_path, map_location=map_location)


def load_scaler_shard(
    manifest: Dict[str, Any],
    run_dir: Path,
    rank: int,
    map_location: Optional[str] = None,
    *,
    root_parent: Optional[Path] = None,
    s3: Optional[S3Uploader] = None,
) -> Optional[Dict[str, Any]]:
    scaler_group = manifest.get("amp_scaler")
    if not scaler_group:
        return None
    shards = scaler_group.get("shards", [])
    match = None
    for shard in shards:
        if int(shard.get("rank", -1)) == int(rank):
            match = shard
            break
    if match is None:
        return None
    base = root_parent if root_parent is not None else run_dir.parent
    shard_path = base / match["key"]
    if root_parent is not None:
        ensure_local(shard_path, root_parent, s3)
    return torch.load(shard_path, map_location=map_location)


def load_rng_state(
    manifest: Dict[str, Any],
    run_dir: Path,
    rank: int,
    *,
    root_parent: Optional[Path] = None,
    s3: Optional[S3Uploader] = None,
) -> Dict[str, Any]:
    keys = manifest.get("rng", {}).get("keys", [])
    match = None
    for entry in keys:
        if int(entry.get("rank", -1)) == int(rank):
            match = entry
            break
    if match is None:
        raise FileNotFoundError(f"rng state for rank {rank} not found in manifest")
    base = root_parent if root_parent is not None else run_dir.parent
    rng_path = base / match["key"]
    if root_parent is not None:
        ensure_local(rng_path, root_parent, s3)
    return json.loads(rng_path.read_text(encoding="utf-8"))
