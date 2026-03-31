from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
import os


@dataclass(frozen=True)
class S3ConfigData:
    bucket: str
    prefix: str = ""
    endpoint_url: Optional[str] = None
    region_name: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None


class S3Uploader:
    def __init__(self, cfg: S3ConfigData) -> None:
        self._cfg = cfg
        self._client = None

    def _client_or_init(self):
        if self._client is not None:
            return self._client
        import boto3  # type: ignore

        self._client = boto3.client(
            "s3",
            endpoint_url=self._cfg.endpoint_url,
            region_name=self._cfg.region_name,
            aws_access_key_id=self._cfg.access_key_id,
            aws_secret_access_key=self._cfg.secret_access_key,
            aws_session_token=self._cfg.session_token,
        )
        return self._client

    def upload(self, local_path: Path, key: str) -> None:
        client = self._client_or_init()
        prefix = self._cfg.prefix.strip("/")
        remote_key = f"{prefix}/{key}" if prefix else key
        client.upload_file(str(local_path), self._cfg.bucket, remote_key)

    def download(self, local_path: Path, key: str) -> None:
        client = self._client_or_init()
        prefix = self._cfg.prefix.strip("/")
        remote_key = f"{prefix}/{key}" if prefix else key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(self._cfg.bucket, remote_key, str(local_path))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def path_to_key(path: Path, root_parent: Path) -> str:
    try:
        return path.relative_to(root_parent).as_posix()
    except ValueError:
        return os.path.relpath(str(path), str(root_parent)).replace("\\", "/")


def file_info(path: Path, root_parent: Path) -> Dict[str, Any]:
    return {
        "key": path_to_key(path, root_parent),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def ensure_local(path: Path, root_parent: Path, s3: Optional[S3Uploader]) -> Path:
    if path.exists():
        return path
    if s3 is None:
        raise FileNotFoundError(f"missing local file: {path}")
    key = path_to_key(path, root_parent)
    s3.download(path, key)
    return path
