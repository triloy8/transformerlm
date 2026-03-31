from __future__ import annotations

from typing import Optional, Dict, Any

from .base import Logger
from .noop import NoOpLogger


class RankZeroLogger(Logger):
    """Proxy that only logs on rank 0.

    If rank != 0 or inner is None, behaves as a NoOpLogger.
    """

    def __init__(self, rank: int, inner: Optional[Logger]):
        self._rank = int(rank)
        self._inner: Logger = inner if (self._rank == 0 and inner is not None) else NoOpLogger()

    def start_run(self, config: Dict[str, Any]) -> Dict[str, str]:
        return self._inner.start_run(config)

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        self._inner.log(data, step=step)

    def log_table(self, key: str, rows: list, step: Optional[int] = None) -> None:
        self._inner.log_table(key, rows, step=step)

    def log_artifact(self, path: str, name: Optional[str] = None, type_: Optional[str] = None) -> None:
        self._inner.log_artifact(path, name=name, type_=type_)

    def finish(self) -> None:
        self._inner.finish()
