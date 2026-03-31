from __future__ import annotations

from typing import Optional, Dict, Any

from .base import Logger


class NoOpLogger(Logger):
    def start_run(self, config: Dict[str, Any]) -> Dict[str, str]:
        return {"run_name": "noop"}

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        return None

    def log_table(self, key: str, rows: list, step: Optional[int] = None) -> None:
        return None

    def finish(self) -> None:
        return None
