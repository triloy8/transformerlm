from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, Any, Optional

from .base import Logger


class ConsoleLogger(Logger):
    def __init__(self, *, prefix: str = "log"):
        self._run_name: Optional[str] = None
        self._prefix = prefix

    def start_run(self, config: Dict[str, Any]) -> Dict[str, str]:
        # Generate a timestamp-based run name
        self._run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        line = {
            "event": "start_run",
            "run_name": self._run_name,
            "config": config,
            "ts": datetime.utcnow().isoformat() + "Z",
        }
        print(f"{self._prefix} " + json.dumps(line, ensure_ascii=False))
        return {"run_name": self._run_name}

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        line = {"ts": datetime.utcnow().isoformat() + "Z", **data}
        if step is not None:
            line["step"] = int(step)
        print(f"{self._prefix} " + json.dumps(line, ensure_ascii=False))

    def log_table(self, key: str, rows: list, step: Optional[int] = None) -> None:
        self.log({key: rows}, step=step)

    def log_artifact(self, path: str, name: Optional[str] = None, type_: Optional[str] = None) -> None:
        line = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "event": "artifact",
            "path": path,
            "name": name or path,
            "type": type_ or "artifact",
        }
        print(f"{self._prefix} " + json.dumps(line, ensure_ascii=False))

    def finish(self) -> None:
        line = {
            "event": "finish",
            "run_name": self._run_name,
            "ts": datetime.utcnow().isoformat() + "Z",
        }
        print(f"{self._prefix} " + json.dumps(line, ensure_ascii=False))
