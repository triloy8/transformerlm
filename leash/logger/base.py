from __future__ import annotations

from typing import Optional, Dict, Any


class Logger:
    def start_run(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Initialize a run. Return info dict with at least {'run_name': str}."""
        raise NotImplementedError

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log a dictionary of metrics/events. 'step' is optional."""
        raise NotImplementedError

    def log_table(self, key: str, rows: list, step: Optional[int] = None) -> None:
        """Log a table-like payload under a key."""
        self.log({key: rows}, step=step)

    def log_artifact(self, path: str, name: Optional[str] = None, type_: Optional[str] = None) -> None:
        """Optionally log a file artifact (noop by default)."""
        return None

    def finish(self) -> None:
        """Close out the run (flush/cleanup)."""
        raise NotImplementedError
