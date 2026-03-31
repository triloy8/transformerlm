from __future__ import annotations

from typing import Dict, Any, Optional

from .base import Logger


class WandbLogger(Logger):
    def __init__(self, entity: Optional[str] = None, project: Optional[str] = None, name: Optional[str] = None):
        self._entity = entity
        self._project = project
        self._name = name
        self._run = None
        self._tables: Dict[str, Any] = {}
        self._table_rows: Dict[str, list[list[Any]]] = {}
        self._last_step: Optional[int] = None

    def _resolve_step(self, step: Optional[int]) -> int:
        if step is None:
            # Keep W&B step monotonic even for background logs that don't carry
            # train iteration information (e.g. streaming diagnostics).
            return 0 if self._last_step is None else int(self._last_step)
        self._last_step = int(step)
        return int(step)

    def start_run(self, config: Dict[str, Any]) -> Dict[str, str]:
        # Import lazily to keep core free of wandb dependency
        import wandb  # type: ignore

        if wandb.run is not None:
            self._run = wandb.run
            wandb.config.update(config, allow_val_change=True)
        else:
            self._run = wandb.init(entity=self._entity, project=self._project, name=self._name, config=config)
        return {"run_name": self._run.name}

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        import wandb  # type: ignore

        wandb.log(data, step=self._resolve_step(step))

    def log_table(self, key: str, rows: list, step: Optional[int] = None) -> None:
        import wandb  # type: ignore

        resolved_step = self._resolve_step(step)
        if isinstance(rows, wandb.Table):
            wandb.log({key: rows}, step=resolved_step)
            return

        table = self._tables.get(key)
        if table is None:
            columns = None
            if rows and isinstance(rows[0], dict):
                columns = list(rows[0].keys())
            else:
                columns = ["noisy_input", "prediction", "target"]
            table = wandb.Table(columns=columns)
            self._tables[key] = table

        row_cache = self._table_rows.setdefault(key, [])
        for row in rows:
            if isinstance(row, dict):
                row_cache.append([row.get(col, "") for col in table.columns])
            elif isinstance(row, (list, tuple)):
                row_cache.append(list(row))

        table = wandb.Table(columns=table.columns, data=row_cache)
        self._tables[key] = table
        wandb.log({key: table}, step=resolved_step)

    def log_artifact(self, path: str, name: Optional[str] = None, type_: Optional[str] = None) -> None:
        try:
            import wandb  # type: ignore

            art = wandb.Artifact(name or path, type=type_ or "artifact")
            art.add_file(path)
            wandb.log_artifact(art)
        except Exception:
            # Stay resilient: logging artifacts is optional.
            pass

    def finish(self) -> None:
        if self._run is not None:
            import wandb  # type: ignore

            wandb.finish()
            self._run = None
