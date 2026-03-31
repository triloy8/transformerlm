from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path
import random

import numpy as np
import torch


def jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return str(value)


def _tupleify(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tupleify(v) for v in value)
    if isinstance(value, dict):
        return {k: _tupleify(v) for k, v in value.items()}
    return value


def capture_rng_state(generator: Optional[torch.Generator]) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": jsonable(random.getstate()),
        "numpy": jsonable(np.random.get_state()),
        "torch": jsonable(torch.random.get_rng_state()),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = jsonable(torch.cuda.get_rng_state_all())
    if generator is not None:
        state["torch_generator"] = jsonable(generator.get_state())
    return state


def restore_rng_state(state: Dict[str, Any], generator: Optional[torch.Generator]) -> bool:
    try:
        random.setstate(_tupleify(state["python"]))
        numpy_state = list(state["numpy"])
        if len(numpy_state) >= 2:
            numpy_state[1] = np.array(numpy_state[1], dtype=np.uint32)
        np.random.set_state(tuple(numpy_state))
        torch.random.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
        if torch.cuda.is_available() and "torch_cuda" in state:
            cuda_states = [torch.tensor(s, dtype=torch.uint8) for s in state["torch_cuda"]]
            torch.cuda.set_rng_state_all(cuda_states)
        if generator is not None and "torch_generator" in state:
            generator.set_state(torch.tensor(state["torch_generator"], dtype=torch.uint8))
    except Exception:
        return False
    return True
