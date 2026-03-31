from __future__ import annotations

import os
from typing import Optional, Any

import torch
import torch.distributed as dist


def setup_process_group(
    *,
    backend: str,
    local_rank: int,
    num_gpus_per_node: int,
    num_nodes: int,
    node_rank: int = 0,
    master_addr: str = "localhost",
    master_port: str = "29500",
) -> tuple[int, int]:
    """Initialize torch.distributed with env setup and optional CUDA device selection.

    Returns `(global_rank, world_size)` for convenience.
    """
    world_size = max(1, int(num_nodes) * int(num_gpus_per_node))
    global_rank = int(node_rank) * int(num_gpus_per_node) + int(local_rank)

    os.environ.setdefault("MASTER_ADDR", str(master_addr))
    os.environ.setdefault("MASTER_PORT", str(master_port))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(global_rank))

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=global_rank, world_size=world_size)

    if backend == "nccl" and torch.cuda.is_available():
        device_id = int(local_rank)
        if torch.cuda.device_count() > 0 and device_id >= torch.cuda.device_count():
            raise RuntimeError(f"local_rank {device_id} >= CUDA device count {torch.cuda.device_count()}")
        torch.cuda.set_device(device_id)

    return global_rank, world_size


def cleanup_process_group() -> None:
    """Destroy the current process group if initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def broadcast_string(s: Optional[str], src: int = 0) -> str:
    """Broadcast a string from src rank to all ranks.

    Expects a process group to be initialized. On non-src ranks, pass None.
    Returns the broadcast value as a string.
    """
    obj_list: list[Any] = [s]
    dist.broadcast_object_list(obj_list, src=src)
    out = obj_list[0]
    return "" if out is None else str(out)


def seed_for_rank(base_seed: int) -> int:
    """Derive a per-rank seed from a base seed."""
    return int(base_seed) + int(get_rank())


def allreduce_mean(x: float | torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> float | torch.Tensor:
    """All-reduce mean for a Python float or a tensor.

    - Returns same type as input (float in, float out; tensor in, tensor out).
    - Chooses device based on backend (nccl->cuda if available, else cpu) for float inputs.
    - No-op if process group is not initialized.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return x

    # Determine device for temporary tensors
    backend = None
    try:
        backend = str(dist.get_backend())
    except Exception:
        backend = None
    device = torch.device("cuda") if (backend == "nccl" and torch.cuda.is_available()) else torch.device("cpu")

    is_float = isinstance(x, (float, int)) and not isinstance(x, bool)
    if is_float:
        t = torch.tensor(float(x), device=device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
        t.div_(get_world_size())
        return float(t.item())

    if isinstance(x, torch.Tensor):
        orig_device = x.device
        t = x.detach().to(device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
        t.div_(get_world_size())
        # move back to original device and dtype
        return t.to(orig_device).type_as(x)

    # Fallback: try to convert to float
    try:
        return allreduce_mean(float(x), group=group)  # type: ignore[arg-type]
    except Exception:
        return x
