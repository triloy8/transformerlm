from dataclasses import dataclass
from typing import List, Dict, Optional
import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


@dataclass
class Bucket:
    params: List[torch.nn.Parameter]
    size_params: int
    ready_count: int = 0
    launched: bool = False
    flat: Optional[torch.Tensor] = None
    splits: Optional[List[int]] = None
    handle: Optional[dist.Work] = None


class DDP(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, world_size: int, bucket_size_mb: int):
        super().__init__()
        self.model = model
        self.world_size = world_size

        self.bucket_size_params = int(bucket_size_mb)
        self._buckets: List[Bucket] = []
        self._param_to_bucket: Dict[torch.nn.Parameter, int] = {}
        self._pending_bucket_idxs: List[int] = []

        self._make_param_buckets()

        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            p.register_post_accumulate_grad_hook(self._on_param_grad_ready)

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.model.load_state_dict(state_dict, *args, **kwargs)

    @torch.no_grad()
    def broadcast_parameters(self, src: int = 0, include_buffers: bool = True) -> None:
        for p in self.model.parameters():
            dist.broadcast(p.data, src=src)
        if include_buffers:
            for b in self.model.buffers():
                dist.broadcast(b.data, src=src)

    def _make_param_buckets(self) -> None:
        params = [p for p in self.model.parameters() if p.requires_grad]

        buckets: List[Bucket] = []
        param_to_bucket: Dict[torch.nn.Parameter, int] = {}

        cap = self.bucket_size_params
        if cap <= 0 or cap >= len(params):
            b = Bucket(params=list(params), size_params=len(params))
            buckets.append(b)
            for q in params:
                param_to_bucket[q] = 0
        else:
            for i in range(0, len(params), cap):
                chunk = params[i:i + cap]
                b = Bucket(params=list(chunk), size_params=len(chunk))
                buckets.append(b)
                b_idx = len(buckets) - 1
                for q in chunk:
                    param_to_bucket[q] = b_idx

        self._buckets = buckets
        self._param_to_bucket = param_to_bucket

    @torch.no_grad()
    def _on_param_grad_ready(self, param: torch.nn.Parameter):
        b_idx = self._param_to_bucket[param]
        bucket = self._buckets[b_idx]

        bucket.ready_count += 1
        if (not bucket.launched) and (bucket.ready_count == bucket.size_params):
            self._launch_bucket_all_reduce(b_idx)

    @torch.no_grad()
    def _launch_bucket_all_reduce(self, b_idx: int):
        bucket = self._buckets[b_idx]

        grads = [p.grad for p in bucket.params if p.grad is not None]
        if not grads:
            bucket.launched = True
            return

        # flatten into a single contiguous buffer
        flat = _flatten_dense_tensors([g.detach() for g in grads])

        # async all-reduce (SUM for gloo compatibility, average after wait)
        handle = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)

        bucket.flat = flat
        bucket.handle = handle
        bucket.launched = True
        self._pending_bucket_idxs.append(b_idx)

    @torch.no_grad()
    def finish_gradient_synchronization(self):
        for b_idx in self._pending_bucket_idxs:
            bucket = self._buckets[b_idx]
            if bucket.handle is None:
                continue

            bucket.handle.wait()
            bucket.flat.div_(self.world_size)

            # unflatten back into grads
            grads = [p.grad for p in bucket.params if p.grad is not None]
            for g, reduced in zip(grads, _unflatten_dense_tensors(bucket.flat, grads)):
                g.copy_(reduced)

            # reset for next iteration
            bucket.ready_count = 0
            bucket.launched = False
            bucket.flat = None
            bucket.handle = None

        self._pending_bucket_idxs.clear()
