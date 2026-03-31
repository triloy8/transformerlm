from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
import torch.distributed as dist
from torch.nn import Parameter

from leash.ddp.utils import get_rank, get_world_size


class OptimizerStateSharding(torch.optim.Optimizer):
    def __init__(self, params: Any, optimizer_cls: type[torch.optim.Optimizer], **kwargs: Any) -> None:
        defaults = kwargs.pop("defaults", {})
        if not isinstance(defaults, dict):
            raise TypeError("defaults must be a dict if provided")

        self._optimizer_cls = optimizer_cls
        self._optimizer_kwargs = dict(kwargs)

        self._rank = int(get_rank())
        world_size = int(get_world_size())
        self._world_size = world_size if world_size > 0 else 1

        self._flat_params: List[Parameter] = []
        self._owner_by_param: Dict[Parameter, int] = {}
        self._ownership_cursor = 0
        self._local_param_groups: List[Dict[str, Any]] = []
        self._optimizer: Optional[torch.optim.Optimizer] = None

        param_groups = self._normalize_param_groups(params)
        super().__init__(param_groups, defaults)

        self._all_param_groups = self.param_groups
        self._rebuild_shards()
        self._optimizer = self._optimizer_cls(self._local_param_groups, **self._optimizer_kwargs)
        self.param_groups = self._all_param_groups
        self._local_param_groups = self._optimizer.param_groups
        self.state = self._optimizer.state

    def _build_local_param_groups(self) -> List[Dict[str, Any]]:
        return [self._make_local_group(group) for group in self._all_param_groups]

    def _sync_local_hyperparams(self) -> None:
        if self._optimizer is None:
            raise RuntimeError("optimizer not initialized")
        for global_group, local_group in zip(self._all_param_groups, self._optimizer.param_groups):
            for key, value in global_group.items():
                if key == "params":
                    continue
                local_group[key] = value

    def _sync_global_hyperparams(self) -> None:
        if self._optimizer is None:
            raise RuntimeError("optimizer not initialized")
        for global_group, local_group in zip(self._all_param_groups, self._optimizer.param_groups):
            for key, value in local_group.items():
                if key == "params":
                    continue
                global_group[key] = value

    def step(self, closure: Optional[Callable[[], Any]] = None, **kwargs: Any) -> Any:
        if self._optimizer is None:
            raise RuntimeError("optimizer not initialized")
        self._sync_local_hyperparams()
        if closure is None:
            loss = self._optimizer.step(**kwargs)
        else:
            loss = self._optimizer.step(closure, **kwargs)
        self._sync_global_hyperparams()

        with torch.no_grad():
            self._sync_parameters()

        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        if self._optimizer is None:
            raise RuntimeError("optimizer not initialized")
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def _sync_parameters(self) -> None:
        if self._world_size <= 1:
            return
        if not dist.is_available() or not dist.is_initialized():
            return

        for p in self._flat_params:
            owner = self._owner_by_param.get(p)
            if owner is None:
                continue
            dist.broadcast(p.data, src=owner)

    def state_dict(self) -> Dict[str, Any]:  # type: ignore[override]
        if self._optimizer is None:
            raise RuntimeError("optimizer not initialized")
        self._sync_local_hyperparams()
        state = self._optimizer.state_dict()
        state["_shard_owners"] = {id(p): owner for p, owner in self._owner_by_param.items()}
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:  # type: ignore[override]
        shard_owners = state_dict.pop("_shard_owners", {})
        owner_map = {int(k): int(v) for k, v in shard_owners.items()}

        self._owner_by_param.clear()
        for idx, p in enumerate(self._flat_params):
            key = id(p)
            if key in owner_map:
                self._owner_by_param[p] = owner_map[key]
            else:
                self._owner_by_param[p] = idx % self._world_size
        if self._optimizer is None:
            raise RuntimeError("optimizer not initialized")
        new_local_groups = self._build_local_param_groups()
        self._optimizer.param_groups = new_local_groups
        self._local_param_groups = self._optimizer.param_groups
        self._ownership_cursor = len(self._flat_params) % self._world_size
        self._optimizer.load_state_dict(state_dict)
        self._sync_global_hyperparams()
        self._sync_local_hyperparams()

    def __getattr__(self, name: str) -> Any:
        if name in {"state", "param_groups"}:
            return super().__getattribute__(name)
        optimizer = self.__dict__.get("_optimizer")
        if optimizer is None:
            raise AttributeError(name)
        return getattr(optimizer, name)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:  # type: ignore[override]
        normalized = self._clone_param_group(param_group)
        super().add_param_group(normalized)
        appended = self.param_groups[-1]

        self._register_params(appended["params"])

        if self._optimizer is None:
            return

        local_group = self._make_local_group(appended)
        if local_group["params"]:
            self._optimizer.add_param_group(local_group)
        else:
            self._optimizer.param_groups.append(local_group)
        self._local_param_groups = self._optimizer.param_groups
        self._sync_local_hyperparams()
        self._sync_global_hyperparams()

    def _normalize_param_groups(self, params: Any) -> List[Dict[str, Any]]:
        if isinstance(params, Parameter):
            return [{"params": [params]}]

        if isinstance(params, dict):
            return [self._clone_param_group(params)]

        if isinstance(params, Iterable):
            materialized = list(params)
            if not materialized:
                return [{"params": []}]

            first = materialized[0]
            if isinstance(first, dict):
                return [self._clone_param_group(group) for group in materialized]

            return [{"params": self._ensure_param_list(materialized)}]

        raise TypeError("params must be an iterable of Parameters or param group dicts")

    def _clone_param_group(self, group: Dict[str, Any]) -> Dict[str, Any]:
        cloned = dict(group)
        params_field = cloned.get("params")
        if isinstance(params_field, Parameter):
            params_list = [params_field]
        elif isinstance(params_field, Iterable):
            params_list = self._ensure_param_list(params_field)
        else:
            raise TypeError("param group missing params iterable")
        cloned["params"] = params_list
        return cloned

    def _ensure_param_list(self, params_iterable: Iterable[Any]) -> List[Parameter]:
        params_list = list(params_iterable)
        for p in params_list:
            if not isinstance(p, Parameter):
                raise TypeError("expected torch.nn.Parameter in params list")
        return params_list

    def _rebuild_shards(self) -> None:
        self._flat_params.clear()
        self._owner_by_param.clear()
        self._ownership_cursor = 0

        for group in self._all_param_groups:
            self._register_params(group["params"])

        self._local_param_groups = self._build_local_param_groups()
        if self._optimizer is not None:
            self._optimizer.param_groups = self._local_param_groups
            self._sync_local_hyperparams()
            self._sync_global_hyperparams()

    def _register_params(self, params: Iterable[Parameter]) -> None:
        for p in params:
            if not isinstance(p, Parameter):
                raise TypeError("expected torch.nn.Parameter")
            if p in self._owner_by_param:
                raise ValueError("parameter already registered in optimizer")
            owner = self._ownership_cursor
            self._owner_by_param[p] = owner
            self._flat_params.append(p)
            self._ownership_cursor = (self._ownership_cursor + 1) % self._world_size

    def _make_local_group(self, group: Dict[str, Any]) -> Dict[str, Any]:
        local = dict(group)
        if self._world_size <= 1:
            local["params"] = list(group["params"])
            return local
        local["params"] = [p for p in group["params"] if self._owner_by_param.get(p) == self._rank]
        return local
