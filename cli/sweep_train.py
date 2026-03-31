import argparse
import json
import os
from pathlib import Path

import torch.multiprocessing as mp

from config import TrainConfig, asdict_pretty, load_train_config
from transformerlm.builders import build_model, build_tokenizer, build_activation_filter
from leash.objectives import build_objective
from leash.trainer import train_ddp
from cli.train import build_train_namespace
from cli.utils import add_config_args, load_config_or_print


def _parse_only_config():
    parser = argparse.ArgumentParser(description="Training with W&B sweep overrides.", allow_abbrev=False)
    add_config_args(parser, type_=Path)
    return parser.parse_args()


def _apply_overrides(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if key.startswith("_") or key in {"wandb_version"} or value is None:
            continue
        if not isinstance(key, str) or not key:
            continue
        parts = key.split(".")
        cursor = base
        for idx, part in enumerate(parts):
            if not isinstance(cursor, dict) or part not in cursor:
                raise ValueError(f"Unknown config path '{key}'")
            if idx == len(parts) - 1:
                cursor[part] = value
            else:
                cursor = cursor[part]
    return base

def _apply_lr_constraints(base: dict, overrides: dict) -> dict:
    ratios = {
        "hidden": 0.1,
        "head": 0.1,
        "embed": 0.1,
        "scalar": 0.05,
    }
    optimizer = base.get("optimizer", {})
    optimizer_name = optimizer.get("optimizer_name")
    muon = optimizer.get("muon", {})
    for group, ratio in ratios.items():
        max_key = f"optimizer.muon.{group}.max_learning_rate"
        min_key = f"optimizer.muon.{group}.min_learning_rate"
        init_key = f"optimizer.muon.{group}.initial_learning_rate"
        if optimizer_name == "muon" and max_key in overrides:
            try:
                max_val = float(overrides[max_key])
            except (TypeError, ValueError):
                continue
            if isinstance(muon, dict) and group in muon:
                if min_key not in overrides:
                    muon[group]["min_learning_rate"] = max_val * ratio
                if init_key not in overrides:
                    muon[group]["initial_learning_rate"] = max_val
    if optimizer_name == "adamw":
        adamw_max_key = "optimizer.max_learning_rate"
        if adamw_max_key in overrides:
            try:
                max_val = float(overrides[adamw_max_key])
            except (TypeError, ValueError):
                return base
            if isinstance(optimizer, dict):
                if "min_learning_rate" not in overrides:
                    optimizer["min_learning_rate"] = max_val * 0.1
                if "initial_learning_rate" not in overrides:
                    optimizer["initial_learning_rate"] = max_val
    return base



def main():
    args_cfg = _parse_only_config()
    cfg_dc = load_config_or_print(load_train_config, args_cfg.config, args_cfg.print_config)
    if cfg_dc is None:
        return
    if cfg_dc.ddp is None:
        raise ValueError("DDP config is required when using transformerlm-sweep-train")

    config_json = os.environ.get("WANDB_CONFIG_JSON")
    overrides = None
    if config_json:
        overrides = json.loads(config_json)
    else:
        import wandb  # type: ignore

        run = wandb.init()
        overrides = dict(run.config)
        run.finish()
    cfg_dict = _apply_overrides(asdict_pretty(cfg_dc), overrides)
    cfg_dict = _apply_lr_constraints(cfg_dict, overrides)
    cfg_dc = TrainConfig.model_validate(cfg_dict)

    ns = build_train_namespace(cfg_dc, str(args_cfg.config))
    config_snapshot = asdict_pretty(cfg_dc)
    nprocs = max(1, int(cfg_dc.ddp.num_gpus_per_node))
    mp.spawn(
        train_ddp,
        args=(
            ns,
            cfg_dc,
            config_snapshot,
            build_model,
            build_tokenizer,
            build_objective,
            build_activation_filter(),
        ),
        nprocs=nprocs,
        join=True,
    )


if __name__ == "__main__":
    main()
