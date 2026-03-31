#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import time
from html import escape as html_escape
from pathlib import Path

import torch
from safetensors.torch import load_file

from config import load_sweep_infer_config
from leash.inference.generate import autoregressive_generate, diffusion_generate
from transformerlm.models import TransformerLM
from transformerlm.models.attention import set_sdp_backend
from transformerlm.tokenizer.tokenizer import Tokenizer
from transformerlm.utils.dtypes import DTYPES
from cli.utils import add_config_args, load_config_or_print


def _normalize_state_dict_keys(state_dict):
    if not state_dict:
        return state_dict
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    return state_dict


def _load_model_and_tokenizer(cfg):
    tokenizer = Tokenizer.from_files(
        vocab_filepath=cfg.tokenizer.vocab_path,
        merges_filepath=cfg.tokenizer.merges_path,
        special_tokens_path=cfg.tokenizer.special_tokens_path,
    )
    set_sdp_backend(cfg.model.attention_sdp_backend)
    model = TransformerLM(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        rope_theta=cfg.model.rope_theta,
        attention_backend=cfg.model.attention_backend,
        device=cfg.model.device,
        dtype=DTYPES[cfg.model.dtype],
    )
    model_state = _normalize_state_dict_keys(load_file(str(cfg.checkpoint.ckpt_path)))
    model.load_state_dict(model_state)
    model.eval()
    guide_model = None
    guidance_mode = "none"
    guide_fallback_strategy = "single_token"
    if getattr(cfg, "guide", None) is not None:
        guidance_mode = str(cfg.guide.guidance_mode)
        guide_fallback_strategy = str(cfg.guide.fallback_strategy)
        if guidance_mode != "none":
            gcfg = cfg.guide.model
            guide_model = TransformerLM(
                vocab_size=gcfg.vocab_size,
                context_length=gcfg.context_length,
                d_model=gcfg.d_model,
                num_layers=gcfg.num_layers,
                num_heads=gcfg.num_heads,
                d_ff=gcfg.d_ff,
                rope_theta=gcfg.rope_theta,
                attention_backend=gcfg.attention_backend,
                device=gcfg.device,
                dtype=DTYPES[gcfg.dtype],
            )
            guide_state = _normalize_state_dict_keys(load_file(str(cfg.guide.checkpoint.ckpt_path)))
            guide_model.load_state_dict(guide_state)
            guide_model.eval()
    return model, tokenizer, guide_model, guidance_mode, guide_fallback_strategy


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sweep inference params via config.")
    add_config_args(parser, type_=str)
    args_cfg = parser.parse_args()
    cfg = load_config_or_print(load_sweep_infer_config, args_cfg.config, args_cfg.print_config)
    if cfg is None:
        return

    prompts = cfg.sweep.prompts or [cfg.inference.prompt]
    temperatures = cfg.sweep.temperatures or [float(cfg.inference.temperature)]
    steps_list = cfg.sweep.steps or [int(cfg.inference.steps)]
    total_lengths = cfg.sweep.total_lengths or [int(cfg.inference.total_length)]
    block_lengths = cfg.sweep.block_lengths or [int(cfg.inference.block_length)]
    cfg_scales = cfg.sweep.cfg_scales or [float(cfg.inference.cfg_scale)]
    remasking = cfg.sweep.remasking or [str(cfg.inference.remasking)]
    generation_mode = str(cfg.inference.generation_mode)
    if cfg.sweep.top_ps is None:
        top_ps = [None if cfg.inference.top_p is None else float(cfg.inference.top_p)]
    else:
        top_ps = cfg.sweep.top_ps
    seeds = cfg.sweep.seeds if cfg.sweep.seeds is not None else [cfg.inference.seed]

    model, tokenizer, guide_model, guidance_mode, guide_fallback_strategy = _load_model_and_tokenizer(cfg)

    output_path = Path(cfg.sweep.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid = list(
        itertools.product(
            prompts,
            temperatures,
            steps_list,
            total_lengths,
            block_lengths,
            cfg_scales,
            remasking,
            top_ps,
            seeds,
        )
    )
    if cfg.sweep.limit is not None:
        grid = grid[: cfg.sweep.limit]

    print(f"grid_size={len(grid)} output={output_path}")

    records = [] if cfg.sweep.html_output_path else None
    with output_path.open("w", encoding="utf-8") as f:
        for idx, (prompt, temp, steps, total_len, block_len, cfg_scale, remask, top_p, seed) in enumerate(grid, 1):
            ids = [tokenizer.encode(prompt)]
            prompt_len = len(ids[0])
            total_len = int(total_len)
            if total_len < prompt_len:
                print(f"skip total_length={total_len} < prompt_len={prompt_len}")
                continue
            if total_len > int(cfg.model.context_length):
                print(f"skip total_length={total_len} > context_length={cfg.model.context_length}")
                continue
            gen_length = total_len - prompt_len

            in_indices = torch.tensor(ids, device=cfg.model.device)
            generator = None
            if seed is not None:
                generator = torch.Generator(device=cfg.model.device)
                generator.manual_seed(int(seed))

            t0 = time.time()
            if gen_length > 0:
                if generation_mode == "ar":
                    out_indices = autoregressive_generate(
                        model,
                        in_indices,
                        gen_length=int(gen_length),
                        temperature=float(temp),
                        top_p=(None if top_p is None else float(top_p)),
                        eos_token_id=(None if cfg.inference.eos_token_id is None else int(cfg.inference.eos_token_id)),
                        logits_eos_inf=bool(cfg.inference.logits_eos_inf),
                        generator=generator,
                    )
                elif generation_mode == "diffusion":
                    out_indices = diffusion_generate(
                        model,
                        in_indices,
                        mask_id=int(cfg.inference.mask_id),
                        eos_token_id=(None if cfg.inference.eos_token_id is None else int(cfg.inference.eos_token_id)),
                        steps=int(steps),
                        gen_length=int(gen_length),
                        block_length=int(block_len),
                        temperature=float(temp),
                        top_p=(None if top_p is None else float(top_p)),
                        cfg_scale=float(cfg_scale),
                        remasking=str(remask),
                        logits_eos_inf=bool(cfg.inference.logits_eos_inf),
                        confidence_eos_eot_inf=bool(cfg.inference.confidence_eos_eot_inf),
                        guide_model=guide_model,
                        guidance_mode=guidance_mode,
                        guide_fallback_strategy=guide_fallback_strategy,
                        generator=generator,
                    )
                else:
                    raise ValueError(f"Unsupported generation_mode: {generation_mode}")
            else:
                out_indices = in_indices
            latency_ms = (time.time() - t0) * 1000.0

            output_string = tokenizer.decode(out_indices[0].tolist())
            rec = {
                "prompt": prompt,
                "output": output_string,
                "temperature": float(temp),
                "steps": int(steps),
                "total_length": int(total_len),
                "block_length": int(block_len),
                "top_p": (None if top_p is None else float(top_p)),
                "cfg_scale": float(cfg_scale),
                "remasking": str(remask),
                "seed": (None if seed is None else int(seed)),
                "mask_id": int(cfg.inference.mask_id),
                "eos_token_id": cfg.inference.eos_token_id,
                "latency_ms": float(latency_ms),
            }
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            if records is not None:
                records.append(rec)
            if cfg.sweep.print_every > 0 and (idx % cfg.sweep.print_every == 0):
                print(f"[{idx}/{len(grid)}] temperature={temp} steps={steps} seed={seed}")

    if records is not None:
        html_path = Path(cfg.sweep.html_output_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "prompt",
            "output",
            "temperature",
            "steps",
            "total_length",
            "block_length",
            "top_p",
            "cfg_scale",
            "remasking",
            "seed",
            "latency_ms",
        ]
        with html_path.open("w", encoding="utf-8") as f:
            f.write("<!doctype html>\n")
            f.write("<html>\n<head>\n<meta charset=\"utf-8\">\n")
            f.write("<title>sweep_infer</title>\n")
            f.write(
                "<style>body{font-family:ui-monospace,Menlo,Consolas,monospace;padding:16px;}"
                "table{border-collapse:collapse;width:100%;}"
                "th,td{border:1px solid #ddd;padding:6px;vertical-align:top;}"
                "th{background:#f6f6f6;text-align:left;}"
                "td{white-space:pre-wrap;}</style>\n"
            )
            f.write("</head>\n<body>\n<table>\n<thead><tr>")
            for col in columns:
                f.write(f"<th>{html_escape(col)}</th>")
            f.write("</tr></thead>\n<tbody>\n")
            for rec in records:
                f.write("<tr>")
                for col in columns:
                    val = rec.get(col)
                    f.write(f"<td>{html_escape('' if val is None else str(val))}</td>")
                f.write("</tr>\n")
            f.write("</tbody>\n</table>\n</body>\n</html>\n")


if __name__ == "__main__":
    main()
