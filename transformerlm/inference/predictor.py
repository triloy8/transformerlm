from typing import Optional

import torch
import time
from safetensors.torch import load_file

from transformerlm.tokenizer.tokenizer import Tokenizer
from transformerlm.models import TransformerLM, TransformerImage, DiTImage, CategoricalFlowImage
from transformerlm.models.attention import set_sdp_backend
from trainkit.inference.generate import (
    autoregressive_generate,
    diffusion_generate,
    image_diffusion_generate,
    categorical_flow_image_generate,
    flow_image_generate,
)
from transformerlm.utils.dtypes import DTYPES
from trainkit.logger import Logger
from trainkit.data.image import dequantize_tokens_to_uint8, flow_pixels_to_uint8


def _normalize_state_dict_keys(state_dict):
    if not state_dict:
        return state_dict
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    return state_dict


def infer_transformer(args, *, logger: Optional[Logger] = None, artifact_path: Optional[str] = None):
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
        special_tokens_path=args.special_tokens_path,
    )
    prompt_text = args.prompt
    ids = [tokenizer.encode(prompt_text)]
    prompt_len = len(ids[0])
    total_length = int(args.total_length)
    if total_length < prompt_len:
        raise ValueError("total_length must be >= prompt token length")
    gen_length = total_length - prompt_len
    if total_length > args.context_length:
        raise ValueError("total_length must be <= model context_length")

    set_sdp_backend(getattr(args, "attention_sdp_backend", "auto"))
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        attention_backend=getattr(args, "attention_backend", "custom"),
        device=args.device,
        dtype=DTYPES[args.dtype],
    )

    model_state = _normalize_state_dict_keys(load_file(str(args.ckpt_path)))
    model.load_state_dict(model_state)
    model.eval()

    guide_model = None
    guidance_mode = str(getattr(args, "guidance_mode", "none"))
    guide_fallback_strategy = str(getattr(args, "guide_fallback_strategy", "single_token"))
    guide_cfg = getattr(args, "guide_model", None)
    guide_ckpt_path = getattr(args, "guide_ckpt_path", None)
    if guide_cfg is not None and guide_ckpt_path is not None and guidance_mode != "none":
        guide_model = TransformerLM(
            vocab_size=guide_cfg.vocab_size,
            context_length=guide_cfg.context_length,
            d_model=guide_cfg.d_model,
            num_layers=guide_cfg.num_layers,
            num_heads=guide_cfg.num_heads,
            d_ff=guide_cfg.d_ff,
            rope_theta=guide_cfg.rope_theta,
            attention_backend=getattr(guide_cfg, "attention_backend", "custom"),
            device=guide_cfg.device,
            dtype=DTYPES[guide_cfg.dtype],
        )
        guide_state = _normalize_state_dict_keys(load_file(str(guide_ckpt_path)))
        guide_model.load_state_dict(guide_state)
        guide_model.eval()

    in_indices = torch.tensor(ids, device=args.device)
    eos_token_id = getattr(args, "eos_token_id", None)
    top_p = getattr(args, "top_p", None)
    cfg_scale = float(getattr(args, "cfg_scale", 0.0))
    remasking = getattr(args, "remasking", "random")
    logits_eos_inf = bool(getattr(args, "logits_eos_inf", False))
    confidence_eos_eot_inf = bool(getattr(args, "confidence_eos_eot_inf", False))
    generation_mode = getattr(args, "generation_mode", "diffusion")
    seed = getattr(args, "seed", None)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=args.device)
        generator.manual_seed(int(seed))

    # Log static sampling parameters once
    if logger is not None:
        logger.log(
            {
                "phase": "infer",
                "params.temperature": float(args.temperature),
                "params.steps": int(args.steps),
                "params.total_length": int(args.total_length),
                "params.gen_length": int(gen_length),
                "params.block_length": int(args.block_length),
                "params.mask_id": int(args.mask_id),
                "params.seed": (None if seed is None else int(seed)),
                "params.eos_token_id": (None if eos_token_id is None else int(eos_token_id)),
                "params.top_p": (None if top_p is None else float(top_p)),
                "params.cfg_scale": float(cfg_scale),
                "params.remasking": str(remasking),
                "params.logits_eos_inf": bool(logits_eos_inf),
                "params.confidence_eos_eot_inf": bool(confidence_eos_eot_inf),
                "params.guidance_mode": str(guidance_mode),
                "params.guide_fallback_strategy": str(guide_fallback_strategy),
            }
        )

    t0 = time.time()
    if gen_length > 0:
        if generation_mode == "ar":
            out_indices = autoregressive_generate(
                model,
                in_indices,
                gen_length=int(gen_length),
                temperature=float(args.temperature),
                top_p=(None if top_p is None else float(top_p)),
                eos_token_id=(None if eos_token_id is None else int(eos_token_id)),
                logits_eos_inf=bool(logits_eos_inf),
                generator=generator,
            )
        elif generation_mode == "diffusion":
            out_indices = diffusion_generate(
                model,
                in_indices,
                mask_id=int(args.mask_id),
                eos_token_id=(None if eos_token_id is None else int(eos_token_id)),
                steps=int(args.steps),
                gen_length=int(gen_length),
                block_length=int(args.block_length),
                temperature=float(args.temperature),
                top_p=(None if top_p is None else float(top_p)),
                cfg_scale=float(cfg_scale),
                remasking=str(remasking),
                logits_eos_inf=bool(logits_eos_inf),
                confidence_eos_eot_inf=bool(confidence_eos_eot_inf),
                guide_model=guide_model,
                guidance_mode=guidance_mode,
                guide_fallback_strategy=guide_fallback_strategy,
                generator=generator,
            )
        else:
            raise ValueError(f"Unsupported generation_mode: {generation_mode}")
    else:
        out_indices = in_indices
    elapsed = time.time() - t0

    out_indices_list = out_indices[0].tolist()
    output_string = tokenizer.decode(out_indices_list)
    if logger is not None:
        logger.log(
            {
                "phase": "infer",
                "text.prompt": (prompt_text[:200] + ("…" if len(prompt_text) > 200 else "")),
                "text.output": (output_string[:200] + ("…" if len(output_string) > 200 else "")),
                "metrics.prompt_len": int(len(ids[0])),
                "metrics.output_len": int(len(out_indices_list)),
                "metrics.latency_ms": float(elapsed * 1000.0),
            }
        )
        print(output_string)

    # Optional artifact logging of full predictions
    if logger is not None and artifact_path:
        try:
            import json
            with open(artifact_path, "w", encoding="utf-8") as f:
                rec = {
                    "prompt": prompt_text,
                    "output": output_string,
                    "temperature": args.temperature,
                    "steps": args.steps,
                    "total_length": args.total_length,
                    "gen_length": gen_length,
                    "block_length": args.block_length,
                    "mask_id": args.mask_id,
                    "seed": seed,
                    "eos_token_id": eos_token_id,
                    "top_p": top_p,
                    "cfg_scale": cfg_scale,
                    "remasking": remasking,
                    "logits_eos_inf": logits_eos_inf,
                    "confidence_eos_eot_inf": confidence_eos_eot_inf,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.log_artifact(artifact_path, name=artifact_path, type_="inference_outputs")
        except Exception:
            pass

    return [output_string]


def infer_image(args, *, logger: Optional[Logger] = None):
    set_sdp_backend(getattr(args, "attention_sdp_backend", "auto"))
    model_type = str(getattr(args, "model_type", "image")).lower()
    if model_type == "image_dit":
        model = DiTImage(
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
            label_vocab_size=args.label_vocab_size,
            attention_backend=getattr(args, "attention_backend", "custom"),
            image_height=getattr(args, "image_height", None),
            image_width=getattr(args, "image_width", None),
            use_rope_2d=bool(getattr(args, "use_rope_2d", False)),
            device=args.device,
            dtype=DTYPES[args.dtype],
        )
    elif model_type == "image_cfm":
        model = CategoricalFlowImage(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
            label_vocab_size=args.label_vocab_size,
            attention_backend=getattr(args, "attention_backend", "custom"),
            image_height=getattr(args, "image_height", None),
            image_width=getattr(args, "image_width", None),
            use_rope_2d=bool(getattr(args, "use_rope_2d", False)),
            device=args.device,
            dtype=DTYPES[args.dtype],
        )
    elif model_type == "image":
        model = TransformerImage(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
            label_vocab_size=args.label_vocab_size,
            attention_backend=getattr(args, "attention_backend", "custom"),
            image_height=getattr(args, "image_height", None),
            image_width=getattr(args, "image_width", None),
            use_rope_2d=bool(getattr(args, "use_rope_2d", False)),
            device=args.device,
            dtype=DTYPES[args.dtype],
        )
    else:
        raise ValueError("infer_image requires model_type to be one of: image, image_dit, image_cfm")
    model.eval()

    model_state = _normalize_state_dict_keys(load_file(str(args.ckpt_path)))
    model.load_state_dict(model_state)

    num_samples = int(args.num_samples)
    label = int(args.label)
    device = args.device
    context = torch.full((num_samples,), label, device=device, dtype=torch.long)
    null_label_id = getattr(args, "null_label_id", None)
    prompt = torch.empty((num_samples, 0), device=device, dtype=torch.long)
    gen_length = int(args.context_length)

    seed = getattr(args, "seed", None)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    cfg_scale = float(args.cfg_scale)
    uncond_context = None
    if cfg_scale > 0.0:
        if null_label_id is None:
            raise ValueError(
                "cfg_scale > 0 for infer_image requires model.null_label_id to be configured "
                "to a dedicated unconditional label embedding"
            )
        uncond_context = torch.full((num_samples,), int(null_label_id), device=device, dtype=torch.long)

    generation_mode = str(getattr(args, "generation_mode", "diffusion")).lower()
    if generation_mode == "diffusion":
        out = image_diffusion_generate(
            model,
            prompt,
            context=context,
            mask_id=int(args.mask_id),
            eos_token_id=None,
            steps=int(args.steps),
            gen_length=int(gen_length),
            block_length=int(args.block_length),
            temperature=float(args.temperature),
            top_p=(None if args.top_p is None else float(args.top_p)),
            cfg_scale=cfg_scale,
            uncond_context=uncond_context,
            remasking=str(args.remasking),
            logits_eos_inf=False,
            confidence_eos_eot_inf=False,
            generator=generator,
        )
    elif generation_mode == "flow":
        out = flow_image_generate(
            model,
            prompt,
            context=context,
            steps=int(args.steps),
            cfg_scale=cfg_scale,
            uncond_context=uncond_context,
            generator=generator,
        )
    elif generation_mode == "categorical_flow":
        out = categorical_flow_image_generate(
            model,
            prompt,
            context=context,
            steps=int(args.steps),
            temperature=float(args.temperature),
            top_p=(None if args.top_p is None else float(args.top_p)),
            cfg_scale=cfg_scale,
            uncond_context=uncond_context,
            prior_type=str(getattr(args, "prior_type", "discunif")),
            generator=generator,
        )
    else:
        raise ValueError("generation_mode must be one of: diffusion, flow, categorical_flow")

    h = getattr(args, "image_height", None)
    w = getattr(args, "image_width", None)
    pixel_bins = int(getattr(args, "pixel_bins", 256))
    if h is None or w is None:
        side = int(gen_length ** 0.5)
        if side * side != gen_length:
            raise ValueError("image_height/image_width must be set when context_length is not a square")
        h = side
        w = side

    from pathlib import Path
    from PIL import Image

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    if num_samples <= 1:
        sample = out[0].detach().cpu().numpy().reshape(int(h), int(w))
        if generation_mode == "flow":
            arr = flow_pixels_to_uint8(sample)
        else:
            tokens = sample.astype("int32")
            arr = dequantize_tokens_to_uint8(tokens, pixel_bins=pixel_bins)
        img = Image.fromarray(arr, mode="L")
        path = out_dir / f"label_{label}_sample_0.png"
        img.save(path)
        outputs.append(str(path))
    else:
        import math

        cols = int(math.ceil(math.sqrt(num_samples)))
        rows = int(math.ceil(num_samples / cols))
        grid = Image.new("L", (cols * int(w), rows * int(h)))
        for i in range(num_samples):
            sample = out[i].detach().cpu().numpy().reshape(int(h), int(w))
            if generation_mode == "flow":
                arr = flow_pixels_to_uint8(sample)
            else:
                tokens = sample.astype("int32")
                arr = dequantize_tokens_to_uint8(tokens, pixel_bins=pixel_bins)
            img = Image.fromarray(arr, mode="L")
            r = i // cols
            c = i % cols
            grid.paste(img, (c * int(w), r * int(h)))
        path = out_dir / f"label_{label}_samples.png"
        grid.save(path)
        outputs.append(str(path))

    if logger is not None:
        logger.log(
            {
                "phase": "infer",
                "metrics.num_samples": int(num_samples),
                "params.label": int(label),
                "params.pixel_bins": int(pixel_bins),
                "params.generation_mode": generation_mode,
                "params.output_dir": str(out_dir),
            }
        )
    return outputs
