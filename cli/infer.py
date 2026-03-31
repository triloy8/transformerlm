import argparse

from config import load_infer_config
from transformerlm.inference.predictor import infer_transformer
from cli.utils import add_config_args, load_config_or_print
from leash.logger import ConsoleLogger


def _parse_only_config():
    parser = argparse.ArgumentParser(description="Inference via config file only.", allow_abbrev=False)
    add_config_args(parser, type_=str)
    return parser.parse_args()


def main():
    args_cfg = _parse_only_config()
    cfg_dc = load_config_or_print(load_infer_config, args_cfg.config, args_cfg.print_config)
    if cfg_dc is None:
        return

    # Build args Namespace expected by infer_transformer
    ns = argparse.Namespace(
        # tokenizer
        merges_path=str(cfg_dc.tokenizer.merges_path),
        vocab_path=str(cfg_dc.tokenizer.vocab_path),
        special_tokens_path=str(cfg_dc.tokenizer.special_tokens_path),
        # model
        vocab_size=cfg_dc.model.vocab_size,
        context_length=cfg_dc.model.context_length,
        d_model=cfg_dc.model.d_model,
        num_layers=cfg_dc.model.num_layers,
        num_heads=cfg_dc.model.num_heads,
        d_ff=cfg_dc.model.d_ff,
        rope_theta=cfg_dc.model.rope_theta,
        attention_backend=cfg_dc.model.attention_backend,
        attention_sdp_backend=cfg_dc.model.attention_sdp_backend,
        device=cfg_dc.model.device,
        dtype=cfg_dc.model.dtype,
        mask_token_id=cfg_dc.model.mask_token_id,
        noise_epsilon=cfg_dc.model.noise_epsilon,
        random_trunc_prob=cfg_dc.model.random_trunc_prob,
        # checkpoint
        ckpt_path=str(cfg_dc.checkpoint.ckpt_path),
        # inference
        prompt=cfg_dc.inference.prompt,
        steps=cfg_dc.inference.steps,
        total_length=cfg_dc.inference.total_length,
        block_length=cfg_dc.inference.block_length,
        temperature=cfg_dc.inference.temperature,
        mask_id=cfg_dc.inference.mask_id,
        seed=cfg_dc.inference.seed,
        eos_token_id=cfg_dc.inference.eos_token_id,
        top_p=cfg_dc.inference.top_p,
        cfg_scale=cfg_dc.inference.cfg_scale,
        remasking=cfg_dc.inference.remasking,
        logits_eos_inf=cfg_dc.inference.logits_eos_inf,
        confidence_eos_eot_inf=cfg_dc.inference.confidence_eos_eot_inf,
        generation_mode=cfg_dc.inference.generation_mode,
        # optional AR guide
        guide_model=(cfg_dc.guide.model if cfg_dc.guide is not None else None),
        guide_ckpt_path=(str(cfg_dc.guide.checkpoint.ckpt_path) if cfg_dc.guide is not None else None),
        guidance_mode=(cfg_dc.guide.guidance_mode if cfg_dc.guide is not None else "none"),
        guide_fallback_strategy=(cfg_dc.guide.fallback_strategy if cfg_dc.guide is not None else "single_token"),
    )
    logger = ConsoleLogger()
    _ = infer_transformer(ns, logger=logger)


if __name__ == "__main__":
    main()
