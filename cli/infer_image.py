import argparse

from config import load_image_infer_config
from transformerlm.inference.predictor import infer_image
from cli.utils import add_config_args, load_config_or_print
from leash.logger import ConsoleLogger


def _parse_only_config():
    parser = argparse.ArgumentParser(description="Image inference via config file only.", allow_abbrev=False)
    add_config_args(parser, type_=str)
    return parser.parse_args()


def main():
    args_cfg = _parse_only_config()
    cfg_dc = load_config_or_print(load_image_infer_config, args_cfg.config, args_cfg.print_config)
    if cfg_dc is None:
        return

    image_height = cfg_dc.model.image_height
    image_width = cfg_dc.model.image_width
    if image_height is None:
        image_height = cfg_dc.inference.image_height
    if image_width is None:
        image_width = cfg_dc.inference.image_width

    ns = argparse.Namespace(
        # model
        model_type=cfg_dc.model.model_type,
        vocab_size=cfg_dc.model.vocab_size,
        pixel_bins=cfg_dc.model.pixel_bins,
        context_length=cfg_dc.model.context_length,
        d_model=cfg_dc.model.d_model,
        num_layers=cfg_dc.model.num_layers,
        num_heads=cfg_dc.model.num_heads,
        d_ff=cfg_dc.model.d_ff,
        rope_theta=cfg_dc.model.rope_theta,
        label_vocab_size=cfg_dc.model.label_vocab_size,
        null_label_id=cfg_dc.model.null_label_id,
        use_rope_2d=cfg_dc.model.use_rope_2d,
        image_height=image_height,
        image_width=image_width,
        attention_backend=cfg_dc.model.attention_backend,
        attention_sdp_backend=cfg_dc.model.attention_sdp_backend,
        device=cfg_dc.model.device,
        dtype=cfg_dc.model.dtype,
        mask_id=cfg_dc.inference.mask_id,
        # checkpoint
        ckpt_path=str(cfg_dc.checkpoint.ckpt_path),
        # inference
        generation_mode=cfg_dc.inference.generation_mode,
        label=cfg_dc.inference.label,
        num_samples=cfg_dc.inference.num_samples,
        steps=cfg_dc.inference.steps,
        block_length=cfg_dc.inference.block_length,
        temperature=cfg_dc.inference.temperature,
        top_p=cfg_dc.inference.top_p,
        cfg_scale=cfg_dc.inference.cfg_scale,
        remasking=cfg_dc.inference.remasking,
        seed=cfg_dc.inference.seed,
        output_dir=str(cfg_dc.inference.output_dir),
    )
    logger = ConsoleLogger()
    _ = infer_image(ns, logger=logger)


if __name__ == "__main__":
    main()
