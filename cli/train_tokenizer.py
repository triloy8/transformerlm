import argparse

from config import load_train_tokenizer_config
from transformerlm.tokenizer.bpe_trainer import train_bpe
from cli.utils import add_config_args, load_config_or_print
from leash.logger import ConsoleLogger


def _parse_only_config():
    parser = argparse.ArgumentParser(description="Train tokenizer via config file only.", allow_abbrev=False)
    add_config_args(parser, type_=str)
    return parser.parse_args()


def main():
    args_cfg = _parse_only_config()
    cfg_dc = load_config_or_print(load_train_tokenizer_config, args_cfg.config, args_cfg.print_config)
    if cfg_dc is None:
        return

    # Build args Namespace compatible with existing train_bpe signature
    ns = argparse.Namespace(
        input_path=str(cfg_dc.input.input_path),
        vocab_size=cfg_dc.input.vocab_size,
        special_tokens=list(cfg_dc.input.special_tokens),
        merges_path=str(cfg_dc.output.merges_path),
        vocab_path=str(cfg_dc.output.vocab_path),
    )
    logger = ConsoleLogger()
    train_bpe(ns, logger=logger)


if __name__ == "__main__":
    main()
