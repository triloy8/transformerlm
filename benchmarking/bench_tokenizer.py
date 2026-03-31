from __future__ import annotations

import argparse
import warnings
from typing import List, Tuple

from cli.utils import add_config_args, load_config_or_print
from config import load_bench_tokenizer_config
from transformerlm.tokenizer import PythonTokenizer, RustTokenizer
from leash.logger import ConsoleLogger

from .common import measure, mean


def _parse_only_config():
    parser = argparse.ArgumentParser(description="Benchmark: tokenizer throughput via config.", allow_abbrev=False)
    add_config_args(parser, type_=str)
    return parser.parse_args()


def main():
    args_cfg = _parse_only_config()
    cfg = load_config_or_print(load_bench_tokenizer_config, args_cfg.config, args_cfg.print_config)
    if cfg is None:
        return

    logger = ConsoleLogger()

    def _load_tokenizers() -> List[Tuple[str, object]]:
        impls: List[Tuple[str, object]] = []
        common_kwargs = dict(
            vocab_filepath=str(cfg.tokenizer.vocab_path),
            merges_filepath=str(cfg.tokenizer.merges_path),
            special_tokens_path=str(cfg.tokenizer.special_tokens_path),
        )

        impls.append(("python", PythonTokenizer.from_files(**common_kwargs)))

        if RustTokenizer is None:
            warnings.warn(
                "Rust tokenizer unavailable for benchmarking; skipping Rust run.",
                RuntimeWarning,
                stacklevel=2,
            )
            return impls

        impls.append(("rust", RustTokenizer.from_files(**common_kwargs)))

        return impls

    texts: List[str] = list(cfg.input.text_list)
    tokenizers = _load_tokenizers()

    run_config = {
        "benchmark": "tokenizer_throughput",
        "texts": len(texts),
        "repeats": cfg.benchmark.repeats,
        "tokenizer_backends": [name for name, _ in tokenizers],
    }
    logger.start_run(run_config)

    for backend_name, tokenizer in tokenizers:
        encoded_once = [tokenizer.encode(t) for t in texts]
        token_counts = [len(x) for x in encoded_once]
        total_tokens = int(sum(token_counts))

        encode_lat_ms: List[float] = []
        encode_tps: List[float] = []
        for r in range(cfg.benchmark.repeats):
            def _encode_all():
                out = []
                for t in texts:
                    out.append(tokenizer.encode(t))
                return out

            out, dt = measure("cpu", _encode_all)
            toks = int(sum(len(x) for x in out))
            lat_ms = dt * 1000.0
            tps = (float(toks) / dt) if dt > 0 else 0.0
            encode_lat_ms.append(lat_ms)
            encode_tps.append(tps)
            logger.log({
                "phase": "bench_tokenizer",
                "tokenizer.backend": backend_name,
                "op": "encode",
                "metrics.latency_ms": lat_ms,
                "metrics.tokens_sec": tps,
                "metrics.tokens": toks,
            }, step=r)

        decode_lat_ms: List[float] = []
        decode_tps: List[float] = []
        for r in range(cfg.benchmark.repeats):
            def _decode_all():
                out = []
                for ids in encoded_once:
                    out.append(tokenizer.decode(ids))
                return out

            out, dt = measure("cpu", _decode_all)
            lat_ms = dt * 1000.0
            tps = (float(total_tokens) / dt) if dt > 0 else 0.0
            decode_lat_ms.append(lat_ms)
            decode_tps.append(tps)
            logger.log({
                "phase": "bench_tokenizer",
                "tokenizer.backend": backend_name,
                "op": "decode",
                "metrics.latency_ms": lat_ms,
                "metrics.tokens_sec": tps,
                "metrics.tokens": total_tokens,
            }, step=r)

        logger.log({
            "phase": "bench_tokenizer",
            "tokenizer.backend": backend_name,
            "event": "summary",
            "metrics.encode.latency_ms.mean": mean(encode_lat_ms),
            "metrics.encode.tokens_sec.mean": mean(encode_tps),
            "metrics.decode.latency_ms.mean": mean(decode_lat_ms),
            "metrics.decode.tokens_sec.mean": mean(decode_tps),
        })

    logger.finish()


if __name__ == "__main__":
    main()
