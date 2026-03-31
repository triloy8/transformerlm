from __future__ import annotations

from typing import Iterable, Iterator, Optional, Protocol
import time
import random

from datasets import load_dataset
from datasets.utils import logging as hf_logging
import torch

from leash.logger import Logger


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]:
        ...


class HFTokenIteratorFactory:
    """Constructs token streams from Hugging Face streaming datasets."""

    def __init__(
        self,
        *,
        dataset_name: str,
        dataset_config: Optional[str],
        split: str,
        text_field: str,
        tokenizer: TokenizerLike,
        context_length: int,
        eot_token_id: int,
        shuffle_buffer_size: int = 0,
        shuffle_seed: Optional[int] = None,
        cache_all: bool = False,
        world_size: int = 1,
        rank: int = 0,
        pad_newline: bool = True,
        logger: Optional[Logger] = None,
        hf_debug_logging: bool = False,
        slow_row_s: float = 10.0,
        slow_encode_s: float = 10.0,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.text_field = text_field
        self.tokenizer = tokenizer
        self.context_length = int(context_length)
        self.eot_token_id = int(eot_token_id)
        self.shuffle_buffer_size = max(0, shuffle_buffer_size)
        self.shuffle_seed = shuffle_seed if shuffle_seed is not None else 0
        self.cache_all = bool(cache_all)
        self.world_size = max(1, world_size)
        self.rank = max(0, min(rank, self.world_size - 1))
        self.pad_newline = pad_newline
        self._epoch = 0
        self._logger = logger
        self._slow_row_s = float(slow_row_s)
        self._slow_encode_s = float(slow_encode_s)
        if hf_debug_logging:
            hf_logging.set_verbosity_debug()

    def _load_dataset(self):
        if self.dataset_config:
            return load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=self.split,
                streaming=not self.cache_all,
            )
        return load_dataset(self.dataset_name, split=self.split, streaming=not self.cache_all)

    def _apply_shuffle(self, dataset):
        if self.shuffle_buffer_size <= 0:
            return dataset
        seed = self.shuffle_seed + self._epoch
        try:
            return dataset.shuffle(buffer_size=self.shuffle_buffer_size, seed=seed)
        except TypeError:
            return dataset.shuffle(seed=seed)

    def _manual_shard(self, iterable: Iterable) -> Iterator:
        for idx, example in enumerate(iterable):
            if idx % self.world_size == self.rank:
                yield example

    def _apply_shard(self, dataset) -> Iterator:
        if self.world_size <= 1:
            return iter(dataset)
        try:
            sharded = dataset.shard(num_shards=self.world_size, index=self.rank)
            num_shards = getattr(sharded, "n_shards", None) or getattr(sharded, "num_shards", None)
            if num_shards is not None and num_shards >= self.world_size:
                return iter(sharded)
            # Fall back when HF dataset cannot provide enough shards.
            return self._manual_shard(iter(dataset))
        except Exception:
            return self._manual_shard(iter(dataset))

    def __call__(self) -> Iterator[list[int]]:
        dataset = self._load_dataset()
        dataset = self._apply_shuffle(dataset)
        rows = iter(self._apply_shard(dataset))
        self._epoch += 1
        if self._logger is not None:
            self._logger.log(
                {
                    "metrics.data/stream_epoch": int(self._epoch),
                    "metrics.data/stream_reset": 1,
                }
            )
        while True:
            row_fetch_start = time.monotonic()
            try:
                row = next(rows)
            except StopIteration:
                break
            row_fetch_elapsed = time.monotonic() - row_fetch_start
            if self._logger is not None and row_fetch_elapsed >= self._slow_row_s:
                self._logger.log(
                    {
                        "metrics.streaming_row_fetch/elapsed_s": float(row_fetch_elapsed),
                        "metrics.streaming_row_fetch/epoch": int(self._epoch),
                        "metrics.streaming_row_fetch/rank": int(self.rank),
                        "metrics.streaming_row_fetch/world_size": int(self.world_size),
                    }
                )

            if row is None:
                continue
            text_value = row.get(self.text_field) if isinstance(row, dict) else None
            if text_value is None:
                continue
            if not isinstance(text_value, str):
                text_value = str(text_value)
            normalized = text_value
            if self.pad_newline and not normalized.endswith("\n"):
                normalized = f"{normalized}\n"
            encode_start = time.monotonic()
            token_ids = self.tokenizer.encode(normalized)
            encode_elapsed = time.monotonic() - encode_start
            if self._logger is not None and encode_elapsed >= self._slow_encode_s:
                self._logger.log(
                    {
                        "metrics.streaming_encode/elapsed_s": float(encode_elapsed),
                        "metrics.streaming_encode/epoch": int(self._epoch),
                        "metrics.streaming_encode/rank": int(self.rank),
                        "metrics.streaming_encode/world_size": int(self.world_size),
                    }
                )
            tokens = list(token_ids)
            tokens.append(self.eot_token_id)
            yield tokens


class StreamingBatcher:
    """Packed-stream batcher that emits fixed-length sequences for batching."""

    def __init__(
        self,
        iterator_factory: HFTokenIteratorFactory,
        *,
        device: str | torch.device,
        logger: Optional[Logger] = None,
    ) -> None:
        self.iterator_factory = iterator_factory
        self.device = torch.device(device)
        self._iterator: Iterator[list[int]] = self.iterator_factory()
        self._buffer: list[int] = []
        self._logger = logger

    def _extend_buffer(self) -> None:
        while True:
            try:
                self._buffer.extend(next(self._iterator))
                return
            except StopIteration:
                self._iterator = self.iterator_factory()

    def draw(self, batch_size: int, context_length: int) -> torch.Tensor:
        if context_length <= 0:
            raise ValueError("context_length must be > 0")
        sequences: list[list[int]] = []
        for _ in range(batch_size):
            while len(self._buffer) < context_length:
                self._extend_buffer()
            seq = self._buffer[:context_length]
            self._buffer = self._buffer[context_length:]
            sequences.append(seq)
        clean_targets = torch.tensor(sequences, dtype=torch.long, device=self.device)
        return clean_targets

    def get_state(self) -> dict:
        state = {
            "buffer": list(self._buffer),
            "exact": False,
        }
        if hasattr(self.iterator_factory, "get_state"):
            state["iterator_factory"] = self.iterator_factory.get_state()
        return state

    def set_state(self, state: dict) -> None:
        self._buffer = list(state.get("buffer", []))
        self._iterator = self.iterator_factory()
        if hasattr(self.iterator_factory, "set_state") and "iterator_factory" in state:
            self.iterator_factory.set_state(state["iterator_factory"])


class RowBatcher:
    """Row-based batcher that pads each row to a fixed context length."""

    def __init__(
        self,
        iterator_factory: HFTokenIteratorFactory,
        *,
        device: str | torch.device,
        pad_token_id: int,
        pad_random_shift: bool = False,
        logger: Optional[Logger] = None,
    ) -> None:
        self.iterator_factory = iterator_factory
        self.device = torch.device(device)
        self.pad_token_id = int(pad_token_id)
        self.pad_random_shift = bool(pad_random_shift)
        self._iterator: Iterator[list[int]] = self.iterator_factory()
        self._logger = logger

    def _next_row(self) -> list[int]:
        while True:
            try:
                return next(self._iterator)
            except StopIteration:
                self._iterator = self.iterator_factory()

    def draw(self, batch_size: int, context_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        if context_length <= 0:
            raise ValueError("context_length must be > 0")
        sequences: list[list[int]] = []
        masks: list[list[bool]] = []
        for _ in range(batch_size):
            tokens = list(self._next_row())
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
            valid_len = len(tokens)
            pad_len = context_length - valid_len
            if pad_len > 0:
                if self.pad_random_shift:
                    left_pad = random.randint(0, pad_len)
                    right_pad = pad_len - left_pad
                    tokens = ([self.pad_token_id] * left_pad) + tokens + ([self.pad_token_id] * right_pad)
                    mask = ([False] * left_pad) + ([True] * valid_len) + ([False] * right_pad)
                else:
                    tokens = tokens + [self.pad_token_id] * pad_len
                    mask = [True] * valid_len + [False] * pad_len
            else:
                mask = [True] * context_length
            sequences.append(tokens)
            masks.append(mask)
        token_tensor = torch.tensor(sequences, dtype=torch.long, device=self.device)
        mask_tensor = torch.tensor(masks, dtype=torch.bool, device=self.device)
        return token_tensor, mask_tensor

    def get_state(self) -> dict:
        state = {
            "exact": False,
        }
        if hasattr(self.iterator_factory, "get_state"):
            state["iterator_factory"] = self.iterator_factory.get_state()
        return state

    def set_state(self, state: dict) -> None:
        self._iterator = self.iterator_factory()
        if hasattr(self.iterator_factory, "set_state") and "iterator_factory" in state:
            self.iterator_factory.set_state(state["iterator_factory"])
