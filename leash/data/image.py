from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import random

import numpy as np
import torch
from datasets import load_dataset


def _tupleify(value):
    if isinstance(value, list):
        return tuple(_tupleify(v) for v in value)
    if isinstance(value, dict):
        return {k: _tupleify(v) for k, v in value.items()}
    return value


def _quantize_pixels_uint8(pixels: np.ndarray, *, pixel_bins: int) -> np.ndarray:
    if pixel_bins == 256:
        return pixels
    # Uniform bucketization over [0, 255] -> [0, pixel_bins - 1].
    quant = (pixels.astype(np.uint16) * int(pixel_bins)) // 256
    return np.minimum(quant, int(pixel_bins) - 1).astype(np.uint8)


def dequantize_tokens_to_uint8(tokens: np.ndarray, *, pixel_bins: int) -> np.ndarray:
    if pixel_bins == 256:
        return tokens.astype(np.uint8)
    # Use bin centers for visualization in [0, 255].
    vals = np.clip(tokens.astype(np.int32), 0, int(pixel_bins) - 1)
    scale = 256.0 / float(pixel_bins)
    restored = np.round((vals + 0.5) * scale - 0.5)
    return np.clip(restored, 0, 255).astype(np.uint8)


def flow_pixels_to_uint8(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float32), -1.0, 1.0)
    restored = np.round((clipped + 1.0) * 127.5)
    return np.clip(restored, 0, 255).astype(np.uint8)


@dataclass
class ImageBatch:
    tokens: torch.Tensor
    labels: torch.Tensor


class DiscreteImageBatcher:
    """Local image batcher that returns fixed-length discrete token sequences."""

    def __init__(
        self,
        *,
        images: np.ndarray,
        labels: Optional[list[int]],
        device: str | torch.device,
        shuffle: bool = True,
        shuffle_seed: Optional[int] = None,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self._images = torch.from_numpy(images).to(dtype=torch.uint8)
        self._labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

        total = int(self._images.shape[0])
        world_size = max(1, int(world_size))
        rank = max(0, min(int(rank), world_size - 1))
        self._indices = list(range(rank, total, world_size))
        if not self._indices:
            raise ValueError("sharded MNIST dataset is empty for this rank")

        self._rng = random.Random(int(shuffle_seed) if shuffle_seed is not None else 0)
        self._shuffle = bool(shuffle)
        self._order = list(self._indices)
        if self._shuffle:
            self._rng.shuffle(self._order)
        self._cursor = 0

        self.sequence_length = int(self._images.shape[1])

    def _next_indices(self, batch_size: int) -> list[int]:
        out: list[int] = []
        for _ in range(batch_size):
            if self._cursor >= len(self._order):
                self._cursor = 0
                if self._shuffle:
                    self._rng.shuffle(self._order)
            out.append(self._order[self._cursor])
            self._cursor += 1
        return out

    def draw(self, batch_size: int, context_length: int) -> ImageBatch | torch.Tensor:
        if context_length != self.sequence_length:
            raise ValueError(
                f"context_length must equal {self.sequence_length} (got {context_length})"
            )
        idxs = self._next_indices(batch_size)
        pixels = self._images[idxs].to(device=self.device, dtype=torch.long)
        if self._labels is None:
            return pixels
        labels = self._labels[idxs].to(device=self.device, dtype=torch.long)
        return ImageBatch(tokens=pixels, labels=labels)

    def get_state(self) -> dict:
        return {
            "order": list(self._order),
            "cursor": int(self._cursor),
            "rng_state": self._rng.getstate(),
            "exact": True,
        }

    def set_state(self, state: dict) -> None:
        if state is None:
            return
        if "order" in state:
            self._order = list(state["order"])
        if "cursor" in state:
            self._cursor = int(state["cursor"])
        if "rng_state" in state:
            self._rng.setstate(_tupleify(state["rng_state"]))


def _load_hf_dataset(dataset_name: str, dataset_config: Optional[str], split: str):
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, split=split, streaming=False)
    return load_dataset(dataset_name, split=split, streaming=False)


def _extract_hf_images(
    dataset,
    *,
    include_label: bool,
    pixel_bins: int = 256,
) -> tuple[np.ndarray, list[int] | None]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    for example in dataset:
        image = example.get("image") if isinstance(example, dict) else None
        if image is None:
            continue
        arr = np.array(image, dtype=np.uint8)
        if arr.ndim > 2:
            arr = arr.squeeze()
        if arr.ndim != 2:
            raise ValueError("image must be 2D")
        images.append(arr.reshape(-1))
        if include_label:
            labels.append(int(example.get("label", 0)))
    if not images:
        raise ValueError("dataset contains no usable images")
    stacked = np.stack(images, axis=0)
    quantized = _quantize_pixels_uint8(stacked, pixel_bins=pixel_bins)
    return quantized, (labels if include_label else None)


def build_mnist_batcher(
    *,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    device: str | torch.device,
    pixel_bins: int = 256,
    shuffle: bool = True,
    shuffle_seed: Optional[int] = None,
    world_size: int = 1,
    rank: int = 0,
) -> DiscreteImageBatcher:
    if pixel_bins <= 1 or pixel_bins > 256:
        raise ValueError("pixel_bins must be in [2, 256]")
    dataset = _load_hf_dataset(dataset_name, dataset_config, split)
    images, labels = _extract_hf_images(dataset, include_label=True, pixel_bins=int(pixel_bins))
    return DiscreteImageBatcher(
        images=images,
        labels=labels,
        device=device,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        world_size=world_size,
        rank=rank,
    )
