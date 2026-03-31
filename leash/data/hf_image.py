from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from datasets import load_dataset

from leash.data.image import DiscreteImageBatcher, _quantize_pixels_uint8


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


__all__ = ["build_mnist_batcher"]
