#!/usr/bin/env bash
set -euo pipefail

# Quick preview of what MNIST samples look like after dataloader binning.
# Usage:
#   bash scripts/preview_mnist_binned_dataloader.sh [pixel_bins] [num_samples] [split] [output_dir]
#
# Example:
#   bash scripts/preview_mnist_binned_dataloader.sh 32 64 train runs/mnist_preview

PIXEL_BINS="${1:-32}"
NUM_SAMPLES="${2:-64}"
SPLIT="${3:-train}"
OUTPUT_DIR="${4:-runs/mnist_binned_preview}"

export PIXEL_BINS
export NUM_SAMPLES
export SPLIT
export OUTPUT_DIR

uv run python - <<'PY'
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from leash.data.image import build_mnist_batcher, dequantize_tokens_to_uint8

pixel_bins = int(os.environ["PIXEL_BINS"])
num_samples = int(os.environ["NUM_SAMPLES"])
split = str(os.environ["SPLIT"])
output_dir = Path(os.environ["OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)

batcher = build_mnist_batcher(
    dataset_name="ylecun/mnist",
    dataset_config=None,
    split=split,
    device="cpu",
    pixel_bins=pixel_bins,
    shuffle=False,
    shuffle_seed=3407,
    world_size=1,
    rank=0,
)

context_length = int(batcher.sequence_length)
side = int(math.isqrt(context_length))
if side * side != context_length:
    raise ValueError(f"Expected square image sequence length, got {context_length}")

drawn = batcher.draw(batch_size=num_samples, context_length=context_length)
if hasattr(drawn, "tokens"):
    tokens = drawn.tokens.detach().cpu().numpy().astype(np.int32)
    labels = drawn.labels.detach().cpu().numpy().astype(np.int32)
else:
    tokens = drawn.detach().cpu().numpy().astype(np.int32)
    labels = np.full((num_samples,), -1, dtype=np.int32)

tokens = tokens.reshape(num_samples, side, side)
dequant = dequantize_tokens_to_uint8(tokens, pixel_bins=pixel_bins)

cols = int(math.ceil(math.sqrt(num_samples)))
rows = int(math.ceil(num_samples / cols))

grid = Image.new("L", (cols * side, rows * side))
for i in range(num_samples):
    r, c = divmod(i, cols)
    img = Image.fromarray(dequant[i], mode="L")
    grid.paste(img, (c * side, r * side))

grid_path = output_dir / f"mnist_binned_{pixel_bins}_{split}_{num_samples}.png"
grid.save(grid_path)

# Optional labeled contact sheet for quick sanity check.
labeled = Image.new("RGB", (cols * side, rows * (side + 12)), color=(0, 0, 0))
draw = ImageDraw.Draw(labeled)
for i in range(num_samples):
    r, c = divmod(i, cols)
    gray = Image.fromarray(dequant[i], mode="L").convert("RGB")
    x = c * side
    y = r * (side + 12)
    labeled.paste(gray, (x, y))
    draw.text((x + 1, y + side), f"y={int(labels[i])}", fill=(255, 255, 255))

labeled_path = output_dir / f"mnist_binned_{pixel_bins}_{split}_{num_samples}_labeled.png"
labeled.save(labeled_path)

print("saved:")
print(f"  {grid_path}")
print(f"  {labeled_path}")
print(
    f"token stats: min={tokens.min()} max={tokens.max()} "
    f"unique={len(np.unique(tokens))}/{pixel_bins}"
)
PY
