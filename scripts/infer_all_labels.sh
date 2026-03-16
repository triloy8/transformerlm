#!/usr/bin/env bash
set -euo pipefail

CONFIG="config/resources/infer_mnist_categorical_flow.toml"

for label in {0..9}; do
  tmp="$(mktemp "/tmp/infer_mnist_label_${label}.XXXXXX.toml")"
  sed -E "s/^label = .*/label = ${label}/" "$CONFIG" > "$tmp"
  echo "Running label=${label} with $tmp"
  uv run transformerlm-infer-image --config "$tmp"
  rm -f "$tmp"
done
