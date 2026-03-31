import numpy as np

from leash.data.image import dequantize_tokens_to_uint8


def _uniform_quantize_uint8(pixels: np.ndarray, bins: int) -> np.ndarray:
    quant = (pixels.astype(np.uint16) * int(bins)) // 256
    return np.minimum(quant, int(bins) - 1).astype(np.uint8)


def test_uniform_quantization_edges():
    pixels = np.array([0, 7, 8, 127, 128, 247, 248, 255], dtype=np.uint8)
    quant = _uniform_quantize_uint8(pixels, bins=32)
    assert quant.tolist() == [0, 0, 1, 15, 16, 30, 31, 31]


def test_dequantize_tokens_to_uint8_bounds_and_dtype():
    tokens = np.array([0, 1, 30, 31], dtype=np.int32)
    restored = dequantize_tokens_to_uint8(tokens, pixel_bins=32)
    assert restored.dtype == np.uint8
    assert restored.min() >= 0
    assert restored.max() <= 255


def test_dequantize_tokens_to_uint8_identity_for_256_bins():
    tokens = np.array([0, 17, 128, 255], dtype=np.uint8)
    restored = dequantize_tokens_to_uint8(tokens, pixel_bins=256)
    assert np.array_equal(restored, tokens)
