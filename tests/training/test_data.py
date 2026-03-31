import numpy as np
import torch

from leash.objectives.data import get_batch


class DummyBatcher:
    def __init__(self, array: np.ndarray):
        self.tokens = torch.from_numpy(array.astype(np.int64))
        if self.tokens.numel() == 0:
            raise ValueError("array must contain tokens")
        self._idx = 0

    def draw(self, batch_size: int, context_length: int) -> torch.Tensor:
        sequences = []
        total_tokens = self.tokens.numel()
        for _ in range(batch_size):
            if self._idx + context_length > total_tokens:
                self._idx = 0
            seq = self.tokens[self._idx : self._idx + context_length]
            self._idx = (self._idx + context_length) % total_tokens
            sequences.append(seq)
        return torch.stack(sequences, dim=0)


class DummyRowBatcher:
    def draw(self, batch_size: int, context_length: int):
        tokens = torch.tensor(
            [
                [1, 2, 3, 0, 0],
                [4, 5, 0, 0, 0],
            ],
            dtype=torch.long,
        )
        mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, False, False, False],
            ],
            dtype=torch.bool,
        )
        return tokens[:batch_size, :context_length], mask[:batch_size, :context_length]


def test_get_batch_metadata_mask_ratio_and_truncation(device):
    arr = np.arange(100, dtype=np.int32)
    generator = torch.Generator(device="cpu").manual_seed(42)
    batcher = DummyBatcher(arr)

    batch = get_batch(
        batcher,
        batch_size=2,
        context_length=8,
        device=str(device),
        mask_token_id=999,
        noise_epsilon=0.05,
        random_trunc_prob=0.0,
        generator=generator,
    )

    assert "mask_ratio" in batch.metadata
    assert 0.0 <= batch.metadata["mask_ratio"] <= 1.0
    assert batch.metadata["random_truncation_applied"] is False


def test_get_batch_random_truncation(device):
    arr = np.arange(100, dtype=np.int32)
    generator = torch.Generator(device="cpu").manual_seed(123)
    batcher = DummyBatcher(arr)

    batch = get_batch(
        batcher,
        batch_size=1,
        context_length=16,
        device=str(device),
        mask_token_id=999,
        noise_epsilon=0.1,
        random_trunc_prob=1.0,
        generator=generator,
    )

    assert batch.clean_targets.shape[1] <= 16
    assert batch.metadata["random_truncation_applied"] is True


def test_get_batch_applies_attention_and_loss_masks(device):
    batcher = DummyRowBatcher()
    batch = get_batch(
        batcher,
        batch_size=2,
        context_length=5,
        device=str(device),
        mask_token_id=999,
        noise_epsilon=0.1,
        random_trunc_prob=0.0,
    )

    assert batch.attention_mask is not None
    assert batch.loss_mask is not None
    assert torch.equal(batch.attention_mask, batch.loss_mask)
    assert torch.all(batch.mask <= batch.attention_mask)
    assert batch.metadata["token_count"] == int(batch.attention_mask.sum().item())
