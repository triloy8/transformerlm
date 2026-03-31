from types import SimpleNamespace

import torch

from leash.objectives.diffusion import DiffusionObjective


class _DummyTokenizer:
    def encode(self, _text: str):
        return []

    def decode(self, _tokens):
        return ""


class _ImageBatcher:
    def __init__(self, tokens: torch.Tensor, labels: torch.Tensor):
        self._tokens = tokens
        self._labels = labels

    def draw(self, batch_size: int, context_length: int):
        assert context_length == self._tokens.shape[1]
        return SimpleNamespace(tokens=self._tokens[:batch_size], labels=self._labels[:batch_size])


def _make_cfg(*, dropout_prob: float, null_label_id: int | None):
    return SimpleNamespace(
        mask_token_id=32,
        vocab_size=33,
        noise_epsilon=1e-3,
        random_trunc_prob=0.0,
        p_mask_override=1.0,
        deterministic_mask=True,
        p_mask_bucket_edges=None,
        null_label_id=null_label_id,
        uncond_label_dropout_prob=dropout_prob,
    )


def test_null_label_dropout_replaces_all_labels_when_prob_one(device):
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long, device=device)
    labels = torch.tensor([7, 9], dtype=torch.long, device=device)
    batcher = _ImageBatcher(tokens, labels)
    cfg = _make_cfg(dropout_prob=1.0, null_label_id=10)
    objective = DiffusionObjective(cfg, _DummyTokenizer())

    batch = objective.get_batch(
        dataset=batcher,
        batch_size=2,
        context_length=4,
        device=str(device),
        generator=torch.Generator(device=str(device)).manual_seed(123),
    )
    assert batch.labels is not None
    assert torch.all(batch.labels == 10)


def test_null_label_dropout_keeps_labels_when_prob_zero(device):
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long, device=device)
    labels = torch.tensor([7, 9], dtype=torch.long, device=device)
    batcher = _ImageBatcher(tokens, labels)
    cfg = _make_cfg(dropout_prob=0.0, null_label_id=10)
    objective = DiffusionObjective(cfg, _DummyTokenizer())

    batch = objective.get_batch(
        dataset=batcher,
        batch_size=2,
        context_length=4,
        device=str(device),
        generator=torch.Generator(device=str(device)).manual_seed(123),
    )
    assert batch.labels is not None
    assert torch.equal(batch.labels, labels)
