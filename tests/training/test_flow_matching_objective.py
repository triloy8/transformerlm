from types import SimpleNamespace

import torch

from leash.objectives.diffusion import FlowMatchingObjective


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


def _make_cfg(*, dropout_prob: float):
    return SimpleNamespace(
        pixel_bins=32,
        random_trunc_prob=0.0,
        null_label_id=10,
        uncond_label_dropout_prob=dropout_prob,
    )


def test_flow_batch_and_loss(device):
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long, device=device)
    labels = torch.tensor([7, 9], dtype=torch.long, device=device)
    batcher = _ImageBatcher(tokens, labels)
    cfg = _make_cfg(dropout_prob=0.0)
    objective = FlowMatchingObjective(cfg, _DummyTokenizer())

    batch = objective.get_batch(
        dataset=batcher,
        batch_size=2,
        context_length=4,
        device=str(device),
        generator=torch.Generator(device=str(device)).manual_seed(123),
    )
    assert batch.noisy_inputs.shape == (2, 4)
    assert batch.target_velocity.shape == (2, 4)
    assert batch.timesteps.shape == (2, 1)

    class _ZeroModel(torch.nn.Module):
        def forward(self, x, t, context=None):
            del t, context
            return torch.zeros_like(x)

    out = objective.forward_with_model(_ZeroModel(), batch)
    assert out is not None
    assert torch.isfinite(out["loss"]).item()


def test_flow_null_label_dropout_replaces_all_labels_when_prob_one(device):
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long, device=device)
    labels = torch.tensor([7, 9], dtype=torch.long, device=device)
    batcher = _ImageBatcher(tokens, labels)
    cfg = _make_cfg(dropout_prob=1.0)
    objective = FlowMatchingObjective(cfg, _DummyTokenizer())

    batch = objective.get_batch(
        dataset=batcher,
        batch_size=2,
        context_length=4,
        device=str(device),
        generator=torch.Generator(device=str(device)).manual_seed(123),
    )
    assert batch.labels is not None
    assert torch.all(batch.labels == 10)
