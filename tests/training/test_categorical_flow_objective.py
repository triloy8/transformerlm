from types import SimpleNamespace

import torch

from trainkit.objectives.categorical_flow import CategoricalFlowObjective
from transformerlm.models import CategoricalFlowImage


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


def test_categorical_flow_batch_and_loss(device):
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long, device=device)
    labels = torch.tensor([7, 9], dtype=torch.long, device=device)
    batcher = _ImageBatcher(tokens, labels)
    cfg = SimpleNamespace(
        vocab_size=8,
        random_trunc_prob=0.0,
        null_label_id=10,
        uncond_label_dropout_prob=0.0,
        categorical_flow_prior_type="discunif",
        categorical_flow_diag_fraction=0.5,
        categorical_flow_inf_weight=1.0,
        categorical_flow_ec_weight=1.0,
        categorical_flow_td_weight=1.0,
    )
    objective = CategoricalFlowObjective(cfg, _DummyTokenizer())
    model = CategoricalFlowImage(
        vocab_size=8,
        context_length=4,
        d_model=16,
        num_layers=1,
        num_heads=2,
        d_ff=32,
        rope_theta=10000.0,
        label_vocab_size=11,
        attention_backend="torch_sdpa",
        device=device,
        dtype=torch.float32,
    ).to(device)

    batch = objective.get_batch(
        dataset=batcher,
        batch_size=2,
        context_length=4,
        device=str(device),
        generator=torch.Generator(device=str(device)).manual_seed(123),
    )
    assert batch.x0_prior.shape == (2, 4, 8)
    out = objective.forward_with_model(model, batch)
    loss = out["loss"]
    assert torch.isfinite(loss).item()
    loss.backward()
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += float(p.grad.norm().item())
    assert grad_norm > 0.0


def test_categorical_flow_null_label_dropout_replaces_all_labels_when_prob_one(device):
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long, device=device)
    labels = torch.tensor([7, 9], dtype=torch.long, device=device)
    batcher = _ImageBatcher(tokens, labels)
    cfg = SimpleNamespace(
        vocab_size=8,
        random_trunc_prob=0.0,
        null_label_id=10,
        uncond_label_dropout_prob=1.0,
        categorical_flow_prior_type="discunif",
        categorical_flow_diag_fraction=0.5,
        categorical_flow_inf_weight=1.0,
        categorical_flow_ec_weight=1.0,
        categorical_flow_td_weight=1.0,
    )
    objective = CategoricalFlowObjective(cfg, _DummyTokenizer())
    batch = objective.get_batch(
        dataset=batcher,
        batch_size=2,
        context_length=4,
        device=str(device),
        generator=torch.Generator(device=str(device)).manual_seed(123),
    )
    assert batch.labels is not None
    assert torch.all(batch.labels == 10)


def test_categorical_flow_model_supports_jvp_attention(device):
    model = CategoricalFlowImage(
        vocab_size=8,
        context_length=4,
        d_model=16,
        num_layers=1,
        num_heads=2,
        d_ff=32,
        rope_theta=10000.0,
        label_vocab_size=11,
        attention_backend="torch_sdpa",
        device=device,
        dtype=torch.float32,
    ).to(device)
    x = torch.full((2, 4, 8), 1.0 / 8.0, device=device)
    s = torch.full((2,), 0.25, device=device)
    t = torch.full((2,), 0.75, device=device)
    labels = torch.tensor([1, 3], dtype=torch.long, device=device)

    out, tangent = torch.func.jvp(
        lambda t_in: torch.softmax(model(x, s, t_in, context=labels, jvp_attention=True), dim=-1),
        (t,),
        (torch.ones_like(t),),
    )
    assert out.shape == (2, 4, 8)
    assert tangent.shape == (2, 4, 8)
    assert torch.isfinite(out).all().item()
    assert torch.isfinite(tangent).all().item()
