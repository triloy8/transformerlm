import torch

from leash.inference.generate import diffusion_generate, image_diffusion_generate, categorical_flow_image_generate
from leash.inference.sampling import compute_transfer_schedule, add_gumbel_noise


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, target_token: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.target_token = target_token

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size, device=input_ids.device)
        logits[..., self.target_token] = 1.0
        return logits


class DummyImageModel(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length

    def forward(self, input_ids: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size, device=input_ids.device)
        # Base preference for token 1 regardless of class.
        logits[..., 1] = 1.1
        # Class-conditional signal on token 2.
        cond = (context == 2).to(logits.dtype).view(batch, 1)
        logits[..., 2] = cond
        return logits


class DummyCategoricalFlowImageModel(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length

    def forward(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        del x, s, t
        batch = context.shape[0]
        seq_len = self.context_length
        logits = torch.zeros(batch, seq_len, self.vocab_size, device=context.device)
        logits[..., 1] = 1.1
        cond = (context == 2).to(logits.dtype).view(batch, 1)
        logits[..., 2] = 0.5 * cond
        return logits


def test_compute_transfer_schedule_basic():
    mask = torch.tensor([[True, True, False, False], [False, False, False, False]])
    schedule = compute_transfer_schedule(mask, steps=2)
    assert schedule.shape == (2, 2)
    assert schedule[0].tolist() == [1, 1]
    assert schedule[1].tolist() == [0, 0]


def test_add_gumbel_noise_temperature_zero():
    logits = torch.randn(2, 3, 5)
    perturbed = add_gumbel_noise(logits, temperature=0.0)
    assert torch.allclose(logits, perturbed)


def test_diffusion_generate_fills_mask_tokens():
    device = torch.device("cpu")
    vocab_size = 6
    context_length = 8
    prompt = torch.tensor([[1, 2]], device=device)
    model = DummyModel(vocab_size=vocab_size, context_length=context_length, target_token=3).to(device)

    output = diffusion_generate(
        model,
        prompt,
        mask_id=5,
        steps=4,
        gen_length=4,
        block_length=2,
        temperature=0.0,
    )

    assert output.shape == (1, 6)
    # Prompt should stay intact
    assert torch.equal(output[:, : prompt.shape[1]], prompt)
    # Newly generated tokens should equal target token from dummy model
    assert torch.all(output[:, prompt.shape[1]:] == 3)


def test_image_diffusion_generate_cfg_uses_unconditional_context():
    device = torch.device("cpu")
    model = DummyImageModel(vocab_size=6, context_length=8).to(device)
    prompt = torch.empty((1, 0), dtype=torch.long, device=device)
    context = torch.tensor([2], dtype=torch.long, device=device)
    uncond_context = torch.tensor([0], dtype=torch.long, device=device)

    # Without CFG, token 1 remains preferred.
    out_no_cfg = image_diffusion_generate(
        model,
        prompt,
        context=context,
        uncond_context=uncond_context,
        mask_id=5,
        steps=2,
        gen_length=4,
        block_length=2,
        cfg_scale=0.0,
        temperature=0.0,
    )
    assert torch.all(out_no_cfg == 1)

    # With CFG, conditional token 2 should become preferred.
    out_cfg = image_diffusion_generate(
        model,
        prompt,
        context=context,
        uncond_context=uncond_context,
        mask_id=5,
        steps=2,
        gen_length=4,
        block_length=2,
        cfg_scale=2.0,
        temperature=0.0,
    )
    assert torch.all(out_cfg == 2)


def test_image_diffusion_generate_requires_uncond_context_when_cfg_enabled():
    device = torch.device("cpu")
    model = DummyImageModel(vocab_size=6, context_length=8).to(device)
    prompt = torch.empty((1, 0), dtype=torch.long, device=device)
    context = torch.tensor([2], dtype=torch.long, device=device)

    try:
        _ = image_diffusion_generate(
            model,
            prompt,
            context=context,
            mask_id=5,
            steps=2,
            gen_length=4,
            block_length=2,
            cfg_scale=1.0,
            temperature=0.0,
        )
    except ValueError as exc:
        assert "uncond_context must be set" in str(exc)
    else:
        raise AssertionError("expected ValueError when cfg_scale > 0 without uncond_context")


def test_categorical_flow_image_generate_cfg_uses_unconditional_context():
    device = torch.device("cpu")
    model = DummyCategoricalFlowImageModel(vocab_size=6, context_length=4).to(device)
    prompt = torch.empty((1, 0), dtype=torch.long, device=device)
    context = torch.tensor([2], dtype=torch.long, device=device)
    uncond_context = torch.tensor([0], dtype=torch.long, device=device)

    out_no_cfg = categorical_flow_image_generate(
        model,
        prompt,
        context=context,
        uncond_context=uncond_context,
        steps=3,
        cfg_scale=0.0,
        temperature=0.0,
        prior_type="uniform",
    )
    assert torch.all(out_no_cfg == 1)

    out_cfg = categorical_flow_image_generate(
        model,
        prompt,
        context=context,
        uncond_context=uncond_context,
        steps=3,
        cfg_scale=2.0,
        temperature=0.0,
        prior_type="uniform",
    )
    assert torch.all(out_cfg == 2)


def test_categorical_flow_image_generate_requires_uncond_context_when_cfg_enabled():
    device = torch.device("cpu")
    model = DummyCategoricalFlowImageModel(vocab_size=6, context_length=4).to(device)
    prompt = torch.empty((1, 0), dtype=torch.long, device=device)
    context = torch.tensor([2], dtype=torch.long, device=device)

    try:
        _ = categorical_flow_image_generate(
            model,
            prompt,
            context=context,
            steps=2,
            cfg_scale=1.0,
            temperature=0.0,
            prior_type="uniform",
        )
    except ValueError as exc:
        assert "uncond_context must be set" in str(exc)
    else:
        raise AssertionError("expected ValueError when cfg_scale > 0 without uncond_context")
