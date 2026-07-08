from types import SimpleNamespace

import torch

from leash.objectives import (
    DiffusionBatch,
    build_objective,
    cross_entropy,
    JointMntpAutoregressiveObjective,
    FlowMatchingObjective,
    CategoricalFlowObjective,
    SumiUniformGiddDiffusionObjective,
    UniformStateDiffusionObjective,
    get_objective_factory,
    uniform_gidd_loss,
    list_objectives,
    register_objective,
)


class _DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [0]

    def decode(self, tokens: list[int]) -> str:
        return ""


def test_build_objective_joint_mntp_ar():
    cfg = SimpleNamespace(
        training_objective="joint-mntp-ar",
        vocab_size=16,
        mask_token_id=15,
        noise_epsilon=1e-3,
        random_trunc_prob=0.0,
        p_mask_override=None,
        deterministic_mask=False,
        joint_diffusion_alpha=0.3,
        joint_diffusion_alpha_end=None,
        joint_alpha_schedule="constant",
        joint_alpha_schedule_start=0.0,
        joint_alpha_schedule_end=1.0,
        max_train_iteration=10,
        p_mask_schedule="none",
        p_mask_start=None,
        p_mask_end=None,
        p_mask_schedule_start=0.0,
        p_mask_schedule_end=1.0,
    )
    objective = build_objective(cfg, _DummyTokenizer())
    assert isinstance(objective, JointMntpAutoregressiveObjective)


def test_build_objective_flow():
    cfg = SimpleNamespace(
        training_objective="flow",
        pixel_bins=32,
        random_trunc_prob=0.0,
        null_label_id=10,
        uncond_label_dropout_prob=0.1,
    )
    objective = build_objective(cfg, _DummyTokenizer())
    assert isinstance(objective, FlowMatchingObjective)


def test_build_objective_categorical_flow():
    cfg = SimpleNamespace(
        training_objective="categorical-flow",
        vocab_size=32,
        random_trunc_prob=0.0,
        null_label_id=10,
        uncond_label_dropout_prob=0.1,
        categorical_flow_prior_type="discunif",
        categorical_flow_diag_fraction=0.5,
        categorical_flow_inf_weight=1.0,
        categorical_flow_ec_weight=1.0,
        categorical_flow_td_weight=1.0,
    )
    objective = build_objective(cfg, _DummyTokenizer())
    assert isinstance(objective, CategoricalFlowObjective)


def test_build_objective_uniform_state_diffusion():
    cfg = SimpleNamespace(
        training_objective="uniform-state-diffusion",
        vocab_size=32,
        mask_token_id=None,
        noise_epsilon=1e-3,
        random_trunc_prob=0.0,
        p_mask_override=None,
        deterministic_mask=False,
        max_train_iteration=10,
        p_mask_schedule="none",
        p_mask_start=None,
        p_mask_end=None,
        p_mask_schedule_start=0.0,
        p_mask_schedule_end=1.0,
    )
    objective = build_objective(cfg, _DummyTokenizer())
    assert isinstance(objective, UniformStateDiffusionObjective)


def test_build_objective_sumi_uniform_gidd_diffusion():
    cfg = SimpleNamespace(
        training_objective="sumi-uniform-gidd-diffusion",
        vocab_size=32,
        mask_token_id=None,
        noise_epsilon=1e-3,
        random_trunc_prob=0.0,
        p_mask_override=None,
        deterministic_mask=False,
        max_train_iteration=10,
        p_mask_schedule="none",
        p_mask_start=None,
        p_mask_end=None,
        p_mask_schedule_start=0.0,
        p_mask_schedule_end=1.0,
        sumi_gidd_beta_is=0.5,
        sumi_gidd_z_loss_strength=0.0,
        sumi_gidd_loss_eps=1e-10,
        sumi_gidd_loss_mask_mode="valid",
    )
    objective = build_objective(cfg, _DummyTokenizer())
    assert isinstance(objective, SumiUniformGiddDiffusionObjective)


def test_sumi_uniform_gidd_diffusion_objective_uses_gidd_loss():
    cfg = SimpleNamespace(
        training_objective="sumi-uniform-gidd-diffusion",
        vocab_size=3,
        mask_token_id=None,
        noise_epsilon=1e-3,
        random_trunc_prob=0.0,
        p_mask_override=None,
        deterministic_mask=False,
        max_train_iteration=10,
        p_mask_schedule="none",
        p_mask_start=None,
        p_mask_end=None,
        p_mask_schedule_start=0.0,
        p_mask_schedule_end=1.0,
        sumi_gidd_beta_is=0.7,
        sumi_gidd_z_loss_strength=0.0,
        sumi_gidd_loss_eps=1e-12,
        sumi_gidd_loss_mask_mode="valid",
    )
    objective = build_objective(cfg, _DummyTokenizer())
    logits = torch.tensor([
        [[1.0, -0.5, 0.25], [0.1, 0.3, -0.2]],
    ])
    noisy = torch.tensor([[0, 2]])
    targets = torch.tensor([[0, 1]])
    t = torch.tensor([0.4])
    batch = DiffusionBatch(
        noisy_inputs=noisy,
        clean_targets=targets,
        mask=torch.tensor([[False, True]]),
        p_mask=t[:, None],
        attention_mask=None,
        loss_mask=None,
        metadata={},
        timesteps=t,
    )

    loss = objective.compute_loss(logits, batch)
    expected = uniform_gidd_loss(
        logits,
        noisy,
        targets,
        t,
        vocab_size=3,
        beta_is=0.7,
        z_loss_strength=0.0,
    )
    assert torch.allclose(loss, expected)


def test_uniform_state_diffusion_objective_uses_unweighted_loss():
    cfg = SimpleNamespace(
        training_objective="uniform-state-diffusion",
        vocab_size=2,
        mask_token_id=None,
        noise_epsilon=1e-3,
        random_trunc_prob=0.0,
        p_mask_override=None,
        deterministic_mask=False,
        max_train_iteration=10,
        p_mask_schedule="none",
        p_mask_start=None,
        p_mask_end=None,
        p_mask_schedule_start=0.0,
        p_mask_schedule_end=1.0,
    )
    objective = build_objective(cfg, _DummyTokenizer())
    logits = torch.log(torch.tensor([
        [[0.7, 0.3], [0.2, 0.8]],
    ], dtype=torch.float32))
    targets = torch.tensor([[0, 1]])
    mask = torch.tensor([[True, True]])
    batch = DiffusionBatch(
        noisy_inputs=targets,
        clean_targets=targets,
        mask=mask,
        p_mask=torch.tensor([[0.25]]),
        attention_mask=None,
        loss_mask=None,
        metadata={},
    )

    loss = objective.compute_loss(logits, batch)
    expected = cross_entropy(logits, targets, reduction="none").mean()
    assert torch.allclose(loss, expected)


def test_objective_registry_lists_builtin_entries():
    names = list_objectives()
    assert "diffusion" in names
    assert "uniform-state-diffusion" in names
    assert "sumi-uniform-gidd-diffusion" in names
    assert "joint-mntp-ar" in names


def test_register_objective_allows_extension():
    class _CustomObjective:
        def __init__(self, cfg, tokenizer) -> None:
            self.cfg = cfg
            self.tokenizer = tokenizer

    register_objective("custom-test", _CustomObjective)
    factory = get_objective_factory("custom-test")
    objective = factory(SimpleNamespace(), _DummyTokenizer())
    assert isinstance(objective, _CustomObjective)
