from types import SimpleNamespace

from trainkit.objectives import (
    build_objective,
    JointMntpAutoregressiveObjective,
    FlowMatchingObjective,
    CategoricalFlowObjective,
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
