import argparse

import pytest
import torch
from pydantic import ValidationError

from config import MuonHiddenConfig, MuonOptimizerConfig
from leash.trainer import _prepare_optimizer_setup
from leash.training.optim import Muon, build_optimizer_param_groups, muon_update_scale


class TinyMuonModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embeddings = torch.nn.Embedding(8, 4)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=True)])
        self.lm_head = torch.nn.Linear(4, 8, bias=False)


def test_muon_hidden_config_accepts_moonshot_scaling():
    cfg = MuonHiddenConfig(adjust_lr_fn="MATCH_RMS_ADAMW")

    assert cfg.adjust_lr_fn == "match_rms_adamw"


def test_muon_hidden_config_rejects_unknown_scaling():
    with pytest.raises(ValidationError, match="muon.hidden.adjust_lr_fn"):
        MuonHiddenConfig(adjust_lr_fn="unknown")


def test_muon_param_groups_propagate_hidden_scaling_mode():
    cfg = MuonOptimizerConfig(hidden=MuonHiddenConfig(adjust_lr_fn="match_rms_adamw"))
    groups = build_optimizer_param_groups(TinyMuonModel(), "muon", cfg)

    hidden_group = next(group for group in groups if group["name"] == "hidden")
    adam_groups = [group for group in groups if not group["use_muon"]]

    assert hidden_group["use_muon"] is True
    assert hidden_group["adjust_lr_fn"] == "match_rms_adamw"
    assert all("adjust_lr_fn" not in group for group in adam_groups)


def test_train_optimizer_setup_preserves_muon_scaling_mode():
    cfg = MuonOptimizerConfig(hidden=MuonHiddenConfig(adjust_lr_fn="match_rms_adamw"))
    args = argparse.Namespace(
        optimizer_name="muon",
        lr_schedule="cosine",
        muon_cfg=cfg,
        initial_learning_rate=0.01,
        max_learning_rate=0.01,
        min_learning_rate=0.001,
        warmup_iters=0,
        cosine_cycle_iters=10,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-10,
        muon_momentum=0.95,
    )

    optimizer_cls, param_groups, kwargs = _prepare_optimizer_setup(args, TinyMuonModel())
    hidden_group = next(group for group in param_groups if group["name"] == "hidden")

    assert optimizer_cls is Muon
    assert kwargs["momentum"] == pytest.approx(0.95)
    assert hidden_group["adjust_lr_fn"] == "match_rms_adamw"


def test_muon_update_scale_matches_supported_modes():
    update = torch.empty(16, 4)

    assert muon_update_scale(update, "original") == pytest.approx(2.0)
    assert muon_update_scale(update, "match_rms_adamw") == pytest.approx(0.8)
    assert muon_update_scale(update, "spectral_unclamped") == pytest.approx(2.0)


def test_moonshot_scaling_changes_muon_update_magnitude():
    grad = torch.ones(16, 4)
    original_param = torch.nn.Parameter(torch.zeros_like(grad))
    moonshot_param = torch.nn.Parameter(torch.zeros_like(grad))
    original_param.grad = grad.clone()
    moonshot_param.grad = grad.clone()

    original_optimizer = Muon(
        [{"params": [original_param], "use_muon": True, "adjust_lr_fn": "original"}],
        lr=1.0,
        momentum=0.0,
        weight_decay=0.0,
    )
    moonshot_optimizer = Muon(
        [{"params": [moonshot_param], "use_muon": True, "adjust_lr_fn": "match_rms_adamw"}],
        lr=1.0,
        momentum=0.0,
        weight_decay=0.0,
    )

    original_optimizer.step()
    moonshot_optimizer.step()

    update_ratio = (moonshot_param.detach().norm() / original_param.detach().norm()).item()
    assert update_ratio == pytest.approx(0.4, rel=5e-3)
