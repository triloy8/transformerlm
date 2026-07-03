import pytest

from cli.sweep_train import _apply_lr_constraints, _apply_overrides, _expand_sweep_overrides


def test_shared_muon_max_learning_rate_expands_to_all_groups():
    overrides = _expand_sweep_overrides({"optimizer.muon.max_learning_rate": 0.001})

    assert "optimizer.muon.max_learning_rate" not in overrides
    for group in ("hidden", "head", "embed", "scalar"):
        assert overrides[f"optimizer.muon.{group}.max_learning_rate"] == pytest.approx(0.001)


def test_shared_muon_max_learning_rate_updates_train_config_dict():
    base = {
        "optimizer": {
            "optimizer_name": "muon",
            "initial_learning_rate": 0.1,
            "max_learning_rate": 1.0,
            "min_learning_rate": 0.01,
            "muon": {
                group: {
                    "initial_learning_rate": 0.1,
                    "max_learning_rate": 1.0,
                    "min_learning_rate": 0.01,
                }
                for group in ("hidden", "head", "embed", "scalar")
            },
        }
    }
    overrides = _expand_sweep_overrides({"optimizer.muon.max_learning_rate": 0.001})

    updated = _apply_lr_constraints(base, overrides)
    updated = _apply_overrides(updated, overrides)

    for group in ("hidden", "head", "embed", "scalar"):
        group_cfg = updated["optimizer"]["muon"][group]
        assert group_cfg["initial_learning_rate"] == pytest.approx(0.0001)
        assert group_cfg["max_learning_rate"] == pytest.approx(0.001)
        assert group_cfg["min_learning_rate"] == pytest.approx(0.00001)
