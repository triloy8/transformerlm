import argparse

import torch

import leash.trainer as trainer


class CompileWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self._orig_mod = module

    def forward(self, *args, **kwargs):
        return self._orig_mod(*args, **kwargs)


class StopAfterOptimizer(RuntimeError):
    pass


def test_train_ddp_builds_optimizer_groups_before_compile(monkeypatch, tmp_path):
    seen_names = []

    def model_builder(_cfg):
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        return model

    def fake_prepare_optimizer_setup(_cfg, model):
        nonlocal seen_names
        seen_names = [name for name, _ in model.named_parameters()]
        return torch.optim.SGD, [{"params": list(model.parameters())}], {"lr": 0.1}

    class FakeDDP(torch.nn.Module):
        def __init__(self, model, _world_size, _bucket_size_mb):
            super().__init__()
            self.model = model

        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)

    class FakeOptimizerStateSharding:
        def __init__(self, *_args, **_kwargs):
            raise StopAfterOptimizer

    class FakeCheckpointManager:
        def __init__(self, *args, **kwargs):
            self.run_dir = tmp_path

        def prepare_run(self, _torch_generator):
            return None

    monkeypatch.setattr(trainer, "setup_process_group", lambda **_kwargs: None)
    monkeypatch.setattr(trainer, "cleanup_process_group", lambda: None)
    monkeypatch.setattr(trainer, "init_logging", lambda *_args, **_kwargs: (None, "run", tmp_path))
    monkeypatch.setattr(trainer, "CheckpointManager", FakeCheckpointManager)
    monkeypatch.setattr(trainer.torch, "compile", lambda model, **_kwargs: CompileWrapper(model))
    monkeypatch.setattr(trainer, "DDP", FakeDDP)
    monkeypatch.setattr(trainer, "_prepare_optimizer_setup", fake_prepare_optimizer_setup)
    monkeypatch.setattr(trainer, "OptimizerStateSharding", FakeOptimizerStateSharding)

    args = argparse.Namespace(
        backend="gloo",
        bucket_size_mb=0,
        compile_backend="inductor",
        compile_dynamic=False,
        compile_enabled=True,
        compile_fullgraph=False,
        compile_mode="default",
        compile_options=None,
        config_path="config/resources/train.toml",
        device="cpu",
        master_addr="127.0.0.1",
        master_port="29500",
        node_rank=0,
        num_gpus_per_node=1,
        num_nodes=1,
        rng_seed=1,
        runs_path=tmp_path,
    )

    try:
        trainer.train_ddp(
            0,
            args,
            argparse.Namespace(checkpointing=None),
            {},
            model_builder,
            lambda _args: object(),
            lambda _cfg, _tokenizer: object(),
        )
    except StopAfterOptimizer:
        pass

    assert seen_names == ["layers.0.weight", "layers.0.bias"]
