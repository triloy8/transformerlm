import json
import textwrap
from copy import deepcopy
from pathlib import Path

import pytest
import tomllib
from pydantic import ValidationError

from config import (
    load_train_config,
    load_infer_config,
    load_image_infer_config,
    load_train_tokenizer_config,
    asdict_pretty,
    TrainConfig,
    InferConfig,
    ImageInferConfig,
    TrainTokenizerConfig,
    BenchInferConfig,
    BenchTokenizerConfig,
)

RESOURCE_ROOT = Path("config/resources")


def write(path: Path, content: str):
    path.write_text(textwrap.dedent(content))


def test_train_config_happy_and_validation(tmp_path: Path):
    # Create dummy tokenizer files
    vocab = tmp_path / "vocab.json"
    merges = tmp_path / "merges.txt"
    special_tokens = tmp_path / "special_tokens.json"
    vocab.write_text("{}")
    merges.write_text("")
    special_tokens.write_text(json.dumps({"<|endoftext|>": 31, "<|mask|>": 32}))

    cfg_path = tmp_path / "train.toml"
    write(cfg_path, f"""
    [model]
    vocab_size = 32
    context_length = 128
    d_model = 8
    num_layers = 1
    num_heads = 2
    d_ff = 16
    rope_theta = 10000.0
    device = "cpu"
    dtype = "float32"
    mask_token_id = 31
    noise_epsilon = 0.01
    random_trunc_prob = 0.0

    [optimizer]
    betas = [0.9, 0.95]
    eps = 1e-8
    weight_decay = 0.01
    initial_learning_rate = 0.001
    max_learning_rate = 0.001
    min_learning_rate = 0.0001
    warmup_iters = 10
    cosine_cycle_iters = 100
    grad_clip_max_l2_norm = 1.0

    [training]
    batch_size = 2
    max_train_iteration = 2
    max_val_iteration = 1
    val_freq_iteration = 1
    seed = 42

    [data]
    runs_path = "{tmp_path.as_posix()}"
    dataset_name = "example/dataset"
    dataset_config = "default"
    train_split = "train"
    val_split = "validation"
    text_field = "text"
    shuffle_buffer_size = 1000
    shuffle_seed = 123

    [data.tokenizer]
    vocab_path = "{vocab.as_posix()}"
    merges_path = "{merges.as_posix()}"
    special_tokens_path = "{special_tokens.as_posix()}"

    [checkpointing]
    enabled = true
    ckpting_save_iter = 2
    best_metric_name = "val_loss"
    best_mode = "min"
    """)

    cfg = load_train_config(cfg_path)
    assert cfg.data.tokenizer.vocab_path.exists()
    assert cfg.data.shuffle_buffer_size == 1000
    assert cfg.optimizer.initial_learning_rate == pytest.approx(0.001)
    assert cfg.training.seed == 42
    dump = cfg.model_dump()
    assert dump["model"]["mask_token_id"] == 31
    assert dump["optimizer"]["initial_learning_rate"] == dump["optimizer"]["max_learning_rate"]
    # pretty dict stringifies paths
    pretty = asdict_pretty(cfg)
    assert isinstance(pretty["data"]["tokenizer"]["vocab_path"], str)

    # Validation error: d_model % num_heads != 0
    bad_cfg = tmp_path / "bad_train.toml"
    write(bad_cfg, cfg_path.read_text().replace("num_heads = 2", "num_heads = 3"))
    with pytest.raises(ValidationError) as exc:
        load_train_config(bad_cfg)
    assert "d_model must be divisible by num_heads" in str(exc.value)


def test_infer_config_happy_and_errors(tmp_path: Path):
    merges = tmp_path / "merges.txt"
    vocab = tmp_path / "vocab.json"
    special_tokens = tmp_path / "special_tokens.json"
    ckpt = tmp_path / "model.ckpt"
    merges.write_text("")
    vocab.write_text("{}")
    special_tokens.write_text(json.dumps({"<|eot|>": 31}))
    ckpt.write_bytes(b"\0\1")

    cfg_path = tmp_path / "infer.toml"
    write(cfg_path, f"""
    [tokenizer]
    merges_path = "{merges.as_posix()}"
    vocab_path = "{vocab.as_posix()}"
    special_tokens_path = "{special_tokens.as_posix()}"

    [model]
    vocab_size = 32
    context_length = 128
    d_model = 8
    num_layers = 1
    num_heads = 2
    d_ff = 16
    rope_theta = 10000.0
    device = "cpu"
    dtype = "float32"

    [checkpoint]
    ckpt_path = "{ckpt.as_posix()}"

    [inference]
    prompt = "hello"
    steps = 32
    total_length = 64
    block_length = 8
    temperature = 1.0
    mask_id = 31
    """)
    cfg = load_infer_config(cfg_path)
    assert cfg.checkpoint.ckpt_path.exists()
    assert cfg.inference.total_length == 64

    # Invalid block_length (must be > 0)
    bad = tmp_path / "infer_bad.toml"
    write(bad, cfg_path.read_text().replace("block_length = 8", "block_length = 0"))
    with pytest.raises(ValidationError) as exc:
        load_infer_config(bad)
    assert "block_length must be > 0" in str(exc.value)

    # Invalid temperature
    bad_t = tmp_path / "infer_bad_t.toml"
    write(bad_t, cfg_path.read_text().replace("temperature = 1.0", "temperature = -0.1"))
    with pytest.raises(ValidationError) as exc:
        load_infer_config(bad_t)
    assert "temperature must be >= 0" in str(exc.value)

    # Extra key should be rejected
    bad_extra = tmp_path / "infer_extra.toml"
    write(bad_extra, cfg_path.read_text().replace('[model]\n', '[model]\nunknown = 123\n', 1))
    with pytest.raises(ValidationError) as exc:
        load_infer_config(bad_extra)
    # errors() contains path ('model', 'unknown')
    assert any(err["loc"] == ("model", "unknown") for err in exc.value.errors())


def test_train_tokenizer_loader(tmp_path: Path):
    merges = tmp_path / "merges.txt"
    vocab = tmp_path / "vocab.json"
    merges.write_text("")
    vocab.write_text("{}")

    # train-tokenizer
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello")
    tt_cfg = tmp_path / "train_tok.toml"
    write(tt_cfg, f"""
    [input]
    input_path = "{corpus.as_posix()}"
    vocab_size = 32
    special_tokens = ["<|eot|>"]

    [output]
    merges_path = "{merges.as_posix()}"
    vocab_path = "{vocab.as_posix()}"
    """)
    cfg_tt = load_train_tokenizer_config(tt_cfg)
    assert cfg_tt.input.input_path.exists()


def test_optimizer_initial_lr_defaults_to_max(tmp_path: Path):
    vocab = tmp_path / "vocab.json"
    merges = tmp_path / "merges.txt"
    special_tokens = tmp_path / "special_tokens.json"
    vocab.write_text("{}")
    merges.write_text("")
    special_tokens.write_text("{}")
    cfg_path = tmp_path / "train_defaults.toml"
    write(cfg_path, f"""
    [model]
    vocab_size = 16
    context_length = 8
    d_model = 8
    num_layers = 1
    num_heads = 2
    d_ff = 16
    rope_theta = 10000.0
    device = "cpu"
    dtype = "float32"

    [optimizer]
    eps = 1e-8
    weight_decay = 0.0
    max_learning_rate = 0.01
    min_learning_rate = 0.001
    warmup_iters = 0
    cosine_cycle_iters = 10
    grad_clip_max_l2_norm = 1.0

    [training]
    batch_size = 2
    max_train_iteration = 2
    max_val_iteration = 1
    val_freq_iteration = 1

    [checkpointing]
    ckpting_save_iter = 2

    [data]
    runs_path = "{tmp_path.as_posix()}"
    dataset_name = "example/dataset"
    train_split = "train"
    val_split = "validation"
    text_field = "text"

    [data.tokenizer]
    vocab_path = "{vocab.as_posix()}"
    merges_path = "{merges.as_posix()}"
    special_tokens_path = "{special_tokens.as_posix()}"
    """)
    cfg = load_train_config(cfg_path)
    assert cfg.optimizer.initial_learning_rate == pytest.approx(cfg.optimizer.max_learning_rate)


def test_train_config_wsd_does_not_require_cosine_cycle_iters(tmp_path: Path):
    vocab = tmp_path / "vocab.json"
    merges = tmp_path / "merges.txt"
    special_tokens = tmp_path / "special_tokens.json"
    vocab.write_text("{}")
    merges.write_text("")
    special_tokens.write_text("{}")
    cfg_path = tmp_path / "train_wsd.toml"
    write(cfg_path, f"""
    [model]
    vocab_size = 16
    context_length = 8
    d_model = 8
    num_layers = 1
    num_heads = 2
    d_ff = 16
    rope_theta = 10000.0
    device = "cpu"
    dtype = "float32"
    mask_token_id = 15

    [optimizer]
    eps = 1e-8
    weight_decay = 0.0
    max_learning_rate = 0.01
    min_learning_rate = 0.001
    warmup_iters = 1
    lr_decay_iters = 10
    wsd_decay_iters = 2
    wsd_decay_style = "exponential"
    lr_schedule = "wsd"
    grad_clip_max_l2_norm = 1.0

    [training]
    batch_size = 2
    max_train_iteration = 2
    max_val_iteration = 1
    val_freq_iteration = 1
    objective = "diffusion"

    [checkpointing]
    ckpting_save_iter = 2

    [data]
    runs_path = "{tmp_path.as_posix()}"
    dataset_name = "example/dataset"
    train_split = "train"
    val_split = "validation"
    text_field = "text"

    [data.tokenizer]
    vocab_path = "{vocab.as_posix()}"
    merges_path = "{merges.as_posix()}"
    special_tokens_path = "{special_tokens.as_posix()}"
    """)
    cfg = load_train_config(cfg_path)
    assert cfg.optimizer.lr_schedule == "wsd"
    assert cfg.optimizer.lr_decay_iters == 10
    assert cfg.optimizer.cosine_cycle_iters == 10


def test_image_model_requires_vocab_size_matches_pixel_bins(tmp_path: Path):
    cfg_path = tmp_path / "train_image_bad.toml"
    write(cfg_path, f"""
    [model]
    model_type = "image"
    vocab_size = 257
    pixel_bins = 32
    context_length = 784
    d_model = 16
    num_layers = 1
    num_heads = 2
    d_ff = 32
    rope_theta = 10000.0
    label_vocab_size = 10
    device = "cpu"
    dtype = "float32"
    mask_token_id = 32
    random_trunc_prob = 0.0

    [optimizer]
    eps = 1e-8
    weight_decay = 0.0
    max_learning_rate = 0.01
    min_learning_rate = 0.001
    warmup_iters = 0
    cosine_cycle_iters = 10
    grad_clip_max_l2_norm = 1.0

    [training]
    batch_size = 2
    max_train_iteration = 2
    max_val_iteration = 1
    val_freq_iteration = 1

    [checkpointing]
    ckpting_save_iter = 2

    [data]
    runs_path = "{tmp_path.as_posix()}"
    dataset_name = "ylecun/mnist"
    train_split = "train"
    val_split = "test"
    text_field = "image"
    pipeline_mode = "mnist"
    """)
    with pytest.raises(ValidationError) as exc:
        load_train_config(cfg_path)
    assert "vocab_size must equal pixel_bins + 1" in str(exc.value)


def test_image_model_null_label_id_must_be_in_range(tmp_path: Path):
    cfg_path = tmp_path / "train_image_bad_null.toml"
    write(cfg_path, f"""
    [model]
    model_type = "image"
    vocab_size = 33
    pixel_bins = 32
    context_length = 784
    d_model = 16
    num_layers = 1
    num_heads = 2
    d_ff = 32
    rope_theta = 10000.0
    label_vocab_size = 10
    null_label_id = 10
    device = "cpu"
    dtype = "float32"
    mask_token_id = 32
    random_trunc_prob = 0.0

    [optimizer]
    eps = 1e-8
    weight_decay = 0.0
    max_learning_rate = 0.01
    min_learning_rate = 0.001
    warmup_iters = 0
    cosine_cycle_iters = 10
    grad_clip_max_l2_norm = 1.0

    [training]
    batch_size = 2
    max_train_iteration = 2
    max_val_iteration = 1
    val_freq_iteration = 1

    [checkpointing]
    ckpting_save_iter = 2

    [data]
    runs_path = "{tmp_path.as_posix()}"
    dataset_name = "ylecun/mnist"
    train_split = "train"
    val_split = "test"
    text_field = "image"
    pipeline_mode = "mnist"
    """)
    with pytest.raises(ValidationError) as exc:
        load_train_config(cfg_path)
    assert "null_label_id must be in [0, label_vocab_size)" in str(exc.value)


def test_uncond_label_dropout_requires_null_label_id(tmp_path: Path):
    cfg_path = tmp_path / "train_image_missing_null.toml"
    write(cfg_path, f"""
    [model]
    model_type = "image"
    vocab_size = 33
    pixel_bins = 32
    context_length = 784
    d_model = 16
    num_layers = 1
    num_heads = 2
    d_ff = 32
    rope_theta = 10000.0
    label_vocab_size = 10
    device = "cpu"
    dtype = "float32"
    mask_token_id = 32
    random_trunc_prob = 0.0

    [optimizer]
    eps = 1e-8
    weight_decay = 0.0
    max_learning_rate = 0.01
    min_learning_rate = 0.001
    warmup_iters = 0
    cosine_cycle_iters = 10
    grad_clip_max_l2_norm = 1.0

    [training]
    batch_size = 2
    max_train_iteration = 2
    max_val_iteration = 1
    val_freq_iteration = 1
    uncond_label_dropout_prob = 0.1

    [checkpointing]
    ckpting_save_iter = 2

    [data]
    runs_path = "{tmp_path.as_posix()}"
    dataset_name = "ylecun/mnist"
    train_split = "train"
    val_split = "test"
    text_field = "image"
    pipeline_mode = "mnist"
    """)
    with pytest.raises(ValidationError) as exc:
        load_train_config(cfg_path)
    assert "uncond_label_dropout_prob > 0 requires model.null_label_id" in str(exc.value)


def _write_text(path: Path, content: str = "") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return str(path)


def _write_bytes(path: Path, data: bytes = b"") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def _patch_tokenizer(tbl: dict, tmp_path: Path) -> None:
    tbl["vocab_path"] = _write_text(tmp_path / "vocab.json", "{}")
    tbl["merges_path"] = _write_text(tmp_path / "merges.txt", "")
    tbl["special_tokens_path"] = _write_text(tmp_path / "special_tokens.json", "{}")


def _patch_train_like(cfg: dict, tmp_path: Path) -> dict:
    _patch_tokenizer(cfg["data"]["tokenizer"], tmp_path)
    return cfg


def _patch_infer_like(cfg: dict, tmp_path: Path) -> dict:
    _patch_tokenizer(cfg["tokenizer"], tmp_path)
    cfg["checkpoint"]["ckpt_path"] = _write_bytes(tmp_path / "ckpt.bin", b"\0\1")
    if "guide" in cfg and "checkpoint" in cfg["guide"]:
        cfg["guide"]["checkpoint"]["ckpt_path"] = _write_bytes(tmp_path / "guide_ckpt.bin", b"\0\1")
    return cfg


def _patch_bench_infer(cfg: dict, tmp_path: Path) -> dict:
    return _patch_infer_like(cfg, tmp_path)


def _patch_train_tokenizer(cfg: dict, tmp_path: Path) -> dict:
    cfg["input"]["input_path"] = _write_text(tmp_path / "corpus.txt", "hello")
    return cfg


def _patch_bench_tokenizer(cfg: dict, tmp_path: Path) -> dict:
    _patch_tokenizer(cfg["tokenizer"], tmp_path)
    return cfg


RESOURCE_CASES = [
    ("train.toml", TrainConfig, _patch_train_like),
    ("infer.toml", InferConfig, _patch_infer_like),
    ("bench_infer.toml", BenchInferConfig, _patch_bench_infer),
    ("bench_tokenizer.toml", BenchTokenizerConfig, _patch_bench_tokenizer),
    ("train_tokenizer.toml", TrainTokenizerConfig, _patch_train_tokenizer),
]


@pytest.mark.parametrize(("filename", "schema", "patcher"), RESOURCE_CASES)
def test_resource_configs_validate(filename: str, schema, patcher, tmp_path: Path):
    raw = tomllib.load((RESOURCE_ROOT / filename).open("rb"))
    patched = patcher(deepcopy(raw), tmp_path)
    cfg = schema.model_validate(patched)
    assert cfg is not None


def test_train_config_accepts_joint_mntp_ar_objective(tmp_path: Path):
    raw = tomllib.load((RESOURCE_ROOT / "train.toml").open("rb"))
    patched = _patch_train_like(deepcopy(raw), tmp_path)
    patched["training"]["objective"] = "joint-mntp-ar"
    cfg = TrainConfig.model_validate(patched)
    assert cfg.training.objective == "joint-mntp-ar"


def test_train_config_accepts_uniform_state_diffusion_objective(tmp_path: Path):
    raw = tomllib.load((RESOURCE_ROOT / "train.toml").open("rb"))
    patched = _patch_train_like(deepcopy(raw), tmp_path)
    patched["model"].pop("mask_token_id", None)
    patched["training"]["objective"] = "uniform-state-diffusion"
    cfg = TrainConfig.model_validate(patched)
    assert cfg.model.mask_token_id is None
    assert cfg.training.objective == "uniform-state-diffusion"


def test_train_config_accepts_flow_with_image_dit(tmp_path: Path):
    raw = tomllib.load((RESOURCE_ROOT / "train_mnist.toml").open("rb"))
    patched = deepcopy(raw)
    patched["model"]["model_type"] = "image_dit"
    patched["training"]["objective"] = "flow"
    cfg = TrainConfig.model_validate(patched)
    assert cfg.model.model_type == "image_dit"
    assert cfg.training.objective == "flow"


def test_train_config_accepts_categorical_flow_with_image_cfm(tmp_path: Path):
    raw = tomllib.load((RESOURCE_ROOT / "train_mnist.toml").open("rb"))
    patched = deepcopy(raw)
    patched["model"]["model_type"] = "image_cfm"
    patched["model"]["vocab_size"] = patched["model"]["pixel_bins"]
    patched["model"].pop("mask_token_id", None)
    patched["training"]["objective"] = "categorical-flow"
    cfg = TrainConfig.model_validate(patched)
    assert cfg.model.model_type == "image_cfm"
    assert cfg.training.objective == "categorical-flow"


def test_image_infer_config_accepts_flow_mode_for_image_dit(tmp_path: Path):
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"123")
    cfg_path = tmp_path / "infer_image_flow.toml"
    write(cfg_path, f"""
    [model]
    model_type = "image_dit"
    vocab_size = 33
    pixel_bins = 32
    context_length = 784
    d_model = 16
    num_layers = 1
    num_heads = 2
    d_ff = 32
    rope_theta = 10000.0
    label_vocab_size = 11
    null_label_id = 10
    use_rope_2d = true
    image_height = 28
    image_width = 28
    attention_backend = "torch_sdpa"
    attention_sdp_backend = "auto"
    device = "cpu"
    dtype = "float32"

    [checkpoint]
    ckpt_path = "{ckpt.as_posix()}"

    [inference]
    generation_mode = "flow"
    label = 3
    num_samples = 2
    steps = 32
    block_length = 784
    temperature = 1.0
    cfg_scale = 1.0
    output_dir = "{(tmp_path / 'out').as_posix()}"
    """)
    cfg = load_image_infer_config(cfg_path)
    assert isinstance(cfg, ImageInferConfig)
    assert cfg.inference.generation_mode == "flow"


def test_image_infer_config_accepts_categorical_flow_mode_for_image_cfm(tmp_path: Path):
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"123")
    cfg_path = tmp_path / "infer_image_categorical_flow.toml"
    write(cfg_path, f"""
    [model]
    model_type = "image_cfm"
    vocab_size = 32
    pixel_bins = 32
    context_length = 784
    d_model = 16
    num_layers = 1
    num_heads = 2
    d_ff = 32
    rope_theta = 10000.0
    label_vocab_size = 11
    null_label_id = 10
    use_rope_2d = true
    image_height = 28
    image_width = 28
    attention_backend = "torch_sdpa"
    attention_sdp_backend = "auto"
    device = "cpu"
    dtype = "float32"

    [checkpoint]
    ckpt_path = "{ckpt.as_posix()}"

    [inference]
    generation_mode = "categorical_flow"
    label = 3
    num_samples = 2
    steps = 8
    block_length = 784
    temperature = 1.0
    cfg_scale = 1.0
    output_dir = "{(tmp_path / 'out').as_posix()}"
    """)
    cfg = load_image_infer_config(cfg_path)
    assert isinstance(cfg, ImageInferConfig)
    assert cfg.inference.generation_mode == "categorical_flow"


def test_train_config_rejects_flow_without_image_dit(tmp_path: Path):
    raw = tomllib.load((RESOURCE_ROOT / "train_mnist.toml").open("rb"))
    patched = deepcopy(raw)
    patched["training"]["objective"] = "flow"
    with pytest.raises(ValidationError) as exc:
        TrainConfig.model_validate(patched)
    assert "requires model.model_type='image_dit'" in str(exc.value)


def test_train_config_rejects_categorical_flow_without_image_cfm(tmp_path: Path):
    raw = tomllib.load((RESOURCE_ROOT / "train_mnist.toml").open("rb"))
    patched = deepcopy(raw)
    patched["training"]["objective"] = "categorical-flow"
    with pytest.raises(ValidationError) as exc:
        TrainConfig.model_validate(patched)
    assert "requires model.model_type='image_cfm'" in str(exc.value)
