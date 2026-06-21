from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ALLOWED_DTYPES = {"float32", "float16", "bfloat16"}
ALLOWED_DEVICES = {"cpu", "cuda"}
ALLOWED_OPTIMIZERS = {"adamw", "muon"}
ALLOWED_MUON_ADJUST_LR_FNS = {"original", "match_rms_adamw", "spectral_unclamped"}
ALLOWED_ATTENTION_BACKENDS = {"custom", "torch_sdpa"}
ALLOWED_SDP_BACKENDS = {"auto", "flash", "mem_efficient", "math"}
ALLOWED_AMP_DTYPES = {"float16", "bfloat16"}


class _BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MuonHiddenConfig(_BaseConfig):
    initial_learning_rate: float = 0.05
    max_learning_rate: float = 0.05
    min_learning_rate: float = 0.05
    momentum: float = 0.95
    weight_decay: float = 0.0
    adjust_lr_fn: str = "original"

    @model_validator(mode="after")
    def _validate_hidden(self):
        self.adjust_lr_fn = self.adjust_lr_fn.lower()
        self._check_lr_range("muon.hidden")
        if not (0 < self.momentum < 1):
            raise ValueError("muon.hidden.momentum must be in (0, 1)")
        if self.weight_decay < 0:
            raise ValueError("muon.hidden.weight_decay must be >= 0")
        if self.adjust_lr_fn not in ALLOWED_MUON_ADJUST_LR_FNS:
            raise ValueError(
                f"muon.hidden.adjust_lr_fn must be one of {sorted(ALLOWED_MUON_ADJUST_LR_FNS)}"
            )
        return self

    def _check_lr_range(self, label: str):
        if self.initial_learning_rate <= 0 or self.min_learning_rate <= 0 or self.max_learning_rate <= 0:
            raise ValueError(f"{label}: learning rates must be > 0")
        if self.min_learning_rate > self.max_learning_rate:
            raise ValueError(f"{label}: min_learning_rate must be <= max_learning_rate")


class MuonAdamGroupConfig(_BaseConfig):
    initial_learning_rate: float
    max_learning_rate: float
    min_learning_rate: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float = 0.0

    @model_validator(mode="after")
    def _validate_group(self):
        if self.initial_learning_rate <= 0 or self.min_learning_rate <= 0 or self.max_learning_rate <= 0:
            raise ValueError("muon group learning rates must be > 0")
        if self.min_learning_rate > self.max_learning_rate:
            raise ValueError("muon group min_learning_rate must be <= max_learning_rate")
        if len(self.betas) != 2 or not (0 <= self.betas[0] < 1 and 0 <= self.betas[1] < 1):
            raise ValueError("muon group betas must be 2 values in [0, 1)")
        if self.eps <= 0:
            raise ValueError("muon group eps must be > 0")
        if self.weight_decay < 0:
            raise ValueError("muon group weight_decay must be >= 0")
        return self


class MuonOptimizerConfig(_BaseConfig):
    hidden: MuonHiddenConfig = Field(default_factory=MuonHiddenConfig)
    head: MuonAdamGroupConfig = Field(
        default_factory=lambda: MuonAdamGroupConfig(
            initial_learning_rate=0.22,
            max_learning_rate=0.22,
            min_learning_rate=0.22,
            betas=(0.8, 0.95),
            eps=1e-10,
            weight_decay=0.0,
        )
    )
    embed: MuonAdamGroupConfig = Field(
        default_factory=lambda: MuonAdamGroupConfig(
            initial_learning_rate=0.6,
            max_learning_rate=0.6,
            min_learning_rate=0.6,
            betas=(0.8, 0.95),
            eps=1e-10,
            weight_decay=0.0,
        )
    )
    scalar: MuonAdamGroupConfig = Field(
        default_factory=lambda: MuonAdamGroupConfig(
            initial_learning_rate=0.04,
            max_learning_rate=0.04,
            min_learning_rate=0.04,
            betas=(0.8, 0.95),
            eps=1e-10,
            weight_decay=0.0,
        )
    )


class OptimizerConfig(_BaseConfig):
    optimizer_name: str = "adamw"
    lr_schedule: str = "cosine"
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float
    weight_decay: float
    initial_learning_rate: Optional[float] = None
    max_learning_rate: float
    min_learning_rate: float
    warmup_iters: int
    cosine_cycle_iters: Optional[int] = None
    lr_decay_iters: Optional[int] = None
    wsd_decay_iters: Optional[int] = None
    wsd_decay_style: str = "exponential"
    grad_clip_max_l2_norm: float
    muon: Optional[MuonOptimizerConfig] = None

    @model_validator(mode="after")
    def _validate_optimizer(self):
        self.optimizer_name = self.optimizer_name.lower()
        if self.optimizer_name not in ALLOWED_OPTIMIZERS:
            raise ValueError(f"optimizer_name must be one of {sorted(ALLOWED_OPTIMIZERS)}")
        self.lr_schedule = self.lr_schedule.lower()
        if self.lr_schedule not in {"cosine", "constant", "constant_with_warmup", "wsd"}:
            raise ValueError("lr_schedule must be one of: cosine, constant, constant_with_warmup, wsd")
        if len(self.betas) != 2 or not (0 <= self.betas[0] < 1 and 0 <= self.betas[1] < 1):
            raise ValueError("optimizer betas must be 2 values in [0, 1)")
        if self.eps <= 0:
            raise ValueError("eps must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.initial_learning_rate is None:
            self.initial_learning_rate = self.max_learning_rate
        for attr in ("initial_learning_rate", "max_learning_rate", "min_learning_rate", "grad_clip_max_l2_norm"):
            if getattr(self, attr) <= 0:
                raise ValueError(f"{attr} must be > 0")
        if self.cosine_cycle_iters is None:
            if self.lr_schedule == "cosine":
                raise ValueError("cosine_cycle_iters must be set when lr_schedule='cosine'")
            self.cosine_cycle_iters = self.lr_decay_iters if self.lr_decay_iters is not None else 0
        if self.lr_decay_iters is None:
            self.lr_decay_iters = self.cosine_cycle_iters
        self.wsd_decay_style = self.wsd_decay_style.lower()
        if self.wsd_decay_style not in {"exponential", "linear", "cosine"}:
            raise ValueError("wsd_decay_style must be one of: exponential, linear, cosine")
        for attr in ("warmup_iters", "cosine_cycle_iters", "lr_decay_iters"):
            if getattr(self, attr) < 0:
                raise ValueError(f"{attr} must be >= 0")
        if self.lr_schedule == "wsd":
            if self.lr_decay_iters <= 0:
                raise ValueError("lr_decay_iters must be > 0 when lr_schedule='wsd'")
            if self.warmup_iters >= self.lr_decay_iters:
                raise ValueError("warmup_iters must be < lr_decay_iters when lr_schedule='wsd'")
            if self.wsd_decay_iters is None:
                raise ValueError("wsd_decay_iters must be set when lr_schedule='wsd'")
            if self.wsd_decay_iters <= 0:
                raise ValueError("wsd_decay_iters must be > 0 when lr_schedule='wsd'")
            if self.wsd_decay_iters > self.lr_decay_iters:
                raise ValueError("wsd_decay_iters must be <= lr_decay_iters when lr_schedule='wsd'")
        if self.min_learning_rate > self.max_learning_rate:
            raise ValueError("min_learning_rate must be <= max_learning_rate")
        if self.initial_learning_rate > self.max_learning_rate:
            raise ValueError("initial_learning_rate must be <= max_learning_rate")
        if self.optimizer_name == "muon":
            if self.muon is None:
                raise ValueError("Muon optimizer requires muon configuration")
            if self.lr_schedule not in {"constant", "constant_with_warmup"}:
                hidden_cfg = self.muon.hidden
                if not (hidden_cfg.min_learning_rate <= hidden_cfg.initial_learning_rate <= hidden_cfg.max_learning_rate):
                    raise ValueError("muon.hidden.initial_learning_rate must be within [min, max]")
                for name in ("head", "embed", "scalar"):
                    group_cfg = getattr(self.muon, name)
                    if not (
                        group_cfg.min_learning_rate <= group_cfg.initial_learning_rate <= group_cfg.max_learning_rate
                    ):
                        raise ValueError(f"muon.{name}.initial_learning_rate must be within [min, max]")
        return self


class ModelConfig(_BaseConfig):
    model_type: str = "lm"
    vocab_size: int
    pixel_bins: int = 256
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float
    label_vocab_size: Optional[int] = None
    null_label_id: Optional[int] = None
    use_rope_2d: bool = False
    image_height: Optional[int] = None
    image_width: Optional[int] = None
    attention_backend: str = "custom"
    attention_sdp_backend: str = "auto"
    device: str
    dtype: str
    mask_token_id: Optional[int] = None
    eot_token_id: Optional[int] = None
    noise_epsilon: float = 1e-3
    random_trunc_prob: float = 0.01

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        if v not in ALLOWED_DEVICES:
            raise ValueError(f"device must be one of {sorted(ALLOWED_DEVICES)}")
        return v

    @field_validator("dtype")
    @classmethod
    def _validate_dtype(cls, v: str) -> str:
        if v not in ALLOWED_DTYPES:
            raise ValueError(f"dtype must be one of {sorted(ALLOWED_DTYPES)}")
        return v

    @model_validator(mode="after")
    def _validate_model(self):
        for field in ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be > 0")
        if not (1 < self.pixel_bins <= 256):
            raise ValueError("pixel_bins must be in [2, 256]")
        self.model_type = self.model_type.lower()
        if self.model_type not in {"lm", "image", "image_dit", "image_cfm"}:
            raise ValueError("model_type must be one of: lm, image, image_dit, image_cfm")
        if self.model_type in {"image", "image_dit", "image_cfm"}:
            if self.label_vocab_size is None or self.label_vocab_size <= 0:
                raise ValueError("label_vocab_size must be > 0 when model_type is image/image_dit/image_cfm")
            if self.model_type == "image":
                # Image diffusion pipeline reserves one token for masking.
                if self.vocab_size != self.pixel_bins + 1:
                    raise ValueError("for model_type='image', vocab_size must equal pixel_bins + 1")
            if self.model_type == "image_cfm":
                if self.vocab_size != self.pixel_bins:
                    raise ValueError("for model_type='image_cfm', vocab_size must equal pixel_bins")
            if self.null_label_id is not None and not (0 <= self.null_label_id < self.label_vocab_size):
                raise ValueError("null_label_id must be in [0, label_vocab_size)")
            if (self.image_height is None) ^ (self.image_width is None):
                raise ValueError("image_height and image_width must be set together when provided")
            if self.image_height is not None and self.image_height <= 0:
                raise ValueError("image_height must be > 0 when provided")
            if self.image_width is not None and self.image_width <= 0:
                raise ValueError("image_width must be > 0 when provided")
            if self.use_rope_2d:
                if self.image_height is None or self.image_width is None:
                    side = int(self.context_length ** 0.5)
                    if side * side != self.context_length:
                        raise ValueError(
                            "use_rope_2d requires image_height/image_width or a square context_length"
                        )
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be > 0")
        self.attention_backend = self.attention_backend.lower()
        if self.attention_backend not in ALLOWED_ATTENTION_BACKENDS:
            raise ValueError(f"attention_backend must be one of {sorted(ALLOWED_ATTENTION_BACKENDS)}")
        self.attention_sdp_backend = self.attention_sdp_backend.lower()
        if self.attention_sdp_backend not in ALLOWED_SDP_BACKENDS:
            raise ValueError(f"attention_sdp_backend must be one of {sorted(ALLOWED_SDP_BACKENDS)}")
        if self.mask_token_id is not None:
            if not (0 <= self.mask_token_id < self.vocab_size):
                raise ValueError("mask_token_id must be in [0, vocab_size)")
        if self.eot_token_id is not None:
            if not (0 <= self.eot_token_id < self.vocab_size):
                raise ValueError("eot_token_id must be in [0, vocab_size)")
        if not (0 < self.noise_epsilon <= 1):
            raise ValueError("noise_epsilon must be in (0, 1]")
        if not (0 <= self.random_trunc_prob <= 1):
            raise ValueError("random_trunc_prob must be in [0, 1]")
        return self


class TrainingConfig(_BaseConfig):
    batch_size: int
    max_train_iteration: int
    max_val_iteration: int
    val_freq_iteration: int
    seed: Optional[int] = None
    repeat_masking_seed: Optional[int] = None
    skip_validation: bool = False
    grad_accum_steps: int = 1
    train_loss_ema_decay: float = 0.0
    amp_enabled: bool = False
    amp_dtype: str = "float16"
    objective: str = "diffusion"
    joint_diffusion_alpha: float = 0.3
    joint_diffusion_alpha_end: Optional[float] = None
    joint_alpha_schedule: str = "constant"
    joint_alpha_schedule_start: float = 0.0
    joint_alpha_schedule_end: float = 1.0
    p_mask_schedule: str = "none"
    p_mask_start: Optional[float] = None
    p_mask_end: Optional[float] = None
    p_mask_schedule_start: float = 0.0
    p_mask_schedule_end: float = 1.0
    eot_mask_loss: bool = False
    p_mask_override: Optional[float] = None
    deterministic_mask: bool = False
    uncond_label_dropout_prob: float = 0.0
    categorical_flow_prior_type: str = "discunif"
    categorical_flow_diag_fraction: float = 0.5
    categorical_flow_inf_weight: float = 1.0
    categorical_flow_ec_weight: float = 1.0
    categorical_flow_td_weight: float = 1.0

    @model_validator(mode="after")
    def _validate_training(self):
        for attr in ("batch_size", "max_train_iteration", "max_val_iteration", "val_freq_iteration"):
            if getattr(self, attr) <= 0:
                raise ValueError(f"{attr} must be > 0")
        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be > 0")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be >= 0 when provided")
        if self.repeat_masking_seed is not None and self.repeat_masking_seed < 0:
            raise ValueError("repeat_masking_seed must be >= 0 when provided")
        if self.train_loss_ema_decay < 0 or self.train_loss_ema_decay >= 1:
            raise ValueError("train_loss_ema_decay must be in [0, 1)")
        self.amp_dtype = self.amp_dtype.lower()
        if self.amp_dtype not in ALLOWED_AMP_DTYPES:
            raise ValueError(f"amp_dtype must be one of {sorted(ALLOWED_AMP_DTYPES)}")
        self.objective = self.objective.lower()
        if self.objective not in {
            "diffusion",
            "megadlm-diffusion",
            "ar",
            "joint-diffusion-ar",
            "joint-mntp-ar",
            "flow",
            "categorical-flow",
        }:
            raise ValueError(
                "objective must be one of: diffusion, megadlm-diffusion, ar, joint-diffusion-ar, joint-mntp-ar, flow, categorical-flow"
            )
        if not (0 <= self.joint_diffusion_alpha <= 1):
            raise ValueError("joint_diffusion_alpha must be in [0, 1]")
        if self.joint_diffusion_alpha_end is not None and not (0 <= self.joint_diffusion_alpha_end <= 1):
            raise ValueError("joint_diffusion_alpha_end must be in [0, 1] when provided")
        self.joint_alpha_schedule = self.joint_alpha_schedule.lower()
        if self.joint_alpha_schedule not in {"constant", "linear", "cosine"}:
            raise ValueError("joint_alpha_schedule must be one of: constant, linear, cosine")
        if not (0 <= self.joint_alpha_schedule_start <= 1):
            raise ValueError("joint_alpha_schedule_start must be in [0, 1]")
        if not (0 <= self.joint_alpha_schedule_end <= 1):
            raise ValueError("joint_alpha_schedule_end must be in [0, 1]")
        if self.joint_alpha_schedule_end < self.joint_alpha_schedule_start:
            raise ValueError("joint_alpha_schedule_end must be >= joint_alpha_schedule_start")
        self.p_mask_schedule = self.p_mask_schedule.lower()
        if self.p_mask_schedule not in {"none", "linear", "cosine"}:
            raise ValueError("p_mask_schedule must be one of: none, linear, cosine")
        if self.p_mask_start is not None and not (0 < self.p_mask_start <= 1):
            raise ValueError("p_mask_start must be in (0, 1] when provided")
        if self.p_mask_end is not None and not (0 < self.p_mask_end <= 1):
            raise ValueError("p_mask_end must be in (0, 1] when provided")
        if not (0 <= self.p_mask_schedule_start <= 1):
            raise ValueError("p_mask_schedule_start must be in [0, 1]")
        if not (0 <= self.p_mask_schedule_end <= 1):
            raise ValueError("p_mask_schedule_end must be in [0, 1]")
        if self.p_mask_schedule_end < self.p_mask_schedule_start:
            raise ValueError("p_mask_schedule_end must be >= p_mask_schedule_start")
        if self.p_mask_schedule in {"linear", "cosine"} and (
            self.p_mask_start is None or self.p_mask_end is None
        ):
            raise ValueError("p_mask_start and p_mask_end must be set when p_mask_schedule is linear/cosine")
        if self.p_mask_override is not None and not (0 < self.p_mask_override <= 1):
            raise ValueError("p_mask_override must be in (0, 1] when provided")
        if not (0 <= self.uncond_label_dropout_prob <= 1):
            raise ValueError("uncond_label_dropout_prob must be in [0, 1]")
        self.categorical_flow_prior_type = self.categorical_flow_prior_type.lower()
        if self.categorical_flow_prior_type not in {"discunif", "uniform"}:
            raise ValueError("categorical_flow_prior_type must be one of: discunif, uniform")
        if not (0 <= self.categorical_flow_diag_fraction <= 1):
            raise ValueError("categorical_flow_diag_fraction must be in [0, 1]")
        if self.categorical_flow_inf_weight < 0:
            raise ValueError("categorical_flow_inf_weight must be >= 0")
        if self.categorical_flow_ec_weight < 0:
            raise ValueError("categorical_flow_ec_weight must be >= 0")
        if self.categorical_flow_td_weight < 0:
            raise ValueError("categorical_flow_td_weight must be >= 0")
        return self


class CompileConfig(_BaseConfig):
    enabled: bool = False
    backend: str = "inductor"
    mode: str = "default"
    fullgraph: bool = False
    dynamic: bool = False
    options: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _validate_compile(self):
        if not self.backend:
            raise ValueError("compile.backend must not be empty")
        if not self.mode:
            raise ValueError("compile.mode must not be empty")
        return self


class TokenizerConfig(_BaseConfig):
    merges_path: Path
    vocab_path: Path
    special_tokens_path: Path

    @model_validator(mode="after")
    def _validate_tokenizer(self):
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"vocab_path not found: {self.vocab_path}")
        if not self.merges_path.exists():
            raise FileNotFoundError(f"merges_path not found: {self.merges_path}")
        if not self.special_tokens_path.exists():
            raise FileNotFoundError(f"special_tokens_path not found: {self.special_tokens_path}")
        return self


class DataConfig(_BaseConfig):
    runs_path: Path
    dataset_name: str
    dataset_config: Optional[str] = None
    train_split: str
    val_split: str
    text_field: str
    tokenizer: Optional[TokenizerConfig] = None
    pipeline_mode: str = "packed"
    pad_token_id: Optional[int] = None
    pad_random_shift: bool = False
    shuffle_buffer_size: int = 0
    shuffle_seed: Optional[int] = None
    cache_all: bool = False
    megatron_train_prefix: Optional[Path] = None
    megatron_val_prefix: Optional[Path] = None

    @model_validator(mode="after")
    def _validate_data(self):
        if not self.dataset_name:
            raise ValueError("dataset_name must not be empty")
        if not self.train_split:
            raise ValueError("train_split must not be empty")
        if not self.val_split:
            raise ValueError("val_split must not be empty")
        self.pipeline_mode = self.pipeline_mode.lower()
        if self.pipeline_mode not in {"packed", "rows", "megatron", "mnist"}:
            raise ValueError("pipeline_mode must be one of: packed, rows, megatron, mnist")
        if self.pipeline_mode != "mnist":
            if not self.text_field:
                raise ValueError("text_field must not be empty")
        if self.pipeline_mode in {"packed", "rows"} and self.tokenizer is None:
            raise ValueError("tokenizer config must be set for packed/rows pipeline")
        if self.pipeline_mode == "rows":
            if self.pad_token_id is None:
                raise ValueError("pad_token_id must be set when pipeline_mode='rows'")
            if self.pad_token_id < 0:
                raise ValueError("pad_token_id must be >= 0")
        if self.pipeline_mode == "megatron":
            if self.megatron_train_prefix is None or self.megatron_val_prefix is None:
                raise ValueError("megatron_train_prefix and megatron_val_prefix must be set when pipeline_mode='megatron'")
        if self.shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be >= 0")
        if self.shuffle_seed is not None and self.shuffle_seed < 0:
            raise ValueError("shuffle_seed must be >= 0 when provided")
        return self


class WandbConfig(_BaseConfig):
    entity: Optional[str] = None
    project: Optional[str] = None
    architecture: Optional[str] = None
    dataset: Optional[str] = None


class LoggingConfig(_BaseConfig):
    backend: Optional[str] = None
    run_name: Optional[str] = None
    architecture: Optional[str] = None
    dataset: Optional[str] = None
    log_activation_norms: bool = False
    log_weight_norms: bool = False
    log_grad_norms: bool = False
    log_p_mask_bucket_loss: bool = False
    p_mask_bucket_edges: Optional[List[float]] = None
    val_log_every: int = 0
    val_log_samples: int = 0

    @model_validator(mode="after")
    def _validate_logging(self):
        if self.val_log_every < 0:
            raise ValueError("val_log_every must be >= 0")
        if self.val_log_samples < 0:
            raise ValueError("val_log_samples must be >= 0")
        if self.val_log_samples > 0 and self.val_log_every == 0:
            raise ValueError("val_log_every must be > 0 when val_log_samples > 0")
        return self


class DdpConfig(_BaseConfig):
    backend: str = "nccl"
    num_nodes: int = 1
    node_rank: int = 0
    num_gpus_per_node: int = 1
    master_addr: str = "localhost"
    master_port: str = "29500"
    bucket_size_mb: int = 0
    nccl_p2p_disable: Optional[bool] = None


class CheckpointingConfig(_BaseConfig):
    enabled: bool = True
    ckpting_save_iter: int
    resume_optimizer: bool = True
    resume_from: Optional[str] = None
    model_init_path: Optional[Path] = None
    best_metric_name: str = "val_loss"
    best_mode: str = "min"
    run_id: Optional[str] = None

    @model_validator(mode="after")
    def _validate_checkpointing(self):
        if self.ckpting_save_iter <= 0:
            raise ValueError("checkpointing.ckpting_save_iter must be > 0")
        if self.best_mode not in {"min", "max"}:
            raise ValueError("checkpointing.best_mode must be 'min' or 'max'")
        if not self.best_metric_name:
            raise ValueError("checkpointing.best_metric_name must not be empty")
        if self.model_init_path is not None and not self.model_init_path.exists():
            raise FileNotFoundError(f"checkpointing.model_init_path not found: {self.model_init_path}")
        return self


class TrainInferConfig(_BaseConfig):
    infer_every: int = 0
    prompts: List[str] = Field(default_factory=list)
    steps: int = 256
    total_length: Optional[int] = None
    block_length: int = 128
    temperature: float = 1.0
    top_p: Optional[float] = None
    cfg_scale: float = 0.0
    remasking: str = "random"
    generation_mode: str = "diffusion"
    logits_eos_inf: bool = False
    confidence_eos_eot_inf: bool = False
    seed: Optional[int] = None

    @model_validator(mode="after")
    def _validate_train_infer(self):
        if self.infer_every < 0:
            raise ValueError("train_infer.infer_every must be >= 0")
        if self.infer_every > 0 and not self.prompts:
            raise ValueError("train_infer.prompts must be set when infer_every > 0")
        if any(not p for p in self.prompts):
            raise ValueError("train_infer.prompts must not contain empty strings")
        if self.steps <= 0:
            raise ValueError("train_infer.steps must be > 0")
        if self.total_length is not None and self.total_length <= 0:
            raise ValueError("train_infer.total_length must be > 0 when provided")
        if self.block_length <= 0:
            raise ValueError("train_infer.block_length must be > 0")
        if self.temperature < 0:
            raise ValueError("train_infer.temperature must be >= 0")
        if self.top_p is not None and (self.top_p < 0 or self.top_p > 1):
            raise ValueError("train_infer.top_p must be between 0 and 1 when provided")
        if self.cfg_scale < 0:
            raise ValueError("train_infer.cfg_scale must be >= 0")
        if self.remasking not in {"low_confidence", "random"}:
            raise ValueError("train_infer.remasking must be one of: low_confidence, random")
        if self.generation_mode not in {"diffusion", "ar", "categorical_flow"}:
            raise ValueError("train_infer.generation_mode must be one of: diffusion, ar, categorical_flow")
        if self.seed is not None and self.seed < 0:
            raise ValueError("train_infer.seed must be >= 0 when provided")
        return self


class TrainConfig(_BaseConfig):
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    data: DataConfig
    compile: Optional[CompileConfig] = None
    train_infer: Optional[TrainInferConfig] = None
    wandb: Optional[WandbConfig] = None
    logging: Optional[LoggingConfig] = None
    ddp: Optional[DdpConfig] = None
    checkpointing: CheckpointingConfig

    @model_validator(mode="after")
    def _validate_train_config(self):
        if self.data.pipeline_mode == "mnist" and self.model.random_trunc_prob > 0:
            raise ValueError("random_trunc_prob must be 0 when pipeline_mode='mnist'")
        if self.training.objective in {"joint-diffusion-ar", "joint-mntp-ar"} and self.model.model_type != "lm":
            raise ValueError(
                "training.objective in {'joint-diffusion-ar', 'joint-mntp-ar'} requires model.model_type='lm'"
            )
        if self.training.objective == "flow" and self.model.model_type != "image_dit":
            raise ValueError("training.objective='flow' requires model.model_type='image_dit'")
        if self.model.model_type == "image_dit" and self.training.objective != "flow":
            raise ValueError("model.model_type='image_dit' requires training.objective='flow'")
        if self.training.objective == "categorical-flow" and self.model.model_type != "image_cfm":
            raise ValueError("training.objective='categorical-flow' requires model.model_type='image_cfm'")
        if self.model.model_type == "image_cfm" and self.training.objective != "categorical-flow":
            raise ValueError("model.model_type='image_cfm' requires training.objective='categorical-flow'")
        if self.training.uncond_label_dropout_prob > 0:
            if self.model.model_type not in {"image", "image_dit", "image_cfm"}:
                raise ValueError("uncond_label_dropout_prob requires model_type in {'image', 'image_dit', 'image_cfm'}")
            if self.model.null_label_id is None:
                raise ValueError("uncond_label_dropout_prob > 0 requires model.null_label_id")
        return self


class CheckpointConfig(_BaseConfig):
    ckpt_path: Path

    @model_validator(mode="after")
    def _validate_checkpoint(self):
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"ckpt_path not found: {self.ckpt_path}")
        return self


class GuideConfig(_BaseConfig):
    model: ModelConfig
    checkpoint: CheckpointConfig
    guidance_mode: str = "ar_agreement"
    fallback_strategy: str = "single_token"

    @model_validator(mode="after")
    def _validate_guide(self):
        self.guidance_mode = self.guidance_mode.lower()
        if self.guidance_mode not in {"none", "ar_agreement"}:
            raise ValueError("guide.guidance_mode must be one of: none, ar_agreement")
        self.fallback_strategy = self.fallback_strategy.lower()
        if self.fallback_strategy not in {"single_token"}:
            raise ValueError("guide.fallback_strategy must be one of: single_token")
        if self.model.model_type != "lm":
            raise ValueError("guide.model.model_type must be 'lm'")
        if self.model.vocab_size <= 0:
            raise ValueError("guide.model.vocab_size must be > 0")
        return self


class InferenceConfig(_BaseConfig):
    prompt: str
    steps: int
    total_length: Optional[int] = None
    block_length: int
    temperature: float = 1.0
    top_p: Optional[float] = None
    mask_id: Optional[int] = None
    seed: Optional[int] = None
    eos_token_id: Optional[int] = None
    cfg_scale: float = 0.0
    remasking: str = "random"
    logits_eos_inf: bool = False
    confidence_eos_eot_inf: bool = False
    generation_mode: str = "diffusion"

    @model_validator(mode="after")
    def _validate_inference(self):
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        if self.steps <= 0:
            raise ValueError("steps must be > 0")
        if self.block_length <= 0:
            raise ValueError("block_length must be > 0")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_p is not None and (self.top_p < 0 or self.top_p > 1):
            raise ValueError("top_p must be between 0 and 1 when provided")
        if self.mask_id is not None and self.mask_id < 0:
            raise ValueError("mask_id must be >= 0")
        if self.generation_mode == "diffusion" and self.mask_id is None:
            raise ValueError("mask_id must be set when generation_mode='diffusion'")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.eos_token_id is not None and self.eos_token_id < 0:
            raise ValueError("eos_token_id must be >= 0")
        if self.total_length is not None and self.total_length <= 0:
            raise ValueError("total_length must be > 0 when provided")
        if self.cfg_scale < 0:
            raise ValueError("cfg_scale must be >= 0")
        if self.remasking not in {"low_confidence", "random"}:
            raise ValueError("remasking must be one of: low_confidence, random")
        if self.generation_mode not in {"diffusion", "ar"}:
            raise ValueError("generation_mode must be one of: diffusion, ar")
        if (self.logits_eos_inf or self.confidence_eos_eot_inf) and self.eos_token_id is None:
            raise ValueError("eos_token_id must be set when EOS suppression is enabled")
        return self


class SweepConfig(_BaseConfig):
    prompts: Optional[List[str]] = None
    temperatures: Optional[List[float]] = None
    steps: Optional[List[int]] = None
    total_lengths: Optional[List[int]] = None
    block_lengths: Optional[List[int]] = None
    cfg_scales: Optional[List[float]] = None
    remasking: Optional[List[str]] = None
    top_ps: Optional[List[float]] = None
    seeds: Optional[List[int]] = None
    output_path: Path = Path("runs/sweep_infer.jsonl")
    html_output_path: Optional[Path] = None
    print_every: int = 1
    limit: Optional[int] = None

    @model_validator(mode="after")
    def _validate_sweep(self):
        if self.prompts is not None and any(not p for p in self.prompts):
            raise ValueError("sweep.prompts must not contain empty strings")
        if self.temperatures is not None and any(t < 0 for t in self.temperatures):
            raise ValueError("sweep.temperatures must be >= 0")
        if self.steps is not None and any(s <= 0 for s in self.steps):
            raise ValueError("sweep.steps must be > 0")
        if self.total_lengths is not None and any(t <= 0 for t in self.total_lengths):
            raise ValueError("sweep.total_lengths must be > 0")
        if self.block_lengths is not None and any(b <= 0 for b in self.block_lengths):
            raise ValueError("sweep.block_lengths must be > 0")
        if self.cfg_scales is not None and any(c < 0 for c in self.cfg_scales):
            raise ValueError("sweep.cfg_scales must be >= 0")
        if self.remasking is not None:
            for r in self.remasking:
                if r not in {"low_confidence", "random"}:
                    raise ValueError("sweep.remasking must be one of: low_confidence, random")
        if self.top_ps is not None and any((p < 0 or p > 1) for p in self.top_ps):
            raise ValueError("sweep.top_ps must be between 0 and 1")
        if self.seeds is not None and any(s < 0 for s in self.seeds):
            raise ValueError("sweep.seeds must be >= 0")
        if self.print_every < 0:
            raise ValueError("sweep.print_every must be >= 0")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("sweep.limit must be > 0 when provided")
        return self


class InferConfig(_BaseConfig):
    tokenizer: TokenizerConfig
    model: ModelConfig
    checkpoint: CheckpointConfig
    inference: InferenceConfig
    guide: Optional[GuideConfig] = None
    logging: Optional[LoggingConfig] = None

    @model_validator(mode="after")
    def _finalize_inference(self):
        inf = self.inference
        updates = {}
        if inf.total_length is None:
            updates["total_length"] = self.model.context_length
        if inf.mask_id is None and self.model.mask_token_id is not None:
            updates["mask_id"] = self.model.mask_token_id
        if updates:
            inf = inf.model_copy(update=updates)
        self.inference = inf
        if self.guide is not None:
            if self.guide.model.vocab_size != self.model.vocab_size:
                raise ValueError("guide.model.vocab_size must match model.vocab_size")
            if self.guide.model.context_length < self.model.context_length:
                raise ValueError("guide.model.context_length must be >= model.context_length")
            if self.guide.model.device != self.model.device:
                raise ValueError("guide.model.device must match model.device")
        return self


class ImageInferenceConfig(_BaseConfig):
    generation_mode: str = "diffusion"
    label: int
    num_samples: int = 4
    steps: int
    block_length: int
    temperature: float = 1.0
    top_p: Optional[float] = None
    mask_id: Optional[int] = None
    seed: Optional[int] = None
    cfg_scale: float = 0.0
    remasking: str = "random"
    prior_type: str = "discunif"
    output_dir: Path = Path("runs/infer_images")
    image_height: Optional[int] = None
    image_width: Optional[int] = None

    @model_validator(mode="after")
    def _validate_image_infer(self):
        self.generation_mode = self.generation_mode.lower()
        if self.generation_mode not in {"diffusion", "flow", "categorical_flow"}:
            raise ValueError("generation_mode must be one of: diffusion, flow, categorical_flow")
        if self.label < 0:
            raise ValueError("label must be >= 0")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be > 0")
        if self.steps <= 0:
            raise ValueError("steps must be > 0")
        if self.block_length <= 0:
            raise ValueError("block_length must be > 0")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_p is not None and (self.top_p < 0 or self.top_p > 1):
            raise ValueError("top_p must be between 0 and 1 when provided")
        if self.mask_id is not None and self.mask_id < 0:
            raise ValueError("mask_id must be >= 0")
        if self.cfg_scale < 0:
            raise ValueError("cfg_scale must be >= 0")
        if self.remasking not in {"low_confidence", "random"}:
            raise ValueError("remasking must be one of: low_confidence, random")
        self.prior_type = self.prior_type.lower()
        if self.prior_type not in {"discunif", "uniform"}:
            raise ValueError("prior_type must be one of: discunif, uniform")
        if (self.image_height is None) ^ (self.image_width is None):
            raise ValueError("image_height and image_width must be set together")
        if self.image_height is not None and self.image_height <= 0:
            raise ValueError("image_height must be > 0 when provided")
        if self.image_width is not None and self.image_width <= 0:
            raise ValueError("image_width must be > 0 when provided")
        return self


class ImageInferConfig(_BaseConfig):
    model: ModelConfig
    checkpoint: CheckpointConfig
    inference: ImageInferenceConfig
    logging: Optional[LoggingConfig] = None

    @model_validator(mode="after")
    def _finalize_image_inference(self):
        if self.model.model_type not in {"image", "image_dit", "image_cfm"}:
            raise ValueError("model.model_type must be one of: image, image_dit, image_cfm for image inference")
        inf = self.inference
        updates = {}
        if inf.generation_mode == "diffusion" and inf.mask_id is None and self.model.mask_token_id is not None:
            updates["mask_id"] = self.model.mask_token_id
        if updates:
            inf = inf.model_copy(update=updates)
        if self.model.model_type == "image_dit" and inf.generation_mode != "flow":
            raise ValueError("model.model_type='image_dit' requires inference.generation_mode='flow'")
        if self.model.model_type == "image" and inf.generation_mode != "diffusion":
            raise ValueError("model.model_type='image' requires inference.generation_mode='diffusion'")
        if self.model.model_type == "image_cfm" and inf.generation_mode != "categorical_flow":
            raise ValueError("model.model_type='image_cfm' requires inference.generation_mode='categorical_flow'")
        self.inference = inf
        return self


class SweepInferConfig(_BaseConfig):
    tokenizer: TokenizerConfig
    model: ModelConfig
    checkpoint: CheckpointConfig
    inference: InferenceConfig
    guide: Optional[GuideConfig] = None
    sweep: SweepConfig
    logging: Optional[LoggingConfig] = None

    @model_validator(mode="after")
    def _finalize_inference(self):
        inf = self.inference
        updates = {}
        if inf.total_length is None:
            updates["total_length"] = self.model.context_length
        if inf.mask_id is None and self.model.mask_token_id is not None:
            updates["mask_id"] = self.model.mask_token_id
        if updates:
            inf = inf.model_copy(update=updates)
        self.inference = inf
        if self.guide is not None:
            if self.guide.model.vocab_size != self.model.vocab_size:
                raise ValueError("guide.model.vocab_size must match model.vocab_size")
            if self.guide.model.context_length < self.model.context_length:
                raise ValueError("guide.model.context_length must be >= model.context_length")
            if self.guide.model.device != self.model.device:
                raise ValueError("guide.model.device must match model.device")
        return self


class TrainTokenizerInputConfig(_BaseConfig):
    input_path: Path
    vocab_size: int
    special_tokens: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_tt_input(self):
        if not self.input_path.exists():
            raise FileNotFoundError(f"input_path not found: {self.input_path}")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        return self


class TrainTokenizerOutputConfig(_BaseConfig):
    merges_path: Path
    vocab_path: Path


class TrainTokenizerConfig(_BaseConfig):
    input: TrainTokenizerInputConfig
    output: TrainTokenizerOutputConfig


class BenchParams(_BaseConfig):
    warmup: int = 2
    repeats: int = 5
    steps: Optional[int] = None
    synchronize: bool = True
    backward: bool = False
    optimizer_step: bool = False
    perplexity_max_batches: Optional[int] = None
    perplexity_batch_size: Optional[int] = None
    perplexity_seed: Optional[int] = None

    @model_validator(mode="after")
    def _validate_bench(self):
        if self.warmup < 0:
            raise ValueError("warmup must be >= 0")
        if self.repeats <= 0:
            raise ValueError("repeats must be > 0")
        if self.steps is not None and self.steps <= 0:
            raise ValueError("steps must be > 0 when provided")
        if self.perplexity_max_batches is not None and self.perplexity_max_batches <= 0:
            raise ValueError("perplexity_max_batches must be > 0 when provided")
        if self.perplexity_batch_size is not None and self.perplexity_batch_size <= 0:
            raise ValueError("perplexity_batch_size must be > 0 when provided")
        if self.perplexity_seed is not None and self.perplexity_seed < 0:
            raise ValueError("perplexity_seed must be >= 0 when provided")
        return self


class BenchDataConfig(_BaseConfig):
    dataset_name: str
    dataset_config: Optional[str] = None
    split: str
    text_field: str
    pipeline_mode: str = "packed"
    pad_token_id: Optional[int] = None
    shuffle_buffer_size: int = 0
    shuffle_seed: Optional[int] = None
    cache_all: bool = False

    @model_validator(mode="after")
    def _validate_bench_data(self):
        if not self.dataset_name:
            raise ValueError("data.dataset_name must not be empty")
        if not self.split:
            raise ValueError("data.split must not be empty")
        if not self.text_field:
            raise ValueError("data.text_field must not be empty")
        self.pipeline_mode = self.pipeline_mode.lower()
        if self.pipeline_mode not in {"packed", "rows"}:
            raise ValueError("data.pipeline_mode must be one of: packed, rows")
        if self.pipeline_mode == "rows":
            if self.pad_token_id is None:
                raise ValueError("data.pad_token_id must be set when pipeline_mode='rows'")
            if self.pad_token_id < 0:
                raise ValueError("data.pad_token_id must be >= 0")
        if self.shuffle_buffer_size < 0:
            raise ValueError("data.shuffle_buffer_size must be >= 0")
        if self.shuffle_seed is not None and self.shuffle_seed < 0:
            raise ValueError("data.shuffle_seed must be >= 0 when provided")
        return self


class BenchInferConfig(_BaseConfig):
    tokenizer: TokenizerConfig
    model: ModelConfig
    checkpoint: CheckpointConfig
    inference: InferenceConfig
    benchmark: BenchParams
    data: Optional[BenchDataConfig] = None
    logging: Optional[LoggingConfig] = None
    optimizer: Optional["OptimizerBenchConfig"] = None

    @model_validator(mode="after")
    def _finalize_bench(self):
        updates = {}
        if self.inference.total_length is None:
            updates["total_length"] = self.model.context_length
        if self.inference.mask_id is None:
            updates["mask_id"] = self.model.mask_token_id
        if updates:
            self.inference = self.inference.model_copy(update=updates)
        if self.benchmark.steps is None:
            self.benchmark = self.benchmark.model_copy(update={"steps": self.model.context_length})
        return self


class OptimizerBenchConfig(_BaseConfig):
    lr: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_max_l2_norm: float = 0.0

    @model_validator(mode="after")
    def _validate_optimizer_bench(self):
        if len(self.betas) != 2 or not (0 <= self.betas[0] < 1 and 0 <= self.betas[1] < 1):
            raise ValueError("optimizer betas must be 2 values in [0, 1)")
        if self.eps <= 0:
            raise ValueError("optimizer eps must be > 0")
        if self.grad_clip_max_l2_norm < 0:
            raise ValueError("grad_clip_max_l2_norm must be >= 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.lr < 0:
            raise ValueError("lr must be >= 0")
        return self


class BenchTokenizerInput(_BaseConfig):
    text_list: List[str]

    @model_validator(mode="after")
    def _validate_text_list(self):
        if not self.text_list:
            raise ValueError("input.text_list must not be empty")
        return self


class BenchTokenizerParams(_BaseConfig):
    repeats: int = 5

    @model_validator(mode="after")
    def _validate_bench_params(self):
        if self.repeats <= 0:
            raise ValueError("benchmark.repeats must be > 0")
        return self


class BenchTokenizerConfig(_BaseConfig):
    tokenizer: TokenizerConfig
    input: BenchTokenizerInput
    benchmark: BenchTokenizerParams
    logging: Optional[LoggingConfig] = None
