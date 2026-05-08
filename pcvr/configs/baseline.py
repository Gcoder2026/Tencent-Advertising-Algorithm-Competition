"""Baseline experiment Config — typed, importable, validated.

One experiment = one Config instance. Trainer dumps `dataclasses.asdict(cfg)`
to `train_config.json` next to `model.pt`; infer.py reconstructs Config(**d).
"""
from dataclasses import dataclass, field
from typing import Dict


_ALLOWED_DENSE_OPTIMIZERS = {"adamw", "sgd"}
_ALLOWED_SPARSE_OPTIMIZERS = {"adagrad"}
_ALLOWED_LOSSES = {"bce", "focal"}
_ALLOWED_NS_TOKENIZERS = {"group", "rankmixer"}
_ALLOWED_SEQ_ENCODERS = {"swiglu", "transformer", "longer"}
_ALLOWED_RANK_MIXER_MODES = {"full", "ffn_only", "none"}


def _default_seq_max_lens() -> Dict[str, int]:
    return {"seq_a": 256, "seq_b": 256, "seq_c": 512, "seq_d": 512}


@dataclass
class Config:
    # ---- Paths (overridden by env vars on platform) ----
    data_dir: str = ""
    schema_path: str = ""           # default: <data_dir>/schema.json
    ckpt_dir: str = ""
    log_dir: str = ""
    tf_events_dir: str = ""
    ns_groups_json: str = ""        # empty -> singleton groups
    resume_from: str = ""           # empty -> train from scratch

    # ---- Data ----
    batch_size: int = 256
    seq_max_lens: Dict[str, int] = field(default_factory=_default_seq_max_lens)
    num_workers: int = 8
    buffer_batches: int = 20
    train_ratio: float = 1.0
    valid_ratio: float = 0.10
    eval_every_n_steps: int = 0     # 0 = end-of-epoch only

    # ---- Model ----
    d_model: int = 64
    emb_dim: int = 64
    num_queries: int = 2
    num_hyformer_blocks: int = 2
    num_heads: int = 4
    seq_encoder_type: str = "swiglu"
    hidden_mult: int = 4
    dropout_rate: float = 0.01
    seq_top_k: int = 50
    seq_causal: bool = False
    action_num: int = 1
    use_time_buckets: bool = True
    rank_mixer_mode: str = "full"
    use_rope: bool = False
    rope_base: float = 10000.0
    emb_skip_threshold: int = 1_000_000
    seq_id_threshold: int = 10_000
    ns_tokenizer_type: str = "rankmixer"
    user_ns_tokens: int = 5
    item_ns_tokens: int = 2

    # ---- Training ----
    num_epochs: int = 999
    patience: int = 5
    seed: int = 42
    device: str = "cuda"

    # ---- Optimizer ----
    dense_optimizer: str = "adamw"
    sparse_optimizer: str = "adagrad"
    lr: float = 1e-4
    sparse_lr: float = 0.05
    sparse_weight_decay: float = 0.0
    weight_decay: float = 0.0

    # ---- LR schedule ----
    warmup_steps: int = 1000
    cosine_decay: bool = True
    min_lr_factor: float = 0.1
    total_steps_hint: int = 0       # 0 -> derived from data + epochs at runtime

    # ---- Mixed precision ----
    use_bf16: bool = True

    # ---- Loss ----
    loss_type: str = "bce"
    focal_alpha: float = 0.1
    focal_gamma: float = 2.0

    # ---- Sparse re-init (KuaiShou MultiEpoch trick) ----
    reinit_sparse_after_epoch: int = 1
    reinit_cardinality_threshold: int = 0

    # ---- Save / resume ----
    save_every_n_steps: int = 0     # 0 -> only at validation new-best
    keep_last_n_step_ckpts: int = 1

    def __post_init__(self) -> None:
        if self.dense_optimizer not in _ALLOWED_DENSE_OPTIMIZERS:
            raise ValueError(
                f"dense_optimizer must be in {sorted(_ALLOWED_DENSE_OPTIMIZERS)}, "
                f"got {self.dense_optimizer!r}")
        if self.sparse_optimizer not in _ALLOWED_SPARSE_OPTIMIZERS:
            raise ValueError(
                f"sparse_optimizer must be in {sorted(_ALLOWED_SPARSE_OPTIMIZERS)}, "
                f"got {self.sparse_optimizer!r}")
        if self.loss_type not in _ALLOWED_LOSSES:
            raise ValueError(
                f"loss_type must be in {sorted(_ALLOWED_LOSSES)}, got {self.loss_type!r}")
        if self.ns_tokenizer_type not in _ALLOWED_NS_TOKENIZERS:
            raise ValueError(
                f"ns_tokenizer_type must be in {sorted(_ALLOWED_NS_TOKENIZERS)}, "
                f"got {self.ns_tokenizer_type!r}")
        if self.seq_encoder_type not in _ALLOWED_SEQ_ENCODERS:
            raise ValueError(
                f"seq_encoder_type must be in {sorted(_ALLOWED_SEQ_ENCODERS)}, "
                f"got {self.seq_encoder_type!r}")
        if self.rank_mixer_mode not in _ALLOWED_RANK_MIXER_MODES:
            raise ValueError(
                f"rank_mixer_mode must be in {sorted(_ALLOWED_RANK_MIXER_MODES)}, "
                f"got {self.rank_mixer_mode!r}")
        if not (0.0 < self.min_lr_factor <= 1.0):
            raise ValueError(f"min_lr_factor must be in (0, 1], got {self.min_lr_factor}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
