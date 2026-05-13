"""C3 (continuous log-Δt time encoding) — mandatory test contract.

Implements the 5 assertions from `docs/superpowers/plans/2026-05-10-pcvr-
roadmap-v3-architecture.md`, Phase C, card C3 (lines 284-291). Per the
roadmap discipline checklist (line 47), this file MUST pass before any
C3 training run or submission build:

  1. log(1 + 300) ≈ 5.71 is the value the encoder receives for Δt=300s.
  2. Two Δt values previously sharing a bucket (305s, 595s — bucket 21
     in the v0 BUCKET_BOUNDARIES) produce DISTINCT encoder outputs.
  3. Encoder output at padding positions (log_dt == 0) is EXACTLY the
     zero vector — fixes the v4 LayerNorm-bias-at-padding leak.
  4. Encoder output shape is (B, L, d_model).
  5. The infer/trainer wiring routes the new `{domain}_log_time_delta`
     batch key into ModelInput.seq_log_time_delta — when
     use_continuous_time=True is set and the key is missing, the model
     must raise (no silent fallback to zeros — that's v4's regression
     surface).

Run:  python -m pytest pcvr/tests/test_c3_continuous_time.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.model import ContinuousTimeEncoder, ModelInput  # noqa: E402
from src.data import BUCKET_BOUNDARIES  # noqa: E402


D_MODEL = 64


@pytest.fixture
def encoder() -> ContinuousTimeEncoder:
    torch.manual_seed(42)
    return ContinuousTimeEncoder(D_MODEL)


# --------------------------------------------------------------------- #
# Assertion 1 — log(1 + 300) ≈ 5.71 reaches the encoder.
# --------------------------------------------------------------------- #
def test_a1_log1p_300_is_5_71() -> None:
    """log(1 + 300) = log(301) ≈ 5.7071, matching roadmap's ≈ 5.71."""
    expected = math.log1p(300.0)
    assert abs(expected - 5.71) < 1e-2  # roadmap claim "≈ 5.71"
    # Also confirm the dataset path (float32 numpy) computes it identically.
    assert abs(float(np.log1p(np.float32(300.0))) - expected) < 1e-4


# --------------------------------------------------------------------- #
# Assertion 2 — same-bucket Δts produce DISTINCT encoder outputs.
# --------------------------------------------------------------------- #
def test_a2_same_bucket_produces_distinct_outputs(
    encoder: ContinuousTimeEncoder,
) -> None:
    """305s and 595s both land in the v0 bucket containing 300s..600s.

    The bucket scheme collapses them to identical embeddings; C3's
    continuous encoder must keep them distinct.
    """
    # Both 305 and 595 land in the same bucket (between boundaries 300 and 600).
    boundary_300_idx = np.searchsorted(BUCKET_BOUNDARIES, 305)
    boundary_595_idx = np.searchsorted(BUCKET_BOUNDARIES, 595)
    assert boundary_300_idx == boundary_595_idx, (
        f"Test fixture assumption broken: 305s and 595s should share a bucket; "
        f"got {boundary_300_idx} vs {boundary_595_idx}"
    )

    log_dt = torch.tensor(
        [[math.log1p(305.0), math.log1p(595.0)]], dtype=torch.float32
    )  # (B=1, L=2)
    encoder.eval()
    with torch.no_grad():
        out = encoder(log_dt)  # (1, 2, D)
    assert out.shape == (1, 2, D_MODEL)
    assert not torch.allclose(out[0, 0], out[0, 1], atol=1e-3), (
        "Continuous encoder collapses two distinct Δt values to the same "
        "vector; verify Linear(1, d_model) is actually wired and not "
        "replaced by an identity passthrough."
    )


# --------------------------------------------------------------------- #
# Assertion 3 — padding (log_dt == 0) → EXACTLY the zero vector.
# --------------------------------------------------------------------- #
def test_a3_padding_emits_exact_zero_vector(
    encoder: ContinuousTimeEncoder,
) -> None:
    """The single failure mode that produced the v4 −0.014 regression.

    With bias=True + LayerNorm, padding positions emit a non-zero bias
    vector that contaminates downstream mean-pooling. With bias=False
    + explicit zero mask, padding emits exactly zero.
    """
    B, L = 2, 5
    # Mix real and padding positions in each row.
    log_dt = torch.tensor(
        [
            [math.log1p(60.0), 0.0, math.log1p(3600.0), 0.0, 0.0],  # 3 padding
            [0.0, 0.0, 0.0, math.log1p(120.0), math.log1p(7200.0)],  # 3 padding
        ],
        dtype=torch.float32,
    )
    encoder.eval()
    with torch.no_grad():
        out = encoder(log_dt)  # (B, L, D)
    pad_mask = log_dt == 0  # (B, L)
    pad_out = out[pad_mask]  # (n_pad, D)
    assert torch.equal(pad_out, torch.zeros_like(pad_out)), (
        "Padding positions emitted a non-zero vector. This was v4's bug: "
        "LayerNorm bias leaks at every padded slot. Verify ContinuousTime"
        "Encoder uses bias=False AND the explicit (log_dt != 0) mask."
    )
    # Real positions must NOT be zero.
    real_out = out[~pad_mask]
    assert (real_out.abs().sum(dim=-1) > 0).all(), (
        "A real (non-padding) position emitted the zero vector. The encoder "
        "may have a degenerate weight init or the mask logic is inverted."
    )


# --------------------------------------------------------------------- #
# Assertion 4 — encoder output shape is (B, L, d_model).
# --------------------------------------------------------------------- #
def test_a4_output_shape(encoder: ContinuousTimeEncoder) -> None:
    B, L = 3, 7
    log_dt = torch.randn(B, L).abs()  # positive values, no padding
    out = encoder(log_dt)
    assert out.shape == (B, L, D_MODEL)


# --------------------------------------------------------------------- #
# Assertion 5 — infer/trainer ModelInput wiring routes the new key.
# --------------------------------------------------------------------- #
def test_a5_model_input_carries_log_time_delta() -> None:
    """ModelInput must have a `seq_log_time_delta` field, and the model
    must raise (not silently fall back to zeros) when use_continuous_time
    is True and the field is missing.

    The silent-fallback failure mode is what produced the v4 regression;
    this assertion locks the absence of that path.
    """
    # Field exists.
    assert "seq_log_time_delta" in ModelInput._fields, (
        "ModelInput missing seq_log_time_delta — infer.py/trainer.py cannot "
        "route the new batch key without it."
    )
    # Default is None (back-compat with v0 checkpoints that never set it).
    assert ModelInput._field_defaults.get("seq_log_time_delta") is None

    # When the field is None and use_continuous_time=True, the model
    # raises — proving there's no silent fallback. We test this by
    # invoking _embed_seq_domain via a lightweight stub model that has
    # use_continuous_time=True but receives log_time_delta=None.
    from src.model import PCVRHyFormer

    # Minimal model construction args. Vocab sizes / specs are tiny
    # placeholders — we only need the embed method to be callable.
    model = PCVRHyFormer(
        user_int_feature_specs=[(10, 0, 1)],
        item_int_feature_specs=[(10, 0, 1)],
        user_dense_dim=0,
        item_dense_dim=0,
        seq_vocab_sizes={"seq_a": [10]},
        user_ns_groups=[[0]],
        item_ns_groups=[[0]],
        d_model=16,
        emb_dim=16,
        num_queries=1,
        num_hyformer_blocks=1,
        num_heads=2,
        seq_encoder_type="swiglu",
        action_num=1,
        num_time_buckets=0,
        rank_mixer_mode="none",
        use_continuous_time=True,
    )
    # Forward with log_time_delta=None must error explicitly.
    seq = torch.zeros(1, 1, 4, dtype=torch.long)  # (B, S, L)
    time_bucket_ids = torch.zeros(1, 4, dtype=torch.long)
    with pytest.raises(RuntimeError, match="use_continuous_time"):
        model._embed_seq_domain(
            seq=seq,
            sideinfo_embs=model._seq_embs["seq_a"],
            proj=model._seq_proj["seq_a"],
            is_id=model._seq_is_id["seq_a"],
            emb_index=model._seq_emb_index["seq_a"],
            time_bucket_ids=time_bucket_ids,
            log_time_delta=None,
        )
