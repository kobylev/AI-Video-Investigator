"""Tests for src.retriever.qb_norm — Querybank Normalization.

Two flavors of test:

  1. Deterministic synthetic embeddings — full control over the math, used
     to verify the perfect-match and ambiguous-match cases hit the right
     confidence range.
  2. Live OpenCLIP encoding — slower, GPU-bound; gated behind a marker so
     the fast tests can run in CI without model weights.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.retriever.qb_norm import (
    SECURITY_DASHCAM_BACKGROUND_QUERIES,
    compute_normalized_similarity,
    encode_background_queries,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _one_hot(d: int, i: int) -> torch.Tensor:
    v = torch.zeros(d)
    v[i] = 1.0
    return v


# -----------------------------------------------------------------------------
# Synthetic-embedding tests
# -----------------------------------------------------------------------------

def test_perfect_match_yields_high_confidence():
    """Query identical to frame, background orthogonal to it — confidence
    should be effectively 100%."""
    d = 64
    frame = _one_hot(d, 0).unsqueeze(0)         # (1, D)
    query = _one_hot(d, 0)                       # (D,)
    bg = torch.stack([_one_hot(d, i + 1) for i in range(20)])  # (20, D), all orthogonal to frame

    result = compute_normalized_similarity(frame, query, bg)

    # Raw cosine should be exactly 1
    assert torch.isclose(result["raw_similarity"], torch.tensor([1.0]))
    # Background cosines are all 0 -> mu_f=0, sigma_f=0 -> clamped to 1e-6 -> huge z
    assert result["z_score"].item() > 100  # very large positive
    assert result["confidence_pct"].item() > 99.0


def test_baseline_match_yields_midrange_confidence():
    """Query similarity equal to mean background similarity -> z=0 ->
    confidence ~= 50%."""
    d = 64
    # Construct: a frame, and 4 background queries each with cosine 0.3 to it,
    # achieved by linear combination of two orthogonal one-hots.
    frame = _one_hot(d, 0).unsqueeze(0)
    # bg_i = 0.3 * e0 + sqrt(1 - 0.09) * e_{i+1}  -> unit norm, cos=0.3
    coeff = math.sqrt(1.0 - 0.3 ** 2)
    bg_rows = []
    for i in range(8):
        v = 0.3 * _one_hot(d, 0) + coeff * _one_hot(d, i + 1)
        bg_rows.append(F.normalize(v, p=2, dim=0))
    bg = torch.stack(bg_rows)
    # Query: same 0.3 cosine to frame, but with yet another orthogonal direction.
    query = 0.3 * _one_hot(d, 0) + coeff * _one_hot(d, 50)
    query = F.normalize(query, p=2, dim=0)

    result = compute_normalized_similarity(frame, query, bg)

    # Raw cosine ~= 0.3, mu_f ~= 0.3 -> z ~= 0 -> confidence ~= 50%
    assert abs(result["raw_similarity"].item() - 0.3) < 1e-4
    assert abs(result["z_score"].item()) < 0.5
    assert 30.0 < result["confidence_pct"].item() < 70.0


def test_below_baseline_yields_low_confidence():
    """Query similarity well below background mean -> low confidence."""
    d = 64
    frame = _one_hot(d, 0).unsqueeze(0)
    # Background: 8 queries each with cosine ~0.4 to the frame.
    high = math.sqrt(1.0 - 0.4 ** 2)
    bg = torch.stack([
        F.normalize(0.4 * _one_hot(d, 0) + high * _one_hot(d, i + 1), p=2, dim=0)
        for i in range(8)
    ])
    # Query: only 0.1 cosine — well below the 0.4 baseline.
    low = math.sqrt(1.0 - 0.1 ** 2)
    query = F.normalize(0.1 * _one_hot(d, 0) + low * _one_hot(d, 50), p=2, dim=0)

    result = compute_normalized_similarity(frame, query, bg)

    assert result["z_score"].item() < -1.0
    assert result["confidence_pct"].item() < 30.0


def test_multi_frame_independent_normalization():
    """Each frame is normalized against its OWN background distribution,
    not a corpus-wide one. A frame that is a 'hub' (high bg mean) should
    get less of a boost than a frame with low bg mean for the same raw."""
    d = 64
    # Frame 0: a non-hub (orthogonal to bg). Frame 1: a hub (matches bg).
    f0 = _one_hot(d, 0)
    f1 = _one_hot(d, 1)
    frames = torch.stack([f0, f1])

    # Background: 8 queries near direction e1 (hub direction).
    bg = torch.stack([
        F.normalize(0.5 * _one_hot(d, 1) + math.sqrt(0.75) * _one_hot(d, i + 2), p=2, dim=0)
        for i in range(8)
    ])

    # User query: cosine 0.4 to BOTH frames (equally).
    coeff = math.sqrt(1.0 - 2 * 0.4 ** 2)
    query = F.normalize(0.4 * f0 + 0.4 * f1 + coeff * _one_hot(d, 60), p=2, dim=0)

    result = compute_normalized_similarity(frames, query, bg)

    # f0 is a non-hub: bg ~ 0 there -> z_0 should be large positive
    # f1 is a hub:     bg ~ 0.5 there -> z_1 should be smaller (or negative)
    assert result["z_score"][0] > result["z_score"][1]
    assert result["confidence_pct"][0] > result["confidence_pct"][1]


def test_input_validation_query_2d_with_multiple_rows():
    d = 64
    frame = _one_hot(d, 0).unsqueeze(0)
    bg = torch.eye(d)[:8]
    bad_query = torch.eye(d)[:2]  # (2, D) — invalid, must be one query
    with pytest.raises(ValueError, match="single query"):
        compute_normalized_similarity(frame, bad_query, bg)


def test_input_validation_dim_mismatch():
    frame = torch.zeros(4, 64)
    bg = torch.zeros(8, 64)
    bad_query = torch.zeros(32)  # wrong D
    with pytest.raises(ValueError, match="Dimensionality mismatch"):
        compute_normalized_similarity(frame, bad_query, bg)


def test_input_validation_singleton_background():
    frame = torch.zeros(4, 64)
    query = torch.zeros(64)
    bad_bg = torch.zeros(1, 64)  # only one query
    with pytest.raises(ValueError, match="at least 2 background"):
        compute_normalized_similarity(frame, query, bad_bg)


def test_default_bank_size():
    """The default bank should be in the spec's 20-30 range."""
    assert 20 <= len(SECURITY_DASHCAM_BACKGROUND_QUERIES) <= 30


# -----------------------------------------------------------------------------
# Live OpenCLIP test (slow; opt-in via marker)
# -----------------------------------------------------------------------------

@pytest.mark.slow
def test_encode_background_queries_with_openclip():
    """Verify end-to-end that OpenCLIP encodes the bank to the expected
    shape and L2-normalization."""
    from src.retriever.openclip_engine import OpenCLIPEngine
    engine = OpenCLIPEngine(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k")
    bg = encode_background_queries(engine)
    assert bg.shape[0] == len(SECURITY_DASHCAM_BACKGROUND_QUERIES)
    # L2-normalized -> each row has norm 1
    norms = bg.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
