"""Querybank Normalization (QB-Norm) for OpenCLIP retrieval scores.

Background
----------
Raw cosine similarity from OpenCLIP for "perfect" visual matches typically
lands in the 0.30-0.35 range, which confuses end users who interpret it as
"30% match". This module re-projects raw cosine scores onto an intuitive
[0, 100] confidence scale by normalizing each frame's response to the user
query against the same frame's distribution of responses to a *background*
set of domain-relevant text queries.

This also mitigates the hubness phenomenon: in high-dimensional vector
spaces, certain frames behave as "hubs" and score moderately high for many
queries indiscriminately. Per-frame z-score normalization penalizes such
hubs because their elevated mean-background-similarity (mu_f) cancels out
the boost from the user query's raw cosine.

The implementation follows the lineage of Bogolin et al. (CVPR 2022,
"Cross Modal Retrieval with Querybank Normalisation") as extended by
Galanopoulos et al. (CVPRW 2025, our flagship reference).

Algorithm
---------
Given frame embeddings F (N x D), a user query q (D,), and a background
querybank Q_bg (M x D), all L2-normalized:

    raw_f       = F @ q^T                         # cosine in [-1, 1]
    bg_matrix   = Q_bg @ F^T                      # (M, N), cosine each pair
    mu_f        = bg_matrix.mean(axis=0)          # (N,) per-frame mean
    sigma_f     = bg_matrix.std(axis=0)           # (N,) per-frame std
    z_f         = (raw_f - mu_f) / sigma_f        # (N,) normalized score
    confidence  = sigmoid(z_f) * 100              # (N,) UI-friendly %

The sigmoid mapping is calibrated for OpenCLIP's typical z-score range:
on real WP8 dashcam data with raw=0.32, mu~0.25, sigma~0.03 we expect
z ~= 2.3 -> sigmoid(2.3) * 100 ~= 91% — which is exactly the "perfect
match should display >90%" UX target.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Domain-specific background query bank
# -----------------------------------------------------------------------------
#
# Curated to span the visual concept space of security CCTV + automotive
# dashcam footage. Diversity matters more than density — these queries
# should collectively excite a wide variety of frames so the per-frame
# (mu_f, sigma_f) estimates are informative. They are NOT meant to be
# orthogonal to typical user queries; they are meant to be representative
# of the "natural noise floor" of text-image similarity in the domain.
#
# 25 queries: vehicles, pedestrians, infrastructure, weather, time-of-day,
# building exteriors, common security scenes.

SECURITY_DASHCAM_BACKGROUND_QUERIES: List[str] = [
    # Vehicles in motion
    "a red car driving on the highway",
    "a white truck overtaking another vehicle",
    "a motorcycle weaving through traffic",
    "a school bus stopped at an intersection",
    "a delivery van parked on the curb",
    # Pedestrians and crowds
    "a person walking at night under a streetlight",
    "a pedestrian crossing the street outside a crosswalk",
    "a crowd of people gathering on a sidewalk",
    "a child running across a parking lot",
    "a person carrying a shopping bag",
    # Traffic events and infrastructure
    "a vehicle running a red traffic light",
    "an empty street at dawn with no vehicles",
    "heavy traffic congestion at rush hour",
    "a stop sign at a rural intersection",
    "a roundabout with multiple cars",
    # Weather and environment
    "rainy weather reducing road visibility",
    "snow covering the road surface",
    "fog over the highway in the morning",
    "bright sunlight glare on the windshield",
    "twilight conditions at sunset",
    # Security and CCTV scenes
    "a person opening a building entrance door",
    "a delivery person approaching a porch",
    "a parking lot at midday with parked cars",
    "an empty office corridor with overhead lights",
    "a security camera mounted on a wall",
]


# -----------------------------------------------------------------------------
# Background bank encoding helper
# -----------------------------------------------------------------------------

def encode_background_queries(
    engine,
    queries: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    """Encode the background querybank with the provided retrieval engine.

    Text encoding is cheap (one forward pass for ~25 short strings) but
    callers should still cache the returned tensor for the lifetime of the
    engine — there is no value re-encoding the same strings per request.

    Args:
        engine: Any object exposing ``get_text_embeddings(List[str])`` that
            returns an L2-normalized ``torch.Tensor`` of shape ``(M, D)``.
            Both ``OpenCLIPEngine`` and ``CLIPEngine`` satisfy this.
        queries: Optional override for the default background bank. Pass an
            empty list and you will get an empty tensor — caller is
            responsible for ensuring at least 2 queries (sigma is undefined
            for n=1).

    Returns:
        ``(M, D)`` L2-normalized tensor of background-query text embeddings.
    """
    bank = list(queries) if queries is not None else SECURITY_DASHCAM_BACKGROUND_QUERIES
    if len(bank) < 2:
        raise ValueError(
            f"Background querybank must contain at least 2 queries "
            f"(got {len(bank)}); standard deviation is undefined otherwise."
        )
    return engine.get_text_embeddings(bank)


# -----------------------------------------------------------------------------
# Core QB-Norm function
# -----------------------------------------------------------------------------

@torch.inference_mode()
def compute_normalized_similarity(
    frame_embeddings: torch.Tensor,
    query_embedding: torch.Tensor,
    background_embeddings: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Apply Querybank Normalization to per-frame retrieval scores.

    All inputs must be L2-normalized (OpenCLIPEngine and CLIPEngine produce
    normalized output by default). Operates entirely on the device of
    ``frame_embeddings``.

    Args:
        frame_embeddings: ``(N, D)`` — frames to score.
        query_embedding:  ``(D,)`` or ``(1, D)`` — user query.
        background_embeddings: ``(M, D)`` with ``M >= 2`` — background bank.

    Returns:
        Dict with three tensors, each of shape ``(N,)``:

        - ``raw_similarity``  : cosine similarity of query to each frame,
                                in ``[-1, 1]``. The router uses this.
        - ``z_score``         : per-frame z-score relative to the bank.
                                Unbounded; typical good matches sit at 2-3.
        - ``confidence_pct``  : ``sigmoid(z) * 100`` in ``[0, 100]``. This
                                is the UX-friendly score for the frontend.

    Raises:
        ValueError: if input dimensionalities do not align.
    """
    # Squeeze the query to (D,) regardless of how it arrived.
    if query_embedding.dim() == 2:
        if query_embedding.shape[0] != 1:
            raise ValueError(
                f"query_embedding must encode a single query; got shape "
                f"{tuple(query_embedding.shape)}"
            )
        query_embedding = query_embedding.squeeze(0)
    if query_embedding.dim() != 1:
        raise ValueError(
            f"query_embedding must be (D,) or (1, D); got shape "
            f"{tuple(query_embedding.shape)}"
        )

    if frame_embeddings.dim() != 2:
        raise ValueError(
            f"frame_embeddings must be (N, D); got shape "
            f"{tuple(frame_embeddings.shape)}"
        )
    if background_embeddings.dim() != 2:
        raise ValueError(
            f"background_embeddings must be (M, D); got shape "
            f"{tuple(background_embeddings.shape)}"
        )

    d = query_embedding.shape[0]
    if frame_embeddings.shape[1] != d or background_embeddings.shape[1] != d:
        raise ValueError(
            f"Dimensionality mismatch: query D={d}, "
            f"frames D={frame_embeddings.shape[1]}, "
            f"background D={background_embeddings.shape[1]}"
        )
    if background_embeddings.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 background queries for a valid sigma; "
            f"got {background_embeddings.shape[0]}"
        )

    # Move everything onto the frame_embeddings device.
    device = frame_embeddings.device
    q  = query_embedding.to(device)
    bg = background_embeddings.to(device)

    # Raw cosine similarity, (N,)
    raw = frame_embeddings @ q  # since both are L2-normalized

    # Background similarity matrix: each row is one bg query's response
    # to all frames. Shape: (M, N)
    bg_sim = bg @ frame_embeddings.T

    # Per-frame mean and (unbiased) std over the M background queries.
    mu_f    = bg_sim.mean(dim=0)
    sigma_f = bg_sim.std(dim=0, unbiased=True).clamp(min=1e-6)

    # Per-frame z-score and confidence percentage.
    z          = (raw - mu_f) / sigma_f
    confidence = torch.sigmoid(z) * 100.0

    return {
        "raw_similarity": raw,
        "z_score":        z,
        "confidence_pct": confidence,
    }
