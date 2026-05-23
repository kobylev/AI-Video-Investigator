"""Temporal Non-Maximum Suppression for retrieved video frames.

Stage 1 (CLIP + FAISS) often returns clusters of near-duplicate frames from
the same scene — e.g., 00:05:29, 00:05:30, 00:05:33 — because the underlying
moments are visually identical. Escalating all of them to Claude wastes
tokens, since the forensic verdict on any one frame in such a cluster will
hold for the rest.

This module implements 1-D greedy NMS: it walks frames in descending
similarity order and keeps each candidate only if it is more than
`time_window_sec` away from every already-kept frame. The result is a
chronologically-sorted set of "scene representatives" — at most one frame
per `time_window_sec` neighborhood, each the strongest of its cluster.

Algorithmic choice: greedy NMS over sequential clustering, because a long
chain of pairwise-close frames (each within the window of its neighbor)
would otherwise collapse to a single representative even when the chain's
endpoints are 30+ seconds apart. For a car-following-car dashcam scene
that lasts 30s, you want multiple representatives, not one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def temporal_deduplicate_frames(
    frames: List[Dict[str, Any]],
    time_window_sec: int = 5,
    high_score_threshold: Optional[float] = None,
    threshold_key: str = "similarity_score",
) -> List[Dict[str, Any]]:
    """Suppress temporally-redundant frames via 1-D NMS with optional
    high-confidence tolerance.

    Default behaviour (``high_score_threshold=None``): classic NMS — for each
    cluster, only the highest-scoring frame survives.

    Tolerance behaviour (``high_score_threshold`` set): if BOTH a candidate
    and the nearby already-kept frame score above ``high_score_threshold``
    on ``threshold_key``, the candidate is kept anyway. This preserves
    critical-action sequences where multiple frames within the time window
    legitimately capture an unfolding event — e.g., a crash visible in three
    consecutive 1-second-apart frames, all scoring 95%+ on QB-Norm
    confidence. Without the tolerance, NMS would aggressively collapse them
    to one representative and zero out the recall on adjacent ground-truth
    frames (the WP8 frame 510 vs 511 case).

    Args:
        frames: List of frame dicts. Each must contain at least
            ``timestamp_sec`` and ``similarity_score``. If
            ``high_score_threshold`` is given, every frame must also contain
            ``threshold_key``.
        time_window_sec: Suppression radius in seconds.
        high_score_threshold: If set, two adjacent frames both scoring above
            this value on ``threshold_key`` are both kept. Typical values:
            ``90.0`` for QB-Norm confidence_pct, ``0.30`` for raw cosine.
        threshold_key: Which dict key the tolerance threshold is compared
            against. Defaults to the same key NMS orders by; set to
            ``"confidence_pct"`` when running tolerance against QB-Norm.

    Returns:
        New list of kept frame dicts, sorted ascending by ``timestamp_sec``.
        Input list is not mutated; dict identities are preserved.

    Raises:
        KeyError: if a required key is missing from any frame.
    """
    if not frames:
        return []

    # NMS ordering is ALWAYS by similarity_score (the canonical retrieval
    # score). The tolerance threshold is a separate orthogonal check.
    by_score_desc = sorted(
        frames,
        key=lambda f: f["similarity_score"],
        reverse=True,
    )

    def _passes_tolerance(candidate: Dict[str, Any], conflicts: List[Dict[str, Any]]) -> bool:
        """True iff candidate AND every conflicting kept frame are both
        above the high-confidence threshold."""
        if high_score_threshold is None:
            return False
        cand_score = candidate.get(threshold_key)
        if cand_score is None or cand_score < high_score_threshold:
            return False
        return all(
            k.get(threshold_key) is not None and k[threshold_key] >= high_score_threshold
            for k in conflicts
        )

    kept: List[Dict[str, Any]] = []
    for candidate in by_score_desc:
        t = candidate["timestamp_sec"]
        conflicts = [k for k in kept if abs(t - k["timestamp_sec"]) <= time_window_sec]
        if not conflicts or _passes_tolerance(candidate, conflicts):
            kept.append(candidate)

    kept.sort(key=lambda f: f["timestamp_sec"])
    return kept
