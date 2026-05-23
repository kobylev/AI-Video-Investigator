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
    max_per_cluster: Optional[int] = None,
    peak_proximity_delta: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Suppress temporally-redundant frames via 1-D NMS with optional
    high-confidence tolerance, hard cluster-size cap, and peak-proximity
    constraint.

    Default behaviour (``high_score_threshold=None``): classic NMS — for each
    cluster, only the highest-scoring frame survives.

    Soft NMS (``high_score_threshold`` set): a candidate within the time
    window of an already-kept frame is kept only if it passes ALL of the
    enabled constraints below.

      1. **Threshold check (always enforced when set):** the candidate AND
         every conflicting kept frame must score at or above
         ``high_score_threshold`` on ``threshold_key``.

      2. **Peak-proximity check (when ``peak_proximity_delta`` is set):** the
         candidate's score must be within ``peak_proximity_delta`` of the
         highest-scoring conflicting kept frame (the de-facto cluster peak).
         Prevents a 98.0 candidate from joining a 99.9 peak's cluster when
         the operator only wants "near-tie" neighbours, not "barely above
         threshold" stragglers.

      3. **Cluster-cap check (when ``max_per_cluster`` is set):** the
         resulting cluster size (existing conflicts + 1) must not exceed
         ``max_per_cluster``. A hard guardrail against a long chain of
         barely-passing frames all surviving by transitive tolerance — e.g.,
         the WP8 frame 271/272/273 case where QB-Norm pushed adjacent
         representatives all to 95-97%, flooding the UI with three
         near-identical porch views.

    Recommended production preset (``src/server.py`` uses these):
        ``high_score_threshold=98.0,  threshold_key="confidence_pct",
          max_per_cluster=2,  peak_proximity_delta=2.0``

    With this preset:
        * Classic NMS handles the typical case (single representative per
          5-second cluster).
        * Soft NMS kicks in only when two frames are both ≥ 98% AND within
          2 percentage points of each other — i.e., a genuine "two cameras
          on the same instant" tie, not a long fade-in.
        * No cluster ever shows more than 2 survivors, regardless of how
          tightly QB-Norm crowds scores at the top.

    Args:
        frames: Frame dicts. Each must contain ``timestamp_sec`` and
            ``similarity_score``. When ``high_score_threshold`` is set,
            every frame must also contain ``threshold_key``.
        time_window_sec: Suppression radius in seconds.
        high_score_threshold: Minimum ``threshold_key`` value for a frame to
            be exempted from suppression by an in-window peer.
        threshold_key: Which dict key the tolerance threshold reads.
        max_per_cluster: Hard cap on survivors within a single
            ``time_window_sec`` neighbourhood. ``None`` = no cap.
        peak_proximity_delta: Maximum ``threshold_key`` gap between a
            candidate and the cluster peak for tolerance to apply.
            ``None`` = no proximity constraint.

    Returns:
        New list of kept frame dicts, sorted ascending by ``timestamp_sec``.
        Input list is not mutated; dict identities are preserved.

    Raises:
        KeyError: if a required key is missing from any frame.
    """
    if not frames:
        return []

    # NMS ordering is ALWAYS by similarity_score (the canonical retrieval
    # score). Tolerance threshold + proximity + cap are orthogonal checks
    # applied per-candidate during the greedy walk.
    by_score_desc = sorted(
        frames,
        key=lambda f: f["similarity_score"],
        reverse=True,
    )

    def _passes_threshold(candidate: Dict[str, Any], conflicts: List[Dict[str, Any]]) -> bool:
        if high_score_threshold is None:
            return False
        cand_score = candidate.get(threshold_key)
        if cand_score is None or cand_score < high_score_threshold:
            return False
        return all(
            k.get(threshold_key) is not None and k[threshold_key] >= high_score_threshold
            for k in conflicts
        )

    def _passes_proximity(candidate: Dict[str, Any], conflicts: List[Dict[str, Any]]) -> bool:
        if peak_proximity_delta is None:
            return True
        cand_score = candidate.get(threshold_key, 0.0)
        peak_score = max(k.get(threshold_key, 0.0) for k in conflicts)
        return (peak_score - cand_score) <= peak_proximity_delta

    def _passes_cluster_cap(conflicts: List[Dict[str, Any]]) -> bool:
        if max_per_cluster is None:
            return True
        # Adding this candidate would make the cluster size = len(conflicts) + 1.
        # Reject when that would exceed the cap.
        return len(conflicts) + 1 <= max_per_cluster

    kept: List[Dict[str, Any]] = []
    for candidate in by_score_desc:
        t = candidate["timestamp_sec"]
        conflicts = [k for k in kept if abs(t - k["timestamp_sec"]) <= time_window_sec]
        if not conflicts:
            kept.append(candidate)
            continue
        if not _passes_threshold(candidate, conflicts):
            continue
        if not _passes_proximity(candidate, conflicts):
            continue
        if not _passes_cluster_cap(conflicts):
            continue
        kept.append(candidate)

    kept.sort(key=lambda f: f["timestamp_sec"])
    return kept
