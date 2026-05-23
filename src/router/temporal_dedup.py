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

from typing import Any, Dict, List


def temporal_deduplicate_frames(
    frames: List[Dict[str, Any]],
    time_window_sec: int = 5,
) -> List[Dict[str, Any]]:
    """Suppress temporally-redundant frames via 1-D NMS.

    Each kept frame is guaranteed to be (a) the highest-scoring frame within
    +/- ``time_window_sec`` of itself, and (b) more than ``time_window_sec``
    away from every other kept frame.

    Args:
        frames: List of frame dicts. Each must contain at least:
            ``timestamp_sec`` (float | int) and ``similarity_score`` (float).
            All other keys (``frame_id``, ``image_path``, ...) are preserved
            on whichever frames survive.
        time_window_sec: Suppression radius in seconds. Two kept frames are
            guaranteed to be strictly more than this many seconds apart.

    Returns:
        New list of kept frame dicts, sorted ascending by ``timestamp_sec``.
        The input list is not mutated; the returned dict objects are the same
        identities as the corresponding inputs (no copy).

    Raises:
        KeyError: if any frame is missing ``timestamp_sec`` or
            ``similarity_score``.
    """
    if not frames:
        return []

    # Greedy NMS — highest-scoring frames win ties for their neighborhood.
    by_score_desc = sorted(
        frames,
        key=lambda f: f["similarity_score"],
        reverse=True,
    )

    kept: List[Dict[str, Any]] = []
    for candidate in by_score_desc:
        t = candidate["timestamp_sec"]
        if all(abs(t - k["timestamp_sec"]) > time_window_sec for k in kept):
            kept.append(candidate)

    kept.sort(key=lambda f: f["timestamp_sec"])
    return kept
