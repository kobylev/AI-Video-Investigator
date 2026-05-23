"""Tests for src.router.temporal_dedup.

The cluster-of-4 case is the canonical demonstration requested in the
specification: four frames from the same scene must collapse to the single
highest-scoring representative.
"""

from __future__ import annotations

from src.router.temporal_dedup import temporal_deduplicate_frames


def test_cluster_of_four_collapses_to_single_winner():
    """A cluster of 4 close frames (all within a 5-second window) must
    reduce to exactly one frame — the highest-scoring one."""
    frames = [
        {"frame_id": 329, "timestamp_sec": 329.0, "similarity_score": 0.85},
        {"frame_id": 330, "timestamp_sec": 330.0, "similarity_score": 0.92},  # winner
        {"frame_id": 332, "timestamp_sec": 332.0, "similarity_score": 0.78},
        {"frame_id": 333, "timestamp_sec": 333.0, "similarity_score": 0.88},
    ]

    result = temporal_deduplicate_frames(frames, time_window_sec=5)

    assert len(result) == 1
    assert result[0]["frame_id"] == 330
    assert result[0]["similarity_score"] == 0.92


def test_temporally_distant_clusters_are_both_preserved():
    """Two separate scene clusters (>> time_window apart) must each yield
    their own winner — output is chronologically sorted."""
    frames = [
        # Cluster A around t=330 (the 4-frame case above)
        {"frame_id": 329, "timestamp_sec": 329.0, "similarity_score": 0.85},
        {"frame_id": 330, "timestamp_sec": 330.0, "similarity_score": 0.92},
        {"frame_id": 332, "timestamp_sec": 332.0, "similarity_score": 0.78},
        {"frame_id": 333, "timestamp_sec": 333.0, "similarity_score": 0.88},
        # Cluster B around t=480 (well outside time_window from A)
        {"frame_id": 480, "timestamp_sec": 480.0, "similarity_score": 0.81},
        {"frame_id": 481, "timestamp_sec": 481.0, "similarity_score": 0.79},
    ]

    result = temporal_deduplicate_frames(frames, time_window_sec=5)

    assert [r["frame_id"] for r in result] == [330, 480]
    assert [r["similarity_score"] for r in result] == [0.92, 0.81]


def test_empty_input_returns_empty_list():
    assert temporal_deduplicate_frames([], time_window_sec=5) == []


def test_input_list_is_not_mutated():
    """The function must not reorder or modify its input."""
    frames = [
        {"timestamp_sec": 5.0, "similarity_score": 0.6},
        {"timestamp_sec": 10.0, "similarity_score": 0.9},
        {"timestamp_sec": 15.0, "similarity_score": 0.7},
    ]
    original_order = [f["timestamp_sec"] for f in frames]

    temporal_deduplicate_frames(frames, time_window_sec=3)

    assert [f["timestamp_sec"] for f in frames] == original_order


def test_window_boundary_is_suppressed():
    """Frames exactly time_window_sec apart are considered 'within' the
    window and the lower-scoring one is suppressed."""
    frames = [
        {"timestamp_sec": 0.0, "similarity_score": 0.9},
        {"timestamp_sec": 5.0, "similarity_score": 0.6},  # exactly window away
    ]

    result = temporal_deduplicate_frames(frames, time_window_sec=5)

    assert len(result) == 1
    assert result[0]["timestamp_sec"] == 0.0


def test_window_boundary_plus_one_is_kept():
    """Frames strictly more than time_window_sec apart are both kept."""
    frames = [
        {"timestamp_sec": 0.0, "similarity_score": 0.9},
        {"timestamp_sec": 5.1, "similarity_score": 0.6},
    ]

    result = temporal_deduplicate_frames(frames, time_window_sec=5)

    assert len(result) == 2
