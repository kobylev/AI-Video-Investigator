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


# -----------------------------------------------------------------------------
# High-score tolerance (the WP8 frame 510 vs 511 fix)
# -----------------------------------------------------------------------------

def test_adjacent_both_high_confidence_are_both_kept():
    """The WP8 regression case: two adjacent frames both scoring >90% on
    QB-Norm should both survive when the tolerance is enabled."""
    frames = [
        # The actual WP8 numbers: frame 510 vs 511 (1s apart, both 90%+)
        {"timestamp_sec": 493.49, "similarity_score": 0.3088, "confidence_pct": 96.1, "fid": 510},
        {"timestamp_sec": 494.46, "similarity_score": 0.3203, "confidence_pct": 98.2, "fid": 511},
    ]

    result = temporal_deduplicate_frames(
        frames,
        time_window_sec=5,
        high_score_threshold=90.0,
        threshold_key="confidence_pct",
    )

    assert [r["fid"] for r in result] == [510, 511]


def test_adjacent_only_one_high_confidence_suppresses_lower():
    """If only ONE of the adjacent frames is above threshold, the lower
    must still be suppressed — no asymmetric exemptions."""
    frames = [
        {"timestamp_sec": 100.0, "similarity_score": 0.20, "confidence_pct": 85.0, "fid": "low"},
        {"timestamp_sec": 101.0, "similarity_score": 0.32, "confidence_pct": 95.0, "fid": "high"},
    ]

    result = temporal_deduplicate_frames(
        frames,
        time_window_sec=5,
        high_score_threshold=90.0,
        threshold_key="confidence_pct",
    )

    assert [r["fid"] for r in result] == ["high"]


def test_tolerance_disabled_default_keeps_old_behaviour():
    """Regression: without high_score_threshold, behaviour is identical to
    the pre-patch NMS — only the highest in each cluster survives."""
    frames = [
        {"timestamp_sec": 100.0, "similarity_score": 0.31, "confidence_pct": 96.0, "fid": "near"},
        {"timestamp_sec": 101.0, "similarity_score": 0.32, "confidence_pct": 98.0, "fid": "peak"},
    ]

    result = temporal_deduplicate_frames(frames, time_window_sec=5)  # no threshold

    assert [r["fid"] for r in result] == ["peak"]


def test_threshold_key_defaults_to_similarity_score():
    """Caller can use raw cosine for both ordering AND tolerance with one knob."""
    frames = [
        {"timestamp_sec": 100.0, "similarity_score": 0.31, "fid": "a"},
        {"timestamp_sec": 101.0, "similarity_score": 0.32, "fid": "b"},
    ]

    # threshold_key defaults to "similarity_score"
    result = temporal_deduplicate_frames(
        frames,
        time_window_sec=5,
        high_score_threshold=0.30,
    )

    assert [r["fid"] for r in result] == ["a", "b"]


def test_long_chain_with_mixed_confidence_correctly_handled():
    """A chain of 5 close frames, only the middle 3 above threshold.
    Expected: peak survives unconditionally, its two high-conf neighbours
    survive via tolerance, the two outer low-conf frames are suppressed
    because at least one conflict (the peak) is in their window."""
    frames = [
        {"timestamp_sec": 99.0,  "similarity_score": 0.28, "confidence_pct": 70.0, "fid": "outer_lo_left"},
        {"timestamp_sec": 100.0, "similarity_score": 0.30, "confidence_pct": 91.0, "fid": "inner_left"},
        {"timestamp_sec": 101.0, "similarity_score": 0.32, "confidence_pct": 98.0, "fid": "peak"},
        {"timestamp_sec": 102.0, "similarity_score": 0.31, "confidence_pct": 95.0, "fid": "inner_right"},
        {"timestamp_sec": 103.0, "similarity_score": 0.27, "confidence_pct": 60.0, "fid": "outer_lo_right"},
    ]

    result = temporal_deduplicate_frames(
        frames,
        time_window_sec=5,
        high_score_threshold=90.0,
        threshold_key="confidence_pct",
    )

    survivors = [r["fid"] for r in result]
    assert "peak" in survivors
    assert "inner_left" in survivors
    assert "inner_right" in survivors
    assert "outer_lo_left" not in survivors
    assert "outer_lo_right" not in survivors
