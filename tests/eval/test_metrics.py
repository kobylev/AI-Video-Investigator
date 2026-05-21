"""Unit tests for the pure metric functions in `src.eval.metrics`.

These tests are intentionally library-only: no real CLIP, no real Claude,
no file I/O. They lock in the precise semantics of each metric so that
future refactors can't silently change a denominator or drop a tie-break.

Run with:
    pytest tests/eval/test_metrics.py -v
"""

from __future__ import annotations

import math

import pytest

from src.eval import metrics as M


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_all_relevant_in_top_k(self):
        # Relevant frames {10, 20}, both appear within top-5.
        assert M.recall_at_k({10, 20}, [10, 5, 20, 7, 9], k=5) == 1.0

    def test_partial_recall(self):
        # 1 of 2 relevant in top-5.
        assert M.recall_at_k({10, 20}, [10, 5, 7, 9, 11], k=5) == 0.5

    def test_no_relevant_in_top_k(self):
        # Both relevant items below cutoff.
        assert M.recall_at_k({10, 20}, [1, 2, 3, 4, 5], k=5) == 0.0

    def test_uses_total_relevant_when_larger(self):
        # 1 enumerated relevant, but annotator says there are 4 total.
        # Hit gives R@5 = 1/4, not 1/1.
        assert M.recall_at_k({10}, [10, 2, 3, 4, 5], k=5, total_relevant=4) == 0.25

    def test_empty_relevant_set_is_zero(self):
        assert M.recall_at_k(set(), [1, 2, 3], k=5) == 0.0

    def test_k_zero_is_zero(self):
        assert M.recall_at_k({10}, [10], k=0) == 0.0


# ---------------------------------------------------------------------------
# Precision@K
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_all_relevant(self):
        # Top-3 are all relevant.
        assert M.precision_at_k({10, 20, 30}, [10, 20, 30, 40, 50], k=3) == 1.0

    def test_half_relevant(self):
        # Top-4 has 2 relevant out of 4 → 0.5.
        assert M.precision_at_k({10, 20}, [10, 5, 20, 7], k=4) == 0.5

    def test_none_relevant(self):
        assert M.precision_at_k({100}, [1, 2, 3, 4, 5], k=5) == 0.0

    def test_k_zero_is_zero(self):
        assert M.precision_at_k({10}, [10], k=0) == 0.0


# ---------------------------------------------------------------------------
# F1@K
# ---------------------------------------------------------------------------

class TestF1AtK:
    def test_perfect_score(self):
        # Both relevant and exactly 2 results → P=1, R=1, F1=1.
        # We need k=2 so precision denominator = 2.
        assert M.f1_at_k({10, 20}, [10, 20], k=2) == pytest.approx(1.0)

    def test_zero_when_no_overlap(self):
        assert M.f1_at_k({1}, [99], k=5) == 0.0

    def test_harmonic_mean_property(self):
        # 2 relevant, top-5 has 1 hit. P=1/5=0.2, R=1/2=0.5, F1=2*0.2*0.5/0.7.
        result = M.f1_at_k({10, 20}, [10, 5, 7, 8, 9], k=5)
        assert result == pytest.approx(2 * 0.2 * 0.5 / 0.7)


# ---------------------------------------------------------------------------
# Reciprocal Rank / MRR
# ---------------------------------------------------------------------------

class TestReciprocalRank:
    def test_first_position(self):
        assert M.reciprocal_rank({10}, [10, 20, 30]) == 1.0

    def test_third_position(self):
        # First relevant at index 2 → rank 3 → 1/3.
        assert M.reciprocal_rank({10}, [1, 2, 10, 3]) == pytest.approx(1.0 / 3.0)

    def test_no_hit_is_zero(self):
        assert M.reciprocal_rank({999}, [1, 2, 3]) == 0.0


# ---------------------------------------------------------------------------
# nDCG@K
# ---------------------------------------------------------------------------

class TestNdcgAtK:
    def test_perfect_ranking_is_one(self):
        # 3 relevant items all packed at the top — perfect ranking → nDCG=1.
        assert M.ndcg_at_k({10, 20, 30}, [10, 20, 30, 40, 50], k=5) == pytest.approx(1.0)

    def test_all_misses_is_zero(self):
        assert M.ndcg_at_k({100}, [1, 2, 3, 4, 5], k=5) == 0.0

    def test_position_matters(self):
        # 1 relevant item. Top of list gives higher DCG than bottom.
        first = M.ndcg_at_k({10}, [10, 1, 2, 3, 4], k=5)
        last = M.ndcg_at_k({10}, [1, 2, 3, 4, 10], k=5)
        assert first > last > 0.0
        # Sanity-check absolute values.
        assert first == pytest.approx(1.0)
        # DCG when sole hit is at rank 5 → 1/log2(6); IDCG → 1/log2(2)=1.
        assert last == pytest.approx(1.0 / math.log2(6))

    def test_empty_relevant_set_is_zero(self):
        assert M.ndcg_at_k(set(), [1, 2, 3], k=5) == 0.0


# ---------------------------------------------------------------------------
# percentile / mean
# ---------------------------------------------------------------------------

class TestSummaryStats:
    def test_percentile_p95(self):
        # 100 values [1..100]; p95 picks the 95th-rank element = 95.
        values = list(range(1, 101))
        assert M.percentile(values, 95) == 95

    def test_percentile_extremes(self):
        values = [10.0, 20.0, 30.0]
        assert M.percentile(values, 0) == 10.0
        assert M.percentile(values, 100) == 30.0

    def test_percentile_empty(self):
        assert M.percentile([], 50) == 0.0

    def test_mean_empty(self):
        assert M.mean([]) == 0.0

    def test_mean_basic(self):
        assert M.mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# aggregate_retrieval
# ---------------------------------------------------------------------------

class TestAggregateRetrieval:
    def test_empty_input_zero_metrics(self):
        out = M.aggregate_retrieval([])
        assert all(v == 0.0 for v in out.values())
        assert set(out.keys()) == {
            "recall_at_1", "recall_at_5", "recall_at_10",
            "precision_at_5", "precision_at_10",
            "f1_at_5", "mrr", "ndcg_at_5", "ndcg_at_10",
        }

    def test_means_across_queries(self):
        per_query = [
            {"recall_at_5": 1.0, "mrr": 1.0},
            {"recall_at_5": 0.0, "mrr": 0.5},
        ]
        out = M.aggregate_retrieval(per_query)
        assert out["recall_at_5"] == pytest.approx(0.5)
        assert out["mrr"] == pytest.approx(0.75)
        # Missing keys default to 0.
        assert out["ndcg_at_10"] == 0.0
