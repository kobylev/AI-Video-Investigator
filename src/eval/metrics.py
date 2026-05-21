"""WP6 - Evaluation Harness: pure metric functions.

Every function here is deterministic and side-effect free, which is what
makes them unit-testable in `tests/eval/test_metrics.py`. They never reach
into the harness or do I/O — the caller passes in plain Python lists.

Retrieval metric conventions
----------------------------
* `relevant_set`  — the set of *corpus indices* (ints) that count as relevant
  ground truth for a query.
* `ranked_indices` — the *corpus indices* the system returned, in rank order
  (highest-confidence first, position 0 = top hit).
* `total_relevant` — the denominator for Recall@K. May be larger than
  `len(relevant_set)` when the annotator enumerated only a subset of truly
  relevant frames.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Set


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hits_in_top_k(relevant_set: Set[int], ranked_indices: Sequence[int], k: int) -> int:
    """Count items from `relevant_set` appearing in the top `k` of `ranked_indices`."""
    if k <= 0 or not ranked_indices:
        return 0
    top = ranked_indices[:k]
    return sum(1 for idx in top if idx in relevant_set)


def percentile(values: Sequence[float], pct: float) -> float:
    """Inclusive-rank percentile, matching what the WP5 dashboards report.

    Returns 0.0 for empty input. `pct` is in [0, 100]. Uses linear-rank
    indexing — equivalent to NumPy's `interpolation='lower'`, so the function
    has no NumPy dependency and is fully deterministic.
    """
    if not values:
        return 0.0
    if pct <= 0:
        return float(min(values))
    if pct >= 100:
        return float(max(values))
    ordered = sorted(values)
    # rank = ceil(pct/100 * n) − 1, clamped into bounds.
    rank = max(0, math.ceil((pct / 100.0) * len(ordered)) - 1)
    return float(ordered[rank])


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


# ---------------------------------------------------------------------------
# Retrieval-quality metrics (per-query)
# ---------------------------------------------------------------------------

def recall_at_k(
    relevant_set: Iterable[int],
    ranked_indices: Sequence[int],
    k: int,
    total_relevant: int = 0,
) -> float:
    """Fraction of ground-truth-relevant items that appear in the top K.

    Denominator is `total_relevant` if provided and positive, otherwise
    `len(relevant_set)`. Returns 0.0 when the denominator is zero — a
    query with no ground truth contributes nothing to Recall@K and is
    expected to be filtered out by the caller in aggregation.
    """
    rel = set(relevant_set)
    denom = total_relevant if total_relevant > 0 else len(rel)
    if denom <= 0:
        return 0.0
    return _hits_in_top_k(rel, ranked_indices, k) / denom


def precision_at_k(
    relevant_set: Iterable[int],
    ranked_indices: Sequence[int],
    k: int,
) -> float:
    """Fraction of top-K results that are relevant. Denominator is K."""
    if k <= 0:
        return 0.0
    rel = set(relevant_set)
    return _hits_in_top_k(rel, ranked_indices, k) / k


def f1_at_k(
    relevant_set: Iterable[int],
    ranked_indices: Sequence[int],
    k: int,
    total_relevant: int = 0,
) -> float:
    """Harmonic mean of Precision@K and Recall@K."""
    p = precision_at_k(relevant_set, ranked_indices, k)
    r = recall_at_k(relevant_set, ranked_indices, k, total_relevant)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def reciprocal_rank(
    relevant_set: Iterable[int],
    ranked_indices: Sequence[int],
) -> float:
    """1 / (1-indexed rank of the first relevant hit). 0.0 if no hit."""
    rel = set(relevant_set)
    for i, idx in enumerate(ranked_indices):
        if idx in rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    relevant_set: Iterable[int],
    ranked_indices: Sequence[int],
    k: int,
) -> float:
    """Binary-relevance nDCG@K.

    DCG@K = Σ rel_i / log2(i + 2)  for i in [0, k)
    IDCG@K = DCG of the perfect ranking (all relevant items first).
    """
    if k <= 0:
        return 0.0
    rel = set(relevant_set)
    dcg = 0.0
    for i, idx in enumerate(ranked_indices[:k]):
        if idx in rel:
            dcg += 1.0 / math.log2(i + 2)
    # Ideal: all relevant items packed at the top, up to k.
    ideal_hits = min(len(rel), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_retrieval(
    per_query: List[Dict[str, float]],
) -> Dict[str, float]:
    """Mean per-query retrieval metrics into the aggregate row.

    Expects each dict to carry keys: recall_at_1, recall_at_5, recall_at_10,
    precision_at_5, precision_at_10, f1_at_5, mrr, ndcg_at_5, ndcg_at_10.
    Missing keys default to 0.0 so the function is safe to call on partial
    runs (e.g. claude_only_stub which has no ranking data).
    """
    keys = (
        "recall_at_1", "recall_at_5", "recall_at_10",
        "precision_at_5", "precision_at_10",
        "f1_at_5", "mrr", "ndcg_at_5", "ndcg_at_10",
    )
    if not per_query:
        return {k: 0.0 for k in keys}
    return {k: mean([q.get(k, 0.0) for q in per_query]) for k in keys}
