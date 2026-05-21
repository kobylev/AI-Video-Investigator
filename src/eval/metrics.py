"""
WP6 - Evaluation Harness: Metric Computation
--------------------------------------------
Pure functions for computing retrieval quality metrics, privacy metrics,
and cost metrics. All functions are deterministic and testable.
"""

from typing import List, Dict, Tuple
import math


# ============================================================================
# Retrieval Quality Metrics
# ============================================================================

def recall_at_k(relevant_ranks: List[int], k: int) -> float:
    """
    Recall@K: Fraction of relevant items that appear in top-K results.

    :param relevant_ranks: List of ranks (0-indexed) where relevant items appear.
                           Example: [0, 2, 5] means relevant items at positions 1, 3, 6.
    :param k: Cutoff rank.
    :return: Recall@K in [0, 1].
    """
    if not relevant_ranks:
        return 0.0
    relevant_in_top_k = sum(1 for r in relevant_ranks if r < k)
    return relevant_in_top_k / len(relevant_ranks)


def precision_at_k(relevant_ranks: List[int], k: int) -> float:
    """
    Precision@K: Fraction of top-K results that are relevant.

    :param relevant_ranks: List of ranks (0-indexed) where relevant items appear.
    :param k: Cutoff rank.
    :return: Precision@K in [0, 1].
    """
    if not relevant_ranks:
        return 0.0
    relevant_in_top_k = sum(1 for r in relevant_ranks if r < k)
    return relevant_in_top_k / k


def f1_at_k(relevant_ranks: List[int], k: int) -> float:
    """
    F1@K: Harmonic mean of Precision@K and Recall@K.

    :param relevant_ranks: List of ranks (0-indexed) where relevant items appear.
    :param k: Cutoff rank.
    :return: F1@K in [0, 1].
    """
    p = precision_at_k(relevant_ranks, k)
    r = recall_at_k(relevant_ranks, k)

    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


def mean_reciprocal_rank(relevant_ranks: List[int]) -> float:
    """
    MRR: Mean of 1 / rank for the first relevant item in each query.

    :param relevant_ranks: List of ranks (0-indexed) where relevant items appear.
    :return: MRR in [0, 1].
    """
    if not relevant_ranks:
        return 0.0
    # Rank is 1-indexed for user interpretation
    first_relevant_rank_1indexed = relevant_ranks[0] + 1
    return 1.0 / first_relevant_rank_1indexed


def ndcg_at_k(scores: List[float], relevance_labels: List[int], k: int) -> float:
    """
    nDCG@K: Normalized Discounted Cumulative Gain.

    Measures ranking quality considering both position and relevance degree.
    Higher positions are discounted less; irrelevant items contribute 0.

    :param scores: List of similarity scores (one per retrieved item, in rank order).
    :param relevance_labels: List of relevance labels (0 or 1) corresponding to scores.
    :param k: Cutoff rank.
    :return: nDCG@K in [0, 1].
    """
    assert len(scores) == len(relevance_labels), "Scores and labels must have same length."

    # Compute DCG@K
    dcg = 0.0
    for i in range(min(k, len(scores))):
        rel = relevance_labels[i]
        # Standard DCG formula: rel / log2(i+2) where i is 0-indexed
        dcg += rel / math.log2(i + 2)

    # Compute ideal DCG (IDCG) assuming perfect ranking
    sorted_labels = sorted(relevance_labels, reverse=True)
    idcg = 0.0
    for i in range(min(k, len(sorted_labels))):
        rel = sorted_labels[i]
        idcg += rel / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


# ============================================================================
# Privacy Metrics
# ============================================================================

def fraction_queries_on_prem(
    query_results: List[Dict],
    corpus_size: int
) -> Tuple[float, int, int]:
    """
    Compute fraction of queries resolved entirely on-premise (no cloud escalation).

    :param query_results: List of query result dicts with 'frames_escalated' key.
    :param corpus_size: Total frames in video corpus.
    :return: (fraction, count_on_prem, total_queries)
    """
    total = len(query_results)
    if total == 0:
        return 0.0, 0, 0

    on_prem_count = sum(1 for qr in query_results if qr.get("frames_escalated", 0) == 0)
    return on_prem_count / total, on_prem_count, total


def fraction_frames_on_prem(
    query_results: List[Dict],
    corpus_size: int
) -> Tuple[float, int, int]:
    """
    Compute fraction of video frames never sent to cloud.

    :param query_results: List of query result dicts with 'frames_escalated' key.
    :param corpus_size: Total frames in video corpus (e.g., 36000 for 10h @ 1fps).
    :return: (fraction_on_prem, frames_escalated_total, corpus_size)
    """
    if corpus_size == 0:
        return 0.0, 0, 0

    total_escalated = sum(qr.get("frames_escalated", 0) for qr in query_results)
    fraction = 1.0 - (total_escalated / corpus_size)
    return fraction, total_escalated, corpus_size


# ============================================================================
# Aggregation Helpers
# ============================================================================

def aggregate_retrieval_metrics(
    query_results: List[Dict],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Aggregate retrieval metrics across all queries.

    Assumes query_results have 'relevant_ranks' key (list of 0-indexed ranks).

    :param query_results: List of query result dicts.
    :param k_values: Cutoff values to compute.
    :return: Dictionary of aggregated metrics.
    """
    recalls = {f"recall_at_{k}": [] for k in k_values}
    precisions = {f"precision_at_{k}": [] for k in k_values}
    f1s = {f"f1_at_{k}": [] for k in k_values}
    mrrs = []

    for qr in query_results:
        relevant_ranks = qr.get("relevant_ranks", [])

        for k in k_values:
            recalls[f"recall_at_{k}"].append(recall_at_k(relevant_ranks, k))
            precisions[f"precision_at_{k}"].append(precision_at_k(relevant_ranks, k))
            f1s[f"f1_at_{k}"].append(f1_at_k(relevant_ranks, k))

        mrrs.append(mean_reciprocal_rank(relevant_ranks))

    # Compute means
    result = {}
    for k, vals in recalls.items():
        result[k] = sum(vals) / len(vals) if vals else 0.0
    for k, vals in precisions.items():
        result[k] = sum(vals) / len(vals) if vals else 0.0
    for k, vals in f1s.items():
        result[k] = sum(vals) / len(vals) if vals else 0.0

    result["mrr"] = sum(mrrs) / len(mrrs) if mrrs else 0.0

    return result


def aggregate_latency_metrics(
    query_results: List[Dict]
) -> Dict[str, float]:
    """
    Aggregate latency metrics (mean and percentiles).

    :param query_results: List of query result dicts with latency fields.
    :return: Dictionary of aggregated latency stats.
    """
    clip_latencies = [qr.get("clip_latency_ms", 0.0) for qr in query_results]
    reasoner_latencies = [qr.get("reasoner_latency_ms", 0.0) for qr in query_results]
    total_latencies = [qr.get("total_latency_ms", 0.0) for qr in query_results]

    def percentile(values, p):
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int((p / 100.0) * len(sorted_vals))
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    return {
        "clip_mean_ms": sum(clip_latencies) / len(clip_latencies) if clip_latencies else 0.0,
        "clip_p95_ms": percentile(clip_latencies, 95),
        "reasoner_mean_ms": sum(reasoner_latencies) / len(reasoner_latencies) if reasoner_latencies else 0.0,
        "reasoner_p95_ms": percentile(reasoner_latencies, 95),
        "total_mean_ms": sum(total_latencies) / len(total_latencies) if total_latencies else 0.0,
        "total_p95_ms": percentile(total_latencies, 95),
    }


def aggregate_cost_metrics(
    query_results: List[Dict],
    price_per_1m_input: float = 1.00,
    price_per_1m_output: float = 5.00,
) -> Dict[str, float]:
    """
    Aggregate token usage and cost.

    :param query_results: List of query result dicts with token fields.
    :param price_per_1m_input: Price per million input tokens (Claude Haiku 4.5).
    :param price_per_1m_output: Price per million output tokens.
    :return: Dictionary of cost aggregates.
    """
    total_input = sum(qr.get("input_tokens", 0) for qr in query_results)
    total_output = sum(qr.get("output_tokens", 0) for qr in query_results)

    input_cost = (total_input / 1_000_000) * price_per_1m_input
    output_cost = (total_output / 1_000_000) * price_per_1m_output
    total_cost = input_cost + output_cost

    num_queries = len(query_results)
    cost_per_query = total_cost / num_queries if num_queries > 0 else 0.0

    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cost_usd": total_cost,
        "cost_per_query_usd": cost_per_query,
    }
