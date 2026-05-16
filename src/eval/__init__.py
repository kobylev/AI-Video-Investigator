"""Evaluation Metrics Module

This module implements the evaluation harness for measuring system performance
across retrieval, reasoning, and end-to-end metrics.

Metrics Implemented:
    Retrieval Metrics (Stage 1 — CLIP):
        - Recall@K (R@1, R@5, R@10)
        - Mean Reciprocal Rank (MRR)
        - Normalized Discounted Cumulative Gain (nDCG)

    Classification Metrics (Stage 2 — Gemini):
        - Precision
        - Recall
        - F1 Score
        - Accuracy (binary relevance)

    End-to-End Metrics (Pipeline):
        - Recall@5 with binary relevance (primary metric)
        - Top-5 F1 (harmonic mean of P/R at K=5)

    System Metrics (Architecture):
        - Latency (p50, p95, p99)
        - Tokens per query
        - Cost per query (USD)
        - Router decision distribution (skip / expand / escalate)

Input Format:
    - Ground truth: JSONL with query_id, relevant_frame_ids
    - Predictions: JSONL with query_id, ranked_frame_ids, scores
    - System logs: JSONL with query_id, latency, tokens, cost, router_decision

Output:
    - Metrics summary (JSON or Markdown table)
    - Per-query breakdown for error analysis
    - Aggregated statistics (mean, std, p50, p95)

Usage Example:
    ```python
    from src.eval import EvaluationHarness

    harness = EvaluationHarness(
        ground_truth="evals/benchmark_ground_truth.jsonl",
        predictions="evals/results/dual_agent_predictions.jsonl"
    )

    metrics = harness.compute_all_metrics()
    print(metrics.to_markdown())
    ```
"""
