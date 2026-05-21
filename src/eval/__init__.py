"""
WP6 - Evaluation Harness for AI Video Investigator
===================================================
Comprehensive evaluation framework for the dual-agent CLIP→Claude pipeline.

Supports three evaluation modes:
  1. clip_only: CLIP retriever baseline (no reasoning)
  2. dual_agent: CLIP + Router + Claude cascade (main system)
  3. claude_only_stub: Naive Claude-only baseline (placeholder for WP7)

Core Components:
  - models.py: Strongly-typed result and config dataclasses
  - metrics.py: Pure metric computation functions
  - harness.py: BenchmarkEvaluator orchestration
  - __main__.py: CLI entry point

Metrics Computed:
  Retrieval:
    - Recall@K (1, 5, 10)
    - Precision@K (5, 10)
    - F1@K (5, 10)
    - MRR (Mean Reciprocal Rank)
    - nDCG@K (5, 10)

  Routing & Privacy:
    - Fraction of queries resolved on-premise (no cloud escalation)
    - Fraction of frames never sent to cloud
    - Number of frames escalated per query

  Cost:
    - Total input/output tokens
    - Estimated cost (USD)
    - Cost per query

  Latency:
    - CLIP retriever latency (mean, p95)
    - Claude reasoner latency (mean, p95)
    - Total end-to-end latency (mean, p95)

Usage:
  python -m src.eval --queries evals/queries.example.jsonl --mode dual_agent
  python -m src.eval --config config.json --mode clip_only

Results are saved to evals/results/ as JSON and CSV for later analysis.
"""

from src.eval.models import (
    EvaluationConfig,
    QueryResult,
    PrivacyMetrics,
    CostMetrics,
    LatencyMetrics,
    AggregateMetrics,
)
from src.eval.harness import BenchmarkEvaluator
from src.eval import metrics

__all__ = [
    "EvaluationConfig",
    "QueryResult",
    "PrivacyMetrics",
    "CostMetrics",
    "LatencyMetrics",
    "AggregateMetrics",
    "BenchmarkEvaluator",
    "metrics",
]
