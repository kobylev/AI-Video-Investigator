"""WP6 - Evaluation Harness for the AI Video Investigator.

Layout
------
    models.py         Dataclasses (EvaluationSample, QueryResult, …)
    metrics.py        Pure functions (recall_at_k, ndcg_at_k, …)
    io.py             JSONL load + JSON/CSV write
    modes.py          Per-mode execution adapters
    harness.py        BenchmarkRunner orchestration
    run_benchmark.py  CLI entry point

Run a benchmark:

    python -m src.eval.run_benchmark --queries evals/queries.example.jsonl --mode dual_agent

See `docs/work_packages/wp6_eval_harness.md` for the full architecture and
metric definitions.
"""

from src.eval.harness import BenchmarkRunner
from src.eval.models import (
    AggregateMetrics,
    CostMetrics,
    EvaluationConfig,
    EvaluationSample,
    LatencyMetrics,
    PrivacyMetrics,
    QueryResult,
    RetrievalMetrics,
    RoutingMetrics,
    MODE_CLAUDE_ONLY_STUB,
    MODE_CLIP_ONLY,
    MODE_DUAL_AGENT,
    VALID_MODES,
)

__all__ = [
    "BenchmarkRunner",
    "EvaluationConfig",
    "EvaluationSample",
    "QueryResult",
    "AggregateMetrics",
    "PrivacyMetrics",
    "CostMetrics",
    "LatencyMetrics",
    "RetrievalMetrics",
    "RoutingMetrics",
    "MODE_CLIP_ONLY",
    "MODE_DUAL_AGENT",
    "MODE_CLAUDE_ONLY_STUB",
    "VALID_MODES",
]
