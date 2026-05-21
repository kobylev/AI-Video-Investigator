"""WP6 - Evaluation Harness: data models.

All result and configuration types live here as dataclasses. The harness is
deliberately ground-truth-driven: every benchmark row carries an explicit set
of relevant frame indices, and every per-query result records the actual
retrieved ranking. That makes Recall@K / Precision@K / nDCG@K meaningful
instead of mocked.

Dataclasses are JSON-serializable via `to_dict()`. We avoid Pydantic so the
harness has no runtime dependency that production retriever code does not
already pull in.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Routing decision vocabulary
# ---------------------------------------------------------------------------

ROUTER_IMMEDIATE = "IMMEDIATE_MATCH"   # CLIP confident; no Claude call
ROUTER_ESCALATED = "ESCALATED"         # Ambiguous band; Claude re-ranks
ROUTER_DROPPED = "DROPPED"             # Below tau_low
ROUTER_CLIP_ONLY = "CLIP_ONLY"         # Mode = clip_only (no routing)
ROUTER_CLAUDE_ONLY = "CLAUDE_ONLY"     # Mode = claude_only_stub (no routing)

MODE_CLIP_ONLY = "clip_only"
MODE_DUAL_AGENT = "dual_agent"
MODE_CLAUDE_ONLY_STUB = "claude_only_stub"
VALID_MODES = (MODE_CLIP_ONLY, MODE_DUAL_AGENT, MODE_CLAUDE_ONLY_STUB)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluationSample:
    """One benchmark query plus its ground-truth labels.

    `relevant_indices` is the set of frame indices (0-based, into the corpus)
    that a human annotator marked as relevant for this query. The harness
    treats anything not in this set as non-relevant. `total_relevant` may be
    larger than `len(relevant_indices)` if the annotator knows there are more
    relevant frames than they enumerated — recall metrics are normalised by
    this value, not by the enumerated count.
    """

    query_id: str
    query_text: str
    event_type: str
    relevant_indices: Tuple[int, ...] = ()
    total_relevant: Optional[int] = None
    expected_relevance_notes: str = ""

    @property
    def effective_total_relevant(self) -> int:
        if self.total_relevant is not None and self.total_relevant > 0:
            return self.total_relevant
        return len(self.relevant_indices)


@dataclass
class EvaluationConfig:
    """Reproducible run configuration. Persisted alongside every result set."""

    mode: str = MODE_DUAL_AGENT
    queries_file: str = "evals/queries.example.jsonl"
    output_dir: str = "evals/results"
    seed: int = 42

    # Router thresholds (mirror src.router.core.BudgetAwareRouter defaults).
    tau_high: float = 0.32
    tau_low: float = 0.24
    max_escalations: int = 5

    # Corpus shape — used for privacy denominators.
    corpus_size: int = 36000          # 10h @ 1 fps
    retrieval_top_k: int = 20

    # Claude Haiku 4.5 list pricing, USD per 1M tokens.
    price_per_1m_input_tokens: float = 1.00
    price_per_1m_output_tokens: float = 5.00
    cost_per_escalated_image: float = 0.0003

    # Stub knobs (only relevant for claude_only_stub mode).
    stub_input_tokens_per_query: int = 75_000
    stub_output_tokens_per_query: int = 2_000
    stub_latency_ms: float = 1_500.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-query result
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """End-to-end record for a single sample.

    `ranked_indices` and `ranked_scores` describe the final ordering the
    system returned (after Claude re-ranking in dual_agent mode, after CLIP
    in clip_only mode). All per-query retrieval metrics are derived from
    this list at aggregation time — see `metrics.py`.
    """

    query_id: str
    query_text: str
    event_type: str

    # Final ranking surfaced to the user.
    ranked_indices: List[int] = field(default_factory=list)
    ranked_scores: List[float] = field(default_factory=list)

    # Retriever (CLIP).
    retriever_latency_ms: float = 0.0
    clip_top_1_score: float = 0.0
    clip_top_5_scores: List[float] = field(default_factory=list)

    # Router.
    router_decision: str = ROUTER_CLIP_ONLY
    router_latency_ms: float = 0.0
    frames_escalated: int = 0
    escalated_indices: List[int] = field(default_factory=list)

    # Reasoner (Claude).
    reasoner_called: bool = False
    reasoner_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0

    # Roll-up.
    total_latency_ms: float = 0.0

    # Ground truth pinned at evaluation time — kept so JSON output is
    # self-describing without needing to re-open the queries file.
    relevant_indices: List[int] = field(default_factory=list)
    total_relevant: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------

@dataclass
class PrivacyMetrics:
    """Privacy preservation — first-class output of the harness."""

    total_queries: int
    queries_resolved_on_prem: int
    total_frames_in_corpus: int
    unique_frames_escalated: int
    total_frame_escalations: int  # may exceed unique count if frames are escalated by multiple queries

    @property
    def fraction_queries_on_prem(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.queries_resolved_on_prem / self.total_queries

    @property
    def fraction_frames_on_prem(self) -> float:
        if self.total_frames_in_corpus == 0:
            return 0.0
        capped = min(self.unique_frames_escalated, self.total_frames_in_corpus)
        return 1.0 - (capped / self.total_frames_in_corpus)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "queries_resolved_on_prem": self.queries_resolved_on_prem,
            "fraction_queries_on_prem": self.fraction_queries_on_prem,
            "total_frames_in_corpus": self.total_frames_in_corpus,
            "unique_frames_escalated": self.unique_frames_escalated,
            "total_frame_escalations": self.total_frame_escalations,
            "fraction_frames_on_prem": self.fraction_frames_on_prem,
        }


@dataclass
class CostMetrics:
    """Token economics. `total_estimated_cost_usd` is the sum of per-query
    estimates recorded during execution; `computed_cost_usd` is recomputed
    from the aggregated token counts as a sanity check."""

    total_input_tokens: int
    total_output_tokens: int
    total_estimated_cost_usd: float
    price_per_1m_input_tokens: float = 1.00
    price_per_1m_output_tokens: float = 5.00

    @property
    def computed_cost_usd(self) -> float:
        return (
            (self.total_input_tokens / 1_000_000) * self.price_per_1m_input_tokens
            + (self.total_output_tokens / 1_000_000) * self.price_per_1m_output_tokens
        )

    def cost_per_query(self, num_queries: int) -> float:
        if num_queries == 0:
            return 0.0
        return self.computed_cost_usd / num_queries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_estimated_cost_usd": self.total_estimated_cost_usd,
            "computed_cost_usd": self.computed_cost_usd,
            "pricing": {
                "input_per_1m_usd": self.price_per_1m_input_tokens,
                "output_per_1m_usd": self.price_per_1m_output_tokens,
            },
        }


@dataclass
class LatencyMetrics:
    """Mean and p95 of each stage's latency, in milliseconds."""

    retriever_mean_ms: float = 0.0
    retriever_p95_ms: float = 0.0
    router_mean_ms: float = 0.0
    router_p95_ms: float = 0.0
    reasoner_mean_ms: float = 0.0
    reasoner_p95_ms: float = 0.0
    total_mean_ms: float = 0.0
    total_p95_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retriever": {"mean_ms": self.retriever_mean_ms, "p95_ms": self.retriever_p95_ms},
            "router": {"mean_ms": self.router_mean_ms, "p95_ms": self.router_p95_ms},
            "reasoner": {"mean_ms": self.reasoner_mean_ms, "p95_ms": self.reasoner_p95_ms},
            "total": {"mean_ms": self.total_mean_ms, "p95_ms": self.total_p95_ms},
        }


@dataclass
class RoutingMetrics:
    """Decision-count breakdown across all queries."""

    immediate_match: int = 0
    escalated: int = 0
    dropped: int = 0
    clip_only: int = 0
    claude_only: int = 0
    avg_frames_escalated_per_query: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalMetrics:
    """Mean retrieval-quality metrics across all queries."""

    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    f1_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregateMetrics:
    """Top-level container persisted as `aggregate_<mode>_<ts>.json`."""

    mode: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_queries: int = 0

    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    routing: RoutingMetrics = field(default_factory=RoutingMetrics)
    privacy: PrivacyMetrics = field(default_factory=lambda: PrivacyMetrics(0, 0, 0, 0, 0))
    cost: CostMetrics = field(default_factory=lambda: CostMetrics(0, 0, 0.0))
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)

    config: Optional[EvaluationConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "mode": self.mode,
                "timestamp": self.timestamp,
                "total_queries": self.total_queries,
            },
            "retrieval_metrics": self.retrieval.to_dict(),
            "routing_metrics": self.routing.to_dict(),
            "privacy_metrics": self.privacy.to_dict(),
            "cost_metrics": self.cost.to_dict(),
            "latency_metrics": self.latency.to_dict(),
            "config": self.config.to_dict() if self.config is not None else None,
        }
