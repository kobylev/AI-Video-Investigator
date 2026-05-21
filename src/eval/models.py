"""
WP6 - Evaluation Harness: Data Models
--------------------------------------
Strongly-typed result and configuration models for reproducible benchmark execution.
All models support JSON serialization for result persistence.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


@dataclass
class EvaluationConfig:
    """Reproducible evaluation configuration."""
    seed: int = 42
    tau_high: float = 0.32
    tau_low: float = 0.24
    max_escalations: int = 5
    cost_per_image: float = 0.0003  # Claude Haiku 4.5 image token cost
    queries_file: str = "evals/queries.example.jsonl"
    output_dir: str = "evals/results"
    mode: str = "dual_agent"  # "clip_only", "dual_agent", "claude_only_stub"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class QueryResult:
    """Result for a single query execution across the pipeline."""
    query_id: str
    query_text: str
    event_type: str
    expected_relevance_notes: str

    # Retrieval stage (CLIP)
    clip_latency_ms: float
    clip_top_k: int
    clip_top_1_score: float
    clip_top_5_scores: List[float] = field(default_factory=list)

    # Router decision
    router_decision: str = ""  # "IMMEDIATE_MATCH", "ESCALATED", "DROPPED"
    frames_escalated: int = 0
    estimated_cost_usd: float = 0.0

    # Reasoner stage (Claude) - if escalated
    reasoner_latency_ms: float = 0.0
    reasoner_called: bool = False
    input_tokens: int = 0
    output_tokens: int = 0

    # Overall
    total_latency_ms: float = 0.0

    # Ground truth (for evaluation)
    ground_truth_relevant: Optional[bool] = None  # Will be populated during evaluation

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class PrivacyMetrics:
    """Aggregated privacy preservation statistics."""
    total_queries: int
    queries_resolved_on_prem: int  # No cloud escalation
    total_frames_in_corpus: int
    frames_escalated_to_cloud: int

    @property
    def fraction_queries_on_prem(self) -> float:
        """Percentage of queries resolved entirely on-premise."""
        if self.total_queries == 0:
            return 0.0
        return self.queries_resolved_on_prem / self.total_queries

    @property
    def fraction_frames_on_prem(self) -> float:
        """Percentage of frames never sent to cloud."""
        if self.total_frames_in_corpus == 0:
            return 0.0
        return 1.0 - (self.frames_escalated_to_cloud / self.total_frames_in_corpus)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "queries_resolved_on_prem": self.queries_resolved_on_prem,
            "fraction_queries_on_prem": self.fraction_queries_on_prem,
            "total_frames_in_corpus": self.total_frames_in_corpus,
            "frames_escalated_to_cloud": self.frames_escalated_to_cloud,
            "fraction_frames_on_prem": self.fraction_frames_on_prem,
        }


@dataclass
class CostMetrics:
    """Aggregated cost statistics."""
    total_input_tokens: int
    total_output_tokens: int
    total_estimated_cost_usd: float
    price_per_1m_input_tokens: float = 1.00  # Claude Haiku 4.5
    price_per_1m_output_tokens: float = 5.00

    @property
    def computed_cost_usd(self) -> float:
        """Compute cost from token counts (validation)."""
        input_cost = (self.total_input_tokens / 1_000_000) * self.price_per_1m_input_tokens
        output_cost = (self.total_output_tokens / 1_000_000) * self.price_per_1m_output_tokens
        return input_cost + output_cost

    @property
    def cost_per_query(self) -> float:
        """Average cost per query (will be computed in aggregation)."""
        # Set during aggregation
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_estimated_cost_usd": self.total_estimated_cost_usd,
            "computed_cost_usd": self.computed_cost_usd,
            "pricing": {
                "input_tokens_per_1m": self.price_per_1m_input_tokens,
                "output_tokens_per_1m": self.price_per_1m_output_tokens,
            }
        }


@dataclass
class LatencyMetrics:
    """Aggregated latency statistics (in milliseconds)."""
    retriever_latencies: List[float] = field(default_factory=list)
    router_latencies: List[float] = field(default_factory=list)
    reasoner_latencies: List[float] = field(default_factory=list)
    total_latencies: List[float] = field(default_factory=list)

    @property
    def mean_retriever_ms(self) -> float:
        if not self.retriever_latencies:
            return 0.0
        return sum(self.retriever_latencies) / len(self.retriever_latencies)

    @property
    def p95_retriever_ms(self) -> float:
        if not self.retriever_latencies:
            return 0.0
        sorted_vals = sorted(self.retriever_latencies)
        idx = int(0.95 * len(sorted_vals))
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    @property
    def mean_reasoner_ms(self) -> float:
        if not self.reasoner_latencies:
            return 0.0
        return sum(self.reasoner_latencies) / len(self.reasoner_latencies)

    @property
    def p95_reasoner_ms(self) -> float:
        if not self.reasoner_latencies:
            return 0.0
        sorted_vals = sorted(self.reasoner_latencies)
        idx = int(0.95 * len(sorted_vals))
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    @property
    def mean_total_ms(self) -> float:
        if not self.total_latencies:
            return 0.0
        return sum(self.total_latencies) / len(self.total_latencies)

    @property
    def p95_total_ms(self) -> float:
        if not self.total_latencies:
            return 0.0
        sorted_vals = sorted(self.total_latencies)
        idx = int(0.95 * len(sorted_vals))
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retriever": {
                "mean_ms": self.mean_retriever_ms,
                "p95_ms": self.p95_retriever_ms,
            },
            "reasoner": {
                "mean_ms": self.mean_reasoner_ms,
                "p95_ms": self.p95_reasoner_ms,
            },
            "total": {
                "mean_ms": self.mean_total_ms,
                "p95_ms": self.p95_total_ms,
            }
        }


@dataclass
class AggregateMetrics:
    """Complete aggregated benchmark results."""
    config: EvaluationConfig
    mode: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_queries: int = 0

    # Retrieval metrics
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    f1_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0

    # Routing metrics
    queries_on_prem_only: int = 0
    queries_escalated: int = 0
    avg_frames_escalated_per_query: float = 0.0

    # Privacy metrics
    privacy: PrivacyMetrics = field(default_factory=lambda: PrivacyMetrics(0, 0, 0, 0))

    # Cost metrics
    cost: CostMetrics = field(default_factory=lambda: CostMetrics(0, 0, 0.0))

    # Latency metrics
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "mode": self.mode,
                "timestamp": self.timestamp,
                "total_queries": self.total_queries,
            },
            "retrieval_metrics": {
                "recall_at_1": self.recall_at_1,
                "recall_at_5": self.recall_at_5,
                "recall_at_10": self.recall_at_10,
                "precision_at_5": self.precision_at_5,
                "precision_at_10": self.precision_at_10,
                "f1_at_5": self.f1_at_5,
                "mrr": self.mrr,
                "ndcg_at_5": self.ndcg_at_5,
                "ndcg_at_10": self.ndcg_at_10,
            },
            "routing_metrics": {
                "queries_on_prem_only": self.queries_on_prem_only,
                "queries_escalated": self.queries_escalated,
                "avg_frames_escalated_per_query": self.avg_frames_escalated_per_query,
            },
            "privacy_metrics": self.privacy.to_dict(),
            "cost_metrics": self.cost.to_dict(),
            "latency_metrics": self.latency.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
