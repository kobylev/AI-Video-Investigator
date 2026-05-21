"""
WP6 - Evaluation Harness: Benchmark Executor
---------------------------------------------
Orchestrates benchmark execution across different evaluation modes:
- clip_only: CLIP retriever baseline
- dual_agent: CLIP + Router + Claude cascade
- claude_only_stub: Placeholder for WP7 (naive baseline)

Handles result persistence, privacy tracking, and cost aggregation.
"""

import json
import logging
import time
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from src.eval.models import (
    EvaluationConfig,
    QueryResult,
    PrivacyMetrics,
    CostMetrics,
    LatencyMetrics,
    AggregateMetrics,
)
from src.eval import metrics as metric_fns

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """
    Orchestrates benchmark execution and result aggregation.

    Supports three evaluation modes:
    1. clip_only: Retriever-only baseline (no reasoning)
    2. dual_agent: CLIP → Router → Claude cascade (main system)
    3. claude_only_stub: Naive baseline (placeholder for WP7)
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        random.seed(config.seed)

        # Ensure output directory exists
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.query_results: List[QueryResult] = []

    def load_queries_from_jsonl(self, path: str) -> List[Dict[str, Any]]:
        """Load benchmark queries from JSONL file."""
        queries = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line))
        logger.info(f"Loaded {len(queries)} queries from {path}")
        return queries

    def evaluate_clip_only(
        self,
        queries: List[Dict[str, Any]],
        retriever: Any,
        index: Any,
        corpus_size: int = 36000,  # 10h @ 1fps
    ) -> List[QueryResult]:
        """
        CLIP-only baseline: Retrieve top-K frames, no reasoning or routing.

        Args:
            queries: List of query dicts with 'query_text' and 'query_id'.
            retriever: CLIP engine (must have get_text_embeddings, compute_similarity).
            index: FAISS index (must support search).
            corpus_size: Total frames in corpus.

        Returns:
            List of QueryResult objects.
        """
        logger.info("Starting CLIP-only evaluation...")
        results = []

        for query_dict in queries:
            query_id = query_dict.get("query_id", "unknown")
            query_text = query_dict.get("query_text", "")
            event_type = query_dict.get("event_type", "unknown")
            expected_notes = query_dict.get("expected_relevance_notes", "")

            # Simulate CLIP retrieval latency
            start_time = time.time()
            time.sleep(0.05)  # Mock latency: 50ms CLIP encoding
            clip_latency_ms = (time.time() - start_time) * 1000

            # Mock: Assume top-1 score and top-5 scores
            clip_top_1_score = random.uniform(0.3, 0.95)
            clip_top_5_scores = [
                clip_top_1_score - (i * 0.05) for i in range(5)
            ]

            result = QueryResult(
                query_id=query_id,
                query_text=query_text,
                event_type=event_type,
                expected_relevance_notes=expected_notes,
                clip_latency_ms=clip_latency_ms,
                clip_top_k=20,
                clip_top_1_score=clip_top_1_score,
                clip_top_5_scores=clip_top_5_scores,
                router_decision="CLIP_ONLY",
                frames_escalated=0,
                estimated_cost_usd=0.0,
                reasoner_called=False,
                total_latency_ms=clip_latency_ms,
            )

            results.append(result)
            logger.debug(f"CLIP-only: {query_id} → top-1 score {clip_top_1_score:.3f}")

        return results

    def evaluate_dual_agent(
        self,
        queries: List[Dict[str, Any]],
        retriever: Any,
        router: Any,
        reasoner: Any,
        index: Any,
        corpus_size: int = 36000,
    ) -> List[QueryResult]:
        """
        Dual-agent evaluation: CLIP → Router → Claude cascade.

        Args:
            queries: List of query dicts.
            retriever: CLIP engine.
            router: BudgetAwareRouter.
            reasoner: Claude reasoner.
            index: FAISS index.
            corpus_size: Total frames.

        Returns:
            List of QueryResult objects.
        """
        logger.info("Starting dual-agent evaluation...")
        results = []

        for query_dict in queries:
            query_id = query_dict.get("query_id", "unknown")
            query_text = query_dict.get("query_text", "")
            event_type = query_dict.get("event_type", "unknown")
            expected_notes = query_dict.get("expected_relevance_notes", "")

            # Stage 1: CLIP retrieval
            clip_start = time.time()
            time.sleep(0.05)  # Mock latency
            clip_latency_ms = (time.time() - clip_start) * 1000

            clip_top_1_score = random.uniform(0.2, 0.95)
            clip_top_5_scores = [
                clip_top_1_score - (i * 0.05) for i in range(5)
            ]

            # Stage 2: Router decision
            router_start = time.time()
            time.sleep(0.01)  # Mock latency
            router_latency_ms = (time.time() - router_start) * 1000

            # Decision logic based on thresholds
            if clip_top_1_score >= router.tau_high:
                router_decision = "IMMEDIATE_MATCH"
                frames_escalated = 0
                reasoner_latency_ms = 0.0
                input_tokens = 0
                output_tokens = 0
            elif clip_top_1_score >= router.tau_low:
                router_decision = "ESCALATED"
                frames_escalated = min(5, random.randint(1, 10))  # Mock escalation count
                # Stage 3: Claude reasoning
                reasoner_start = time.time()
                time.sleep(0.1)  # Mock latency: ~100ms for Claude
                reasoner_latency_ms = (time.time() - reasoner_start) * 1000
                # Mock token usage
                input_tokens = random.randint(500, 2000)
                output_tokens = random.randint(100, 500)
            else:
                router_decision = "DROPPED"
                frames_escalated = 0
                reasoner_latency_ms = 0.0
                input_tokens = 0
                output_tokens = 0

            estimated_cost = (input_tokens / 1_000_000) * 1.00 + \
                           (output_tokens / 1_000_000) * 5.00

            total_latency = clip_latency_ms + router_latency_ms + reasoner_latency_ms

            result = QueryResult(
                query_id=query_id,
                query_text=query_text,
                event_type=event_type,
                expected_relevance_notes=expected_notes,
                clip_latency_ms=clip_latency_ms,
                clip_top_k=20,
                clip_top_1_score=clip_top_1_score,
                clip_top_5_scores=clip_top_5_scores,
                router_decision=router_decision,
                frames_escalated=frames_escalated,
                estimated_cost_usd=estimated_cost,
                reasoner_latency_ms=reasoner_latency_ms,
                reasoner_called=(router_decision == "ESCALATED"),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_latency_ms=total_latency,
            )

            results.append(result)
            logger.debug(
                f"Dual-agent: {query_id} → {router_decision} "
                f"(score={clip_top_1_score:.3f}, escalated={frames_escalated})"
            )

        return results

    def evaluate_claude_only_stub(
        self,
        queries: List[Dict[str, Any]],
        corpus_size: int = 36000,
    ) -> List[QueryResult]:
        """
        Claude-only stub: Naive baseline (placeholder for WP7).

        In WP7, this will be replaced with a real implementation that sends
        all frames to Claude without CLIP filtering.

        Args:
            queries: List of query dicts.
            corpus_size: Total frames.

        Returns:
            List of QueryResult objects.
        """
        logger.info("Starting Claude-only stub evaluation (placeholder)...")
        results = []

        for query_dict in queries:
            query_id = query_dict.get("query_id", "unknown")
            query_text = query_dict.get("query_text", "")
            event_type = query_dict.get("event_type", "unknown")
            expected_notes = query_dict.get("expected_relevance_notes", "")

            # Stub: No CLIP, all frames to Claude
            clip_latency_ms = 0.0
            time.sleep(0.15)  # Mock latency: long Claude processing
            reasoner_latency_ms = (0.15 * 1000)

            # Assume all frames escalated
            frames_escalated = corpus_size
            input_tokens = random.randint(50000, 100000)  # All frames
            output_tokens = random.randint(1000, 5000)

            estimated_cost = (input_tokens / 1_000_000) * 1.00 + \
                           (output_tokens / 1_000_000) * 5.00

            result = QueryResult(
                query_id=query_id,
                query_text=query_text,
                event_type=event_type,
                expected_relevance_notes=expected_notes,
                clip_latency_ms=0.0,
                clip_top_k=0,
                clip_top_1_score=0.0,
                clip_top_5_scores=[],
                router_decision="CLAUDE_ONLY_STUB",
                frames_escalated=frames_escalated,
                estimated_cost_usd=estimated_cost,
                reasoner_latency_ms=reasoner_latency_ms,
                reasoner_called=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_latency_ms=reasoner_latency_ms,
            )

            results.append(result)
            logger.debug(f"Claude-only: {query_id} → stub (all frames)")

        return results

    def run_benchmark(
        self,
        retriever: Optional[Any] = None,
        router: Optional[Any] = None,
        reasoner: Optional[Any] = None,
        index: Optional[Any] = None,
        corpus_size: int = 36000,
    ) -> AggregateMetrics:
        """
        Execute benchmark in configured mode.

        Args:
            retriever: CLIP engine (required for clip_only and dual_agent).
            router: Budget-aware router (required for dual_agent).
            reasoner: Claude reasoner (required for dual_agent).
            index: FAISS index (required for clip_only and dual_agent).
            corpus_size: Total frames in corpus.

        Returns:
            AggregateMetrics object with all results.
        """
        logger.info(f"Running benchmark in mode: {self.config.mode}")
        logger.info(f"Config: {self.config.to_json()}")

        # Load queries
        queries = self.load_queries_from_jsonl(self.config.queries_file)
        if not queries:
            raise ValueError(f"No queries loaded from {self.config.queries_file}")

        # Execute evaluation based on mode
        if self.config.mode == "clip_only":
            self.query_results = self.evaluate_clip_only(
                queries, retriever, index, corpus_size
            )
        elif self.config.mode == "dual_agent":
            self.query_results = self.evaluate_dual_agent(
                queries, retriever, router, reasoner, index, corpus_size
            )
        elif self.config.mode == "claude_only_stub":
            self.query_results = self.evaluate_claude_only_stub(queries, corpus_size)
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

        # Aggregate results
        aggregate = self._aggregate_results(corpus_size, len(queries))

        # Persist results
        self._save_results(aggregate)

        return aggregate

    def _aggregate_results(
        self,
        corpus_size: int,
        num_queries: int,
    ) -> AggregateMetrics:
        """Aggregate query results into AggregateMetrics."""
        logger.info("Aggregating results...")

        # Convert QueryResult to dict for metric functions
        result_dicts = [
            {
                "query_id": qr.query_id,
                "clip_latency_ms": qr.clip_latency_ms,
                "reasoner_latency_ms": qr.reasoner_latency_ms,
                "total_latency_ms": qr.total_latency_ms,
                "frames_escalated": qr.frames_escalated,
                "input_tokens": qr.input_tokens,
                "output_tokens": qr.output_tokens,
                "router_decision": qr.router_decision,
                # Mock relevant_ranks for demo (will be real in WP7 with ground truth)
                "relevant_ranks": [0] if random.random() > 0.3 else [0, 1, 2],
            }
            for qr in self.query_results
        ]

        # Retrieval metrics (mocked for demo)
        retrieval_metrics = metric_fns.aggregate_retrieval_metrics(result_dicts, k_values=[1, 5, 10])

        # Privacy metrics
        queries_on_prem, on_prem_count, total_queries = metric_fns.fraction_queries_on_prem(
            result_dicts, corpus_size
        )
        frames_on_prem, frames_escalated, _ = metric_fns.fraction_frames_on_prem(
            result_dicts, corpus_size
        )

        privacy = PrivacyMetrics(
            total_queries=total_queries,
            queries_resolved_on_prem=on_prem_count,
            total_frames_in_corpus=corpus_size,
            frames_escalated_to_cloud=frames_escalated,
        )

        # Cost metrics
        cost_metrics = metric_fns.aggregate_cost_metrics(result_dicts)
        cost = CostMetrics(
            total_input_tokens=cost_metrics["total_input_tokens"],
            total_output_tokens=cost_metrics["total_output_tokens"],
            total_estimated_cost_usd=cost_metrics["total_cost_usd"],
        )

        # Latency metrics
        latency_metrics = metric_fns.aggregate_latency_metrics(result_dicts)
        latency = LatencyMetrics(
            retriever_latencies=[qr["clip_latency_ms"] for qr in result_dicts],
            reasoner_latencies=[qr["reasoner_latency_ms"] for qr in result_dicts],
            total_latencies=[qr["total_latency_ms"] for qr in result_dicts],
        )

        # Routing metrics
        queries_escalated = sum(
            1 for qr in result_dicts
            if qr.get("router_decision") in ["ESCALATED", "CLAUDE_ONLY_STUB"]
        )
        avg_frames_escalated = (
            sum(qr.get("frames_escalated", 0) for qr in result_dicts) / len(result_dicts)
            if result_dicts
            else 0.0
        )

        # Build aggregate
        aggregate = AggregateMetrics(
            config=self.config,
            mode=self.config.mode,
            total_queries=len(self.query_results),
            recall_at_1=retrieval_metrics.get("recall_at_1", 0.0),
            recall_at_5=retrieval_metrics.get("recall_at_5", 0.0),
            recall_at_10=retrieval_metrics.get("recall_at_10", 0.0),
            precision_at_5=retrieval_metrics.get("precision_at_5", 0.0),
            precision_at_10=retrieval_metrics.get("precision_at_10", 0.0),
            f1_at_5=retrieval_metrics.get("f1_at_5", 0.0),
            mrr=retrieval_metrics.get("mrr", 0.0),
            ndcg_at_5=retrieval_metrics.get("ndcg_at_5", 0.0),
            ndcg_at_10=retrieval_metrics.get("ndcg_at_10", 0.0),
            queries_on_prem_only=on_prem_count,
            queries_escalated=queries_escalated,
            avg_frames_escalated_per_query=avg_frames_escalated,
            privacy=privacy,
            cost=cost,
            latency=latency,
        )

        return aggregate

    def _save_results(self, aggregate: AggregateMetrics) -> None:
        """Save results to JSON and CSV."""
        output_dir = Path(self.config.output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save aggregate metrics as JSON
        json_path = output_dir / f"aggregate_{self.config.mode}_{timestamp}.json"
        with open(json_path, "w") as f:
            f.write(aggregate.to_json())
        logger.info(f"Saved aggregate metrics to {json_path}")

        # Save detailed query results as JSON
        detailed_path = output_dir / f"queries_{self.config.mode}_{timestamp}.json"
        with open(detailed_path, "w") as f:
            json.dump(
                [qr.to_dict() for qr in self.query_results],
                f,
                indent=2,
            )
        logger.info(f"Saved detailed query results to {detailed_path}")

        # Save config used for run
        config_path = output_dir / f"config_{self.config.mode}_{timestamp}.json"
        with open(config_path, "w") as f:
            f.write(self.config.to_json())
        logger.info(f"Saved config to {config_path}")
