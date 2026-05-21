"""WP6 - Evaluation Harness: orchestration.

`BenchmarkRunner.run()` is the single entry point that loads samples,
executes one mode against each, computes per-query retrieval metrics
against ground truth, aggregates into `AggregateMetrics`, and persists
JSON + CSV outputs.

Privacy aggregation deliberately tracks *unique* frames escalated rather
than summed escalations — sending the same frame to Claude for two
different queries still only leaks one frame from the corpus. This is the
correct interpretation for the privacy-percentage metric.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime
from typing import List, Optional

from src.eval import metrics as M
from src.eval.io import load_samples, write_outputs
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
    ROUTER_CLAUDE_ONLY,
    ROUTER_CLIP_ONLY,
    ROUTER_DROPPED,
    ROUTER_ESCALATED,
    ROUTER_IMMEDIATE,
    MODE_CLAUDE_ONLY_STUB,
    MODE_CLIP_ONLY,
    MODE_DUAL_AGENT,
    VALID_MODES,
)
from src.eval.modes import (
    run_claude_only_stub,
    run_clip_only,
    run_dual_agent,
)

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs a single benchmark configuration end-to-end.

    Components (retriever / router / reasoner) are optional. When `None`,
    the mode adapters fall back to deterministic stand-ins so the harness
    can run in CI without GPUs or API keys. WP7 will inject real instances
    via these parameters when comparing against live infrastructure.
    """

    def __init__(
        self,
        config: EvaluationConfig,
        retriever: Optional[object] = None,
        router: Optional[object] = None,
        reasoner: Optional[object] = None,
    ) -> None:
        if config.mode not in VALID_MODES:
            raise ValueError(
                f"Unknown evaluation mode: {config.mode!r}. "
                f"Expected one of: {VALID_MODES}"
            )
        self.config = config
        self.retriever = retriever
        self.router = router
        self.reasoner = reasoner

    # -----------------------------------------------------------------
    # Per-query metric computation
    # -----------------------------------------------------------------

    @staticmethod
    def _per_query_retrieval(result: QueryResult) -> dict:
        rel = result.relevant_indices
        total = result.total_relevant
        ranked = result.ranked_indices
        return {
            "recall_at_1": M.recall_at_k(rel, ranked, 1, total),
            "recall_at_5": M.recall_at_k(rel, ranked, 5, total),
            "recall_at_10": M.recall_at_k(rel, ranked, 10, total),
            "precision_at_5": M.precision_at_k(rel, ranked, 5),
            "precision_at_10": M.precision_at_k(rel, ranked, 10),
            "f1_at_5": M.f1_at_k(rel, ranked, 5, total),
            "mrr": M.reciprocal_rank(rel, ranked),
            "ndcg_at_5": M.ndcg_at_k(rel, ranked, 5),
            "ndcg_at_10": M.ndcg_at_k(rel, ranked, 10),
        }

    # -----------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------

    def _execute_one(
        self,
        sample: EvaluationSample,
        rng: random.Random,
    ) -> QueryResult:
        mode = self.config.mode
        if mode == MODE_CLIP_ONLY:
            return run_clip_only(sample, self.config, rng, retriever=self.retriever)
        if mode == MODE_DUAL_AGENT:
            return run_dual_agent(
                sample,
                self.config,
                rng,
                retriever=self.retriever,
                router=self.router,
                reasoner=self.reasoner,
            )
        if mode == MODE_CLAUDE_ONLY_STUB:
            return run_claude_only_stub(sample, self.config, rng)
        # VALID_MODES guard in __init__ makes this unreachable.
        raise AssertionError(f"unreachable mode: {mode}")

    def run(self, samples: Optional[List[EvaluationSample]] = None) -> AggregateMetrics:
        if samples is None:
            samples = load_samples(self.config.queries_file)
        if not samples:
            raise ValueError(f"No samples loaded from {self.config.queries_file}")

        # Fixed seed → deterministic stand-ins → reproducible run.
        rng = random.Random(self.config.seed)

        logger.info(
            "Running benchmark mode=%s samples=%d corpus_size=%d seed=%d",
            self.config.mode, len(samples), self.config.corpus_size, self.config.seed,
        )

        results: List[QueryResult] = []
        per_query_metrics: List[dict] = []
        for sample in samples:
            qr = self._execute_one(sample, rng)
            results.append(qr)
            per_query_metrics.append(self._per_query_retrieval(qr))

        aggregate = self._aggregate(results, per_query_metrics)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = write_outputs(
            self.config.output_dir,
            self.config.mode,
            aggregate,
            results,
            self.config,
            timestamp=timestamp,
        )
        logger.info("Wrote %d artefacts to %s", len(paths), self.config.output_dir)
        for p in paths:
            logger.info("  - %s", p)
        return aggregate

    # -----------------------------------------------------------------
    # Aggregation
    # -----------------------------------------------------------------

    def _aggregate(
        self,
        results: List[QueryResult],
        per_query_metrics: List[dict],
    ) -> AggregateMetrics:
        retr_dict = M.aggregate_retrieval(per_query_metrics)
        retrieval = RetrievalMetrics(**retr_dict)

        # Routing decision counts.
        routing = RoutingMetrics()
        total_escalated_frames = 0
        for r in results:
            if r.router_decision == ROUTER_IMMEDIATE:
                routing.immediate_match += 1
            elif r.router_decision == ROUTER_ESCALATED:
                routing.escalated += 1
            elif r.router_decision == ROUTER_DROPPED:
                routing.dropped += 1
            elif r.router_decision == ROUTER_CLIP_ONLY:
                routing.clip_only += 1
            elif r.router_decision == ROUTER_CLAUDE_ONLY:
                routing.claude_only += 1
            total_escalated_frames += r.frames_escalated
        routing.avg_frames_escalated_per_query = (
            total_escalated_frames / len(results) if results else 0.0
        )

        # Privacy: count *unique* frames sent to cloud across all queries.
        # The claude_only_stub mode escalates the entire corpus per query
        # but does not enumerate frame indices; we recognise that case and
        # cap the unique-frames denominator at the corpus size.
        unique_frames: set[int] = set()
        for r in results:
            if r.router_decision == ROUTER_CLAUDE_ONLY:
                # Conceptually every frame leaks; recorded explicitly below.
                continue
            unique_frames.update(r.escalated_indices)
        if any(r.router_decision == ROUTER_CLAUDE_ONLY for r in results):
            unique_escalated = self.config.corpus_size
        else:
            unique_escalated = len(unique_frames)
        queries_resolved_on_prem = sum(1 for r in results if r.frames_escalated == 0)
        privacy = PrivacyMetrics(
            total_queries=len(results),
            queries_resolved_on_prem=queries_resolved_on_prem,
            total_frames_in_corpus=self.config.corpus_size,
            unique_frames_escalated=unique_escalated,
            total_frame_escalations=total_escalated_frames,
        )

        # Cost: sum tokens, then have CostMetrics recompute USD as a check.
        total_in = sum(r.input_tokens for r in results)
        total_out = sum(r.output_tokens for r in results)
        total_cost = sum(r.estimated_cost_usd for r in results)
        cost = CostMetrics(
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            total_estimated_cost_usd=total_cost,
            price_per_1m_input_tokens=self.config.price_per_1m_input_tokens,
            price_per_1m_output_tokens=self.config.price_per_1m_output_tokens,
        )

        # Latency.
        retr_l = [r.retriever_latency_ms for r in results]
        rout_l = [r.router_latency_ms for r in results]
        reas_l = [r.reasoner_latency_ms for r in results]
        tot_l = [r.total_latency_ms for r in results]
        latency = LatencyMetrics(
            retriever_mean_ms=M.mean(retr_l),
            retriever_p95_ms=M.percentile(retr_l, 95),
            router_mean_ms=M.mean(rout_l),
            router_p95_ms=M.percentile(rout_l, 95),
            reasoner_mean_ms=M.mean(reas_l),
            reasoner_p95_ms=M.percentile(reas_l, 95),
            total_mean_ms=M.mean(tot_l),
            total_p95_ms=M.percentile(tot_l, 95),
        )

        return AggregateMetrics(
            mode=self.config.mode,
            total_queries=len(results),
            retrieval=retrieval,
            routing=routing,
            privacy=privacy,
            cost=cost,
            latency=latency,
            config=self.config,
        )
