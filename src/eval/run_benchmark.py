"""WP6 - Evaluation Harness: CLI entry point.

Invoke with either:
    python -m src.eval.run_benchmark --queries evals/queries.example.jsonl --mode dual_agent
    python -m src.eval                 --queries evals/queries.example.jsonl --mode dual_agent

Both forms call into `main()` below.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from src.eval.harness import BenchmarkRunner
from src.eval.models import AggregateMetrics, EvaluationConfig, VALID_MODES

logger = logging.getLogger("src.eval.run_benchmark")


def _load_config_file(path: str) -> EvaluationConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return EvaluationConfig(**data)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval.run_benchmark",
        description="WP6 Evaluation Harness for the AI Video Investigator (Claude Haiku 4.5).",
    )
    parser.add_argument("--mode", required=True, choices=list(VALID_MODES),
                        help="Which system mode to evaluate.")
    parser.add_argument("--queries", default=None,
                        help="Path to JSONL benchmark file (overrides config).")
    parser.add_argument("--config", default=None,
                        help="Path to a JSON config file. CLI flags override its values.")
    parser.add_argument("--output-dir", default=None,
                        help="Directory where result artefacts are written.")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for the deterministic stand-ins.")
    parser.add_argument("--tau-high", type=float, default=None,
                        help="Router high-confidence threshold.")
    parser.add_argument("--tau-low", type=float, default=None,
                        help="Router low-confidence threshold (ambiguous band lower bound).")
    parser.add_argument("--max-escalations", type=int, default=None,
                        help="Maximum frames escalated to Claude per query.")
    parser.add_argument("--corpus-size", type=int, default=None,
                        help="Total number of frames in the corpus (used for privacy denominators).")
    parser.add_argument("--retrieval-top-k", type=int, default=None,
                        help="How many frames CLIP returns per query.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG-level logging.")
    return parser


def _resolve_config(args: argparse.Namespace) -> EvaluationConfig:
    if args.config:
        cfg = _load_config_file(args.config)
    else:
        cfg = EvaluationConfig()
    cfg.mode = args.mode
    if args.queries is not None:
        cfg.queries_file = args.queries
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.tau_high is not None:
        cfg.tau_high = args.tau_high
    if args.tau_low is not None:
        cfg.tau_low = args.tau_low
    if args.max_escalations is not None:
        cfg.max_escalations = args.max_escalations
    if args.corpus_size is not None:
        cfg.corpus_size = args.corpus_size
    if args.retrieval_top_k is not None:
        cfg.retrieval_top_k = args.retrieval_top_k
    return cfg


def _print_summary(aggregate: AggregateMetrics) -> None:
    line = "=" * 72
    print(line)
    print(f" Mode:        {aggregate.mode}")
    print(f" Queries:     {aggregate.total_queries}")
    print(f" Timestamp:   {aggregate.timestamp}")
    print(line)
    r = aggregate.retrieval
    print(" Retrieval")
    print(f"   Recall@1 / @5 / @10:     {r.recall_at_1:.3f} / {r.recall_at_5:.3f} / {r.recall_at_10:.3f}")
    print(f"   Precision@5 / @10:       {r.precision_at_5:.3f} / {r.precision_at_10:.3f}")
    print(f"   F1@5:                    {r.f1_at_5:.3f}")
    print(f"   MRR:                     {r.mrr:.3f}")
    print(f"   nDCG@5 / @10:            {r.ndcg_at_5:.3f} / {r.ndcg_at_10:.3f}")
    rt = aggregate.routing
    print(" Routing")
    print(f"   immediate / escalated / dropped / clip_only / claude_only:")
    print(f"     {rt.immediate_match} / {rt.escalated} / {rt.dropped} / {rt.clip_only} / {rt.claude_only}")
    print(f"   Avg frames escalated:    {rt.avg_frames_escalated_per_query:.2f}")
    p = aggregate.privacy
    print(" Privacy")
    print(f"   Queries on-prem:         {p.fraction_queries_on_prem:.1%} "
          f"({p.queries_resolved_on_prem}/{p.total_queries})")
    print(f"   Frames on-prem:          {p.fraction_frames_on_prem:.4f} "
          f"({p.unique_frames_escalated} unique escalated / {p.total_frames_in_corpus} corpus)")
    c = aggregate.cost
    print(" Cost (Claude Haiku 4.5)")
    print(f"   Input tokens:            {c.total_input_tokens:,}")
    print(f"   Output tokens:           {c.total_output_tokens:,}")
    print(f"   Estimated cost (USD):    ${c.total_estimated_cost_usd:.4f}")
    print(f"   Cost / query:            ${c.cost_per_query(aggregate.total_queries):.4f}")
    lat = aggregate.latency
    print(" Latency (ms)")
    print(f"   Retriever mean/p95:      {lat.retriever_mean_ms:.1f} / {lat.retriever_p95_ms:.1f}")
    print(f"   Router    mean/p95:      {lat.router_mean_ms:.1f} / {lat.router_p95_ms:.1f}")
    print(f"   Reasoner  mean/p95:      {lat.reasoner_mean_ms:.1f} / {lat.reasoner_p95_ms:.1f}")
    print(f"   Total     mean/p95:      {lat.total_mean_ms:.1f} / {lat.total_p95_ms:.1f}")
    print(line)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        config = _resolve_config(args)
    except Exception as exc:
        logger.error("Failed to assemble config: %s", exc)
        return 2

    if not Path(config.queries_file).exists():
        logger.error("Queries file not found: %s", config.queries_file)
        return 2

    try:
        runner = BenchmarkRunner(config)
        aggregate = runner.run()
    except Exception as exc:
        logger.exception("Benchmark execution failed: %s", exc)
        return 1

    _print_summary(aggregate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
