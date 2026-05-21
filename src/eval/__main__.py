"""
WP6 - Evaluation Harness: CLI Entry Point
------------------------------------------
Execute benchmarks from command line:
  python -m src.eval --queries evals/queries.example.jsonl --mode dual_agent
  python -m src.eval --config config.json --mode clip_only
"""

import argparse
import logging
import json
import sys
from pathlib import Path

from src.eval.models import EvaluationConfig
from src.eval.harness import BenchmarkEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config_from_file(config_path: str) -> EvaluationConfig:
    """Load evaluation config from JSON file."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return EvaluationConfig(**config_dict)


def main():
    parser = argparse.ArgumentParser(
        description="WP6 Evaluation Harness for AI Video Investigator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run dual-agent benchmark
  python -m src.eval --queries evals/queries.example.jsonl --mode dual_agent

  # Run CLIP-only baseline
  python -m src.eval --queries evals/queries.example.jsonl --mode clip_only

  # Load config from file
  python -m src.eval --config my_config.json --mode dual_agent

  # Override config settings
  python -m src.eval --queries evals/queries.example.jsonl --mode dual_agent \\
    --tau-high 0.35 --tau-low 0.25 --seed 123
        """,
    )

    # Config input (either file or inline)
    config_group = parser.add_mutually_exclusive_group(required=False)
    config_group.add_argument(
        "--config",
        type=str,
        help="Path to JSON config file",
    )

    # Evaluation mode (required)
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["clip_only", "dual_agent", "claude_only_stub"],
        help="Evaluation mode",
    )

    # Query input (optional if config provided)
    parser.add_argument(
        "--queries",
        type=str,
        help="Path to JSONL queries file (default: from config or evals/queries.example.jsonl)",
    )

    # Output directory
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evals/results",
        help="Output directory for results (default: evals/results)",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (default: 42)",
    )

    # Router thresholds
    parser.add_argument(
        "--tau-high",
        type=float,
        help="High confidence threshold (default: 0.32)",
    )
    parser.add_argument(
        "--tau-low",
        type=float,
        help="Low confidence threshold (default: 0.24)",
    )

    # Max escalations
    parser.add_argument(
        "--max-escalations",
        type=int,
        help="Maximum frames to escalate per query (default: 5)",
    )

    # Verbosity
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build config
    try:
        if args.config:
            logger.info(f"Loading config from {args.config}")
            config = load_config_from_file(args.config)
        else:
            config = EvaluationConfig()

        # Override with CLI arguments
        config.mode = args.mode
        if args.queries:
            config.queries_file = args.queries
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.seed is not None:
            config.seed = args.seed
        if args.tau_high is not None:
            config.tau_high = args.tau_high
        if args.tau_low is not None:
            config.tau_low = args.tau_low
        if args.max_escalations is not None:
            config.max_escalations = args.max_escalations

    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    logger.info(f"Evaluation Config:\n{config.to_json()}")

    # Validate queries file exists
    if not Path(config.queries_file).exists():
        logger.error(f"Queries file not found: {config.queries_file}")
        sys.exit(1)

    # Run benchmark
    try:
        evaluator = BenchmarkEvaluator(config)
        # Note: In WP6 demo mode, we run without real component instances.
        # These will be injected in WP7+ when real evaluation against live systems occurs.
        aggregate = evaluator.run_benchmark(
            retriever=None,  # Mock mode
            router=None,
            reasoner=None,
            index=None,
            corpus_size=36000,  # 10h @ 1fps
        )

        logger.info("=" * 80)
        logger.info("BENCHMARK COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nMode: {config.mode}")
        logger.info(f"Queries: {aggregate.total_queries}")
        logger.info(f"\nRetrieval Metrics (Top-5):")
        logger.info(f"  Recall@5:    {aggregate.recall_at_5:.3f}")
        logger.info(f"  Precision@5: {aggregate.precision_at_5:.3f}")
        logger.info(f"  F1@5:        {aggregate.f1_at_5:.3f}")
        logger.info(f"  nDCG@5:      {aggregate.ndcg_at_5:.3f}")
        logger.info(f"  MRR:         {aggregate.mrr:.3f}")
        logger.info(f"\nRouting Metrics:")
        logger.info(f"  On-prem only:       {aggregate.queries_on_prem_only}/{aggregate.total_queries}")
        logger.info(f"  Avg frames escalated: {aggregate.avg_frames_escalated_per_query:.1f}")
        logger.info(f"\nPrivacy Metrics:")
        logger.info(f"  Queries on-prem:    {aggregate.privacy.fraction_queries_on_prem:.1%}")
        logger.info(f"  Frames on-prem:     {aggregate.privacy.fraction_frames_on_prem:.1%}")
        logger.info(f"\nCost Metrics:")
        logger.info(f"  Total input tokens:   {aggregate.cost.total_input_tokens:,}")
        logger.info(f"  Total output tokens:  {aggregate.cost.total_output_tokens:,}")
        logger.info(f"  Total cost (USD):     ${aggregate.cost.total_estimated_cost_usd:.4f}")
        logger.info(f"  Cost per query:       ${aggregate.cost.computed_cost_usd / aggregate.total_queries:.4f}")
        logger.info(f"\nLatency Metrics:")
        logger.info(f"  CLIP mean:    {aggregate.latency.mean_retriever_ms:.1f}ms")
        logger.info(f"  CLIP p95:     {aggregate.latency.p95_retriever_ms:.1f}ms")
        logger.info(f"  Total mean:   {aggregate.latency.mean_total_ms:.1f}ms")
        logger.info(f"  Total p95:    {aggregate.latency.p95_total_ms:.1f}ms")
        logger.info("\n" + "=" * 80)

    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
