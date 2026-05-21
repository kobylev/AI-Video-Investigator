"""WP6 - Evaluation Harness: I/O helpers.

Centralised so the harness can stay focused on orchestration. All file
writes are timestamped and grouped under a single timestamp per run, so
JSON/CSV/config artefacts produced by one invocation can be paired up
from the filenames alone.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

from src.eval.models import AggregateMetrics, EvaluationConfig, EvaluationSample, QueryResult


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def load_samples(path: str) -> List[EvaluationSample]:
    """Load benchmark samples from a JSONL file.

    Each line is parsed as JSON. Recognised fields:
      - query_id (required)
      - query_text (required)
      - event_type (optional, default "unknown")
      - relevant_indices (optional, list[int])
      - total_relevant (optional, int)
      - expected_relevance_notes (optional, str)

    Missing optional fields default sensibly so partially-annotated
    benchmark files still load — but their retrieval metrics will be 0.0.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")

    samples: List[EvaluationSample] = []
    with p.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON ({exc})") from exc

            if "query_id" not in row or "query_text" not in row:
                raise ValueError(
                    f"{path}:{line_no} missing required field: query_id / query_text"
                )

            samples.append(
                EvaluationSample(
                    query_id=str(row["query_id"]),
                    query_text=str(row["query_text"]),
                    event_type=str(row.get("event_type", "unknown")),
                    relevant_indices=tuple(int(i) for i in row.get("relevant_indices", [])),
                    total_relevant=row.get("total_relevant"),
                    expected_relevance_notes=str(row.get("expected_relevance_notes", "")),
                )
            )
    return samples


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

# Stable column order for the per-query CSV. Anything machine-consumed
# downstream (WP7 comparison, future dashboards) should depend on this
# exact set of column names — extend only by appending.
QUERY_CSV_COLUMNS: Tuple[str, ...] = (
    "query_id",
    "query_text",
    "event_type",
    "router_decision",
    "frames_escalated",
    "reasoner_called",
    "input_tokens",
    "output_tokens",
    "estimated_cost_usd",
    "retriever_latency_ms",
    "router_latency_ms",
    "reasoner_latency_ms",
    "total_latency_ms",
    "clip_top_1_score",
    "total_relevant",
    "num_relevant_indices",
    "ranked_indices",          # JSON-encoded list (CSV-safe)
    "relevant_indices",        # JSON-encoded list (CSV-safe)
)


def _query_row(result: QueryResult) -> dict:
    return {
        "query_id": result.query_id,
        "query_text": result.query_text,
        "event_type": result.event_type,
        "router_decision": result.router_decision,
        "frames_escalated": result.frames_escalated,
        "reasoner_called": result.reasoner_called,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "estimated_cost_usd": f"{result.estimated_cost_usd:.6f}",
        "retriever_latency_ms": f"{result.retriever_latency_ms:.3f}",
        "router_latency_ms": f"{result.router_latency_ms:.3f}",
        "reasoner_latency_ms": f"{result.reasoner_latency_ms:.3f}",
        "total_latency_ms": f"{result.total_latency_ms:.3f}",
        "clip_top_1_score": f"{result.clip_top_1_score:.4f}",
        "total_relevant": result.total_relevant,
        "num_relevant_indices": len(result.relevant_indices),
        "ranked_indices": json.dumps(result.ranked_indices),
        "relevant_indices": json.dumps(result.relevant_indices),
    }


def write_outputs(
    output_dir: str,
    mode: str,
    aggregate: AggregateMetrics,
    query_results: Iterable[QueryResult],
    config: EvaluationConfig,
    timestamp: str | None = None,
) -> List[Path]:
    """Persist a run's JSON aggregate, JSON per-query results, CSV per-query
    results, and the run config. Returns the list of paths written, in the
    order they were created."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    written: List[Path] = []

    aggregate_path = out / f"aggregate_{mode}_{ts}.json"
    with aggregate_path.open("w", encoding="utf-8") as fh:
        json.dump(aggregate.to_dict(), fh, indent=2)
    written.append(aggregate_path)

    queries_json = out / f"queries_{mode}_{ts}.json"
    results = list(query_results)
    with queries_json.open("w", encoding="utf-8") as fh:
        json.dump([qr.to_dict() for qr in results], fh, indent=2)
    written.append(queries_json)

    queries_csv = out / f"queries_{mode}_{ts}.csv"
    with queries_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(QUERY_CSV_COLUMNS))
        writer.writeheader()
        for qr in results:
            writer.writerow(_query_row(qr))
    written.append(queries_csv)

    config_path = out / f"config_{mode}_{ts}.json"
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(config.to_dict(), fh, indent=2)
    written.append(config_path)

    return written
