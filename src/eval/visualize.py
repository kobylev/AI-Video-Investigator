"""WP6 — Results visualiser.

Reads the latest `aggregate_<mode>_<ts>.json` per mode from a results
directory and emits presentation-ready PNG charts plus a single-row-per-
mode CSV summary.

Run with:

    python -m src.eval.visualize --results-dir evals/results --latest-per-mode

Charts are deterministic given the same inputs (Plotly's static export
via Kaleido is stable; no random sampling or styling jitter is
introduced). Missing modes are skipped with a warning rather than
crashing — the CLI exits with status 2 only if no modes resolve at all.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import plotly.graph_objects as go
import plotly.io as pio

from src.eval import plots
from src.eval.models import VALID_MODES

logger = logging.getLogger("src.eval.visualize")

AGGREGATE_PATTERN = re.compile(
    r"^aggregate_(?P<mode>[a-z_]+)_(?P<ts>\d{8}_\d{6})\.json$"
)

CSV_COLUMNS = (
    "mode",
    "total_queries",
    "recall_at_5",
    "f1_at_5",
    "mrr",
    "ndcg_at_5",
    "fraction_queries_on_prem",
    "fraction_frames_on_prem",
    "total_estimated_cost_usd",
    "cost_per_query_usd",
    "total_mean_ms",
    "total_p95_ms",
    "reasoner_mean_ms",
    "source_file",
)


@dataclass(frozen=True)
class ResolvedRun:
    """Bookkeeping for one mode's latest aggregate file."""

    mode: str
    path: Path
    timestamp: str
    payload: dict


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def discover_latest(results_dir: Path) -> Dict[str, ResolvedRun]:
    """For each known mode, find the most recent `aggregate_<mode>_<ts>.json`
    in `results_dir`. Returns a dict keyed by mode; absent modes are simply
    not present."""
    by_mode: Dict[str, ResolvedRun] = {}
    if not results_dir.is_dir():
        return by_mode

    candidates: Dict[str, List[tuple[str, Path]]] = {m: [] for m in VALID_MODES}
    for f in results_dir.iterdir():
        if not f.is_file():
            continue
        m = AGGREGATE_PATTERN.match(f.name)
        if not m:
            continue
        mode = m.group("mode")
        if mode not in candidates:
            continue
        candidates[mode].append((m.group("ts"), f))

    for mode, hits in candidates.items():
        if not hits:
            continue
        ts, path = max(hits, key=lambda t: t[0])  # timestamps are sortable as strings
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping %s — could not read JSON (%s)", path, exc)
            continue
        by_mode[mode] = ResolvedRun(mode=mode, path=path, timestamp=ts, payload=payload)
    return by_mode


# ---------------------------------------------------------------------------
# CSV summary
# ---------------------------------------------------------------------------

def _summary_row(run: ResolvedRun) -> Dict[str, object]:
    p = run.payload
    n = max(int(p["metadata"]["total_queries"]), 1)
    cost_total = float(p["cost_metrics"]["computed_cost_usd"])
    return {
        "mode": run.mode,
        "total_queries": p["metadata"]["total_queries"],
        "recall_at_5": round(float(p["retrieval_metrics"]["recall_at_5"]), 4),
        "f1_at_5": round(float(p["retrieval_metrics"]["f1_at_5"]), 4),
        "mrr": round(float(p["retrieval_metrics"]["mrr"]), 4),
        "ndcg_at_5": round(float(p["retrieval_metrics"]["ndcg_at_5"]), 4),
        "fraction_queries_on_prem": round(float(p["privacy_metrics"]["fraction_queries_on_prem"]), 4),
        "fraction_frames_on_prem": round(float(p["privacy_metrics"]["fraction_frames_on_prem"]), 6),
        "total_estimated_cost_usd": round(cost_total, 6),
        "cost_per_query_usd": round(cost_total / n, 6),
        "total_mean_ms": round(float(p["latency_metrics"]["total"]["mean_ms"]), 2),
        "total_p95_ms": round(float(p["latency_metrics"]["total"]["p95_ms"]), 2),
        "reasoner_mean_ms": round(float(p["latency_metrics"]["reasoner"]["mean_ms"]), 2),
        "source_file": run.path.name,
    }


def write_summary_csv(runs: Mapping[str, ResolvedRun], path: Path) -> Path:
    """Write `wp6_presentation_summary.csv` with one row per mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Emit rows in the canonical mode order so the spreadsheet reads
    # consistently across runs.
    ordered = [runs[m] for m in plots.MODE_ORDER if m in runs]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for run in ordered:
            writer.writerow(_summary_row(run))
    return path


# ---------------------------------------------------------------------------
# Chart rendering
# ---------------------------------------------------------------------------

CHART_BUILDERS = (
    ("quality.png", plots.build_quality_chart),
    ("privacy.png", plots.build_privacy_chart),
    ("cost.png", plots.build_cost_chart),
    ("latency.png", plots.build_latency_chart),
    ("scorecard.png", plots.build_scorecard_chart),
)


def _payloads(runs: Mapping[str, ResolvedRun]) -> Dict[str, dict]:
    return {m: r.payload for m, r in runs.items()}


def render_charts(
    runs: Mapping[str, ResolvedRun],
    out_dir: Path,
    *,
    scale: float = 2.0,
) -> List[Path]:
    """Render every chart in `CHART_BUILDERS` to `out_dir` as PNG.

    `scale=2.0` gives Retina-quality 2560×1440 PNGs from the default
    1280×720 Plotly layout — sharp on a 4K projector, still cheap to
    drop into a Keynote/PowerPoint slide.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = _payloads(runs)
    written: List[Path] = []
    for filename, builder in CHART_BUILDERS:
        fig = builder(payloads)
        target = out_dir / filename
        pio.write_image(fig, str(target), format="png", scale=scale)
        written.append(target)
        logger.info("Wrote chart %s", target)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval.visualize",
        description=(
            "Generate presentation-ready PNG charts and a one-row-per-mode "
            "CSV summary from WP6 result artefacts."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default="evals/results",
        help="Directory containing aggregate_<mode>_<ts>.json files (default: evals/results).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write charts and the summary CSV. Default: <results-dir>/charts/.",
    )
    parser.add_argument(
        "--latest-per-mode",
        action="store_true",
        default=True,
        help="(Default) Use the latest aggregate file per mode.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Plotly PNG export scale factor (default 2.0 → 2560×1440).",
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG-level logging.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else results_dir / "charts"

    runs = discover_latest(results_dir)
    missing = [m for m in VALID_MODES if m not in runs]
    for m in missing:
        logger.warning("No aggregate file found for mode %r — chart will omit it.", m)

    if not runs:
        logger.error(
            "No aggregate files found under %s. Run "
            "`python -m src.eval.run_benchmark --mode <mode>` first.",
            results_dir,
        )
        return 2

    logger.info("Resolved %d mode(s): %s", len(runs), ", ".join(sorted(runs)))
    for run in runs.values():
        logger.info("  %s ← %s", run.mode, run.path.name)

    charts = render_charts(runs, out_dir, scale=args.scale)
    csv_path = write_summary_csv(runs, out_dir / "wp6_presentation_summary.csv")

    print("=" * 72)
    print(f" Charts written ({len(charts)}):")
    for c in charts:
        print(f"   - {c}")
    print(f" CSV summary:")
    print(f"   - {csv_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
