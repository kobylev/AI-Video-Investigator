"""V1 vs V2 retriever A/B comparator.

Loads the two aggregate JSON artefacts produced by the WP6 harness and prints
a comparison table for Recall@5, F1@5, and retriever p95 latency.

Exits with status 0 only if all acceptance criteria from
docs/V2_OPENCLIP_MIGRATION.md (Acceptance Criteria for Merge) are met:

    1. Recall@5(V2) - Recall@5(V1) >= +0.05
    2. F1@5(V2)     - F1@5(V1)     >= +0.05
    3. p95 retriever latency ratio <= 1.5x
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def _load_latest_aggregate(results_dir: str) -> dict:
    matches = sorted(glob.glob(str(Path(results_dir) / "aggregate_*.json")))
    if not matches:
        raise FileNotFoundError(f"no aggregate_*.json in {results_dir}")
    with open(matches[-1], "r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    v1 = _load_latest_aggregate("evals/results/v1_baseline")
    v2 = _load_latest_aggregate("evals/results/v2_openclip")

    r1, r2 = v1["retrieval"], v2["retrieval"]
    l1, l2 = v1["latency"], v2["latency"]

    print(f"{'Metric':<28} {'V1 (OpenAI CLIP)':>20} {'V2 (OpenCLIP)':>20} {'Delta':>10}")
    print("-" * 80)

    rows = [
        ("Recall@5",            r1["recall_at_5"],          r2["recall_at_5"]),
        ("F1@5",                r1["f1_at_5"],              r2["f1_at_5"]),
        ("nDCG@5",              r1["ndcg_at_5"],            r2["ndcg_at_5"]),
        ("MRR",                 r1["mrr"],                  r2["mrr"]),
        ("Retriever mean (ms)", l1["retriever_mean_ms"],    l2["retriever_mean_ms"]),
        ("Retriever p95 (ms)",  l1["retriever_p95_ms"],     l2["retriever_p95_ms"]),
    ]
    for name, a, b in rows:
        print(f"{name:<28} {a:>20.4f} {b:>20.4f} {b - a:>+10.4f}")

    recall_gain   = r2["recall_at_5"] - r1["recall_at_5"]
    f1_gain       = r2["f1_at_5"]     - r1["f1_at_5"]
    latency_ratio = l2["retriever_p95_ms"] / max(l1["retriever_p95_ms"], 1e-6)

    failed = []
    if recall_gain   < 0.05: failed.append(f"Recall@5 gain {recall_gain:+.3f} < +0.05")
    if f1_gain       < 0.05: failed.append(f"F1@5 gain {f1_gain:+.3f} < +0.05")
    if latency_ratio > 1.5:  failed.append(f"Latency ratio {latency_ratio:.2f}x > 1.5x")

    print()
    if failed:
        print("FAIL - acceptance criteria not met:")
        for reason in failed:
            print(f"  - {reason}")
        return 1
    print("PASS - V2 meets all acceptance criteria; safe to merge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
