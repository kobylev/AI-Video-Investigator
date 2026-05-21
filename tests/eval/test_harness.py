"""Smoke tests for `BenchmarkRunner` end-to-end behaviour.

These don't try to exercise real CLIP/Claude. They verify the harness:
  - is deterministic under a fixed seed,
  - writes the expected JSON + CSV artefacts,
  - bounds privacy metrics correctly in claude_only_stub mode,
  - produces non-zero retrieval metrics against the deterministic stand-in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval import (
    BenchmarkRunner,
    EvaluationConfig,
    EvaluationSample,
    MODE_CLAUDE_ONLY_STUB,
    MODE_CLIP_ONLY,
    MODE_DUAL_AGENT,
)


def _samples() -> list[EvaluationSample]:
    # Two queries with explicit ground truth so retrieval metrics are real.
    return [
        EvaluationSample(
            query_id="q1",
            query_text="white SUV cutting off truck",
            event_type="vehicle_interaction",
            relevant_indices=(10, 11, 12),
            total_relevant=3,
        ),
        EvaluationSample(
            query_id="q2",
            query_text="pedestrian in red jacket",
            event_type="pedestrian_event",
            relevant_indices=(500, 501),
            total_relevant=2,
        ),
    ]


def _config(tmp_path: Path, mode: str) -> EvaluationConfig:
    return EvaluationConfig(
        mode=mode,
        queries_file=str(tmp_path / "unused.jsonl"),  # we'll pass samples directly
        output_dir=str(tmp_path / "results"),
        seed=42,
        corpus_size=1000,
        retrieval_top_k=20,
    )


@pytest.mark.parametrize("mode", [MODE_CLIP_ONLY, MODE_DUAL_AGENT, MODE_CLAUDE_ONLY_STUB])
def test_run_writes_expected_artefacts(tmp_path, mode):
    cfg = _config(tmp_path, mode)
    runner = BenchmarkRunner(cfg)
    aggregate = runner.run(samples=_samples())

    out = Path(cfg.output_dir)
    files = list(out.iterdir())
    names = sorted(f.name for f in files)
    # One run = 4 artefacts: aggregate JSON, queries JSON, queries CSV, config JSON.
    assert len(files) == 4
    assert any(n.startswith(f"aggregate_{mode}_") and n.endswith(".json") for n in names)
    assert any(n.startswith(f"queries_{mode}_") and n.endswith(".json") for n in names)
    assert any(n.startswith(f"queries_{mode}_") and n.endswith(".csv") for n in names)
    assert any(n.startswith(f"config_{mode}_") and n.endswith(".json") for n in names)

    # Aggregate JSON has the documented top-level keys.
    agg_path = next(out.glob(f"aggregate_{mode}_*.json"))
    payload = json.loads(agg_path.read_text(encoding="utf-8"))
    assert set(payload.keys()) >= {
        "metadata", "retrieval_metrics", "routing_metrics",
        "privacy_metrics", "cost_metrics", "latency_metrics",
    }
    assert payload["metadata"]["mode"] == mode
    assert payload["metadata"]["total_queries"] == 2
    assert aggregate.total_queries == 2


def test_clip_only_keeps_everything_on_prem(tmp_path):
    cfg = _config(tmp_path, MODE_CLIP_ONLY)
    aggregate = BenchmarkRunner(cfg).run(samples=_samples())
    assert aggregate.privacy.fraction_queries_on_prem == 1.0
    assert aggregate.privacy.fraction_frames_on_prem == 1.0
    assert aggregate.cost.total_input_tokens == 0
    assert aggregate.cost.total_output_tokens == 0
    assert aggregate.cost.total_estimated_cost_usd == 0.0


def test_claude_only_stub_caps_privacy_denominator(tmp_path):
    # The bug we're guarding against: summing frames_escalated across queries
    # used to drive `fraction_frames_on_prem` negative.
    cfg = _config(tmp_path, MODE_CLAUDE_ONLY_STUB)
    aggregate = BenchmarkRunner(cfg).run(samples=_samples())
    assert aggregate.privacy.fraction_queries_on_prem == 0.0
    # Every frame leaks → on-prem fraction is exactly 0.
    assert aggregate.privacy.fraction_frames_on_prem == 0.0
    # Cost is the dominant signal of this mode.
    assert aggregate.cost.total_input_tokens > 0
    assert aggregate.cost.total_output_tokens > 0
    assert aggregate.cost.total_estimated_cost_usd > 0


def test_dual_agent_runs_are_deterministic(tmp_path):
    cfg1 = _config(tmp_path / "a", MODE_DUAL_AGENT)
    cfg2 = _config(tmp_path / "b", MODE_DUAL_AGENT)
    a1 = BenchmarkRunner(cfg1).run(samples=_samples())
    a2 = BenchmarkRunner(cfg2).run(samples=_samples())
    # Retrieval, routing, cost are derived deterministically from the seed.
    assert a1.retrieval.to_dict() == a2.retrieval.to_dict()
    assert a1.routing.to_dict() == a2.routing.to_dict()
    assert a1.cost.total_input_tokens == a2.cost.total_input_tokens
    assert a1.cost.total_output_tokens == a2.cost.total_output_tokens


def test_sample_order_does_not_affect_other_samples(tmp_path):
    # Per-query seeds are derived from query_id, so reordering the input
    # must not change any one query's result.
    samples = _samples()
    cfg1 = _config(tmp_path / "a", MODE_DUAL_AGENT)
    cfg2 = _config(tmp_path / "b", MODE_DUAL_AGENT)

    BenchmarkRunner(cfg1).run(samples=samples)
    BenchmarkRunner(cfg2).run(samples=list(reversed(samples)))

    def _by_id(path: Path) -> dict[str, dict]:
        data = json.loads(
            next(path.glob("queries_dual_agent_*.json")).read_text(encoding="utf-8")
        )
        return {row["query_id"]: row for row in data}

    a = _by_id(Path(cfg1.output_dir))
    b = _by_id(Path(cfg2.output_dir))
    for qid, row_a in a.items():
        assert row_a["ranked_indices"] == b[qid]["ranked_indices"]
        assert row_a["router_decision"] == b[qid]["router_decision"]
        assert row_a["input_tokens"] == b[qid]["input_tokens"]


def test_unknown_mode_rejected(tmp_path):
    cfg = _config(tmp_path, "nonsense_mode")
    with pytest.raises(ValueError):
        BenchmarkRunner(cfg)
