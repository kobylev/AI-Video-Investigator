# `evals/` — Benchmarks, Ground Truth, Results

Inputs and outputs for the WP6 evaluation harness. Implementation lives
under [src/eval/](../src/eval/); deep documentation is in
[docs/work_packages/wp6_eval_harness.md](../docs/work_packages/wp6_eval_harness.md).

---

## Quick start

```bash
# Main system (CLIP → Router → Claude Haiku 4.5)
python -m src.eval.run_benchmark --queries evals/queries.example.jsonl --mode dual_agent

# Retriever-only baseline
python -m src.eval.run_benchmark --queries evals/queries.example.jsonl --mode clip_only

# Naive "all frames to Claude" baseline (placeholder for WP7)
python -m src.eval.run_benchmark --queries evals/queries.example.jsonl --mode claude_only_stub
```

`python -m src.eval` is also accepted and forwards to the same entry point.

Every invocation writes four timestamped artefacts to `evals/results/`:

```
aggregate_<mode>_<ts>.json   # top-level metrics (the headline numbers)
queries_<mode>_<ts>.json     # per-query records, full fidelity
queries_<mode>_<ts>.csv      # per-query records, spreadsheet-friendly
config_<mode>_<ts>.json      # the EvaluationConfig the run used
```

Same `--seed` plus same `--queries` plus same `--mode` ⇒ byte-identical
results across `.json` files (timestamps differ).

---

## WP6 Evaluation Snapshot

Latest benchmark modes were run on `evals/queries.example.jsonl` and summarized by the WP6 harness.

| Mode | Recall@5 | F1@5 | Queries On-Prem | Cost / Query (USD) | Mean Total Latency (ms) |
|---|---:|---:|---:|---:|---:|
| clip_only | 0.587 | 0.460 | 100.0% | 0.0000 | 51.1 |
| dual_agent | 0.587 | 0.460 | 80.0% | 0.0006 | 111.0 |
| claude_only_stub\* | 0.742 | 0.627 | 0.0% | 0.0850 | 1500.0 |

\* `claude_only_stub` is a simulated upper-bound baseline, not a production run.

Presentation-ready charts are available in [`results/charts/`](results/charts/); see [Generating presentation charts](#generating-presentation-charts) below to regenerate them.

---

## Query file format

JSONL — one query per line. Required fields are `query_id` and
`query_text`; the rest are optional but recommended.

```json
{
  "query_id": "veh_001",
  "query_text": "white SUV cutting off truck in left lane",
  "event_type": "vehicle_interaction",
  "relevant_indices": [142, 143, 144, 1287],
  "total_relevant": 4,
  "expected_relevance_notes": "Frames showing a white SUV performing a sudden lane change…"
}
```

| Field | Type | Purpose |
| --- | --- | --- |
| `query_id` | str | Stable identifier; carried into every result row. |
| `query_text` | str | The natural-language query. |
| `event_type` | str | One of `vehicle_interaction`, `pedestrian_event`, `object_of_interest`, `traffic_violation`, `ambient_scene` (free-form; used only for slicing). |
| `relevant_indices` | list[int] | Corpus frame indices the annotator marked relevant. Drives Recall@K, Precision@K, F1@5, MRR, nDCG@K. |
| `total_relevant` | int | Recall@K denominator. May exceed `len(relevant_indices)` when annotators only enumerated a subset (common for `ambient_scene`). Defaults to `len(relevant_indices)` if omitted. |
| `expected_relevance_notes` | str | Free-text guidance for downstream annotation reviews. |

A query with no `relevant_indices` is still valid — it will contribute
0.0 to every retrieval metric. Privacy, cost, and latency are still
recorded for it.

---

## Evaluation modes

| Mode | What runs | Expected shape |
| --- | --- | --- |
| `clip_only` | CLIP retriever only — no router, no Claude. | 100% on-prem, $0 cost, fast latency. Sets the floor for retrieval quality. |
| `dual_agent` | CLIP → BudgetAwareRouter (τ_low/τ_high) → Claude Haiku 4.5 on the ambiguous band. | Most queries resolved on-prem; small Claude bill when CLIP is unsure. |
| `claude_only_stub` | Naive baseline that conceptually ships every corpus frame to Claude. WP7 will replace the stub with the real implementation. | Highest accuracy ceiling, but ~$0.08/query and 0% on-prem. Motivates the cascade. |

Modes are selected with `--mode`. The harness rejects unknown values.

---

## What the harness measures

**Retrieval quality** (against the JSONL ground truth):
- `Recall@{1,5,10}` — fraction of relevant frames the system surfaced in top-K.
- `Precision@{5,10}` — fraction of top-K that was relevant.
- `F1@5` — harmonic mean of P@5 and R@5.
- `MRR` — 1 / rank of the first relevant hit, averaged.
- `nDCG@{5,10}` — position-weighted ranking quality.

**Routing**:
- Counts of `IMMEDIATE_MATCH`, `ESCALATED`, `DROPPED`, plus the mode-specific
  `CLIP_ONLY` and `CLAUDE_ONLY` decisions.
- Average frames escalated per query.

**Privacy** (first-class outputs):
- Fraction of queries resolved entirely on-prem (zero escalations).
- Fraction of frames never sent to cloud, counting **unique** frames so
  re-escalating the same frame across queries isn't double-counted.

**Cost** (Claude Haiku 4.5 list pricing — $1/M input, $5/M output by default):
- Total input/output tokens, total estimated USD, computed USD (sanity
  check from token totals), and USD per query.

**Latency** (in ms):
- Retriever, router, reasoner, and end-to-end. Mean + p95 for each.

---

## Adding your own benchmark

```jsonl
{"query_id": "custom_001", "query_text": "…", "event_type": "vehicle_interaction", "relevant_indices": [101, 102], "total_relevant": 2}
{"query_id": "custom_002", "query_text": "…", "event_type": "ambient_scene",     "relevant_indices": [9000, 9001, 9002], "total_relevant": 50}
```

```bash
python -m src.eval.run_benchmark --queries evals/my_queries.jsonl --mode dual_agent --corpus-size 36000
```

`--corpus-size` matters because it's the denominator of the
"frames-on-prem" privacy metric. Default is 36000 (10h @ 1fps).

---

## CLI reference

```
--mode {clip_only,dual_agent,claude_only_stub}   (required)
--queries PATH               JSONL benchmark file
--config PATH                Optional JSON config (CLI flags override values)
--output-dir DIR             Where artefacts are written
--seed INT                   RNG seed; deterministic stand-ins use this
--tau-high FLOAT             Router high-confidence threshold (default 0.32)
--tau-low FLOAT              Router low-confidence threshold (default 0.24)
--max-escalations INT        Max frames escalated per query (default 5)
--corpus-size INT            Total frames in the corpus (default 36000)
--retrieval-top-k INT        How many frames CLIP returns per query (default 20)
-v, --verbose                DEBUG-level logging
```

---

## Reproducibility

The harness seeds a single `random.Random(seed)` once at the start of
`BenchmarkRunner.run()`. Per-query RNGs are derived from that root plus
`query_id`, so adding or removing a query does not change the results
for other queries.

Real components (CLIP, Router, Claude) are passed in via the
`BenchmarkRunner(config, retriever=…, router=…, reasoner=…)` constructor;
when omitted, deterministic stand-ins keyed by the seed take over. Two
runs with the same `--seed`, `--mode`, and `--queries` will produce
identical aggregate JSON content (modulo the embedded timestamp).

---

## Generating presentation charts

Once you have at least one `aggregate_<mode>_<ts>.json` for each mode you
want to show, render slide-ready charts with:

```bash
python -m src.eval.visualize --results-dir evals/results --latest-per-mode
```

Outputs land in `evals/results/charts/`:

| File | Purpose |
| --- | --- |
| `quality.png`    | Recall@5 / F1@5 / MRR / nDCG@5 grouped by mode. |
| `privacy.png`    | Fraction of queries and unique frames kept on-prem. |
| `cost.png`       | USD per query, with the dual-agent vs. simulated-stub ratio called out in the subtitle. |
| `latency.png`    | End-to-end mean vs. reasoner mean latency per mode. |
| `scorecard.png`  | Normalised radar over quality / privacy / cost-efficiency / speed. |
| `wp6_presentation_summary.csv` | One row per mode with the same numbers as the charts, for the appendix slide. |

Every chart subtitle calls out that `claude_only_stub` is a simulated
upper-bound baseline rather than a live system. The chart code is
deterministic — same inputs produce the same PNGs.

Useful flags:

```
--results-dir DIR      Directory to scan for aggregate_*.json (default: evals/results)
--output-dir DIR       Where to write charts (default: <results-dir>/charts/)
--scale FLOAT          Plotly PNG export scale factor (default 2.0 → 2560×1440)
-v, --verbose          DEBUG-level logging
```

If a mode has no aggregate file in the results directory, the visualiser
logs a warning and continues. If *zero* modes resolve the CLI exits with
status 2.

---

## Tests

```bash
pytest tests/eval/ -v
```

`tests/eval/test_metrics.py` locks in the precise semantics of every
metric function. `tests/eval/test_harness.py` checks that runs are
deterministic, all four artefacts are written, and the
`claude_only_stub` privacy denominator is bounded.

---

## Related

- [`../src/eval/`](../src/eval/) — implementation.
- [`../docs/work_packages/wp6_eval_harness.md`](../docs/work_packages/wp6_eval_harness.md) — architecture & metric definitions in depth.
- [`../src/router/core.py`](../src/router/core.py) — the real `BudgetAwareRouter` whose thresholds the stand-in mirrors.
- [`../src/reasoner/claude_engine.py`](../src/reasoner/claude_engine.py) — the real Claude Haiku 4.5 client.
