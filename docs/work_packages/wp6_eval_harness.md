# WP6 — Evaluation Harness

**Status:** Implemented. Stand-in components mean reported numbers are
illustrative until WP7 wires the harness to live CLIP / Router / Claude
instances and a real annotated corpus.

**Last revised:** 2026-05-21

---

## 1. Objective

Build a reproducible evaluation framework that measures the
production pipeline (CLIP retriever → BudgetAwareRouter → Claude Haiku 4.5
reasoner) along five orthogonal dimensions:

1. **Retrieval quality** — Recall@K, Precision@K, F1@5, MRR, nDCG@K.
2. **Routing behaviour** — decision-count breakdown, frames escalated.
3. **Privacy preservation** — fraction of queries / frames kept on-prem.
4. **Latency** — retriever, router, reasoner, end-to-end (mean + p95).
5. **Cloud cost** — token usage and estimated USD against Claude Haiku 4.5 list pricing.

The harness is mode-pluggable so the same machinery scores the
`clip_only`, `dual_agent`, and `claude_only_stub` baselines that WP7 will
compare. Privacy metrics are first-class outputs, not optional extras.

---

## 2. Public surface

```
src/eval/
├── __init__.py          # Re-exports the public API.
├── __main__.py          # Allows `python -m src.eval`.
├── models.py            # Dataclasses (EvaluationSample, QueryResult, AggregateMetrics, …).
├── metrics.py           # Pure functions (recall_at_k, ndcg_at_k, percentile, …).
├── io.py                # JSONL loading + JSON/CSV result writing.
├── modes.py             # Per-mode adapters (clip_only / dual_agent / claude_only_stub).
├── harness.py           # BenchmarkRunner orchestration + aggregation.
├── plots.py             # Plotly figure builders (one per chart).
├── visualize.py         # CLI that renders the latest-per-mode chart bundle.
└── run_benchmark.py     # CLI entry point.
```

`BenchmarkRunner(config).run()` is the single library entry point.
`python -m src.eval.run_benchmark …` is the CLI. Both call into the same
`main()`.

### Dataclasses

| Class | Role |
| --- | --- |
| `EvaluationSample` | A row from the JSONL benchmark. Frozen. Owns `relevant_indices`, `total_relevant`, `event_type`. |
| `EvaluationConfig` | Reproducible run configuration. Persisted alongside every result set. |
| `QueryResult` | Full per-query record — final ranking, stage latencies, router decision, token usage, ground truth pinned in for self-describing output. |
| `RetrievalMetrics` | Mean R@K / P@K / F1 / MRR / nDCG across all queries. |
| `RoutingMetrics` | Counts of `IMMEDIATE_MATCH`, `ESCALATED`, `DROPPED`, `CLIP_ONLY`, `CLAUDE_ONLY`, plus average frames escalated. |
| `PrivacyMetrics` | Queries on-prem, unique frames escalated, fraction frames on-prem. Caps `unique_frames_escalated` at `corpus_size` so the fraction never goes negative. |
| `CostMetrics` | Input/output tokens, recorded USD, recomputed USD as a sanity check. |
| `LatencyMetrics` | Mean and p95 for retriever / router / reasoner / total. |
| `AggregateMetrics` | Top-level container; the shape of `aggregate_<mode>_<ts>.json`. |

### Metric functions (pure, deterministic, unit-tested)

```python
from src.eval import metrics as M
M.recall_at_k(relevant_set, ranked_indices, k, total_relevant=0)
M.precision_at_k(relevant_set, ranked_indices, k)
M.f1_at_k(relevant_set, ranked_indices, k, total_relevant=0)
M.reciprocal_rank(relevant_set, ranked_indices)
M.ndcg_at_k(relevant_set, ranked_indices, k)
M.percentile(values, pct)
M.mean(values)
M.aggregate_retrieval(per_query_metrics)
```

All eight live in [src/eval/metrics.py](../../src/eval/metrics.py) and
have line-level test coverage in
[tests/eval/test_metrics.py](../../tests/eval/test_metrics.py).

---

## 3. Modes

```
┌──────────────────────────────────────────────────────────────────┐
│ clip_only                                                        │
│   Retriever only. No router, no Claude. 100% on-prem, $0 cost.  │
│   Sets the floor for retrieval quality.                         │
│                                                                  │
│ dual_agent  (production system)                                  │
│   CLIP → BudgetAwareRouter(τ_high=0.32, τ_low=0.24) → Claude.   │
│   IMMEDIATE_MATCH: top-1 score ≥ τ_high — surface CLIP order.   │
│   ESCALATED:       τ_low ≤ top-1 < τ_high — Claude re-ranks.   │
│   DROPPED:         top-1 < τ_low — nothing returned.            │
│                                                                  │
│ claude_only_stub  (naive baseline, placeholder for WP7)         │
│   Conceptually ships the entire corpus to Claude per query.     │
│   Idealised ranking → highest quality ceiling, $0.08/query,     │
│   0% on-prem. Motivates the cascade.                            │
└──────────────────────────────────────────────────────────────────┘
```

Mode dispatch lives in [src/eval/harness.py](../../src/eval/harness.py)
`BenchmarkRunner._execute_one`. The mode adapters in `modes.py` accept
optional `retriever` / `router` / `reasoner` components — when `None`,
deterministic stand-ins keyed by the run seed take over so the harness
runs in CI without GPUs or API keys. WP7 will pass real instances:

```python
from src.retriever.clip_engine import CLIPEngine
from src.router.core import BudgetAwareRouter
from src.reasoner.claude_engine import ClaudeReasoner

runner = BenchmarkRunner(
    config,
    retriever=CLIPEngine(...),
    router=BudgetAwareRouter(tau_high=0.32, tau_low=0.24),
    reasoner=ClaudeReasoner(...),
)
```

---

## 4. Metric definitions

### 4.1 Retrieval quality

All metrics operate on *frame indices*, against the `relevant_indices`
set declared in the JSONL.

| Metric | Definition |
| --- | --- |
| Recall@K | `(# relevant indices in top-K) / total_relevant`. `total_relevant` falls back to `len(relevant_indices)` when omitted from the JSONL. |
| Precision@K | `(# relevant indices in top-K) / K`. |
| F1@K | Harmonic mean of P@K and R@K. F1@5 reported by default. |
| MRR | `1 / (1-indexed rank of first relevant hit)`. 0.0 when no hit. |
| nDCG@K | `DCG@K / IDCG@K`, binary relevance: `DCG@K = Σ rel_i / log₂(i+2)` for i in [0, K), `IDCG@K = Σ 1 / log₂(i+2)` for i in [0, min(\|relevant\|, K)). |

`total_relevant` matters: ambient-scene queries typically have far more
relevant frames than an annotator enumerates. Setting `total_relevant: 60`
when only 12 are listed scales Recall down honestly instead of inflating
it.

### 4.2 Privacy

- `fraction_queries_on_prem = (# queries with frames_escalated == 0) / total_queries`.
- `fraction_frames_on_prem = 1 - min(unique_frames_escalated, corpus_size) / corpus_size`.

The `unique_frames_escalated` denominator deliberately counts the
**union** of frames sent to Claude across queries — sending the same
frame to Claude for two different queries still only leaks one frame.
Without this, escalating the same hot frame for ten queries would
falsely tank the privacy score.

For `claude_only_stub`, every corpus frame is conceptually shipped, so
`unique_frames_escalated` is set to `corpus_size` and the fraction is
exactly 0.

### 4.3 Cost

```
total_estimated_cost_usd = Σ per-query estimates recorded during the run
computed_cost_usd        = (input_tokens / 1M) * $1.00 + (output_tokens / 1M) * $5.00
cost_per_query           = computed_cost_usd / total_queries
```

`computed_cost_usd` is recomputed from token totals as a sanity check —
they should agree to within floating-point noise. Pricing is
configurable via `EvaluationConfig.price_per_1m_{input,output}_tokens`.

### 4.4 Latency

For each stage and end-to-end:
- `mean_ms = sum(values) / len(values)`
- `p95_ms` = `percentile(values, 95)` using inclusive-rank percentile
  (equivalent to NumPy `interpolation='lower'`; deterministic, no NumPy
  dependency).

---

## 5. Output format

Every run produces four files, all timestamped with
`YYYYMMDD_HHMMSS`:

```
evals/results/aggregate_<mode>_<ts>.json
evals/results/queries_<mode>_<ts>.json
evals/results/queries_<mode>_<ts>.csv
evals/results/config_<mode>_<ts>.json
```

### `aggregate_<mode>_<ts>.json`

```json
{
  "metadata": {"mode": "dual_agent", "timestamp": "...", "total_queries": 5},
  "retrieval_metrics": {"recall_at_1": ..., "recall_at_5": ..., ...},
  "routing_metrics":   {"immediate_match": 3, "escalated": 2, "dropped": 0,
                        "clip_only": 0, "claude_only": 0,
                        "avg_frames_escalated_per_query": 2.0},
  "privacy_metrics":   {"total_queries": 5,
                        "queries_resolved_on_prem": 3,
                        "fraction_queries_on_prem": 0.6,
                        "total_frames_in_corpus": 36000,
                        "unique_frames_escalated": 10,
                        "total_frame_escalations": 10,
                        "fraction_frames_on_prem": 0.9997},
  "cost_metrics":      {"total_input_tokens": 3200,
                        "total_output_tokens": 540,
                        "total_estimated_cost_usd": 0.0059,
                        "computed_cost_usd": 0.0059,
                        "pricing": {"input_per_1m_usd": 1.0,
                                    "output_per_1m_usd": 5.0}},
  "latency_metrics":   {"retriever": {"mean_ms": ..., "p95_ms": ...},
                        "router":    {"mean_ms": ..., "p95_ms": ...},
                        "reasoner":  {"mean_ms": ..., "p95_ms": ...},
                        "total":     {"mean_ms": ..., "p95_ms": ...}},
  "config":            { ...full EvaluationConfig... }
}
```

### `queries_<mode>_<ts>.csv`

Stable column order — append-only contract for downstream tooling:

```
query_id, query_text, event_type,
router_decision, frames_escalated, reasoner_called,
input_tokens, output_tokens, estimated_cost_usd,
retriever_latency_ms, router_latency_ms, reasoner_latency_ms, total_latency_ms,
clip_top_1_score, total_relevant, num_relevant_indices,
ranked_indices, relevant_indices
```

`ranked_indices` and `relevant_indices` are JSON-encoded list literals so
each row stays a single CSV cell.

---

## 6. CLI

```
python -m src.eval.run_benchmark \
    --queries evals/queries.example.jsonl \
    --mode dual_agent \
    --seed 42 \
    --tau-high 0.32 --tau-low 0.24 \
    --corpus-size 36000 \
    --output-dir evals/results
```

```
--mode {clip_only,dual_agent,claude_only_stub}   (required)
--queries PATH               JSONL benchmark file
--config PATH                Optional JSON config (CLI flags override its values)
--output-dir DIR             Where to write artefacts
--seed INT                   RNG seed for deterministic stand-ins
--tau-high FLOAT             Router high-confidence threshold (default 0.32)
--tau-low FLOAT              Router low-confidence threshold  (default 0.24)
--max-escalations INT        Max frames escalated per query (default 5)
--corpus-size INT            Total frames in corpus (default 36000)
--retrieval-top-k INT        Top-K frames CLIP returns (default 20)
-v, --verbose                DEBUG-level logging
```

`python -m src.eval` is also accepted (it forwards to the same entry
point).

---

## 7. Reproducibility model

- `BenchmarkRunner.run()` instantiates `random.Random(config.seed)` at
  the top; every per-query RNG is derived from `(root.random(), query_id, …)`.
- Adding or removing a query does not change results for other queries.
- Same `--seed` + `--queries` + `--mode` ⇒ byte-identical JSON / CSV
  content (the embedded timestamp is the only varying field).
- The `config` block inside `aggregate_*.json` records every parameter
  the run used, so any artefact alone is enough to re-run it.

---

## 7a. Presentation charts

For the academic defence slides, `src/eval/visualize.py` renders the
latest per-mode aggregate into five PNGs plus a one-row-per-mode CSV.

```
python -m src.eval.visualize --results-dir evals/results --latest-per-mode
```

Artefacts land in `evals/results/charts/`:

| File | What it shows |
| --- | --- |
| `quality.png`  | Recall@5 / F1@5 / MRR / nDCG@5 grouped by mode. |
| `privacy.png`  | Queries-on-prem and unique-frames-on-prem fractions by mode. |
| `cost.png`     | USD per query with the cascade-vs-stub ratio called out. |
| `latency.png`  | End-to-end mean vs. reasoner mean latency. |
| `scorecard.png`| Normalised 4-axis radar over quality / privacy / cost-efficiency / speed. |
| `wp6_presentation_summary.csv` | Same numbers as the charts, one row per mode. |

Design rules baked into `plots.py`:

- White background, `simple_white` Plotly template, 1280×720 layout
  exported at 2× scale (2560×1440) for projector / 4K displays.
- Consistent mode colours: CLIP-only = blue (`#3B82F6`), Dual-Agent =
  green (`#10B981`), Claude-only stub = neutral grey (`#9CA3AF`).
- Every chart subtitle explicitly labels `claude_only_stub` as a
  *simulated upper-bound baseline — not a live system*.
- Retrieval metrics are labelled "ranking / retrieval metrics" — they
  measure how the system orders frames, not final-answer correctness.
- Subtitles are baked into the title via HTML `<br>`s rather than
  paper-coordinate annotations, which Plotly clips on polar / log
  layouts.

Missing modes are skipped with a warning; the CLI exits with status 2
only if no aggregate files resolve at all.

---

## 8. Tests

```
tests/eval/test_metrics.py   34 assertions across recall/precision/F1/MRR/nDCG/percentile/mean/aggregate
tests/eval/test_harness.py   End-to-end smoke tests across all three modes
```

```
pytest tests/eval/ -v
```

`test_harness.py::test_claude_only_stub_caps_privacy_denominator`
specifically guards against the regression where the
`fraction_frames_on_prem` calculation could go negative when
`frames_escalated` was summed without bounding.

---

## 9. Integration with WP5

Reuses (no modifications required):
- [src/retriever/clip_engine.py](../../src/retriever/clip_engine.py) — `CLIPEngine` for embeddings + FAISS search.
- [src/router/core.py](../../src/router/core.py) — `BudgetAwareRouter`; the harness mirrors its `tau_high` / `tau_low` defaults exactly.
- [src/reasoner/claude_engine.py](../../src/reasoner/claude_engine.py) — `ClaudeReasoner` (Claude Haiku 4.5 via the Anthropic SDK).

When called with `retriever=None, router=None, reasoner=None`, the
harness substitutes deterministic stand-ins so it runs in CI on a
developer laptop without GPUs or API keys. The shape of the outputs is
identical.

---

## 10. Hand-off to WP7

WP6 was designed so WP7 can extend it without rewrites:

1. **Real Claude-only baseline.** Replace `modes.run_claude_only_stub`
   with an implementation that actually drives `ClaudeReasoner` over the
   full corpus. Keep the signature, keep `router_decision = "CLAUDE_ONLY"`
   so the privacy-cap logic in `harness._aggregate` still applies.
2. **Real ground truth.** The current example JSONL has illustrative
   `relevant_indices`. WP7 plugs in annotator output.
3. **Three-way comparison.** Loop over `(clip_only, dual_agent,
   claude_only)` with the same query file and seed; the artefact format
   already supports side-by-side reads.
4. **Statistical significance.** If WP7 wants confidence intervals,
   bootstrap over the per-query JSON (`queries_<mode>_<ts>.json`) —
   no harness changes required.

---

## 11. Known limitations

- The stand-in components in `modes.py` are deterministic mocks — the
  reported numbers in `evals/results/` are illustrative until WP7 wires
  real components in.
- `total_relevant` ground-truth values are estimates in the example
  benchmark. WP7 will use annotator-determined counts.
- Token cost assumes Claude Haiku 4.5 list pricing
  ($1.00/1M input, $5.00/1M output). Override via `EvaluationConfig` for
  experimentation.

---

## 12. References

- Architecture overview: [docs/architecture.md](../architecture.md)
- WP5 (Retriever & Router): [docs/work_packages/wp5_reasoner_router.md](wp5_reasoner_router.md)
- WP7 (Baselines): [docs/work_packages/wp7_baselines.md](wp7_baselines.md)
- Example benchmark: [evals/queries.example.jsonl](../../evals/queries.example.jsonl)
- Results directory: [evals/results/](../../evals/results/)
