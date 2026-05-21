# Evaluation Harness — AI Video Investigator (WP6)

This directory contains benchmarks, ground truth, and results for the AI Video Investigator project.

---

## Quick Start

Run a benchmark evaluation:

```bash
# Dual-agent (main system)
python -m src.eval --queries evals/queries.example.jsonl --mode dual_agent

# CLIP-only baseline
python -m src.eval --queries evals/queries.example.jsonl --mode clip_only

# Claude-only naive baseline (placeholder for WP7)
python -m src.eval --queries evals/queries.example.jsonl --mode claude_only_stub

# Custom configuration
python -m src.eval --queries evals/queries.example.jsonl --mode dual_agent \
  --tau-high 0.35 --tau-low 0.25 --seed 42
```

Results are saved to `results/` as JSON files with timestamps.

---

## Directory Structure

```
evals/
├── queries.example.jsonl          # 5 example benchmark queries
├── results/                       # Generated results (JSON/CSV)
│   ├── aggregate_dual_agent_*.json
│   ├── queries_dual_agent_*.json
│   └── config_dual_agent_*.json
├── README.md                      # This file
```

---

## Benchmark Structure

### Query Format (`queries.example.jsonl`)

JSONL file with one query per line:

```json
{
  "query_id": "veh_001",
  "query_text": "white SUV cutting off truck in left lane",
  "event_type": "vehicle_interaction",
  "expected_relevance_notes": "Should match frames showing white SUV performing sudden lane change..."
}
```

Query fields:
- `query_id`: Unique identifier (e.g., "veh_001", "ped_001")
- `query_text`: Natural language query for semantic search
- `event_type`: Category (vehicle_interaction, pedestrian_event, object_of_interest, traffic_violation, ambient_scene)
- `expected_relevance_notes`: Guide for annotators on expected ground truth

### Event Types

1. **vehicle_interaction** — Traffic interactions (lane changes, following, merging)
2. **pedestrian_event** — Pedestrian activities (jaywalking, crosswalk behavior)
3. **object_of_interest** — Specific vehicles/objects by visual description
4. **traffic_violation** — Rule violations (red lights, speeding, wrong-way)
5. **ambient_scene** — Environmental/context queries (weather, time, traffic density)

---

## Evaluation Modes

### 1. CLIP-Only Baseline
- **Mode:** `clip_only`
- **What:** CLIP retriever only, no reasoning or routing
- **Expected:**
  - High recall (CLIP is comprehensive but not precise)
  - Zero cost (no Claude calls)
  - Fast latency (sub-100ms)

```bash
python -m src.eval --queries evals/queries.example.jsonl --mode clip_only
```

### 2. Dual-Agent (Main System)
- **Mode:** `dual_agent`
- **What:** CLIP → Router (τ_high=0.32, τ_low=0.24) → Claude Haiku 4.5
- **Expected:**
  - Balanced accuracy (CLIP + Claude re-ranking)
  - Low cost (40-60% queries skip Claude)
  - Privacy-first (99% on-premise)
  - ~250ms p95 latency

```bash
python -m src.eval --queries evals/queries.example.jsonl --mode dual_agent
```

### 3. Claude-Only Stub (Naive Baseline)
- **Mode:** `claude_only_stub`
- **What:** Placeholder for WP7 (will process all frames)
- **Expected:**
  - Best accuracy but prohibitive cost
  - Demonstrates cost-quality trade-off
  - Motivates the dual-agent approach

```bash
python -m src.eval --queries evals/queries.example.jsonl --mode claude_only_stub
```

---

## Success Metrics

### Retrieval Metrics (Information Retrieval Standard)
- **Recall@K (R@1, R@5, R@10)** — Fraction of relevant items found in top-K
- **Precision@K (P@5, P@10)** — Fraction of top-K results that are relevant
- **F1@K** — Harmonic mean of precision and recall
- **MRR (Mean Reciprocal Rank)** — Speed to first relevant result
- **nDCG@K** — Ranking quality considering position and relevance

### Privacy Metrics (Novel)
- **Fraction of Queries On-Premise** — % with zero cloud escalation (target: ≥60%)
- **Fraction of Frames On-Premise** — % of video never sent to cloud (target: ≥99%)

### Cost Metrics (Token Economics)
- **Total Input Tokens** — Accumulated Claude input tokens
- **Total Output Tokens** — Accumulated Claude output tokens
- **Cost per Query (USD)** — Average cost (target: <$0.01)

### Latency Metrics (Milliseconds)
- **CLIP Latency** — Local embedding + search (mean, p95)
- **Claude Latency** — Cloud reasoning (mean, p95)
- **Total Latency** — End-to-end (mean, p95, target: <3000ms)

---

## Output Format

### Aggregate Results (`results/aggregate_<mode>_<timestamp>.json`)

High-level summary metrics:

```json
{
  "metadata": {"mode": "dual_agent", "total_queries": 5},
  "retrieval_metrics": {
    "recall_at_1": 0.80,
    "recall_at_5": 0.95,
    "f1_at_5": 0.92,
    "mrr": 0.65
  },
  "privacy_metrics": {
    "fraction_queries_on_prem": 0.60,
    "fraction_frames_on_prem": 0.9997
  },
  "cost_metrics": {
    "total_cost_usd": 0.0055,
    "cost_per_query_usd": 0.0011
  },
  "latency_metrics": {
    "total_mean_ms": 200,
    "total_p95_ms": 450
  }
}
```

### Detailed Query Results (`results/queries_<mode>_<timestamp>.json`)

Per-query breakdown for debugging and analysis:

```json
[
  {
    "query_id": "veh_001",
    "query_text": "white SUV cutting off truck",
    "clip_latency_ms": 52,
    "router_decision": "ESCALATED",
    "frames_escalated": 5,
    "reasoner_latency_ms": 145,
    "input_tokens": 1200,
    "output_tokens": 150,
    "total_latency_ms": 197
  }
]
```

---

## Interpretation Guide

### Baseline Comparison (WP7)

```
Metric              | CLIP-Only | Dual-Agent | Claude-Only
==================|===========|============|===========
Recall@5          |   0.65    |   0.85     |   0.92
Privacy (on-prem) |   1.00    |   0.99     |   0.00
Cost (per query)  |  $0.00    |   $0.005   |   $0.38
Latency (p95)     |   80ms    |   250ms    |  2500ms
```

**Interpretation:**
- **CLIP-Only:** Cheap & fast but inaccurate (privacy-focused retrieval only)
- **Dual-Agent:** Best trade-off (accuracy + privacy + cost + latency)
- **Claude-Only:** Best accuracy but prohibitive for production

---

## Adding Custom Queries

Create a new JSONL file:

```json
{"query_id": "custom_001", "query_text": "my query", "event_type": "vehicle_interaction", "expected_relevance_notes": "..."}
{"query_id": "custom_002", "query_text": "another query", "event_type": "pedestrian_event", "expected_relevance_notes": "..."}
```

Run evaluation:

```bash
python -m src.eval --queries evals/custom_queries.jsonl --mode dual_agent
```

---

## Reproducibility

Ensure deterministic results by using the same seed:

```bash
# Run 1
python -m src.eval --queries evals/queries.example.jsonl --mode dual_agent --seed 42

# Run 2 (identical results)
python -m src.eval --queries evals/queries.example.jsonl --mode dual_agent --seed 42
```

---

## Documentation

- **Full WP6 documentation:** docs/work_packages/wp6_eval_harness.md
- **Metric definitions:** See WP6 doc (Recall@K, nDCG, privacy metrics)
- **Data models:** src/eval/models.py (EvaluationConfig, QueryResult, AggregateMetrics)
- **Metric functions:** src/eval/metrics.py (pure, testable functions)
- **Harness implementation:** src/eval/harness.py (BenchmarkEvaluator)

---

**Last Updated:** 2026-05-21  
**Status:** ✅ Complete (WP6)  
**Author:** Koby Lev
