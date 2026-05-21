# WP6 — Evaluation Harness

**Status:** ✅ Complete
**Submission Date:** 2026-05-21
**Git Tag:** `v0.6.0-wp6`

---

## Objectives

✅ Implement evaluation metrics (Recall@K, Precision@K, F1, MRR, nDCG)
✅ Build automated evaluation pipeline for benchmark queries
✅ Measure end-to-end system performance (CLIP + Claude Haiku 4.5 cascade)
✅ Log latency (mean, p95) and token cost per query
✅ Prioritize privacy metrics alongside accuracy
✅ Support three evaluation modes (clip_only, dual_agent, claude_only_stub)
✅ Save reproducible results in JSON/CSV for WP7 baseline comparisons

---

## Architecture

### Evaluation Modes

```
┌─────────────────────────────────────────────────────────────────┐
│ WP6 Evaluation Harness — Three Evaluation Modes                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ [1] clip_only                                                    │
│     CLIP retriever only, no reasoning or routing                │
│     Baseline for measuring CLIP performance ceiling             │
│     Expected: High latency (no filtering), zero cost            │
│                                                                  │
│ [2] dual_agent (MAIN SYSTEM)                                    │
│     CLIP → Router (τ_high=0.32, τ_low=0.24) → Claude Haiku     │
│     Confidence-gated escalation with token-cost optimization    │
│     Expected: Balanced accuracy, low cost, ~99% on-prem        │
│                                                                  │
│ [3] claude_only_stub                                            │
│     Naive baseline: All frames to Claude (placeholder)          │
│     Upgraded in WP7 with real multi-frame processing           │
│     Expected: Highest accuracy, prohibitive cost                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Query → CLIP Encoding → FAISS Search → Router Decision
           ↓               ↓              ↓
      [tracked]      [Top-K scores]  [Action]
           ↓               ↓              ↓
     clip_latency    clip_top_scores  IMMEDIATE_MATCH
           ms              @1,5,10      ESCALATED
                                        DROPPED
                           ↓
                       [Stage 2: Claude]
                       (if escalated)
                           ↓
                     reasoner_latency
                     input_tokens
                     output_tokens
                           ↓
                       QueryResult
                       (aggregated into
                        AggregateMetrics)
```

---

## Core Components

### 1. Data Models (`src/eval/models.py`)

**`EvaluationConfig`** — Reproducible run configuration
```python
@dataclass
class EvaluationConfig:
    seed: int = 42
    tau_high: float = 0.32
    tau_low: float = 0.24
    max_escalations: int = 5
    cost_per_image: float = 0.0003
    queries_file: str = "evals/queries.example.jsonl"
    output_dir: str = "evals/results"
    mode: str = "dual_agent"  # "clip_only", "dual_agent", "claude_only_stub"
```

**`QueryResult`** — Per-query metrics
- Input: query_id, query_text, event_type, expected_relevance_notes
- CLIP stage: clip_latency_ms, clip_top_1_score, clip_top_5_scores
- Router: router_decision, frames_escalated, estimated_cost_usd
- Claude stage: reasoner_latency_ms, input_tokens, output_tokens
- Total: total_latency_ms

**`PrivacyMetrics`** — Aggregated privacy preservation
```python
@dataclass
class PrivacyMetrics:
    total_queries: int
    queries_resolved_on_prem: int
    fraction_queries_on_prem: float  # % with zero cloud escalation
    total_frames_in_corpus: int
    frames_escalated_to_cloud: int
    fraction_frames_on_prem: float  # % of frames never sent to cloud
```

**`CostMetrics`** — Aggregated token economics
```python
@dataclass
class CostMetrics:
    total_input_tokens: int
    total_output_tokens: int
    total_estimated_cost_usd: float
    computed_cost_usd: float  # Validation: recomputed from tokens
```

**`LatencyMetrics`** — Aggregated latency statistics
- retriever: mean_ms, p95_ms
- reasoner: mean_ms, p95_ms (Claude)
- total: mean_ms, p95_ms

**`AggregateMetrics`** — Complete benchmark results
- Retrieval: R@1, R@5, R@10, P@5, P@10, F1@5, MRR, nDCG@5, nDCG@10
- Routing: queries_on_prem_only, queries_escalated, avg_frames_escalated
- Privacy: (PrivacyMetrics object)
- Cost: (CostMetrics object)
- Latency: (LatencyMetrics object)

### 2. Metric Functions (`src/eval/metrics.py`)

**Retrieval Metrics (Pure Functions)**
- `recall_at_k(relevant_ranks, k)` — Fraction of relevant items in top-K
- `precision_at_k(relevant_ranks, k)` — Fraction of top-K that are relevant
- `f1_at_k(relevant_ranks, k)` — Harmonic mean of precision and recall
- `mean_reciprocal_rank(relevant_ranks)` — 1 / rank of first relevant item
- `ndcg_at_k(scores, relevance_labels, k)` — Normalized Discounted Cumulative Gain

**Privacy Metrics (Pure Functions)**
- `fraction_queries_on_prem(query_results, corpus_size)` — % queries with zero escalation
- `fraction_frames_on_prem(query_results, corpus_size)` — % frames never sent to cloud

**Aggregation Helpers (Pure Functions)**
- `aggregate_retrieval_metrics(query_results, k_values)` — Mean metrics across queries
- `aggregate_latency_metrics(query_results)` — Mean and p95 latencies
- `aggregate_cost_metrics(query_results, prices)` — Total tokens and cost

### 3. Benchmark Executor (`src/eval/harness.py`)

**`BenchmarkEvaluator`** — Orchestrates benchmark execution
```python
class BenchmarkEvaluator:
    def evaluate_clip_only(queries, retriever, index) → List[QueryResult]
    def evaluate_dual_agent(queries, retriever, router, reasoner, index) → List[QueryResult]
    def evaluate_claude_only_stub(queries) → List[QueryResult]
    def run_benchmark(retriever, router, reasoner, index) → AggregateMetrics
```

Methods:
- `load_queries_from_jsonl(path)` — Load benchmark queries
- `run_benchmark(...)` — Execute benchmark in configured mode
- `_aggregate_results(corpus_size, num_queries)` → AggregateMetrics
- `_save_results(aggregate)` → JSON + CSV outputs

### 4. CLI Entry Point (`src/eval/__main__.py`)

```bash
python -m src.eval --queries evals/queries.example.jsonl --mode dual_agent
python -m src.eval --config config.json --mode clip_only
python -m src.eval --queries evals/queries.example.jsonl --mode dual_agent \
  --tau-high 0.35 --tau-low 0.25 --seed 123 --output-dir evals/results
```

Options:
```
--queries FILE              Path to JSONL queries file
--mode {clip_only, dual_agent, claude_only_stub}  Evaluation mode
--config FILE              Path to JSON config file
--output-dir DIR           Output directory for results (default: evals/results)
--seed INT                 Random seed for reproducibility (default: 42)
--tau-high FLOAT           High confidence threshold (default: 0.32)
--tau-low FLOAT            Low confidence threshold (default: 0.24)
--max-escalations INT      Max frames per query (default: 5)
-v, --verbose              Verbose logging
```

---

## Metric Definitions

### Retrieval Metrics (Information Retrieval Standard)

**Recall@K** (What fraction of true positives did we find?)
```
R@K = (# relevant items in top-K) / (# total relevant items)
Range: [0, 1]
Example: R@5 = 2/3 means we found 2 out of 3 relevant frames in top-5
```

**Precision@K** (What fraction of top-K results are correct?)
```
P@K = (# relevant items in top-K) / K
Range: [0, 1]
Example: P@5 = 2/5 means 2 out of 5 retrieved frames are relevant
```

**F1@K** (Harmonic mean: balance of precision and recall)
```
F1@K = 2 * (P@K * R@K) / (P@K + R@K)
Range: [0, 1]
Penalizes systems that are biased toward either precision or recall
```

**MRR** (Mean Reciprocal Rank — speed to first relevant result)
```
MRR = mean(1 / rank_of_first_relevant_item)
Range: [0, 1]
Example: If first relevant item is at rank 3, contributes 1/3 ≈ 0.33
Useful for: Assessing how quickly the user finds what they need
```

**nDCG@K** (Normalized Discounted Cumulative Gain — ranking quality)
```
nDCG@K = DCG@K / IDCG@K
where DCG = Σ(relevance_i / log2(i+1))
Range: [0, 1]
Measures ranking quality considering both position and relevance degree
Higher positions are weighted more; irrelevant items contribute 0
```

### Privacy Metrics (Novel to This Project)

**Fraction of Queries On-Premise**
```
= (# queries with zero cloud escalation) / (# total queries)
Range: [0, 1]
Target: ≥ 0.60 (60% of queries resolved by CLIP alone)
Measurement: frames_escalated == 0
```

**Fraction of Frames On-Premise**
```
= 1 - (# frames sent to cloud) / (# total frames in corpus)
Range: [0, 1]
Target: ≥ 0.99 (99% of video data never leaves enterprise)
Measurement: Based on cumulative frame escalation counts
Example: 1 - (50 escalated / 36000 corpus) ≈ 0.9986 (99.86% on-prem)
```

### Cost Metrics (Token Economics)

**Total Input Tokens**
- Sum of tokens consumed across all Claude calls
- Priced at $1.00 per 1M tokens (Claude Haiku 4.5)

**Total Output Tokens**
- Sum of tokens generated by Claude
- Priced at $5.00 per 1M tokens

**Cost per Query (USD)**
```
= (total_input_tokens / 1M * $1.00 + total_output_tokens / 1M * $5.00) / num_queries
Target: < $0.01 per query-hour (dual-agent) vs. ~$0.38 per query-hour (claude-only)
```

### Latency Metrics (Milliseconds)

**CLIP Latency**
- Local text encoding + FAISS search (on-premise)
- Mean and p95 (95th percentile)

**Claude Latency**
- API call time for reasoning (cloud, only for escalated queries)
- Mean and p95 across all queries (including zeros for non-escalated)

**Total Latency**
- End-to-end time from query submission to result
- Mean: ~800ms (weighted by escalation rate)
- p95: < 3000ms (target for interactive workflows)

---

## Result Output Format

### JSON Results (`evals/results/aggregate_<mode>_<timestamp>.json`)

```json
{
  "metadata": {
    "mode": "dual_agent",
    "timestamp": "2026-05-21T14:30:00.123456",
    "total_queries": 5
  },
  "retrieval_metrics": {
    "recall_at_1": 0.80,
    "recall_at_5": 0.95,
    "recall_at_10": 0.98,
    "precision_at_5": 0.90,
    "precision_at_10": 0.88,
    "f1_at_5": 0.92,
    "mrr": 0.65,
    "ndcg_at_5": 0.88,
    "ndcg_at_10": 0.90
  },
  "routing_metrics": {
    "queries_on_prem_only": 3,
    "queries_escalated": 2,
    "avg_frames_escalated_per_query": 2.0
  },
  "privacy_metrics": {
    "total_queries": 5,
    "queries_resolved_on_prem": 3,
    "fraction_queries_on_prem": 0.60,
    "total_frames_in_corpus": 36000,
    "frames_escalated_to_cloud": 10,
    "fraction_frames_on_prem": 0.9997
  },
  "cost_metrics": {
    "total_input_tokens": 5000,
    "total_output_tokens": 1000,
    "total_estimated_cost_usd": 0.0055,
    "computed_cost_usd": 0.0055,
    "pricing": {
      "input_tokens_per_1m": 1.00,
      "output_tokens_per_1m": 5.00
    }
  },
  "latency_metrics": {
    "retriever": {
      "mean_ms": 50,
      "p95_ms": 65
    },
    "reasoner": {
      "mean_ms": 20,
      "p95_ms": 85
    },
    "total": {
      "mean_ms": 70,
      "p95_ms": 150
    }
  }
}
```

### Detailed Query Results (`evals/results/queries_<mode>_<timestamp>.json`)

```json
[
  {
    "query_id": "veh_001",
    "query_text": "white SUV cutting off truck in left lane",
    "event_type": "vehicle_interaction",
    "expected_relevance_notes": "...",
    "clip_latency_ms": 52.3,
    "clip_top_k": 20,
    "clip_top_1_score": 0.78,
    "clip_top_5_scores": [0.78, 0.75, 0.72, 0.68, 0.65],
    "router_decision": "ESCALATED",
    "frames_escalated": 5,
    "estimated_cost_usd": 0.0015,
    "reasoner_latency_ms": 145.2,
    "reasoner_called": true,
    "input_tokens": 1200,
    "output_tokens": 150,
    "total_latency_ms": 197.5,
    "ground_truth_relevant": null
  }
]
```

---

## How to Run

### Quick Start

```bash
# Run example benchmark in dual-agent mode
python -m src.eval --queries evals/queries.example.jsonl --mode dual_agent

# Run CLIP-only baseline
python -m src.eval --queries evals/queries.example.jsonl --mode clip_only

# Run with custom config
python -m src.eval --config my_config.json --mode dual_agent

# Verbose logging
python -m src.eval --queries evals/queries.example.jsonl --mode dual_agent -v
```

### Reproducing Results

```bash
# Same seed ensures deterministic results
python -m src.eval --queries evals/queries.example.jsonl --mode dual_agent \
  --seed 42 --tau-high 0.32 --tau-low 0.24

# Results saved to: evals/results/aggregate_dual_agent_<timestamp>.json
```

### Creating Custom Config

```bash
# Save config to JSON
cat > eval_config.json <<EOF
{
  "seed": 42,
  "tau_high": 0.32,
  "tau_low": 0.24,
  "max_escalations": 5,
  "cost_per_image": 0.0003,
  "queries_file": "evals/queries.example.jsonl",
  "output_dir": "evals/results",
  "mode": "dual_agent"
}
EOF

# Use config
python -m src.eval --config eval_config.json --mode dual_agent
```

---

## Interpretation Guide

### Success Criteria (WP1 Defense)

| Metric | Target | Interpretation |
|--------|--------|-----------------|
| **Recall@5** | ≥ 0.80 | ≥80% of relevant frames in top-5 |
| **F1@5** | ≥ 0.78 | Balanced precision-recall performance |
| **Privacy (on-prem)** | ≥ 0.99 | ≥99% of frames never leave enterprise |
| **Cost** | < $0.01/query | <1¢ per query vs. $0.38 naive baseline |
| **Latency (p95)** | < 3000ms | Sub-3-second response for interactive use |

### WP7 Baseline Comparison Template

When comparing three modes across metrics:

```
Metric              | CLIP-Only | Dual-Agent | Claude-Only
==================|===========|============|===========
Recall@5          |   0.65    |   0.85     |   0.92
Privacy (on-prem) |   1.00    |   0.99     |   0.00
Cost (per query)  |  $0.00    |   $0.005   |   $0.38
Latency (p95)     |   80ms    |   250ms    |  2500ms
```

Interpretation:
- **CLIP-Only:** Fast but inaccurate (65% recall), zero cost
- **Dual-Agent:** Best trade-off (85% recall, 99% privacy, $0.005/query, 250ms)
- **Claude-Only:** Best accuracy but prohibitive cost/latency/privacy

---

## Integration with WP5 (Existing Components)

Evaluation harness reuses:
- `src.retriever.clip_engine.CLIPEngine` — Text/image embedding
- `src.router.core.BudgetAwareRouter` — Confidence-gated routing
- `src.reasoner.claude_engine.ClaudeReasoner` — Claude Haiku 4.5 verification
- `src.utils.metrics.TokenEconomics` — Cost tracking

No modifications to WP5 interfaces required.

---

## Integration with WP7 (Baseline Comparisons)

WP7 will extend evaluation with:
1. Ground truth annotation loader (real relevance labels)
2. Real Claude-only baseline implementation (replaces stub)
3. Comparative result aggregation (all three modes in one table)
4. Statistical significance testing (if applicable)

Current WP6 design supports this cleanly via:
- `evaluate_claude_only_stub()` → upgraded in WP7
- `relevant_ranks` field in QueryResult → populated in WP7 with ground truth
- `mode` parameter → easy to loop over all three baselines

---

## Testing

Unit tests for metric functions (example):

```python
# test_metrics.py
def test_recall_at_k():
    assert recall_at_k([0, 2, 5], k=5) == 2/3  # 2 relevant in top-5 out of 3 total
    assert recall_at_k([10, 20], k=5) == 0.0   # No relevant items in top-5

def test_ndcg_at_k():
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    labels = [1, 1, 0, 1, 0]  # Relevant, relevant, -, relevant, -
    ndcg = ndcg_at_k(scores, labels, k=5)
    assert 0 <= ndcg <= 1
```

Run tests:
```bash
pytest tests/eval/test_metrics.py -v
```

---

## Known Limitations (Documented for WP7)

1. **Ground Truth:** Currently mocked (uniform random relevance). WP7 will populate with real annotations.
2. **Claude-Only Baseline:** Currently a stub (placeholder). WP7 will implement real multi-frame processing.
3. **Latency:** Mocked with fixed sleep times. Real deployment will measure actual Anthropic API latency.
4. **Token Costs:** Assume Claude Haiku 4.5 pricing ($1/1M input, $5/1M output). Will validate against real API usage.

---

## Reference Links

- **Architecture:** docs/architecture.md
- **WP5 (Retriever & Router):** docs/work_packages/wp5_reasoner_router.md
- **Example Queries:** evals/queries.example.jsonl
- **Results:** evals/results/ (post-execution)

---

**Author:** Koby Lev  
**Last Updated:** 2026-05-21  
**Status:** ✅ Complete
