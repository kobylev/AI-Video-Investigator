# Evaluation Benchmark

## AI Video Investigator — Query Benchmark Design

This directory contains the evaluation benchmark for measuring system performance across retrieval, reasoning, and end-to-end metrics.

---

## Benchmark Specification

### Target Scale

- **Queries:** ≥100 natural-language queries
- **Video Corpus:** ≥10 hours of BDD100K dashcam footage
- **Ground Truth:** Manual annotation of relevant frame ranges per query (binary relevance)
- **Train/Val/Test Split:** 60% / 20% / 20%

### Query Taxonomy

All queries are categorized into **5 event types** to ensure diversity:

| Event Type | Description | Example Query | Target Count |
|------------|-------------|---------------|--------------|
| **Vehicle Interaction** | Collisions, near-misses, aggressive driving, lane changes | "white SUV cutting off truck in left lane" | 20+ |
| **Pedestrian Event** | Jaywalking, crossings, falls, near-misses | "pedestrian in red jacket running across street" | 20+ |
| **Object-of-Interest** | Specific vehicle (color, type, plate), clothing, road sign | "blue sedan with license plate starting with 'ABC'" | 20+ |
| **Traffic Violation** | Red light running, stop sign violations, illegal turns | "red car running red light at intersection" | 20+ |
| **Ambient Scene** | Weather, time-of-day, location type, traffic density | "rainy nighttime highway with heavy traffic" | 20+ |

**Total:** 100+ queries distributed evenly across event types (validated via chi-squared test).

---

## File Formats

### `queries.example.jsonl`

5 example queries demonstrating the schema. Each line is a JSON object:

```json
{
  "query_id": "veh_001",
  "query_text": "red car running red light at intersection",
  "event_type": "traffic_violation",
  "expected_relevance_notes": "Should match frames showing red vehicles crossing intersection threshold while signal is red"
}
```

**Fields:**
- `query_id` (str): Unique identifier (format: `{event_type_abbrev}_{number}`)
- `query_text` (str): Natural-language query
- `event_type` (str): One of 5 event types (lowercase, underscores)
- `expected_relevance_notes` (str): Guidance for annotators on what constitutes a relevant frame

### `ground_truth.jsonl` (WP3 Deliverable)

Full benchmark with manually annotated ground truth. Each line:

```json
{
  "query_id": "veh_001",
  "query_text": "red car running red light at intersection",
  "event_type": "traffic_violation",
  "video_id": "bdd100k_video_0042",
  "relevant_frame_ids": [1523, 1524, 1525, 1526],
  "relevant_frame_ranges": [[1523, 1526]],
  "annotation_confidence": "high",
  "annotator_notes": "Clear red light violation at timestamp 00:25:23"
}
```

### `predictions.jsonl` (System Output)

Model predictions for evaluation. Each line:

```json
{
  "query_id": "veh_001",
  "ranked_frame_ids": [1524, 1525, 3201, 1523, 7890],
  "scores": [0.92, 0.88, 0.76, 0.74, 0.68],
  "latency_ms": 2634,
  "tokens_consumed": 8160,
  "cost_usd": 0.0102,
  "router_decision": "escalate"
}
```

**Fields:**
- `ranked_frame_ids` (list[int]): Top-K frame IDs (typically K=5 for final output)
- `scores` (list[float]): Relevance scores (0.0–1.0)
- `latency_ms` (int): End-to-end wall-clock time
- `tokens_consumed` (int): Gemini API tokens (0 if CLIP-only)
- `cost_usd` (float): API cost in USD
- `router_decision` (str): "skip" | "expand" | "escalate"

---

## Evaluation Metrics

### Retrieval Metrics (CLIP Stage)

- **Recall@K:** Was at least one relevant frame in the top-K?
- **Mean Reciprocal Rank (MRR):** Average of 1/rank of first relevant frame
- **Normalized Discounted Cumulative Gain (nDCG):** Rank-weighted relevance score

### Classification Metrics (Gemini Stage)

- **Precision:** Fraction of returned frames that are relevant
- **Recall:** Fraction of relevant frames that were returned
- **F1 Score:** Harmonic mean of precision and recall
- **Accuracy:** Binary relevance correctness

### End-to-End Metrics (Pipeline)

- **Recall@5:** Was at least one relevant frame in the top-5? (primary metric)
- **Top-5 F1:** Harmonic mean of precision and recall at K=5

### System Metrics (Architecture)

- **Latency (p50, p95, p99):** Distribution of query latencies
- **Tokens per query:** Mean and std of Gemini token consumption
- **Cost per query:** Mean cost in USD
- **Router decision distribution:** Percentage of queries that were skipped / expanded / escalated

---

## Benchmark Construction Workflow (WP3)

1. **Select Videos:** Curate 10+ hours of BDD100K footage with diverse conditions
2. **Author Queries:** Use template-driven approach (20 queries × 5 event types)
3. **Extract Frames:** Sample videos at 1 fps, store frame IDs with timestamps
4. **Annotate Ground Truth:** Manually label relevant frame ranges for each query
5. **Validate Diversity:** Measure lexical diversity, check event-type distribution
6. **Split Dataset:** 60% train, 20% val, 20% test (stratified by event type)
7. **Commit Benchmark:** Push `ground_truth.jsonl` to `evals/`

---

## Results Directory

All evaluation results are stored in `evals/results/` with timestamped filenames:

- `wp4_clip_baseline_YYYYMMDD.json` — CLIP-only baseline
- `wp6_dual_agent_YYYYMMDD.json` — Dual-agent system
- `wp7_gemini_only_YYYYMMDD.json` — Gemini-only baseline
- `wp8_ablation_no_router_YYYYMMDD.json` — Ablation studies

**Format:** Each result file contains:
- System configuration (K, τ_high, τ_low, model versions)
- Aggregate metrics (mean, std, p50, p95 for all metrics)
- Per-query predictions (linked JSONL)
- Failure analysis (false positives, false negatives)

---

## Usage Example

```python
from src.eval import EvaluationHarness

# Load benchmark
harness = EvaluationHarness(
    ground_truth="evals/ground_truth.jsonl",
    predictions="evals/results/wp6_dual_agent_20260520.jsonl"
)

# Compute all metrics
metrics = harness.compute_all_metrics()

# Display summary
print(f"Recall@5: {metrics['recall_at_5']:.3f}")
print(f"Top-5 F1: {metrics['top5_f1']:.3f}")
print(f"Mean Latency (p95): {metrics['latency_p95']:.2f}s")
print(f"Mean Cost: ${metrics['cost_mean']:.4f}")

# Export to Markdown table
harness.export_markdown("evals/results/wp6_summary.md")
```

---

**Maintained by:** Koby Lev | **Last Updated:** 2026-05-16
