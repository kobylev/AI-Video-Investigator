# Evaluation Harness — AI Video Investigator

This directory contains the benchmarks, ground truth, and results for the AI Video Investigator project.

---

## Benchmark Structure

- `queries.example.jsonl`: Example semantic queries.
- `results/`: Directory for experiment output logs.

---

## Success Metrics

### Retrieval Stage (CLIP)
- **Recall@K (R@1, R@5, R@10)**
- **MRR (Mean Reciprocal Rank)**

### Reasoning Stage (**Claude Haiku 4.5**)
- **Precision / Recall / F1-Score**
- **Accuracy**
- **Token Efficiency:** Tokens consumed per query-hour.

### Baseline Definitions
1. **CLIP-only:** No reasoning agent.
2. **Claude-only:** All frames analyzed by **Claude Haiku 4.5** (naive baseline).
3. **Dual-agent (Proposed):** Confidence-gated routing to **Claude**.

---
**Last Updated:** 2026-05-21
