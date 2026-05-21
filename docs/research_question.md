# Research Question

## AI Video Investigator — Measurable Hypothesis

This document formalizes the research question guiding the AI Video Investigator project, ensuring falsifiability, quantifiability, and clear baseline comparisons.

---

## Primary Research Question

> **To what extent does a dual-agent CLIP→Claude routing architecture improve top-5 retrieval F1 and reduce inference token cost on long-form dashcam footage, compared to (a) a CLIP-only retrieval baseline and (b) a Claude-only frame-analysis baseline, evaluated on a benchmark of ≥100 natural-language queries over ≥10 hours of curated video?**

---

## Decomposition

### Independent Variable

**System Architecture:**
1. **Dual-Agent (This Work):** CLIP retrieval → Confidence-gated router → **Claude** re-ranking
2. **Baseline A (CLIP-Only):** CLIP retrieval → Top-K results (no reasoning)
3. **Baseline B (Claude-Only):** All frames sent to **Claude Haiku 4.5** for ranking

### Dependent Variables (Measured Outcomes)

1. **Top-5 Retrieval F1** 
   - **Target:** ≥ 0.80
   - **Claude-Only Baseline:** ~0.85 (estimated upper bound)

2. **Inference Token Cost**
   - **Target:** >90% reduction vs. naive baseline
   - **Claude-Only Baseline:** ~$0.30 per query-hour (brute force estimate)

### Controlled Variables

- **Claude Model:** `claude-haiku-4-5-20251001` (same API version across all conditions)

---

## Hypotheses

### H1: Accuracy Hypothesis

**Claim:** The dual-agent system will achieve **top-5 F1 ≥ 0.80**, outperforming CLIP-only by **≥15 accuracy points**.

**Rationale:**
- CLIP provides high recall.
- **Claude** provides high precision (correctly re-ranks ambiguous cases).

---

### H2: Cost Efficiency Hypothesis

**Claim:** The dual-agent system will reduce token consumption by **>90%** compared to **Claude-only**, while maintaining accuracy within 5 F1 points of **Claude-only**.

---

### H3: Latency Hypothesis

**Claim:** The dual-agent system will achieve **p95 end-to-end latency < 3 seconds**.

**Rationale:**
- **Claude reasoning:** ~2.5s (parallelized tool-use calls).

---

## Success Criteria (Composite)

| Metric | Threshold | Comparison |
|--------|-----------|------------|
| Top-5 F1 | ≥ 0.80 | Absolute target |
| F1 Lift over CLIP-only | ≥ +15 points | Relative improvement |
| Token Reduction vs. **Claude-only** | > 90% | Cost efficiency |
| Latency (p95) | < 3 seconds | Usability threshold |
| F1 Gap vs. **Claude-only** | ≤ 5 points | Accuracy preservation |

---

**Last Updated:** 2026-05-21
**Status:** WP5 — Reasoner & Router Integration
