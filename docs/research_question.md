# Research Question

## AI Video Investigator — Measurable Hypothesis

This document formalizes the research question guiding the AI Video Investigator project, ensuring falsifiability, quantifiability, and clear baseline comparisons.

---

## Primary Research Question

> **To what extent does a dual-agent CLIP→Gemini routing architecture improve top-5 retrieval F1 and reduce inference token cost on long-form dashcam footage, compared to (a) a CLIP-only retrieval baseline and (b) a Gemini-only frame-analysis baseline, evaluated on a benchmark of ≥100 natural-language queries over ≥10 hours of curated video?**

---

## Decomposition

### Independent Variable

**System Architecture:**
1. **Dual-Agent (This Work):** CLIP retrieval → Confidence-gated router → Gemini re-ranking
2. **Baseline A (CLIP-Only):** CLIP retrieval → Top-K results (no reasoning)
3. **Baseline B (Gemini-Only):** All frames sent to Gemini 1.5 Pro for ranking

### Dependent Variables (Measured Outcomes)

1. **Top-5 Retrieval F1** — Harmonic mean of precision and recall at K=5
   - **Target:** ≥ 0.80
   - **CLIP-Only Baseline:** ~0.65 (estimated)
   - **Gemini-Only Baseline:** ~0.85 (estimated, upper bound)

2. **Inference Token Cost** — Total tokens consumed per query
   - **Target:** < 10,000 tokens/query (< $0.0125 per query at $1.25/1M tokens)
   - **CLIP-Only Baseline:** ~0 tokens (embedding-only)
   - **Gemini-Only Baseline:** ~930,000 tokens/query-hour (~$1.16 per query-hour)

### Controlled Variables

- **Dataset:** BDD100K dashcam footage (fixed distribution)
- **Query Benchmark:** 100+ natural-language queries, manually annotated ground truth
- **CLIP Model:** ViT-L/14 (frozen, off-the-shelf)
- **Gemini Model:** `gemini-1.5-pro-latest` (same API version across all conditions)
- **Evaluation Protocol:** Same metrics (R@K, MRR, nDCG, F1, Accuracy) applied to all systems

---

## Hypotheses

### H1: Accuracy Hypothesis

**Claim:** The dual-agent system will achieve **top-5 F1 ≥ 0.80**, outperforming CLIP-only by **≥15 accuracy points**.

**Rationale:**
- CLIP provides high recall (gets relevant frames into top-K)
- Gemini provides high precision (correctly re-ranks ambiguous cases)
- Confidence gating preserves easy cases (high CLIP similarity) while escalating hard cases

**Falsifiability:** If F1 < 0.80 OR (F1_dual - F1_clip) < 15 points, hypothesis is rejected.

---

### H2: Cost Efficiency Hypothesis

**Claim:** The dual-agent system will reduce token consumption by **>90%** compared to Gemini-only, while maintaining accuracy within 5 F1 points of Gemini-only.

**Rationale:**
- Confidence gating prevents unnecessary Gemini calls for high-confidence CLIP results
- Only top-K candidates (K=20, default) are sent to Gemini vs. all frames (~36,000 for 10h video)
- Token reduction: (36,000 - 20) / 36,000 ≈ 99.94% in best case; conservatively >90% with gating overhead

**Falsifiability:** If token reduction < 90% OR |F1_dual - F1_gemini| > 5 points, hypothesis is rejected.

---

### H3: Latency Hypothesis

**Claim:** The dual-agent system will achieve **p95 end-to-end latency < 3 seconds** for 10-hour video corpus.

**Rationale:**
- CLIP retrieval: <100ms (FAISS ANN search)
- Gemini reasoning: ~2.5s (for K=20 frames, parallelized API calls)
- Routing overhead: <50ms
- Total: ~2.65s < 3s target

**Falsifiability:** If p95 latency ≥ 3 seconds on benchmark, hypothesis is rejected.

---

## Success Criteria (Composite)

The project is considered **successful** if ALL of the following hold on the held-out test set (20% of benchmark):

| Metric | Threshold | Comparison |
|--------|-----------|------------|
| Top-5 F1 | ≥ 0.80 | Absolute target |
| F1 Lift over CLIP-only | ≥ +15 points | Relative improvement |
| Token Reduction vs. Gemini-only | > 90% | Cost efficiency |
| Latency (p95) | < 3 seconds | Usability threshold |
| F1 Gap vs. Gemini-only | ≤ 5 points | Accuracy preservation |

**Partial Success:** If 4 out of 5 criteria are met, identify the failure mode and propose mitigation for future work.

---

## Evaluation Protocol

### Benchmark Composition

- **Queries:** ≥100 natural-language queries spanning 5 event types:
  1. Vehicle interactions (collision, near-miss, aggressive driving)
  2. Pedestrian events (jaywalking, fall, crowd behavior)
  3. Object-of-interest (specific vehicle color, clothing, license plate)
  4. Traffic violations (red light, stop sign, illegal turn)
  5. Ambient scene (weather, time-of-day, location type)

- **Video Corpus:** ≥10 hours of BDD100K dashcam footage
- **Ground Truth:** Manual annotation of relevant frame ranges per query (binary relevance)
- **Train/Val/Test Split:** 60% / 20% / 20%

### Metrics (Stage-Specific)

| Stage | Metrics |
|-------|---------|
| CLIP Retrieval | Recall@1, Recall@5, Recall@10, MRR, nDCG |
| Gemini Reasoning | Precision, Recall, F1, Accuracy (binary relevance) |
| End-to-End Pipeline | Recall@5 (with binary relevance), Latency (p95), Tokens/query |

**Primary Metric for RQ:** Top-5 F1 (end-to-end)
**Secondary Metrics:** Token cost, latency

---

## Relationship to Flagship Work

This research question builds upon:

> Galanopoulos et al., *"An LLM Framework for Long-form Video Retrieval,"* **CVPRW 2025**.

**Adopted Elements:**
- Retrieve-then-reason paradigm
- Rank-based evaluation protocol (R@K, MRR, nDCG)
- Use of frontier LLM for re-ranking

**Novel Contributions:**
1. **Domain adaptation:** Security/dashcam footage vs. general long-form video
2. **Confidence-gated routing:** Threshold-based escalation to minimize LLM calls (not present in flagship work)
3. **Dual-baseline comparison:** Explicit comparison against both CLIP-only and Gemini-only (flagship work lacks cost baseline)
4. **Token-cost optimization:** Formal measurement of cost efficiency as a success metric

---

## Open Questions (To Be Resolved in WP3–WP6)

1. **Optimal confidence thresholds (τ_high, τ_low):** Should they be global or event-type-specific?
2. **CLIP domain gap magnitude:** How much does performance degrade on dashcam vs. web images?
3. **Gemini prompt sensitivity:** What is the marginal gain of structured JSON output vs. free-text?
4. **Minimum query count per event class:** How many queries are needed for statistical significance?

---

**Version:** 0.1.0 | **Status:** WP1 — Planning | **Last Updated:** 2026-05-16
