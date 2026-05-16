# Flagship Paper Notes

## Reference Work Analysis

**Full Citation:**
> Galanopoulos et al., *"An LLM Framework for Long-form Video Retrieval,"* **CVPRW 2025** (Computer Vision and Pattern Recognition Workshops).

This document summarizes the flagship paper's contributions and clarifies the relationship between that work and the AI Video Investigator project.

---

## Paper Summary

### Core Contribution

The paper introduces a **retrieve-then-reason framework** for semantic video search, where a lightweight retrieval model (e.g., CLIP) filters a large corpus down to candidate segments, which are then analyzed by a large language model (LLM) with vision capabilities for fine-grained re-ranking.

### Key Components

1. **Dense Retrieval Stage:**
   - Uses a pre-trained vision-language model (CLIP or similar) to encode video frames
   - Performs approximate nearest-neighbor (ANN) search to retrieve top-K candidates
   - Goal: High recall at low computational cost

2. **LLM Re-Ranking Stage:**
   - Sends top-K candidate frames/segments to a frontier multimodal LLM (e.g., GPT-4V, Gemini)
   - LLM performs deep reasoning to assess relevance to the natural-language query
   - Outputs: Re-ranked list with relevance scores and rationales

3. **Evaluation Protocol:**
   - Benchmark: Long-form video datasets (movie summaries, instructional videos, etc.)
   - Metrics: **Recall@K, Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (nDCG)**
   - Baselines: CLIP-only retrieval, full-video LLM analysis

### Key Findings

- **Accuracy:** Retrieve-then-reason consistently outperforms CLIP-only by 10–20 points in retrieval accuracy
- **Efficiency:** Reduces LLM token consumption by 95–99% compared to full-video analysis
- **Trade-offs:** Performance depends heavily on retrieval recall (if relevant frames not in top-K, LLM cannot recover)

---

## Adoption: What We Use From This Work

### 1. The Retrieve-Then-Reason Paradigm

**What it is:** A two-stage pipeline where a fast retriever (CLIP) narrows the search space before an expensive reasoner (Gemini) performs deep analysis.

**Why we adopt it:**
- Proven effective on long-form video (flagship work demonstrated on 1–3 hour videos)
- Aligns with our constraint of <3s latency and <$0.10/query-hour cost
- Balances accuracy and efficiency

**Implementation in our work:**
- Stage 1: CLIP ViT-L/14 + FAISS ANN search
- Stage 2: Gemini 1.5 Pro re-ranking of top-K candidates

---

### 2. Rank-Based Evaluation Metrics

**What it is:** Measuring retrieval quality with **Recall@K, MRR, and nDCG** instead of binary accuracy.

**Why we adopt it:**
- Rank-based metrics are the standard in information retrieval research
- Captures whether the correct frame appears "high enough" in the ranked list (not just "somewhere" in the list)
- Enables fair comparison with the flagship work's results

**Implementation in our work:**
- CLIP retrieval stage evaluated with R@1, R@5, R@10, MRR, nDCG
- End-to-end pipeline evaluated with R@5 (binary relevance)

---

### 3. Natural-Language Query Interface

**What it is:** Users express information needs in free-form text (e.g., "car swerving to avoid pedestrian") rather than keywords or tags.

**Why we adopt it:**
- Mirrors the flagship work's user model
- Aligns with our persona (Yossi, fleet safety analyst) who is non-technical
- Leverages CLIP's text-image joint embedding space

**Implementation in our work:**
- Query input: plain-text strings
- CLIP text encoder: same ViT-L/14 used in flagship work

---

## Divergence: Where We Differ From This Work

### 1. Domain Shift — Security/Dashcam vs. General Video

**Flagship work domain:**
- General long-form video (movies, instructional content, YouTube)
- Rich visual variety, high production quality
- Queries often narrative-based ("scene where protagonist enters the building")

**Our domain:**
- Security and dashcam footage
- Constrained visual context (fixed camera angle, outdoor/traffic scenes)
- Queries event-based ("red car running red light") and object-based ("pedestrian in red jacket")

**Implication:**
- CLIP may suffer from **domain gap** (trained on web images, not dashcam footage)
- Our benchmark must reflect the unique query distribution of security use cases
- Potential mitigation: Fine-tune CLIP on dashcam data (deferred to WP5 as risk mitigation)

---

### 2. Confidence-Gated Routing Layer

**What it is:** A decision module between retrieval and reasoning that routes queries based on CLIP's confidence score.

**Why this is novel:**

The flagship work sends **all top-K candidates to the LLM** for every query. Our system adds a **confidence gate**:

```
IF top-1 CLIP similarity > τ_high:
    → Return immediately (LLM call skipped)
ELIF top-1 CLIP similarity < τ_low:
    → Expand K and escalate to LLM
ELSE:
    → Escalate top-K to LLM (standard retrieve-then-reason)
```

**Motivation:**
- **Cost optimization:** Some queries are trivially answered by CLIP alone (e.g., "daytime highway scene")
- **Latency reduction:** Skipping LLM calls saves 2+ seconds per query
- **Token economics:** Target >90% token reduction requires aggressive gating

**Flagship work does not implement this.** Their token savings come solely from reducing corpus size (top-K vs. full video), not from conditional LLM invocation.

---

### 3. Dual-Baseline Comparison

**Flagship work baselines:**
1. CLIP-only retrieval
2. (Implicitly) Full-video LLM analysis (for cost comparison)

**Our baselines:**
1. **CLIP-only:** Same as flagship work
2. **Gemini-only:** Explicit implementation where ALL frames are sent to Gemini (not just top-K)

**Why this matters:**
- The flagship work does not formally measure the **cost-accuracy trade-off** against a Gemini-only upper bound
- Our dual-baseline setup quantifies:
  - **Accuracy gain over CLIP-only** (measures reasoning value)
  - **Cost reduction vs. Gemini-only** (measures efficiency gain)
  - **Accuracy preservation vs. Gemini-only** (measures quality loss from gating)

**Deliverable:** Explicit comparison table showing all three systems on the same benchmark.

---

### 4. Token-Cost as a First-Class Metric

**Flagship work:**
- Reports token counts informally
- Focuses on accuracy metrics (R@K, MRR, nDCG)
- Does not set explicit cost targets

**Our work:**
- **Token cost is a success criterion:** Must achieve >90% reduction vs. Gemini-only
- **Cost-per-query-hour:** Tracked and reported as `< $0.10` target
- **Cost-accuracy Pareto frontier:** Sweep τ_high/τ_low to plot trade-off curve

**Rationale:** Industrial deployment requires proving economic viability, not just technical feasibility.

---

## Methodological Alignment

Despite the divergences above, our methodology remains **compatible** with the flagship work's evaluation protocol:

| Aspect | Flagship Work | Our Work | Alignment? |
|--------|---------------|----------|------------|
| Retrieval Model | CLIP ViT-L/14 | CLIP ViT-L/14 | ✅ Identical |
| Reasoner | GPT-4V | Gemini 1.5 Pro | ⚠️ Different model, same capability tier |
| Metrics | R@K, MRR, nDCG | R@K, MRR, nDCG + F1 + cost | ✅ Superset |
| Benchmark Size | 100+ queries | 100+ queries | ✅ Comparable |
| Video Length | 1–3 hours | 10+ hours | ✅ Longer (more challenging) |

**Conclusion:** Results will be directly comparable on shared metrics (R@K, MRR, nDCG), with our additional cost/latency metrics providing extra dimensions of analysis.

---

## Relationship Statement (For Preparatory Report)

**Suggested phrasing for WP1 defense:**

> This project adopts the **retrieve-then-reason paradigm** and **rank-based evaluation protocol** established by Galanopoulos et al. (CVPRW 2025) as the foundational architecture. We diverge in three key areas: (1) domain adaptation to security/dashcam footage, (2) introduction of a **confidence-gated router** to minimize LLM token consumption, and (3) dual-baseline comparison against both CLIP-only and Gemini-only systems with token cost as a first-class success metric. Our work validates whether the flagship framework generalizes to event-based retrieval in constrained visual domains while meeting industrial cost constraints.

---

## Open Questions for Future Work

1. **Would the flagship work's findings hold with our confidence-gating layer?**
   - Hypothesis: Yes, but with 40–60% fewer LLM calls
   - Resolution: WP5–WP6 empirical evaluation

2. **How does CLIP domain gap affect the retrieve-then-reason pipeline?**
   - Flagship work uses general video; we use dashcam footage
   - If CLIP recall drops, does Gemini re-ranking still provide +15 point lift?
   - Resolution: WP4 CLIP baseline + WP6 ablation study

3. **Could we improve upon the flagship work's LLM prompting strategy?**
   - Flagship work uses free-text prompts; we use structured JSON output
   - Does schema enforcement improve consistency without sacrificing accuracy?
   - Resolution: WP5 prompt engineering experiments

---

**Version:** 0.1.0 | **Status:** WP1 — Planning | **Last Updated:** 2026-05-16
