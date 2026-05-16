# Product Requirements Document

## AI Video Investigator — Dual-Agent Semantic Video Retrieval

This document defines the functional and non-functional requirements for the AI Video Investigator system, a dual-agent architecture combining CLIP retrieval with Gemini 1.5 Pro reasoning for semantic search over long-form security and dashcam footage.

---

## 1. User & Persona

**Primary Persona: Yossi, Fleet Safety Analyst**

- **Role:** Safety compliance officer at a logistics company managing 200 delivery vehicles
- **Pain Point:** Must review 12–24 hours of dashcam footage per incident report (near-miss, customer complaint, insurance claim)
- **Current Workflow:** Manual scrubbing at 4x speed, taking 3–6 hours per investigation
- **Technical Context:** Non-technical user; needs natural-language query interface
- **Success Criteria:** Reduce investigation time from hours to minutes while maintaining high recall

**Secondary Persona: Security Operations Center (SOC) Analyst**

- **Role:** Monitoring 50+ CCTV feeds across a commercial campus
- **Pain Point:** Retrospective searches for suspects, vehicles, or events described in witness reports
- **Current Workflow:** Keyword tags (if available) or brute-force scanning
- **Technical Context:** Comfortable with dashboards; needs API-first integration

---

## 2. Problem Statement

Security and fleet operators face **cognitive overload at industrial scale** when searching long-form video. Current solutions force a false choice:

- **Traditional keyword tagging:** Fast but brittle—misses semantic queries like "pedestrian running across street"
- **Frontier VLM analysis (e.g., Gemini 1.5 Pro):** Accurate but prohibitively expensive (~$1.16/query-hour) and slow (~60s latency)

**Core Tension:** Users need the precision of multimodal LLMs at the cost and latency of traditional retrieval systems.

---

## 3. Goals & Non-Goals

### Goals

1. **Enable natural-language video search** over 10+ hours of footage with sub-3-second response time
2. **Achieve top-5 F1 ≥ 0.80** on a curated benchmark of security/dashcam queries
3. **Reduce token cost by >90%** compared to a Gemini-only baseline
4. **Deliver +15 point accuracy lift** over CLIP-only retrieval
5. **Provide reproducible evaluation harness** for academic validation

### Non-Goals (WP1 Scope)

- Real-time streaming video analysis (focus: retrospective search)
- Fine-tuning CLIP on domain-specific data (use off-the-shelf ViT-L/14)
- Multi-modal query inputs (e.g., sketch-based search)
- Production deployment infrastructure (MVP proof-of-concept only)

---

## 4. Functional Requirements

### FR1: Natural-Language Query Interface

- **Input:** Text query (e.g., "red sedan running red light at intersection")
- **Output:** Ranked list of top-K video frames with timestamps and confidence scores
- **Constraint:** Query must be processed end-to-end in <3 seconds (p95)

### FR2: Dual-Agent Pipeline

1. **Stage 1 — CLIP Retrieval:**
   - Encode video frames offline using CLIP ViT-L/14
   - Index embeddings with FAISS for sub-millisecond ANN search
   - Return top-K candidates (K configurable, default K=20)

2. **Stage 2 — Confidence-Gated Router:**
   - If CLIP top-1 similarity > τ_high: return immediately (high-confidence shortcut)
   - If CLIP top-1 similarity < τ_low: expand K and escalate to Gemini
   - Else: escalate top-K to Gemini for re-ranking

3. **Stage 3 — Gemini Reasoner:**
   - Analyze each candidate frame with structured prompt
   - Output: relevance score (0-1), rationale, detected objects
   - Re-rank candidates and return top-5

### FR3: Event-Specific Prompt Templates

Provide domain-specific prompts for:
- Vehicle interactions (collision, near-miss, aggressive driving)
- Pedestrian events (jaywalking, fall, crowd behavior)
- Object-of-interest (specific vehicle color/type, clothing, license plate)
- Traffic violations (red light, stop sign, illegal turn)
- Ambient scene queries (weather condition, time-of-day, location type)

### FR4: Evaluation Harness

- **Benchmark:** ≥100 queries over ≥10 hours of annotated dashcam footage
- **Metrics:** R@1, R@5, R@10, MRR, nDCG (retrieval stage); Precision, Recall, F1, Accuracy (reasoning stage)
- **Baselines:** CLIP-only, Gemini-only (full-video)
- **Reproducibility:** Queries, ground truth, and evaluation scripts committed to repo

---

## 5. Non-Functional Requirements

### NFR1: Latency

- **Target:** End-to-end p95 latency < 3 seconds for 10-hour video corpus
- **Breakdown:** CLIP retrieval <100ms, Gemini reasoning <2.5s (for K=20 candidates)

### NFR2: Cost

- **Target:** < $0.10 per query-hour of footage
- **Constraint:** Gemini token usage must be minimized via confidence gating
- **Measurement:** Track tokens-per-query and cost-per-query in evaluation logs

### NFR3: Accuracy

- **Target:** Top-5 F1 ≥ 0.80 on benchmark
- **Constraint:** Must outperform CLIP-only baseline by ≥15 accuracy points
- **Validation:** Measured on held-out test set (20% of benchmark)

### NFR4: Reproducibility

- All experiments versioned with Git tags (v0.X.0-wpX)
- Evaluation results committed to `evals/results/` with timestamp and config
- Dependencies pinned in `requirements.txt`

---

## 6. Success Metrics

| Metric | Target | Baseline | Measurement Method |
|--------|--------|----------|-------------------|
| Top-5 F1 | ≥ 0.80 | CLIP-only: ~0.65 | Harmonic mean of P/R at K=5 |
| Accuracy Lift | +15 pts | CLIP-only | Binary relevance on test set |
| Latency (p95) | < 3s | Gemini-only: ~60s | End-to-end wall-clock time |
| Cost/query-hour | < $0.10 | Gemini-only: $1.16 | Gemini API token count × pricing |
| Token Reduction | > 90% | Gemini-only baseline | (Baseline tokens - Ours) / Baseline |

---

## 7. Out of Scope (WP1)

- **Fine-tuning CLIP:** Off-the-shelf ViT-L/14 used; fine-tuning deferred to WP5 as risk mitigation
- **Temporal reasoning:** Frame-level retrieval only; no cross-frame event detection
- **Multi-camera fusion:** Single-camera queries only
- **Real-time inference:** Batch processing acceptable for MVP
- **Production SLA:** Academic proof-of-concept; no uptime guarantee

---

## 8. Open Questions

1. **Optimal confidence thresholds (τ_high, τ_low):** Should these be learned per-event-type or global?
2. **CLIP domain gap:** How significant is the performance drop on dashcam footage vs. CLIP's training distribution?
3. **Gemini prompt engineering:** What is the marginal gain of structured JSON output vs. free-text rationale?
4. **Benchmark diversity:** What is the minimum number of queries per event class to ensure statistical validity?

**Resolution Path:** Questions 1 and 3 will be addressed empirically in WP5–WP6. Questions 2 and 4 are scoped into WP3 (data acquisition).

---

**Version:** 0.1.0 | **Status:** WP1 — Planning | **Last Updated:** 2026-05-16
