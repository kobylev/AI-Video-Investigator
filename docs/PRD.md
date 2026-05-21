# Product Requirements Document

## AI Video Investigator — Dual-Agent Semantic Video Retrieval

This document defines the functional and non-functional requirements for the AI Video Investigator system, a dual-agent architecture combining CLIP retrieval with Claude Haiku 4.5 reasoning for semantic search over long-form security and dashcam footage.

---

## 1. User & Persona

**Primary Persona: Yossi, Fleet Safety Analyst**

- **Role:** Safety compliance officer at a logistics company managing 200 delivery vehicles
- **Pain Point:** Must review 12–24 hours of dashcam footage per incident report (near-miss, customer complaint, insurance claim)
- **Current Workflow:** Manual scrubbing at 4x speed, taking 3–6 hours per investigation
- **Success Criteria:** Reduce investigation time from hours to minutes while maintaining high recall

---

## 2. Problem Statement

Security and fleet operators face **cognitive overload at industrial scale** when searching long-form video. Current solutions force a false choice:

- **Traditional keyword tagging:** Fast but brittle—misses semantic queries like "pedestrian running across street"
- **Frontier VLM analysis (e.g., Claude Haiku 4.5):** Accurate but prohibitively expensive (~$0.30/query-hour) and slow (~60s latency)

**Core Tension:** Users need the precision of multimodal LLMs at the cost and latency of traditional retrieval systems.

---

## 3. Goals & Non-Goals

### Goals

1. **Enable natural-language video search** over 10+ hours of footage with sub-3-second response time
2. **Achieve top-5 F1 ≥ 0.80** on a curated benchmark of security/dashcam queries
3. **Reduce token cost by >90%** compared to a Claude-only baseline
4. **Deliver +15 point accuracy lift** over CLIP-only retrieval
5. **Provide reproducible evaluation harness** for academic validation

---

## 4. Functional Requirements

### FR2: Dual-Agent Pipeline

1. **Stage 1 — CLIP Retrieval:**
   - Encode video frames offline using CLIP ViT-L/14
   - Index embeddings with FAISS for sub-millisecond ANN search
   - Return top-K candidates (K configurable, default K=20)

2. **Stage 2 — Confidence-Gated Router:**
   - If CLIP top-1 similarity > τ_high: return immediately (high-confidence shortcut)
   - If CLIP top-1 similarity < τ_low: expand K and escalate to Claude
   - Else: escalate top-K to Claude for re-ranking

3. **Stage 3 — Claude Reasoner:**
   - Analyze each candidate frame with structured prompt + tool-use
   - Output: `is_event_present`, `confidence_score`, `forensic_summary`
   - Re-rank candidates and return top-5

---

## 5. Non-Functional Requirements

### NFR1: Latency

- **Target:** End-to-end p95 latency < 3 seconds for 10-hour video corpus
- **Breakdown:** CLIP retrieval <100ms, Claude reasoning <2.5s (for K=20 candidates)

### NFR2: Cost

- **Target:** < $0.10 per query-hour of footage
- **Constraint:** Cloud token usage must be minimized via confidence gating

---

## 6. Success Metrics

| Metric | Target | Baseline | Measurement Method |
|--------|--------|----------|-------------------|
| Top-5 F1 | ≥ 0.80 | CLIP-only: ~0.65 | Harmonic mean of P/R at K=5 |
| Accuracy Lift | +15 pts | CLIP-only | Binary relevance on test set |
| Latency (p95) | < 3s | Claude-only: ~60s | End-to-end wall-clock time |
| Cost/query-hour | < $0.10 | Claude-only: ~$0.30 | Anthropic API token count × pricing |
| Token Reduction | > 90% | Claude-only baseline | (Baseline tokens - Ours) / Baseline |

---

**Version:** 0.1.0 | **Status:** WP5 — Reasoner & Router Integration | **Last Updated:** 2026-05-21
