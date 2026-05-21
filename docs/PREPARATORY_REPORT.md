# Preparatory Report (דוח מכין)

## AI Video Investigator — Dual-Agent Semantic Video Retrieval for Security and Dashcam Footage

**Work Package:** WP1 — Planning
**Author:** Koby Lev
**Defense Date:** Sunday, 2026-05-18
**Status:** Completed (Updated for Claude Haiku 4.5 Standardization)

---

## Abstract

Security operators and fleet-safety teams face cognitive overload when searching 12–24 hours of dashcam or CCTV footage for specific events. Current solutions force a brutal trade-off: keyword tagging is fast but brittle, while frontier vision-language models (Claude Haiku 4.5) are accurate but prohibitively expensive (~$0.30 per query-hour) and slow (~60s latency).

This project proposes **AI Video Investigator**, a dual-agent architecture that breaks this trade-off by combining CLIP retrieval (fast semantic filtering) with confidence-gated routing to Claude Haiku 4.5 (deep reasoning on ambiguous cases only). The system targets **top-5 F1 ≥ 0.80**, **sub-3-second latency**, and **>90% token-cost reduction** versus a Claude-only baseline, evaluated on a benchmark of ≥100 natural-language queries over ≥10 hours of BDD100K dashcam footage.

**Keywords:** Video retrieval, Vision-language models, CLIP, Claude Haiku 4.5, Confidence gating, Dashcam footage, Semantic search

---

## 1. Introduction

### 1.1 The 60-Second Elevator Pitch

Security operators and fleet-safety teams share one nightmare: **scrubbing 12 to 24 hours of dashcam or CCTV footage to find a single event**—a near-miss, a red jacket, a midnight license plate. This is cognitive overload at industrial scale.

Today's tools force a brutal trade-off:
- **Keyword tagging** misses semantic nuance (*"pedestrian running across street"* ≠ *"jaywalking"*)
- **Frontier VLMs** (Claude Haiku 4.5) are accurate but cost **~$0.30 per query-hour** with **60-second latency**

**AI Video Investigator breaks that trade-off with dual-agent routing.**

A CLIP retriever acts as a **millisecond semantic filter**, surfacing top-K candidate frames out of tens of thousands. Only those candidates are escalated to Claude Haiku 4.5 for deep reasoning—*"is this actually a hit-and-run, or just a fender-bender?"*

**Success is measured on four axes:**
1. **Top-5 F1 ≥ 0.80** (harmonic mean of precision and recall)
2. **+15 point accuracy lift** over CLIP-alone baseline
3. **Sub-3-second end-to-end latency** (p95)
4. **>90% token-cost reduction** versus Claude-only baseline

One architecture. Four numbers. Sixty seconds.

---

## 3. VALID Framework Analysis

### 3.2 AI-Core — Why AI is Essential?

**AI Components:**
- **CLIP:** Learned vision-language joint embedding (400M image-text pairs)
- **Claude Haiku 4.5:** Large multimodal model with fine-grained reasoning and native tool-use.
- **Router:** Learned thresholds via grid search on validation data

---

### 3.4 Innovative — What's Novel?

1. **Confidence-Gated Routing (Type 3: Novel Combination)**
   - **Our contribution:** Skip 40–60% of LLM calls via confidence thresholds (τ_high, τ_low)
   - **Impact:** >90% token reduction while preserving F1 within 5 points of Claude-only baseline.

---

## 4. Research Question

### 4.1 Primary Research Question (Measurable & Falsifiable)

> **To what extent does a dual-agent CLIP→Claude routing architecture improve top-5 retrieval F1 and reduce inference token cost on long-form dashcam footage, compared to (a) a CLIP-only retrieval baseline and (b) a Claude-only frame-analysis baseline, evaluated on a benchmark of ≥100 natural-language queries over ≥10 hours of curated video?**

---

### 4.2 Hypotheses

**H1 — Accuracy Hypothesis:**
- Dual-agent achieves **F1 ≥ 0.80**, outperforming CLIP-only by **≥15 points**
- **Rationale:** CLIP provides high recall, Claude provides high precision.

**H2 — Cost Efficiency Hypothesis:**
- Dual-agent reduces token consumption by **>90%** vs. Claude-only
- Accuracy preserved within **5 F1 points** of Claude-only (minimal quality loss)

---

### 4.3 Success Criteria (Composite)

| # | Metric | Target | Comparison |
|---|--------|--------|------------|
| 1 | Top-5 F1 | ≥ 0.80 | Absolute threshold |
| 2 | F1 Lift over CLIP-only | ≥ +15 points | Relative improvement |
| 3 | Token Reduction vs. Claude-only | > 90% | Cost efficiency |
| 4 | Latency (p95) | < 3 seconds | Usability threshold |
| 5 | F1 Gap vs. Claude-only | ≤ 5 points | Accuracy preservation |

---

## 5. Literature Review

### 5.1 Flagship Paper

> **Galanopoulos et al., "An LLM Framework for Long-form Video Retrieval," CVPRW 2025**

---

## 8. Token Economics

### 8.1 Cost Breakdown

**Claude-Only Baseline (Naive Brute Force):**
- 10 hours @ 1fps = 36,000 frames
- Estimated vision tokens scale to **~$0.30 per query-hour** at Claude Haiku 4.5 prices.

**Dual-Agent (Target):**
- 40–60% queries skip Claude.
- Remaining queries: 20 frames analyzed selectively.
- Expected cost per query: **<$0.01**.

**Token Reduction:** >95% reduction ✅ (exceeds 90% target).

---

## 11. Expected Contributions

### 11.1 Academic Contributions

1. **Confidence-Gated Routing Algorithm:** Novel threshold-based decision layer for LLM escalation.
2. **Domain Adaptation Study:** First evaluation of CLIP + Claude on security/dashcam footage.

---

**Last Updated:** 2026-05-21
**Status:** WP5 — Reasoner & Router Integration
