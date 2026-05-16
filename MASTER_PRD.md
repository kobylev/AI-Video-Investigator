# AI Video Investigator — Master Product Requirements Document

> **The Complete Project Specification**
> Dual-Agent Semantic Video Retrieval for Security and Dashcam Footage

**Version:** 0.1.0 | **Status:** WP1 — Planning | **Author:** Koby Lev | **Last Updated:** 2026-05-16

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Problem](#2-the-problem)
3. [The Solution](#3-the-solution)
4. [Research Question](#4-research-question)
5. [Success Criteria](#5-success-criteria)
6. [System Architecture](#6-system-architecture)
7. [User Personas](#7-user-personas)
8. [Functional Requirements](#8-functional-requirements)
9. [Non-Functional Requirements](#9-non-functional-requirements)
10. [Technical Implementation](#10-technical-implementation)
11. [Evaluation Strategy](#11-evaluation-strategy)
12. [Risk Management](#12-risk-management)
13. [10-Week Roadmap](#13-10-week-roadmap)
14. [Relationship to Existing Work](#14-relationship-to-existing-work)
15. [Deliverables](#15-deliverables)
16. [Open Questions](#16-open-questions)

---

## 1. Executive Summary

### The 60-Second Pitch

Security operators and fleet-safety teams share one nightmare: **scrubbing 12 to 24 hours of dashcam or CCTV footage to find a single event**—a near-miss, a red jacket, a midnight license plate. This is cognitive overload at industrial scale, and today's tools force a brutal trade-off.

- **Keyword tagging** misses semantic nuance
- **Feeding raw video into Gemini 1.5 Pro** costs over $1.16 per query-hour and stalls any interactive workflow

**AI Video Investigator breaks that trade-off with dual-agent routing.**

A CLIP retriever acts as a millisecond semantic filter, surfacing the top-K candidate frames out of tens of thousands. Only those candidates are escalated to Gemini 1.5 Pro for deep reasoning—*"is this actually a hit-and-run, or just a fender-bender?"*

### The Four Numbers

Success is measured on four axes:

1. **Top-5 F1 ≥ 0.80** (harmonic mean of precision and recall)
2. **+15 point accuracy lift** over CLIP-alone baseline
3. **Sub-3-second end-to-end latency** (p95)
4. **>90% token-cost reduction** versus Gemini-only baseline

One architecture. Four numbers. Sixty seconds.

---

## 2. The Problem

### Problem Statement

Security and fleet operators face **cognitive overload at industrial scale** when searching long-form video. Current solutions force a false choice:

**Option A: Traditional Keyword Tagging**
- ✅ Fast (metadata-based search)
- ❌ Brittle (misses semantic queries like "pedestrian running across street")
- ❌ Low recall (requires perfect metadata annotation)
- **Example failure:** Query for "near-miss with cyclist" returns zero results because video was tagged "traffic incident" with no mention of cyclist

**Option B: Frontier VLM Analysis (Gemini 1.5 Pro)**
- ✅ Accurate (understands semantic nuance)
- ✅ Flexible (natural-language queries)
- ❌ Prohibitively expensive (~$1.16 per query-hour of footage)
- ❌ Slow (~60 seconds latency for 1 hour of video @ 1fps = 3,600 frames)
- **Cost breakdown:** 1 hour @ 1fps = 3,600 frames × 258 tokens/frame = 928,800 tokens × $1.25/1M tokens = $1.16

**The Core Tension:** Users need the precision of multimodal LLMs at the cost and latency of traditional retrieval systems.

### Market Context

- **Fleet management market:** 200M+ commercial vehicles worldwide (source: Berg Insight 2024)
- **Security camera market:** 1B+ surveillance cameras globally (source: IHS Markit 2025)
- **Current manual review cost:** $50–$150 per incident (3–6 hours @ $15–25/hour analyst labor)
- **Opportunity:** Reduce per-incident cost to <$5 while improving recall from ~60% to >80%

---

## 3. The Solution

### System Overview

**AI Video Investigator** is a dual-agent architecture that combines:

1. **CLIP Retriever (Fast Filter)** — Encodes 10+ hours of video offline, performs sub-100ms semantic search
2. **Confidence-Gated Router (Decision Layer)** — Routes queries based on CLIP confidence scores
3. **Gemini 1.5 Pro Reasoner (Deep Analysis)** — Analyzes only the ambiguous cases requiring multimodal reasoning

### Key Innovation: Confidence-Gated Routing

This is the **novel contribution** over existing retrieve-then-reason systems:

```
IF top-1 CLIP similarity > τ_high (e.g., 0.85):
    → Return immediately (LLM call skipped, saves $0.01 and 2.5s)

ELIF top-1 CLIP similarity < τ_low (e.g., 0.60):
    → Expand K from 20 to 50 and escalate to Gemini (hedge against false negatives)

ELSE:
    → Escalate top-K to Gemini (standard retrieve-then-reason)
```

**Impact:**
- 40–60% of queries skip the Gemini call entirely (high-confidence matches)
- Remaining queries get full Gemini reasoning on only 20–50 frames (not 3,600)
- **Token reduction:** From 928,800 tokens/query-hour (Gemini-only) to <10,000 tokens/query (dual-agent) = **>98% reduction**

---

## 4. Research Question

### Primary Research Question (Measurable & Falsifiable)

> **To what extent does a dual-agent CLIP→Gemini routing architecture improve top-5 retrieval F1 and reduce inference token cost on long-form dashcam footage, compared to (a) a CLIP-only retrieval baseline and (b) a Gemini-only frame-analysis baseline, evaluated on a benchmark of ≥100 natural-language queries over ≥10 hours of curated video?**

### Why This Question Matters

1. **Falsifiable:** If F1 < 0.80 OR token reduction < 90%, hypothesis is rejected
2. **Quantitative:** Four measurable outcomes (F1, accuracy lift, latency, cost)
3. **Dual baselines:** Bounds the solution space (cost-accuracy Pareto frontier)
4. **Generalizable:** Tests whether retrieve-then-reason paradigm (proven on general video) works for security/dashcam domain

### Hypotheses

**H1 — Accuracy Hypothesis:**
- Dual-agent achieves **F1 ≥ 0.80**, outperforming CLIP-only by **≥15 points**
- CLIP provides high recall, Gemini provides high precision
- Confidence gating preserves easy cases while escalating hard cases

**H2 — Cost Efficiency Hypothesis:**
- Dual-agent reduces token consumption by **>90%** vs. Gemini-only
- Accuracy remains within 5 F1 points of Gemini-only (minimal quality loss)

**H3 — Latency Hypothesis:**
- Dual-agent achieves **p95 latency < 3 seconds** for 10-hour corpus
- CLIP retrieval: <100ms, Gemini reasoning: <2.5s, routing overhead: <50ms

---

## 5. Success Criteria

### Composite Success Matrix

The project is **successful** if ALL of the following hold on the held-out test set:

| # | Metric | Target | Comparison | Measurement Method |
|---|--------|--------|------------|-------------------|
| 1 | **Top-5 F1** | ≥ 0.80 | Absolute threshold | Harmonic mean of P/R at K=5 |
| 2 | **F1 Lift over CLIP-only** | ≥ +15 points | Relative improvement | F1_dual - F1_clip ≥ 0.15 |
| 3 | **Token Reduction vs. Gemini-only** | > 90% | Cost efficiency | (Tokens_gemini - Tokens_dual) / Tokens_gemini |
| 4 | **Latency (p95)** | < 3 seconds | Usability threshold | End-to-end wall-clock time |
| 5 | **F1 Gap vs. Gemini-only** | ≤ 5 points | Accuracy preservation | \|F1_dual - F1_gemini\| ≤ 0.05 |

**Partial Success:** If 4 out of 5 criteria met, identify failure mode and propose mitigation for future work.

### Baseline Performance Estimates

| System | Top-5 F1 | Latency (p95) | Cost/query-hour | Tokens/query |
|--------|----------|---------------|-----------------|--------------|
| **CLIP-only** | ~0.65 | <0.1s | ~$0 | 0 |
| **Gemini-only** | ~0.85 | ~60s | ~$1.16 | 928,800 |
| **Dual-Agent (Target)** | **≥0.80** | **<3s** | **<$0.10** | **<10,000** |

---

## 6. System Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      OFFLINE INDEXING PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│  Video Files (10h) → Frame Extraction (1fps) → CLIP Encoder     │
│       ↓                                                          │
│  FAISS Index (36,000 frames × 768-dim embeddings)               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     QUERY-TIME PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│  Natural Language Query                                          │
│       ↓                                                          │
│  [1] CLIP Text Encoder → 768-dim query embedding                │
│       ↓                                                          │
│  [2] FAISS ANN Search → Top-20 candidate frames                 │
│       ↓                                                          │
│  [3] CONFIDENCE ROUTER:                                          │
│      ├─ High Confidence (sim > 0.85) → Return Top-1 (SKIP LLM)  │
│      ├─ Low Confidence (sim < 0.60) → Expand to Top-50          │
│      └─ Ambiguous (0.60 ≤ sim ≤ 0.85) → Escalate Top-20         │
│       ↓                                                          │
│  [4] Gemini 1.5 Pro Reasoner:                                   │
│      - Analyze each candidate frame with structured prompt      │
│      - Output: relevance_score (0-1), rationale, objects        │
│       ↓                                                          │
│  [5] Re-Rank by Gemini Scores → Return Top-5                    │
└─────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. CLIP Retriever
- **Model:** OpenAI CLIP ViT-L/14 (off-the-shelf, frozen)
- **Embedding Dimension:** 768
- **Index Type:** FAISS IVF or HNSW
- **Search Metric:** Cosine similarity (L2-normalized inner product)
- **Performance:** <100ms for 36,000 frames

#### 2. Confidence-Gated Router
- **Input:** Top-K CLIP results with similarity scores
- **Decision Logic:** Compare top-1 score against thresholds
- **Thresholds (Initial):**
  - τ_high = 0.85 (90th percentile of positive pairs)
  - τ_low = 0.60 (50th percentile of positive pairs)
- **Tuning:** Grid search on validation set (WP5)
- **Impact:** 40–60% query reduction to Gemini (estimated)

#### 3. Gemini Reasoner
- **Model:** `gemini-1.5-pro-latest` via Google Generative AI SDK
- **Input:** Top-K frames (K=20 default) + event-type-specific prompt
- **Output Schema (JSON):**
  ```json
  {
    "relevance_score": 0.0 to 1.0,
    "rationale": "1-2 sentence explanation",
    "detected_objects": ["object1", "object2"],
    "confidence": "high" | "medium" | "low"
  }
  ```
- **Cost:** ~$0.01 per query (8,160 tokens @ $1.25/1M tokens)
- **Latency:** ~2.5s for K=20 (parallelized API calls)

---

## 7. User Personas

### Primary Persona: Yossi, Fleet Safety Analyst

**Profile:**
- **Role:** Safety compliance officer at logistics company (200 vehicles)
- **Age:** 35–45
- **Technical Skill:** Low (not a developer, uses dashboards and reports)
- **Daily Task:** Review incident reports, pull dashcam footage, find relevant events

**Pain Points:**
1. **Time Sink:** Scrubbing 12–24 hours of footage per incident takes 3–6 hours
2. **Cognitive Overload:** After 2 hours of scrubbing, miss critical details
3. **Missed Events:** Fast-forwarding at 4x speed often skips the exact moment of interest
4. **No Semantic Search:** Can't query "red car cutting off our truck" — must watch everything

**Success Criteria for Yossi:**
- Reduce investigation time from **3–6 hours to <10 minutes**
- Find the relevant moment **>80% of the time** (current: ~60%)
- Use **natural language**, not technical queries or frame numbers

**User Journey (Current State):**
1. Receive incident report: "Driver complained of near-miss on Route 5 around 2 PM Tuesday"
2. Identify vehicle from fleet database → find dashcam device ID
3. Download 24 hours of footage (entire day, don't know exact time)
4. Open in VLC, scrub at 4x speed for 3–6 hours
5. Manually tag relevant clips, export to insurance
6. **Frustration:** Often miss the exact moment, have to re-watch

**User Journey (With AI Video Investigator):**
1. Receive incident report
2. Upload footage to system (or point to existing corpus)
3. Type query: *"near-miss with vehicle cutting into our lane on Route 5"*
4. **System returns top-5 frames in <3 seconds**
5. Review 5 frames (2 minutes), select the relevant one
6. Export clip with timestamp → submit to insurance
7. **Satisfaction:** Found it in 5 minutes, confident it's the right moment

---

### Secondary Persona: Security Operations Center (SOC) Analyst

**Profile:**
- **Role:** Monitor 50+ CCTV cameras across commercial campus
- **Age:** 25–35
- **Technical Skill:** Medium (comfortable with dashboards, APIs, scripting)
- **Daily Task:** Respond to security incidents, pull footage for investigations

**Pain Points:**
1. **Retrospective Search:** Incident reported 24 hours later, must search yesterday's footage across 50 cameras
2. **Vague Descriptions:** Witness says "person in red jacket" but doesn't remember which camera or exact time
3. **Keyword Tags:** Cameras don't auto-tag footage; must manually label or scrub
4. **Multi-Camera Correlation:** Event might span multiple camera angles

**Success Criteria for SOC:**
- Query across **multiple camera feeds** with single natural-language query
- Reduce **time-to-evidence from hours to minutes**
- **API integration** with existing security platform (Genetec, Milestone)

---

## 8. Functional Requirements

### FR1: Natural-Language Query Interface

**Input:**
- Plain-text query (1–50 words)
- Examples:
  - "red sedan running red light at intersection"
  - "pedestrian in yellow jacket crossing street outside crosswalk"
  - "two vehicles colliding in left lane"

**Output:**
- Ranked list of top-5 video frames
- For each frame:
  - Frame ID
  - Timestamp (HH:MM:SS)
  - Relevance score (0.0–1.0)
  - Thumbnail preview
  - Rationale (if Gemini was used)
  - Detected objects

**Constraints:**
- Query must be processed in **<3 seconds (p95 latency)**
- Must support **100+ concurrent queries** (for SOC use case)

---

### FR2: Dual-Agent Pipeline Stages

#### Stage 1: CLIP Retrieval
- **Input:** Natural-language query
- **Process:**
  1. Encode query text with CLIP text encoder
  2. Search FAISS index for top-K frames by cosine similarity
  3. Return K candidate frames with scores
- **Output:** List of (frame_id, score) tuples, sorted descending
- **Performance:** <100ms

#### Stage 2: Confidence-Gated Router
- **Input:** Top-K CLIP results
- **Process:**
  1. Evaluate top-1 similarity score
  2. Apply routing logic (skip / expand / escalate)
  3. Route to Gemini or return directly
- **Output:** Routing decision + frames to analyze
- **Performance:** <50ms

#### Stage 3: Gemini Reasoner
- **Input:** Frames selected by router + query + prompt template
- **Process:**
  1. Construct event-type-specific prompt
  2. Call Gemini API for each frame (parallelized)
  3. Parse JSON responses
  4. Re-rank by relevance_score
- **Output:** Top-5 frames with rationales
- **Performance:** <2.5s for K=20

---

### FR3: Event-Specific Prompt Templates

The system provides **5 curated prompt templates** (see `docs/prompts/prompt_book.md`):

| Event Type | Description | Example Query |
|------------|-------------|---------------|
| **Vehicle Interaction** | Collisions, near-misses, aggressive driving | "white SUV cutting off truck in left lane" |
| **Pedestrian Event** | Jaywalking, crossings, falls | "pedestrian running across street outside crosswalk" |
| **Object-of-Interest** | Vehicle color/type/plate, clothing, signs | "blue sedan with license plate starting 'ABC'" |
| **Traffic Violation** | Red light, stop sign, illegal turn | "red car running red light" |
| **Ambient Scene** | Weather, time-of-day, traffic density | "rainy nighttime highway" |

Each template includes:
- Role priming ("You are a forensic video analyst...")
- Analysis steps (chain-of-thought)
- Structured JSON output schema
- Example query/response pairs

---

### FR4: Evaluation Harness

**Benchmark Requirements:**
- **Queries:** ≥100 natural-language queries
- **Video Corpus:** ≥10 hours of BDD100K dashcam footage
- **Ground Truth:** Manual annotation of relevant frame ranges
- **Train/Val/Test Split:** 60% / 20% / 20% (stratified by event type)

**Metrics (Comprehensive):**

| Stage | Metrics |
|-------|---------|
| **CLIP Retrieval** | Recall@1, Recall@5, Recall@10, MRR, nDCG |
| **Gemini Reasoning** | Precision, Recall, F1, Accuracy (binary relevance) |
| **End-to-End Pipeline** | Recall@5, Top-5 F1 |
| **System** | Latency (p50, p95, p99), Tokens/query, Cost/query |

**Baseline Implementations:**
1. **CLIP-only:** Top-K retrieval without reasoning
2. **Gemini-only:** All frames sent to Gemini for ranking
3. **Dual-agent (This Work):** CLIP + Router + Gemini

---

## 9. Non-Functional Requirements

### NFR1: Performance (Latency)

| Metric | Target | Rationale |
|--------|--------|-----------|
| **End-to-End Latency (p95)** | <3s | Interactive use case; users won't wait 60s |
| **CLIP Retrieval Latency** | <100ms | ANN search on 36K frames |
| **Router Overhead** | <50ms | Simple threshold comparison |
| **Gemini Reasoning Latency** | <2.5s | K=20 frames, parallelized API calls |

**Latency Budget Breakdown (Worst Case):**
```
CLIP:    100ms
Router:   50ms
Gemini: 2,500ms
Total:  2,650ms < 3,000ms target ✅
```

---

### NFR2: Cost (Token Economics)

| System | Tokens/Query | Cost/Query | Cost/Query-Hour (10h corpus) |
|--------|--------------|------------|------------------------------|
| Gemini-only | 928,800 | $1.16 | $1.16 |
| Dual-agent (Target) | <10,000 | <$0.0125 | <$0.10 |
| **Reduction** | **>98%** | **>98%** | **>91%** |

**Cost Control Mechanisms:**
1. **Confidence gating:** Skip 40–60% of Gemini calls
2. **Top-K limiting:** Only analyze 20–50 frames (not 3,600)
3. **Caching:** Store Gemini responses to avoid re-queries
4. **Batch API:** Use Gemini batch endpoint for lower cost (if available)

---

### NFR3: Accuracy

| Metric | Target | Baseline (CLIP-only) | Baseline (Gemini-only) |
|--------|--------|---------------------|------------------------|
| **Top-5 F1** | ≥0.80 | ~0.65 | ~0.85 |
| **Accuracy Lift** | +15 pts | — | — |
| **Recall@5** | ≥0.85 | ~0.75 | ~0.90 |

**Quality Assurance:**
- Manual review of 10% of predictions
- Failure analysis for false positives/negatives
- Ablation studies (remove router, use different prompts, vary K)

---

### NFR4: Reproducibility

- **Git Tagging:** Each WP completion tagged (`v0.X.0-wpX`)
- **Dependency Pinning:** `requirements.txt` with exact versions
- **Experiment Logging:** All results committed to `evals/results/` with:
  - Config (K, τ_high, τ_low, model versions)
  - Metrics (JSON format)
  - Predictions (JSONL with query_id, frame_ids, scores)
- **Data Provenance:** BDD100K video IDs documented in `data/bdd100k_video_ids.txt`

---

## 10. Technical Implementation

### Technology Stack

| Component | Technology | Version | Justification |
|-----------|------------|---------|---------------|
| **Video Processing** | ffmpeg | latest | Industry standard, reliable frame extraction |
| **CLIP Encoder** | OpenAI CLIP ViT-L/14 | openai | Off-the-shelf, no fine-tuning required |
| **Vector Index** | FAISS (CPU) | faiss-cpu | Facebook's battle-tested ANN library |
| **Reasoner** | Gemini 1.5 Pro | `gemini-1.5-pro-latest` | State-of-art multimodal VLM |
| **Backend** | Python 3.9+ | 3.9+ | Ecosystem compatibility (PyTorch, FAISS) |
| **Validation** | Pydantic | latest | Type-safe JSON parsing |
| **Testing** | pytest | latest | Unit tests for each module |

### Repository Structure

```
.
├── src/                    # Source code (organized by function)
│   ├── retriever/          # CLIP encoder + FAISS search
│   ├── router/             # Confidence-gated routing logic
│   ├── reasoner/           # Gemini API wrapper
│   ├── pipeline/           # End-to-end orchestration
│   └── eval/               # Metrics implementation
│
├── docs/                   # Documentation (organized by topic)
│   ├── PRD.md              # Product requirements
│   ├── architecture.md     # System design + Mermaid diagram
│   ├── research_question.md
│   ├── flagship_paper_notes.md
│   ├── risk_register.md
│   ├── prompts/prompt_book.md
│   └── work_packages/      # WP1-WP10 progress docs
│
├── evals/                  # Evaluation benchmark
│   ├── queries.example.jsonl
│   ├── ground_truth.jsonl  (WP3)
│   └── results/            # Experiment logs
│
├── data/                   # Video corpus (gitignored)
│   ├── raw/bdd100k/
│   ├── processed/frames/
│   ├── embeddings/
│   └── indices/
│
├── deliverables/           # Submitted PDFs per WP
└── notebooks/              # Experimental notebooks
```

---

## 11. Evaluation Strategy

### Benchmark Design (WP3)

**Query Authoring:**
- Template-driven approach: 20 queries × 5 event types = 100 base queries
- Add 10–20 edge cases (nighttime, rain, occlusion)
- Ensure lexical diversity (unique tokens / total tokens > 0.6)
- Validate event-type distribution (chi-squared test for uniformity)

**Ground Truth Annotation:**
- Manual annotation by primary author (you)
- 10% cross-validation by second annotator (if available)
- Binary relevance: frame is relevant (1) or not (0)
- Annotate frame ranges (start_frame, end_frame) for each query

**Dataset Split:**
| Split | Percentage | Purpose |
|-------|------------|---------|
| Train | 60% | Hyperparameter tuning (τ_high, τ_low) |
| Validation | 20% | Model selection, prompt engineering |
| Test | 20% | Final evaluation (held-out, untouched) |

---

### Metrics Computation

#### Retrieval Metrics (R@K, MRR, nDCG)

```python
# Recall@K: Was at least one relevant frame in top-K?
def recall_at_k(predictions, ground_truth, k):
    hits = 0
    for query_id in predictions:
        top_k = predictions[query_id][:k]
        relevant = ground_truth[query_id]
        if any(frame_id in relevant for frame_id in top_k):
            hits += 1
    return hits / len(predictions)

# Mean Reciprocal Rank
def mean_reciprocal_rank(predictions, ground_truth):
    reciprocal_ranks = []
    for query_id in predictions:
        ranked_list = predictions[query_id]
        relevant = ground_truth[query_id]
        for rank, frame_id in enumerate(ranked_list, 1):
            if frame_id in relevant:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
```

#### Classification Metrics (Precision, Recall, F1)

```python
# Top-K Precision: Fraction of top-K that are relevant
def precision_at_k(predictions, ground_truth, k):
    precisions = []
    for query_id in predictions:
        top_k = predictions[query_id][:k]
        relevant = ground_truth[query_id]
        tp = sum(1 for frame_id in top_k if frame_id in relevant)
        precisions.append(tp / k)
    return sum(precisions) / len(precisions)

# Top-K F1
def f1_at_k(predictions, ground_truth, k):
    p = precision_at_k(predictions, ground_truth, k)
    r = recall_at_k(predictions, ground_truth, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0
```

---

## 12. Risk Management

### Top 6 Risks (Likelihood × Impact)

| # | Risk | L | I | Score | Mitigation |
|---|------|---|---|-------|------------|
| 1 | **Dataset Access Constraints** | 2 | 3 | 6 | Verify BDD100K access by WP2; fallback: Waymo, YouTube, CARLA |
| 2 | **CLIP Domain Gap** | 2 | 3 | 6 | Measure R@20 in WP4; if <0.70, fine-tune CLIP in WP5 |
| 3 | **Gemini Rate Limits** | 3 | 2 | 6 | Async API calls + caching; upgrade to paid tier if needed |
| 4 | **Gemini API Cost Overrun** | 2 | 2 | 4 | Set billing alerts ($25, $50, $75); fallback: GPT-4o mini |
| 5 | **Benchmark Diversity** | 1 | 2 | 2 | Template-driven query authoring; measure lexical diversity |
| 6 | **WP1 Defense Pivot** | 1 | 3 | 3 | Modular design; router can be removed without breaking system |

**Risk Monitoring:**
- Weekly review (WP5 onward)
- Per-WP retrospective
- Update likelihood/impact as new info emerges

---

## 13. 10-Week Roadmap

| WP | Title | Duration | Key Deliverables | Git Tag |
|----|-------|----------|------------------|---------|
| **WP1** | Planning & Preparatory Report | Weeks 1–2 | Research question, architecture, risk register, repo scaffolding | `v0.1.0-wp1` |
| **WP2** | Business Plan | Week 3 | Market analysis, user personas, go-to-market strategy | `v0.2.0-wp2` |
| **WP3** | Data Acquisition & Benchmark | Week 3–4 | BDD100K download, 100+ queries, ground truth annotation | `v0.3.0-wp3` |
| **WP4** | Retriever Implementation | Week 4–5 | CLIP encoding, FAISS indexing, CLIP-only baseline | `v0.4.0-wp4` |
| **WP5** | Reasoner & Router Integration | Week 5–6 | Gemini API wrapper, prompt templates, router logic | `v0.5.0-wp5` |
| **WP6** | Evaluation Harness | Week 6–7 | Metrics implementation, dual-agent evaluation | `v0.6.0-wp6` |
| **WP7** | Baseline Comparisons | Week 7–8 | Gemini-only baseline, full benchmark run on 3 systems | `v0.7.0-wp7` |
| **WP8** | Results Analysis | Week 8 | Ablation studies, failure analysis, visualizations | `v0.8.0-wp8` |
| **WP9** | Final Report | Week 9 | 30–50 page report with all results | `v0.9.0-wp9` |
| **WP10** | Defense | Week 10 | Presentation, Q&A, final submission | `v1.0.0-wp10` |

---

## 14. Relationship to Existing Work

### Flagship Reference

> Galanopoulos et al., *"An LLM Framework for Long-form Video Retrieval,"* **CVPRW 2025**

### What We Adopt

1. **Retrieve-Then-Reason Paradigm:** Two-stage pipeline (fast filter → deep analysis)
2. **Rank-Based Evaluation:** R@K, MRR, nDCG metrics
3. **Natural-Language Queries:** Free-form text input (not keywords)

### Where We Diverge (Novel Contributions)

1. **Domain:** Security/dashcam footage (not general long-form video)
2. **Confidence-Gated Router:** Threshold-based escalation to minimize LLM calls (not in flagship work)
3. **Dual-Baseline Comparison:** Explicit cost/accuracy trade-off vs. both CLIP-only and Gemini-only
4. **Token-Cost as Success Metric:** Formal measurement of economic viability

---

## 15. Deliverables

### Per Work Package

| WP | Document | Code | Data |
|----|----------|------|------|
| WP1 | Preparatory report (PDF) | Repo scaffolding | — |
| WP2 | Business plan (PDF) | — | — |
| WP3 | Data acquisition report | Frame extraction scripts | BDD100K subset, ground truth |
| WP4 | Retriever report | CLIP encoder, FAISS index | CLIP embeddings, FAISS index |
| WP5 | Integration report | Router, reasoner modules | Prompt templates |
| WP6 | Evaluation report | Metrics harness | Dual-agent results |
| WP7 | Baseline comparison | Baseline implementations | All 3 system results |
| WP8 | Results analysis | Ablation scripts | Failure analysis, plots |
| WP9 | Final report (PDF) | Final codebase | — |
| WP10 | Defense slides (PDF) | — | — |

### Final Artifacts (WP10)

1. **GitHub Repository:** All code, docs, evaluation harness
2. **Final Report:** 30–50 page PDF with full results
3. **Benchmark Dataset:** 100+ queries + ground truth (published)
4. **Pretrained Models:** FAISS index + router checkpoints (if applicable)
5. **Defense Presentation:** Slides + demo video

---

## 16. Open Questions

### To Be Resolved in WP3–WP6

1. **Optimal Confidence Thresholds:**
   - Should τ_high and τ_low be global or event-type-specific?
   - **Resolution:** Grid search on validation set (WP5)

2. **CLIP Domain Gap Magnitude:**
   - How much does CLIP performance degrade on dashcam vs. web images?
   - **Resolution:** Measure R@20 on WP4 baseline; if <0.70, trigger fine-tuning

3. **Gemini Prompt Sensitivity:**
   - What's the marginal gain of structured JSON vs. free-text rationale?
   - **Resolution:** Ablation study in WP6

4. **Minimum Query Count per Event Class:**
   - How many queries needed for statistical significance?
   - **Resolution:** Power analysis in WP3 (target: 20+ per class)

5. **Router Decision Distribution:**
   - What percentage of queries will actually skip Gemini?
   - **Resolution:** Empirical measurement on validation set (WP6)

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **ANN** | Approximate Nearest Neighbor (fast similarity search) |
| **CLIP** | Contrastive Language-Image Pretraining (OpenAI's vision-language model) |
| **FAISS** | Facebook AI Similarity Search (vector index library) |
| **F1** | Harmonic mean of precision and recall |
| **MRR** | Mean Reciprocal Rank (average 1/rank of first relevant result) |
| **nDCG** | Normalized Discounted Cumulative Gain (rank-weighted relevance metric) |
| **VLM** | Vision-Language Model (multimodal AI that understands images + text) |
| **τ_high, τ_low** | Confidence thresholds for router (tau high, tau low) |
| **R@K** | Recall at K (was relevant item in top-K results?) |

---

## Appendix B: Quick Reference

### Success Criteria (1-Pager)

**Passing Criteria:**
- ✅ Top-5 F1 ≥ 0.80
- ✅ F1 lift over CLIP-only ≥ +15 points
- ✅ Token reduction >90% vs. Gemini-only
- ✅ Latency (p95) <3 seconds
- ✅ F1 gap vs. Gemini-only ≤5 points

**If 4/5 pass:** Partial success; document failure mode

---

**Document Maintained By:** Koby Lev
**Repository:** [AI-Video-Investigator](https://github.com/YOUR_USERNAME/AI-Video-Investigator)
**Last Updated:** 2026-05-16
**Status:** WP1 — Planning (Defense: Sunday, 2026-05-18)
