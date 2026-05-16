# Preparatory Report (דוח מכין)

## AI Video Investigator — Dual-Agent Semantic Video Retrieval for Security and Dashcam Footage

**Work Package:** WP1 — Planning
**Author:** Koby Lev
**Defense Date:** Sunday, 2026-05-18
**Status:** Ready for Defense

---

## Abstract

Security operators and fleet-safety teams face cognitive overload when searching 12–24 hours of dashcam or CCTV footage for specific events. Current solutions force a brutal trade-off: keyword tagging is fast but brittle, while frontier vision-language models (Gemini 1.5 Pro) are accurate but prohibitively expensive (~$1.16 per query-hour) and slow (~60s latency).

This project proposes **AI Video Investigator**, a dual-agent architecture that breaks this trade-off by combining CLIP retrieval (fast semantic filtering) with confidence-gated routing to Gemini 1.5 Pro (deep reasoning on ambiguous cases only). The system targets **top-5 F1 ≥ 0.80**, **sub-3-second latency**, and **>90% token-cost reduction** versus a Gemini-only baseline, evaluated on a benchmark of ≥100 natural-language queries over ≥10 hours of BDD100K dashcam footage.

**Keywords:** Video retrieval, Vision-language models, CLIP, Gemini, Confidence gating, Dashcam footage, Semantic search

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [VALID Framework Analysis](#3-valid-framework-analysis)
4. [Research Question](#4-research-question)
5. [Literature Review](#5-literature-review)
6. [System Architecture](#6-system-architecture)
7. [Methodology](#7-methodology)
8. [Token Economics](#8-token-economics)
9. [Risk Management](#9-risk-management)
10. [Project Plan](#10-project-plan-10-week-gantt-chart)
11. [Expected Contributions](#11-expected-contributions)
12. [References](#12-references)

---

## 1. Introduction

### 1.1 The 60-Second Elevator Pitch

Security operators and fleet-safety teams share one nightmare: **scrubbing 12 to 24 hours of dashcam or CCTV footage to find a single event**—a near-miss, a red jacket, a midnight license plate. This is cognitive overload at industrial scale.

Today's tools force a brutal trade-off:
- **Keyword tagging** misses semantic nuance (*"pedestrian running across street"* ≠ *"jaywalking"*)
- **Frontier VLMs** (Gemini 1.5 Pro) are accurate but cost **$1.16 per query-hour** with **60-second latency**

**AI Video Investigator breaks that trade-off with dual-agent routing.**

A CLIP retriever acts as a **millisecond semantic filter**, surfacing top-K candidate frames out of tens of thousands. Only those candidates are escalated to Gemini 1.5 Pro for deep reasoning—*"is this actually a hit-and-run, or just a fender-bender?"*

**Success is measured on four axes:**
1. **Top-5 F1 ≥ 0.80** (harmonic mean of precision and recall)
2. **+15 point accuracy lift** over CLIP-alone baseline
3. **Sub-3-second end-to-end latency** (p95)
4. **>90% token-cost reduction** versus Gemini-only baseline

One architecture. Four numbers. Sixty seconds.

---

### 1.2 Market Context

- **Fleet Management Market:** 200M+ commercial vehicles worldwide (Berg Insight 2024)
- **Surveillance Camera Market:** 1B+ cameras globally (IHS Markit 2025)
- **Current Manual Review Cost:** $50–$150 per incident (3–6 hours @ $15–25/hour analyst labor)
- **Opportunity:** Reduce per-incident cost to <$5 while improving recall from ~60% to >80%

**Value Proposition:**
| Metric | Current State | With AI Video Investigator | Value Created |
|--------|---------------|----------------------------|---------------|
| Investigation Time | 3–6 hours | <10 minutes | 95% time reduction |
| Cost per Incident | $50–$150 | <$5 | 90–97% cost reduction |
| Recall (Find Rate) | ~60% | >80% | +20 points accuracy |

---

## 2. Problem Statement

### 2.1 User Personas

**Primary Persona: Yossi, Fleet Safety Analyst**
- **Role:** Safety compliance officer at logistics company (200 vehicles)
- **Pain Point:** Must review 12–24 hours of dashcam footage per incident (near-miss, complaint, insurance claim)
- **Current Workflow:** Manual scrubbing at 4x speed, taking 3–6 hours per investigation
- **Success Criteria:** Reduce investigation time to <10 minutes while maintaining >80% recall

**Secondary Persona: Security Operations Center (SOC) Analyst**
- **Role:** Monitor 50+ CCTV cameras across commercial campus
- **Pain Point:** Retrospective searches for incidents reported 24+ hours later
- **Current Workflow:** Brute-force scanning or keyword tags (if available)
- **Success Criteria:** Query across multi-camera feeds in <3 seconds

---

### 2.2 The Core Tension

Security and fleet operators face a **false choice**:

**Option A: Keyword Tagging**
- ✅ Fast (metadata-based search)
- ❌ Brittle (misses semantic queries)
- ❌ Low recall (requires perfect annotation)

**Option B: Frontier VLM (Gemini 1.5 Pro)**
- ✅ Accurate (semantic understanding)
- ❌ Expensive (~$1.16 per query-hour)
- ❌ Slow (~60s latency for 1h video @ 1fps)

**The Gap:** Users need the precision of multimodal LLMs at the cost and latency of traditional retrieval systems.

---

## 3. VALID Framework Analysis

### 3.1 Value — Who Benefits?

**Primary Beneficiaries:**
1. **Fleet Safety Analysts:** 95% time reduction (6h → 10min per incident)
2. **SOC Operators:** Near-instant evidence retrieval (<3s vs. hours)
3. **Insurance Companies:** Faster claims processing + reduced fraud

**Market Scale:** $50B+ annually (video analytics market, TAM)

---

### 3.2 AI-Core — Why AI is Essential?

**Non-AI approaches fail because:**
1. **Semantic Gap:** Query *"pedestrian running across street outside crosswalk"* requires understanding action (running), spatial context (outside crosswalk), and scene semantics—impossible with rule-based systems
2. **Scale Impossibility:** 10h video @ 1fps = 36,000 frames; manual review at 4x speed = 2.5 hours (cannot scale to fleet operations)
3. **Multimodal Reasoning:** Query *"red car running red light"* requires visual recognition + spatial reasoning + temporal context (was car past stop line when signal turned red?)

**AI Components:**
- **CLIP:** Learned vision-language joint embedding (400M image-text pairs)
- **Gemini:** Large multimodal model with fine-grained reasoning
- **Router:** Learned thresholds via grid search on validation data

---

### 3.3 Learned — What Learns from Data?

1. **CLIP Encoder (Pre-Trained):** Joint embedding space learned via contrastive learning on 400M pairs
2. **Gemini 1.5 Pro (Pre-Trained):** Multimodal reasoning learned on trillions of tokens
3. **Confidence Router (This Work):** Thresholds τ_high, τ_low learned via grid search on validation set
4. **Evaluation Harness:** Ground-truth relevance annotations (manual labeling)

**Optional:** Fine-tune CLIP if domain gap detected (R@20 < 0.70 on dashcam footage)

---

### 3.4 Innovative — What's Novel?

**Four Innovations:**

1. **Confidence-Gated Routing (Type 3: Novel Combination)**
   - **Gap in flagship work:** Galanopoulos et al. send *all queries* to LLM (no conditional routing)
   - **Our contribution:** Skip 40–60% of LLM calls via confidence thresholds (τ_high, τ_low)
   - **Impact:** >90% token reduction while preserving F1 within 5 points of Gemini-only

2. **Domain Adaptation to Security/Dashcam (Type 2: Novel Application)**
   - **Gap:** Flagship work evaluates on general video (movies, instructional); CLIP trained on web images
   - **Our contribution:** First application to constrained dashcam domain with event-based query taxonomy
   - **Impact:** If CLIP fails (domain gap), document gap and propose fine-tuning; if succeeds, validate zero-shot generalization

3. **Event-Type-Specific Prompt Engineering (Type 3: Novel Combination)**
   - **Gap:** Flagship work uses generic prompts; no existing dashcam prompt library
   - **Our contribution:** 5 curated templates (vehicle interactions, pedestrian events, traffic violations, object-of-interest, ambient scenes) with JSON schemas
   - **Impact:** Structured output enables reproducible evaluation + ablation study potential

4. **Dual-Baseline Evaluation with Token-Cost Metric (Type 4: Novel Evaluation)**
   - **Gap:** Flagship work reports token counts informally; CLIP benchmarks ignore cost
   - **Our contribution:** Token-cost as first-class success criterion + cost-accuracy Pareto frontier
   - **Impact:** Economic feasibility proof (validates industrial deployment viability)

---

### 3.5 Doable — Why Feasible in 10 Weeks?

**Evidence:**
1. **No model training required:** Use pre-trained CLIP + Gemini API
2. **Data accessible:** BDD100K publicly available (free academic registration) + fallback datasets
3. **Compute affordable:** CPU-only for MVP; Gemini API ~$5–10 total for full evaluation
4. **Risks mitigated:** 6 critical risks, all with documented fallback plans
5. **Timeline realistic:** 270h ÷ 10 weeks = 27h/week (full-time academic project)
6. **Progress proof:** WP1 is 90% complete in Week 2 (on track)

**Scoped MVP:** Proof-of-concept (not production system); no real-time streaming, no multi-camera fusion, no custom model training.

---

## 4. Research Question

### 4.1 Primary Research Question (Measurable & Falsifiable)

> **To what extent does a dual-agent CLIP→Gemini routing architecture improve top-5 retrieval F1 and reduce inference token cost on long-form dashcam footage, compared to (a) a CLIP-only retrieval baseline and (b) a Gemini-only frame-analysis baseline, evaluated on a benchmark of ≥100 natural-language queries over ≥10 hours of curated video?**

---

### 4.2 Hypotheses

**H1 — Accuracy Hypothesis:**
- Dual-agent achieves **F1 ≥ 0.80**, outperforming CLIP-only by **≥15 points**
- **Rationale:** CLIP provides high recall (gets relevant frames into top-K), Gemini provides high precision (correctly re-ranks ambiguous cases)

**H2 — Cost Efficiency Hypothesis:**
- Dual-agent reduces token consumption by **>90%** vs. Gemini-only
- Accuracy preserved within **5 F1 points** of Gemini-only (minimal quality loss)

**H3 — Latency Hypothesis:**
- Dual-agent achieves **p95 latency < 3 seconds** for 10-hour corpus
- **Breakdown:** CLIP <100ms, Router <50ms, Gemini <2.5s

---

### 4.3 Success Criteria (Composite)

| # | Metric | Target | Comparison |
|---|--------|--------|------------|
| 1 | Top-5 F1 | ≥ 0.80 | Absolute threshold |
| 2 | F1 Lift over CLIP-only | ≥ +15 points | Relative improvement |
| 3 | Token Reduction vs. Gemini-only | > 90% | Cost efficiency |
| 4 | Latency (p95) | < 3 seconds | Usability threshold |
| 5 | F1 Gap vs. Gemini-only | ≤ 5 points | Accuracy preservation |

**Passing Criteria:** ALL 5 metrics must be satisfied.
**Partial Success:** 4/5 metrics → document failure mode, propose mitigation for future work.

---

## 5. Literature Review

### 5.1 Flagship Paper

> **Galanopoulos et al., "An LLM Framework for Long-form Video Retrieval," CVPRW 2025**

**Key Contributions:**
- Propose retrieve-then-reason paradigm: dense retriever (CLIP) → LLM re-ranking
- Benchmark on general long-form video (movies, instructional content)
- Report 95% token reduction vs. full-video LLM analysis
- Evaluate with rank-based metrics (R@K, MRR, nDCG)

**Relationship to This Work:**

| Aspect | Flagship Work | Our Work |
|--------|---------------|----------|
| **Paradigm** | Retrieve-then-reason | ✅ Adopted |
| **Metrics** | R@K, MRR, nDCG | ✅ Adopted + F1, Latency, Token-cost |
| **Domain** | General long-form video | ❌ Security/dashcam (novel application) |
| **Routing** | All queries → LLM | ❌ Confidence-gated (novel contribution) |
| **Cost Analysis** | Informal token counts | ❌ First-class metric (novel evaluation) |

**Gap Analysis:**
1. **No conditional routing:** Flagship work sends all queries to LLM; we introduce confidence gating to skip 40–60% of calls
2. **No domain adaptation study:** They use general video; we test on constrained dashcam domain and document domain gap
3. **No economic feasibility proof:** They focus on accuracy; we make token-cost a success criterion

---

### 5.2 Supporting Literature

**Vision-Language Models:**
- **Radford et al. (2021), "Learning Transferable Visual Models From Natural Language Supervision" (CLIP):** Foundational work on contrastive vision-language pretraining; 76% zero-shot ImageNet accuracy
- **Alayrac et al. (2022), "Flamingo: A Visual Language Model for Few-Shot Learning":** Demonstrates multimodal reasoning; inspiration for Gemini architecture

**Video Retrieval:**
- **Miech et al. (2020), "End-to-End Learning of Visual Representations from Uncurated Instructional Videos":** Dense video retrieval with text-video joint embeddings
- **Liu et al. (2021), "HiT: Hierarchical Transformer for Video-Text Retrieval":** Temporal reasoning over video frames

**Domain Adaptation:**
- **Ganin et al. (2016), "Domain-Adversarial Training of Neural Networks":** Foundational work on domain adaptation; motivates CLIP fine-tuning fallback if domain gap detected

**Dashcam/Autonomous Driving Datasets:**
- **Yu et al. (2020), "BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning":** Our source dataset (100K videos, object detection annotations)

---

### 5.3 Open Source Baseline

**Primary Baseline: `rom1504/clip-retrieval`**
- **Repository:** https://github.com/rom1504/clip-retrieval
- **Description:** Battle-tested CLIP indexing and ANN search infrastructure
- **License:** MIT (compatible with academic use)
- **Usage:** We use this for CLIP encoding + FAISS indexing (not reinvent the wheel)

**What We Build On Top:**
1. **Router Module:** Confidence-gated decision logic (not in clip-retrieval)
2. **Gemini Reasoner:** API wrapper + prompt templates (not in clip-retrieval)
3. **Evaluation Harness:** Metrics for dual-agent pipeline (not in clip-retrieval)

**Other Open-Source Components:**
- **OpenAI CLIP ViT-L/14:** Off-the-shelf encoder (frozen weights)
- **FAISS (Facebook AI Similarity Search):** Vector index library
- **Google Generative AI SDK:** Gemini API wrapper

---

## 6. System Architecture

### 6.1 High-Level Data Flow

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

(See `docs/architecture.md` for full Mermaid diagram)

---

### 6.2 Component Specifications

| Component | Technology | Performance Target |
|-----------|------------|-------------------|
| **CLIP Retriever** | OpenAI CLIP ViT-L/14 (768-dim) | <100ms for 36K frames |
| **Vector Index** | FAISS IVF or HNSW (cosine similarity) | <100ms ANN search |
| **Confidence Router** | Threshold comparison (τ_high=0.85, τ_low=0.60) | <50ms overhead |
| **Gemini Reasoner** | `gemini-1.5-pro-latest` via API | <2.5s for K=20 frames |

**Latency Budget (Worst Case):**
```
CLIP:    100ms
Router:   50ms
Gemini: 2,500ms
──────────────
Total:  2,650ms < 3,000ms target ✅
```

---

## 7. Methodology

### 7.1 Dataset: BDD100K

- **Source:** Berkeley DeepDrive 100K (https://www.bdd100k.com/)
- **Content:** 100K dashcam videos (40 seconds each, 720p resolution)
- **Subset:** 10 hours (900 videos, ~1% of full dataset)
- **License:** Academic and non-commercial use permitted
- **Sampling Strategy:** Stratified by scene type (urban, highway, residential), time-of-day (day/night), weather (clear/rain/fog)

---

### 7.2 Benchmark Curation (WP3)

**Query Authoring:**
- Template-driven approach: 20 queries × 5 event types = 100 base queries
- Event types: Vehicle interactions, Pedestrian events, Traffic violations, Object-of-interest, Ambient scenes
- Add 10–20 edge cases (nighttime, rain, occlusion)

**Ground Truth Annotation:**
- Manual annotation: binary relevance (relevant=1, irrelevant=0)
- Annotate frame ranges (start_frame, end_frame) for each query

**Dataset Split:**
- Train: 60% (hyperparameter tuning for router thresholds)
- Validation: 20% (prompt engineering, model selection)
- Test: 20% (final evaluation, held-out)

---

### 7.3 Evaluation Metrics

**Retrieval Metrics (CLIP Stage):**
- Recall@K (R@1, R@5, R@10): Was at least one relevant frame in top-K?
- Mean Reciprocal Rank (MRR): Average 1/rank of first relevant result
- Normalized Discounted Cumulative Gain (nDCG): Rank-weighted relevance

**Classification Metrics (Gemini Stage):**
- Precision: Fraction of returned frames that are relevant
- Recall: Fraction of relevant frames that were returned
- F1 Score: Harmonic mean of precision and recall
- Accuracy: Binary relevance correctness

**End-to-End Metrics (Pipeline):**
- Recall@5: Was at least one relevant frame in top-5? (primary metric)
- Top-5 F1: Harmonic mean of P/R at K=5

**System Metrics (Architecture):**
- Latency (p50, p95, p99): Distribution of query latencies
- Tokens per query: Mean and std of Gemini token consumption
- Cost per query: Mean cost in USD
- Router decision distribution: % queries skipped / expanded / escalated

---

### 7.4 Baselines

**Baseline 1: CLIP-Only**
- Query → CLIP text encoder → FAISS search → Top-5
- **Purpose:** Measure reasoning value (F1 lift from Gemini)

**Baseline 2: Gemini-Only**
- Query → Send all 36,000 frames to Gemini → Rank → Top-5
- **Purpose:** Measure efficiency gain (token reduction from routing)

**Baseline 3: Dual-Agent (This Work)**
- Query → CLIP → Router → Gemini (conditional) → Top-5
- **Purpose:** Target F1 ≥ 0.80, >90% token reduction, <3s latency

---

## 8. Token Economics

### 8.1 Cost Breakdown

**Gemini-Only Baseline (Worst Case):**
```
10 hours @ 1fps = 36,000 frames
36,000 frames × 258 tokens/frame = 9,288,000 tokens
9,288,000 tokens × $1.25/1M tokens = $11.61 per query (for 10h corpus)
```
**Corrected to per-query-hour:**
```
$11.61 ÷ 10 hours = $1.16 per query-hour
```

**Dual-Agent (Target):**
```
40–60% queries skip Gemini (high confidence)
Remaining 40–60% queries: 20 frames × 408 tokens/frame = 8,160 tokens
Expected tokens/query = 0.5 × 0 + 0.5 × 8,160 = 4,080 tokens/query
4,080 tokens × $1.25/1M tokens = $0.0051 per query
```
**Per 10-hour corpus:**
```
$0.0051 × 10 hours = $0.051 per query-hour
```

**Token Reduction:**
```
($1.16 - $0.051) / $1.16 = 95.6% reduction ✅ (exceeds 90% target)
```

---

### 8.2 Fleet-Scale Economics

**Scenario:** 100-vehicle fleet, 800 hours of footage per day, 10 queries per day

**Gemini-Only Cost:**
```
10 queries × $1.16/query-hour × 800 hours = $9,280 per day
$9,280 × 365 days = $3.39M per year
```

**Dual-Agent Cost:**
```
10 queries × $0.051/query-hour × 800 hours = $408 per day
$408 × 365 days = $149K per year
```

**Savings:** $3.24M per year (96% cost reduction)

**Insight:** Token economics is not academic curiosity—it's deployment feasibility proof.

---

## 9. Risk Management

(See `docs/risk_register.md` for full table with likelihood × impact × mitigation)

**Top 3 Critical Risks (Score ≥ 6):**

| Risk | L×I | Mitigation |
|------|-----|------------|
| **Dataset Access** | 2×3=6 | Verify BDD100K access by WP2; fallback: Waymo, YouTube, CARLA |
| **CLIP Domain Gap** | 2×3=6 | Measure R@20 in WP4; if <0.70, fine-tune CLIP in WP5 (LoRA, 2–3 days) |
| **Gemini Rate Limits** | 3×2=6 | Async API calls + caching; upgrade to paid tier if needed ($0 base cost) |

**All risks have documented fallback plans.** No unmitigated high-impact risks.

---

## 10. Project Plan (10-Week Gantt Chart)

### 10.1 Work Package Timeline

| Week | WP | Title | Key Deliverables | Git Tag |
|------|-----|-------|------------------|---------|
| 1–2 | **WP1** | Planning & Preparatory Report | Research question, architecture, risk register, repo scaffolding | `v0.1.0-wp1` |
| 3 | **WP2** | Business Plan | Market analysis, user personas, go-to-market strategy | `v0.2.0-wp2` |
| 3–4 | **WP3** | Data Acquisition & Benchmark | BDD100K download, 100+ queries, ground truth annotation | `v0.3.0-wp3` |
| 4–5 | **WP4** | Retriever Implementation | CLIP encoding, FAISS indexing, CLIP-only baseline | `v0.4.0-wp4` |
| 5–6 | **WP5** | Reasoner & Router Integration | Gemini API wrapper, prompt templates, router logic | `v0.5.0-wp5` |
| 6–7 | **WP6** | Evaluation Harness | Metrics implementation, dual-agent evaluation | `v0.6.0-wp6` |
| 7–8 | **WP7** | Baseline Comparisons | Gemini-only baseline, full benchmark run on 3 systems | `v0.7.0-wp7` |
| 8 | **WP8** | Results Analysis | Ablation studies, failure analysis, visualizations | `v0.8.0-wp8` |
| 9 | **WP9** | Final Report | 30–50 page report with all results | `v0.9.0-wp9` |
| 10 | **WP10** | Defense | Presentation, Q&A, final submission | `v1.0.0-wp10` |

---

### 10.2 Visual Gantt Chart (ASCII)

```
Week:  1   2   3   4   5   6   7   8   9   10
WP1:  ████████
WP2:          ████
WP3:          ████████
WP4:              ████████
WP5:                  ████████
WP6:                      ████████
WP7:                          ████████
WP8:                              ████
WP9:                                  ████
WP10:                                     ████
```

**Critical Path:** WP1 → WP3 → WP4 → WP5 → WP6 → WP7 → WP8 → WP9 → WP10

**Buffer:** Weeks 8–10 have lighter workload (WP8: 20h, WP9: 30h, WP10: 20h) to accommodate delays.

---

## 11. Expected Contributions

### 11.1 Academic Contributions

1. **Confidence-Gated Routing Algorithm:** Novel threshold-based decision layer for LLM escalation (not in existing retrieve-then-reason systems)
2. **Domain Adaptation Study:** First evaluation of CLIP + Gemini on security/dashcam footage (documents domain gap or validates zero-shot generalization)
3. **Reproducible Benchmark:** 100+ natural-language queries + ground truth for dashcam retrieval (open-sourced for future work)
4. **Token-Cost Optimization Framework:** Cost-accuracy Pareto frontier for retrieve-then-reason systems (generalizable to other domains)

---

### 11.2 Industrial Contributions

1. **Economic Feasibility Proof:** Demonstrate that frontier VLMs can be deployed at fleet scale (96% cost reduction)
2. **Latency-Cost Trade-Off Analysis:** Quantify when to escalate vs. trust retrieval alone (decision framework for practitioners)
3. **Event-Type-Specific Prompt Library:** 5 curated templates for dashcam analysis (reusable for security applications)

---

### 11.3 Software Artifacts

1. **GitHub Repository:** Complete codebase (retriever, router, reasoner, pipeline, evaluation harness)
2. **FAISS Index:** Pre-built index for 10-hour BDD100K subset (accelerates future research)
3. **Evaluation Scripts:** Metrics computation (R@K, MRR, nDCG, F1, Accuracy, Latency, Token-cost)
4. **Prompt Templates:** JSON schemas for 5 event types (vehicle, pedestrian, traffic violation, object, ambient)

---

## 12. References

### 12.1 Flagship Paper

- **Galanopoulos et al. (2025).** "An LLM Framework for Long-form Video Retrieval." *Computer Vision and Pattern Recognition Workshops (CVPRW)*.

### 12.2 Vision-Language Models

- **Radford, A., Kim, J. W., Hallacy, C., et al. (2021).** "Learning Transferable Visual Models From Natural Language Supervision." *ICML*.
- **Alayrac, J.-B., Donahue, J., Luc, P., et al. (2022).** "Flamingo: A Visual Language Model for Few-Shot Learning." *NeurIPS*.

### 12.3 Video Retrieval

- **Miech, A., Zhukov, D., Alayrac, J.-B., et al. (2020).** "End-to-End Learning of Visual Representations from Uncurated Instructional Videos." *CVPR*.
- **Liu, S., Fan, H., Qian, S., et al. (2021).** "HiT: Hierarchical Transformer for Video-Text Retrieval." *ACM MM*.

### 12.4 Domain Adaptation

- **Ganin, Y., Ustinova, E., Ajakan, H., et al. (2016).** "Domain-Adversarial Training of Neural Networks." *JMLR*.

### 12.5 Datasets

- **Yu, F., Chen, H., Wang, X., et al. (2020).** "BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning." *CVPR*.

### 12.6 Open Source Baselines

- **rom1504/clip-retrieval.** GitHub Repository. https://github.com/rom1504/clip-retrieval (Accessed 2026-05-16)

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
| **LMM** | Large Multimodal Model (VLM with billions of parameters, e.g., Gemini) |
| **τ_high, τ_low** | Confidence thresholds for router (tau high, tau low) |

---

## Appendix B: GitHub Repository Structure

See: https://github.com/YOUR_USERNAME/AI-Video-Investigator

```
.
├── docs/               # All documentation
├── src/                # Source code (retriever, router, reasoner, pipeline, eval)
├── evals/              # Benchmark queries + results
├── data/               # BDD100K subset (gitignored)
├── deliverables/       # Submitted PDFs per WP
├── notebooks/          # Experimental notebooks
├── README.md           # Project overview
├── MASTER_PRD.md       # Comprehensive PRD
└── WORK_PACKAGES.md    # 10-WP roadmap dashboard
```

---

**Document Version:** 1.0 (Ready for Defense)
**Author:** Koby Lev
**Last Updated:** 2026-05-16
**Status:** WP1 — Planning Complete
**Next Milestone:** Defense (Sunday, 2026-05-18) → Tag `v0.1.0-wp1` → Begin WP2
