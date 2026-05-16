# WP1 — Planning & Preparatory Report

**Status:** 🔄 In Progress
**Submission Date:** 2026-05-18 (target) — Defense scheduled for Sunday
**Git Tag:** `v0.1.0-wp1` (to be created post-defense)

---

## Objectives

- Define the research question with measurable success criteria
- Establish the system architecture (CLIP → Router → Gemini)
- Identify and mitigate project risks (dataset access, API costs, CLIP domain gap, rate limits)
- Create a 10-week roadmap with work package breakdown
- Scaffold the GitHub repository structure for reproducibility

---

## Deliverables

- [x] **Preparatory Report (Draft):** 10–15 page document covering problem statement, literature review, methodology, and Gantt chart
- [x] **GitHub Repository:** Public repo with initial scaffolding ([AI-Video-Investigator](https://github.com/))
- [x] **Architecture Diagram:** Mermaid flowchart of dual-agent pipeline ([docs/architecture.md](../architecture.md))
- [x] **Research Question:** Falsifiable, quantitative hypothesis ([docs/research_question.md](../research_question.md))
- [x] **Risk Register:** ≥4 risks with likelihood × impact × mitigation ([docs/risk_register.md](../risk_register.md))
- [x] **Flagship Paper Analysis:** Relationship to Galanopoulos et al. (CVPRW 2025) ([docs/flagship_paper_notes.md](../flagship_paper_notes.md))
- [ ] **Defense Presentation:** Slides or notes for Sunday's defense meeting
- [ ] **Final Report PDF:** Exported to `deliverables/wp1/` post-defense

---

## Key Decisions

### Decision 1: Dual-Agent Architecture with Confidence Gating

- **Context:** Need to balance accuracy (frontier VLM reasoning) with cost/latency (retrieval speed)
- **Options Considered:**
  1. CLIP-only retrieval (fast, cheap, but low accuracy)
  2. Gemini-only analysis (accurate, but $1.16/query-hour and 60s latency)
  3. Retrieve-then-reason (flagship work approach)
  4. **Confidence-gated retrieve-then-reason** (this work)
- **Choice:** Option 4 — Dual-agent with router
- **Rationale:**
  - CLIP provides high recall at <100ms latency
  - Gemini provides precision on ambiguous cases
  - **Confidence gating** (τ_high, τ_low thresholds) skips Gemini calls for easy queries, reducing cost by 40–60%
  - This is a **novel contribution** over the flagship work
- **Impact:** System targets F1 ≥ 0.80, <$0.10/query-hour, <3s latency
- **Reference:** [docs/architecture.md](../architecture.md), [docs/research_question.md](../research_question.md)

### Decision 2: BDD100K as Primary Dataset

- **Context:** Need dashcam footage for domain-specific evaluation
- **Options Considered:**
  1. BDD100K (Berkeley DeepDrive, 100K dashcam videos)
  2. Waymo Open Dataset (self-driving car data)
  3. YouTube dashcam compilations (scraped, licensing unclear)
  4. CARLA simulator (synthetic data)
- **Choice:** BDD100K
- **Rationale:**
  - Public, academic-friendly license
  - Diverse driving conditions (urban, highway, weather, time-of-day)
  - Pre-existing annotations (object detection, lane markings) can be leveraged
  - Large enough to curate 10+ hour subset
- **Impact:** WP3 will focus on dataset acquisition and curation
- **Reference:** [data/README.md](../../data/README.md), [docs/risk_register.md](../risk_register.md) (Risk #1)

### Decision 3: Off-the-Shelf CLIP (No Fine-Tuning in WP1)

- **Context:** CLIP may suffer from domain gap (trained on web images, not dashcam footage)
- **Options Considered:**
  1. Use CLIP ViT-L/14 off-the-shelf (WP1 baseline)
  2. Fine-tune CLIP on BDD100K object detection annotations (deferred to WP5)
  3. Use alternative retriever (SigLIP, BLIP-2)
- **Choice:** Option 1 for WP1; keep Option 2 as WP5 risk mitigation
- **Rationale:**
  - Establish baseline performance first (scientific rigor)
  - Fine-tuning is time-intensive and may not be necessary
  - Aligns with flagship work (they used off-the-shelf CLIP)
  - If R@20 < 0.70 in WP4, trigger fine-tuning fallback
- **Impact:** WP4 will measure CLIP-only baseline to quantify domain gap
- **Reference:** [docs/flagship_paper_notes.md](../flagship_paper_notes.md), [docs/risk_register.md](../risk_register.md) (Risk #3)

### Decision 4: Dual-Baseline Comparison (CLIP-only + Gemini-only)

- **Context:** Need to prove value of dual-agent routing over single-agent alternatives
- **Options Considered:**
  1. Compare only against CLIP-only (standard retrieval baseline)
  2. Compare only against Gemini-only (standard VLM baseline)
  3. **Compare against both** (comprehensive evaluation)
- **Choice:** Option 3 — Dual baselines
- **Rationale:**
  - CLIP-only measures accuracy gain from reasoning
  - Gemini-only measures cost/latency reduction from retrieval
  - Together, they bound the solution space (cost-accuracy Pareto frontier)
  - Flagship work lacks explicit Gemini-only comparison
- **Impact:** WP7 will implement both baselines and run full benchmark
- **Reference:** [docs/research_question.md](../research_question.md)

---

## Outputs Feeding Next WP

- **Architecture Document:** WP2 (Business Plan) will reference system design for market positioning
- **Research Question:** WP3 (Data Acquisition) will design benchmark to answer this question
- **Risk Register:** WP2–WP10 will monitor and update risks as project progresses
- **Repository Scaffolding:** All future WPs will commit code/data/results to this structure
- **Flagship Paper Notes:** WP6 (Evaluation) will use the same rank-based metrics for comparability

---

## Open Issues / Carried Forward

- **Issue 1: Optimal Confidence Thresholds (τ_high, τ_low)**
  - **Description:** Initial values (0.85, 0.60) are estimated from literature; need empirical tuning
  - **Resolution:** WP5 will perform grid search on validation set
  - **Question:** Should thresholds be global or event-type-specific?

- **Issue 2: CLIP Domain Gap Magnitude**
  - **Description:** Unknown how much CLIP performance degrades on dashcam footage
  - **Resolution:** WP4 will measure R@K on curated benchmark
  - **Trigger:** If R@20 < 0.70, activate fine-tuning contingency in WP5

- **Issue 3: Gemini API Rate Limits**
  - **Description:** Free tier = 60 RPM, paid tier = 300 RPM; unclear if 100-query benchmark will hit limits
  - **Resolution:** WP5 will implement async API calls with exponential backoff and caching
  - **Contingency:** Upgrade to paid tier ($50 budget) if rate limits block progress

- **Issue 4: Benchmark Query Diversity**
  - **Description:** Need to ensure 100 queries cover all 5 event types without bias
  - **Resolution:** WP3 will use template-driven query authoring (20 queries × 5 event types)
  - **Validation:** Measure lexical diversity and chi-squared test for even distribution

---

## Defense Preparation (Sunday, 2026-05-18)

### The 60-Second Elevator Pitch

_To be delivered unprompted in the first 90 seconds:_

> Security operators and fleet-safety teams face cognitive overload at industrial scale: scrubbing 12 to 24 hours of dashcam footage to find a single event—a near-miss, a red jacket, a midnight license plate. Today's tools force a brutal trade-off. Keyword tagging misses semantic nuance. Feeding raw video into a frontier model like Gemini 1.5 Pro costs over a dollar per query-hour and stalls any interactive workflow.
>
> **AI Video Investigator breaks that trade-off with dual-agent routing.** A CLIP retriever acts as a millisecond semantic filter, surfacing the top-K candidate frames. Only those candidates are escalated to Gemini 1.5 Pro for deep reasoning—*"is this actually a hit-and-run, or just a fender-bender?"*
>
> Success is measured on four axes: top-5 retrieval **F1 above 0.80**, **a 15+ point accuracy lift** over CLIP-alone, **sub-3-second end-to-end latency**, and **over 90% token-cost reduction** versus a Gemini-only baseline. One architecture. Four numbers. Sixty seconds.

### Anticipated Questions & Prepared Answers

**Q1: Token Economics**
> "Why not just send the whole video to Gemini 1.5 Pro?"

**A1:** Cost, latency, and architectural discipline. One hour of video at 1 fps = 930K tokens = $1.16 per query-hour. A 100-vehicle fleet generates $9,000/day in API costs. Gemini reasoning takes 60+ seconds, destroying interactive workflows. The retrieve-then-reason pattern is canonical in IR—CLIP encodes once, every query is sub-millisecond cosine search, and Gemini analyzes only the 0.1% of frames that need reasoning.

**Q2: Metrics Strategy**
> "How do you combine rank-based and classification metrics?"

**A2:** Each metric measures a different pipeline stage. CLIP retrieval (ranking problem) → R@K, MRR, nDCG. Gemini reasoning (classification problem) → Precision, Recall, F1. End-to-end → R@5 with binary relevance. System metrics → latency and token cost. Five metric families, each justified by what it measures. No cherry-picking—every baseline scored on every metric.

**Q3: GitHub Baseline**
> "What's your unique contribution?"

**A3:** Three open-source foundations (clip-retrieval, CLIP ViT-L/14, Gemini API) + three novel pieces: (1) confidence-gated router to minimize LLM calls, (2) domain-specific prompt pack for dashcam events, (3) reproducible dual-baseline benchmark. The novelty is the orchestration layer, not the components.

### See Also

This work package references the **WP1 Defense Preparation** document for detailed Q&A simulation and tactical defense notes:
- [WP1_DEFENSE_PREP.md](../../WP1_DEFENSE_PREP.md)

---

## Retrospective (Post-Defense)

_To be filled in after WP1 defense on 2026-05-18_

### What Went Well

- TBD

### What Didn't Go Well

- TBD

### Lessons Learned

- TBD

### Defense Feedback Summary

- TBD

### Time Spent

- **Planned:** 40 hours (Week 1 + Week 2)
- **Actual:** TBD
- **Delta:** TBD

---

**Author:** Koby Lev
**Last Updated:** 2026-05-16
