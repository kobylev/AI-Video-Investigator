# Risk Register

## AI Video Investigator — Project Risks & Mitigations

This document identifies key risks to the successful completion of the AI Video Investigator project, scored by likelihood and impact, with mitigation strategies for each.

---

## Risk Scoring Framework

- **Likelihood:** Low (1), Medium (2), High (3)
- **Impact:** Low (1), Medium (2), High (3)
- **Risk Score:** Likelihood × Impact (range: 1–9)
- **Priority:** Critical (7–9), High (4–6), Medium (2–3), Low (1)

---

## Risk 1: Dataset Access & Licensing Constraints

### Description

The project depends on the **BDD100K dataset** (Berkeley DeepDrive) for dashcam footage. If the dataset is unavailable, paywalled, or requires institutional approval, data acquisition (WP3) will be delayed or blocked.

**Affected Work Packages:** WP3 (Data Acquisition), WP4 (Retriever), WP6 (Evaluation)

### Likelihood × Impact

- **Likelihood:** Medium (2) — BDD100K is publicly available but requires registration
- **Impact:** High (3) — No dataset = no experiments
- **Risk Score:** **6 (High Priority)**

### Mitigation Strategy

1. **Primary:** Verify BDD100K access **before WP2** (within 1 week)
   - Register for dataset access immediately
   - Download a 10-hour subset (100 videos) to local storage
   - Confirm license permits academic use

2. **Fallback Dataset:**
   - **Waymo Open Dataset** (alternative dashcam source, academic-friendly license)
   - **YouTube dashcam compilations** (scrape with youtube-dl, check Creative Commons licenses)
   - **Synthetic data:** Use CARLA simulator to generate labeled dashcam footage (last resort)

3. **Benchmark Curation Contingency:**
   - If full 10-hour corpus unavailable, proceed with ≥5 hours + 50 queries
   - Adjust success criteria proportionally (e.g., ≥50 queries instead of 100)

**Responsible Party:** Koby Lev (complete by end of WP2)

---

## Risk 2: Gemini API Cost Overrun

### Description

Gemini 1.5 Pro API pricing is **~$1.25 per million input tokens**. If experiments exceed budget (e.g., due to debugging, failed runs, or underestimated token counts), the project may become financially infeasible.

**Affected Work Packages:** WP5 (Reasoner), WP6 (Evaluation), WP7 (Baselines)

### Likelihood × Impact

- **Likelihood:** Medium (2) — Cost controls planned, but estimation is uncertain
- **Impact:** Medium (2) — Could force scope reduction (fewer queries, smaller benchmark)
- **Risk Score:** **4 (High Priority)**

### Mitigation Strategy

1. **Cost Monitoring:**
   - Set up **Google Cloud billing alerts** at $25, $50, $75 thresholds
   - Log token usage per experiment to `evals/results/token_log.csv`
   - Track cumulative spend in weekly status reports

2. **Cost Control Measures:**
   - **Batch API calls:** Use Gemini's batch API to reduce per-request overhead
   - **Reduce K during development:** Use K=5 for debugging, K=20 for final eval
   - **Cache Gemini responses:** Store API responses locally; avoid re-running identical queries
   - **Subset evaluation:** Run full benchmark on 20% sample first, extrapolate cost before full run

3. **Fallback Model:**
   - If Gemini costs exceed $100, switch to **GPT-4o mini** (cheaper alternative, ~$0.15/1M tokens)
   - Trade-off: Slightly lower accuracy, but still multimodal VLM capability
   - Update research question to compare "frontier VLM" instead of "Gemini specifically"

**Responsible Party:** Koby Lev (monitor weekly starting WP5)

---

## Risk 3: CLIP Domain Gap on Dashcam Footage

### Description

CLIP was trained on **web images (LAION-5B)**, which may not generalize well to dashcam footage (low resolution, motion blur, constrained viewpoints, weather/lighting variation). If CLIP retrieval recall is poor, the retrieve-then-reason pipeline may fail.

**Affected Work Packages:** WP4 (Retriever), WP6 (Evaluation), WP7 (Baselines)

### Likelihood × Impact

- **Likelihood:** Medium (2) — Domain shift is known in CV literature; magnitude uncertain
- **Impact:** High (3) — If CLIP recall <0.70, Gemini cannot recover (garbage in, garbage out)
- **Risk Score:** **6 (High Priority)**

### Mitigation Strategy

1. **Early Baseline Measurement (WP4):**
   - Implement CLIP-only baseline **first** (before router/reasoner)
   - Measure Recall@20 on a 20-query validation set
   - If R@20 < 0.70, proceed to fallback plan

2. **Model Selection Alternatives:**
   - **OpenCLIP ViT-H/14:** Larger CLIP variant, may generalize better
   - **SigLIP:** Google's successor to CLIP, trained on cleaner data
   - **BLIP-2:** Bootstrapped VLM with stronger image understanding

3. **Fine-Tuning Contingency (WP5):**
   - If off-the-shelf CLIP fails, fine-tune on **BDD100K object detection annotations**
   - Use LoRA (low-rank adaptation) to minimize compute cost
   - Budget: 2–3 days of fine-tuning on single GPU
   - Accept fine-tuning as a contribution (domain adaptation of CLIP to dashcam footage)

4. **Adjust Success Criteria:**
   - If CLIP baseline F1 is 0.50 instead of 0.65, target dual-agent F1 ≥ 0.75 (still +15 point lift)

**Responsible Party:** Koby Lev (validate in WP4, decide on mitigation by end of WP4)

---

## Risk 4: Gemini API Rate Limits & Latency Variability

### Description

Google Cloud enforces **rate limits** on Gemini API (e.g., 60 requests/minute for free tier, 300 requests/minute for paid). If evaluation requires 100 queries × 20 frames = 2,000 API calls, this could take 30+ minutes or hit throttling errors.

Additionally, Gemini latency may be **variable** (p95 > p50 by 2–3x), risking the <3s latency target.

**Affected Work Packages:** WP5 (Reasoner), WP6 (Evaluation)

### Likelihood × Impact

- **Likelihood:** High (3) — Rate limits are documented; latency variance is common in cloud APIs
- **Impact:** Medium (2) — Slows evaluation timeline; may require asynchronous architecture
- **Risk Score:** **6 (High Priority)**

### Mitigation Strategy

1. **Parallelize API Calls with Backoff:**
   - Use `asyncio` + exponential backoff to handle rate limit errors gracefully
   - Batch requests into chunks of 50–100 to stay under rate limits
   - Implement retry logic with jitter to avoid thundering herd

2. **Caching Layer:**
   - Store Gemini API responses in `evals/results/gemini_cache.jsonl`
   - For repeated experiments (e.g., hyperparameter tuning), load from cache instead of re-querying
   - Reduces API calls by 80–90% during development

3. **Upgrade to Paid Tier:**
   - If rate limits block progress, upgrade to **Google Cloud paid tier** ($0 base cost, pay-per-use)
   - Paid tier increases rate limit to 300 RPM (5x improvement)
   - Budget: $50 contingency fund for API costs

4. **Latency Monitoring:**
   - Log p50, p95, p99 latency per API call
   - If p95 > 2.5s, file issue with Google Cloud support or switch to batch API
   - Fallback: Report latency **without** Gemini overhead (isolate CLIP+routing latency)

5. **Alternative: Self-Hosted Model:**
   - If Gemini latency/rate limits are prohibitive, switch to **LLaVA 1.6 (34B)** self-hosted on GPU
   - Trade-off: Lower accuracy, but full control over latency/throughput
   - Requires access to GPU cluster (check university resources)

**Responsible Party:** Koby Lev (implement caching in WP5, monitor latency in WP6)

---

## Risk 5: Insufficient Benchmark Query Diversity

### Description

The evaluation benchmark requires **≥100 queries** spanning 5 event types. If query authoring is rushed or biased, the benchmark may lack diversity, leading to overfitting or non-generalizable results.

**Affected Work Packages:** WP3 (Data Acquisition), WP6 (Evaluation)

### Likelihood × Impact

- **Likelihood:** Low (1) — Query authoring is manual and controllable
- **Impact:** Medium (2) — Poor benchmark = weak experimental validation
- **Risk Score:** **2 (Medium Priority)**

### Mitigation Strategy

1. **Query Authoring Protocol (WP3):**
   - Use a **template-driven approach** to ensure coverage:
     - 20 queries per event type × 5 event types = 100 base queries
     - Add 10–20 edge cases (nighttime, rain, occlusion)
   - Involve a second annotator to review query quality

2. **Diversity Metrics:**
   - Measure query **lexical diversity** (unique tokens / total tokens)
   - Ensure no event type is over-represented (chi-squared test)

3. **Pilot Study:**
   - Run CLIP-only baseline on first 20 queries
   - If all queries trivially answered (>90% accuracy), add harder queries
   - Iterate until baseline accuracy is 60–70% (challenging but not impossible)

**Responsible Party:** Koby Lev (complete in WP3)

---

## Risk 6: WP1 Defense Feedback Requires Architecture Pivot

### Description

If the lecturer/advisor provides critical feedback during the WP1 defense (e.g., "dual-agent routing is over-engineered" or "benchmark is too small"), the project scope may need significant revision, delaying WP2–WP3.

**Affected Work Packages:** WP2 (Business Plan), WP3 (Data Acquisition)

### Likelihood × Impact

- **Likelihood:** Low (1) — Architecture is grounded in published work (Galanopoulos et al.)
- **Impact:** High (3) — Scope change could cascade to all downstream WPs
- **Risk Score:** **3 (Medium Priority)**

### Mitigation Strategy

1. **Proactive Alignment:**
   - Share architecture diagram and research question with advisor **before** defense (if possible)
   - Incorporate feedback into final preparatory report

2. **Modular Design:**
   - Ensure system components (retriever, router, reasoner) are **decoupled**
   - If router is removed, system degrades gracefully to standard retrieve-then-reason
   - If benchmark size is challenged, argue for "proof-of-concept" and commit to expansion in future work

3. **Defense Preparation:**
   - Rehearse answers to Q1–Q3 from WP1_DEFENSE_PREP.md (token economics, metrics, GitHub baseline)
   - Lead with the 60-second elevator pitch to frame the discussion

**Responsible Party:** Koby Lev (prepare defense by Saturday night)

---

## Summary Table

| # | Risk | Likelihood | Impact | Score | Priority | Mitigation Owner |
|---|------|------------|--------|-------|----------|------------------|
| 1 | Dataset Access Constraints | 2 | 3 | 6 | High | Koby Lev |
| 2 | Gemini API Cost Overrun | 2 | 2 | 4 | High | Koby Lev |
| 3 | CLIP Domain Gap | 2 | 3 | 6 | High | Koby Lev |
| 4 | Gemini Rate Limits & Latency | 3 | 2 | 6 | High | Koby Lev |
| 5 | Insufficient Benchmark Diversity | 1 | 2 | 2 | Medium | Koby Lev |
| 6 | WP1 Defense Feedback Pivot | 1 | 3 | 3 | Medium | Koby Lev |

**Top 3 Risks (Score ≥ 6):**
1. Dataset Access (6) — Mitigation: Verify BDD100K access by end of WP2
2. CLIP Domain Gap (6) — Mitigation: Baseline measurement in WP4, fine-tuning contingency in WP5
3. Gemini Rate Limits (6) — Mitigation: Caching layer + async API calls in WP5

---

## Review Cadence

- **Weekly:** Monitor Gemini API costs and token usage (starting WP5)
- **Per WP:** Re-assess likelihood/impact as new information emerges
- **Post-WP1 Defense:** Update based on advisor feedback

---

**Version:** 0.1.0 | **Status:** WP1 — Planning | **Last Updated:** 2026-05-16
