# AI Video Investigator — WP1 Defense Preparation

> Internal preparation document for the Preparatory Report defense (דוח מכין).
> **Project:** AI Video Investigator — Dual-Agent Semantic Video Retrieval
> **Work Package:** WP1 — Planning
> **Status:** Pre-defense (Updated for Claude Haiku 4.5)

---

## 1. The 60-Second Elevator Pitch ("The Hook")

> Security operators and fleet-safety teams share one nightmare: scrubbing 12 to 24 hours of dashcam or CCTV footage to find a single event — a near-miss, a red jacket, a midnight license plate. This is cognitive overload at industrial scale, and today's tools force a brutal trade-off. Keyword tagging misses semantic nuance. Feeding raw video into a frontier model like Claude Haiku 4.5 costs ~$0.30 per query-hour and stalls any interactive workflow.
>
> **AI Video Investigator breaks that trade-off with dual-agent routing.** A CLIP retriever acts as a millisecond semantic filter, surfacing the top-K candidate frames out of tens of thousands. Only those candidates are escalated to Claude Haiku 4.5 for deep reasoning — *"is this actually a hit-and-run, or just a fender-bender?"*
>
> Success is measured on four axes: top-5 retrieval **F1 above 0.80**, **a 15+ point accuracy lift** over CLIP-alone, **sub-3-second end-to-end latency**, and **over 90% token-cost reduction** versus a Claude-only baseline. One architecture. Four numbers. Sixty seconds.

---

## 3. The Murder Board — Defense Q&A Simulation

### Q1 — Token Economics

> **"Why not just send the whole video to Claude Haiku 4.5? It has multimodal vision. Your dual-agent system is over-engineered."**

**Your answer — say this almost verbatim:**

> Three reasons: cost, latency, and architectural discipline.
>
> **On cost:** Brute-force cloud processing over an hour of 1fps video requires thousands of vision-token calls. Even with Haiku's aggressive pricing, this scales linearly to ~$0.30 per query-hour. A fleet of 100 vehicles queried ten times a day becomes a major OpEx line item.
>
> **On latency:** Cloud reasoning over thousands of frames takes minutes. That destroys any interactive workflow. Analysts will not wait.
>
> **On architecture:** this is the canonical **retrieve-then-re-rank** pattern. We get the precision of an Anthropic frontier model on the candidates where it matters, at the cost of a local CLIP model on the 99.9% of frames that are obviously irrelevant.

---

### Q2 — Metrics Strategy

> **"How exactly do you combine the rank-based scoring from your flagship paper with traditional F1 and accuracy?"**

**Your answer:**
- **Stage 1, CLIP retrieval:** ranking problem (Recall@K, MRR).
- **Stage 2, Claude reasoning:** classification problem (Precision, Recall, F1).
- **End-to-end:** R@5 with binary relevance.

---

### Q3 — GitHub Baseline

**Foundations:**
1. `rom1504/clip-retrieval` (CLIP + FAISS indexing)
2. OpenAI's CLIP (ViT-L/14)
3. Anthropic's Claude Haiku 4.5 API

**Novelty:**
1. Confidence-gated router (conditional escalation)
2. Domain-specific prompt pack (5 security event types, JSON tool-use)
3. Reproducible dual-agent evaluation harness.

---

## 4. Next Immediate Steps

```text
AI-Video-Investigator/
├── src/
│   ├── reasoner/claude_engine.py  # Anthropic SDK implementation
│   ├── router/core.py             # Confidence-gated escalation logic
```

---

**בהצלחה.**
