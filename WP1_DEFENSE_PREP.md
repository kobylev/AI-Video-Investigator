# AI Video Investigator — WP1 Defense Preparation

> Internal preparation document for the Preparatory Report defense (דוח מכין).
> **Project:** AI Video Investigator — Dual-Agent Semantic Video Retrieval
> **Work Package:** WP1 — Planning
> **Status:** Pre-defense

---

## Table of Contents

1. [The 60-Second Elevator Pitch](#1-the-60-second-elevator-pitch-the-hook)
2. [WP1 Readiness Checklist](#2-wp1-readiness-checklist)
3. [The Murder Board — Defense Q&A Simulation](#3-the-murder-board--defense-qa-simulation)
4. [Next Immediate Steps — Bridge to WP2](#4-next-immediate-steps--bridge-to-wp2)

---

## 1. The 60-Second Elevator Pitch ("The Hook")

> Security operators and fleet-safety teams share one nightmare: scrubbing 12 to 24 hours of dashcam or CCTV footage to find a single event — a near-miss, a red jacket, a midnight license plate. This is cognitive overload at industrial scale, and today's tools force a brutal trade-off. Keyword tagging misses semantic nuance. Feeding raw video into a frontier model like Gemini 1.5 Pro costs over a dollar per query-hour and stalls any interactive workflow.
>
> **AI Video Investigator breaks that trade-off with dual-agent routing.** A CLIP retriever acts as a millisecond semantic filter, surfacing the top-K candidate frames out of tens of thousands. Only those candidates are escalated to Gemini 1.5 Pro for deep reasoning — *"is this actually a hit-and-run, or just a fender-bender?"*
>
> Success is measured on four axes: top-5 retrieval **F1 above 0.80**, **a 15+ point accuracy lift** over CLIP-alone, **sub-3-second end-to-end latency**, and **over 90% token-cost reduction** versus a Gemini-only baseline. One architecture. Four numbers. Sixty seconds.

**Delivery notes:** ~160 words. Pause after "brutal trade-off" and again after "dual-agent routing" — those are the two beat-drops. End on the four numbers; do not soften them with a hedge.

---

## 2. WP1 Readiness Checklist

Print this. Tick it off Saturday night. Anything unchecked is a vulnerability the lecturer will find.

### Document Deliverables

- [ ] **Preparatory Report** finalized in Markdown or LaTeX, exported to PDF (10–15 pages typical for WP1).
- [ ] **Title page** with full Hebrew/English title, your name, lecturer's name, course code, date.
- [ ] **Abstract** (150–200 words) — a written version of the elevator pitch above.
- [ ] **Problem statement** with a named user persona (e.g., "Yossi, fleet safety analyst at a logistics company managing 200 vehicles").
- [ ] **Literature review section** anchored by the flagship reference and 4–6 supporting papers.

### Flagship Paper Integration

- [ ] Full citation included:
  > Galanopoulos et al., *"An LLM Framework for Long-form Video Retrieval,"* **CVPRW 2025**.
- [ ] Explicit paragraph titled **"Relationship to Flagship Work"** that states:
  - **(a)** what you adopt from it — the rank-based evaluation protocol, the retrieve-then-reason paradigm.
  - **(b)** where you diverge — domain (security/dashcam vs. general long-form) and the confidence-gated routing layer.

### The Measurable Research Question

Adopt or adapt this exact phrasing:

> *To what extent does a dual-agent CLIP→Gemini routing architecture improve top-5 retrieval F1 and reduce inference token cost on long-form dashcam footage, compared to (a) a CLIP-only retrieval baseline and (b) a Gemini-only frame-analysis baseline, evaluated on a benchmark of ≥100 natural-language queries over ≥10 hours of curated video?*

This sentence is gold because it is **falsifiable, quantitative, and names its baselines**. Memorize it.

### Methodology Section

- [ ] System architecture diagram (CLIP → Router → Gemini → Ranked Output).
- [ ] Dataset declaration: BDD100K for dashcam, plus a self-curated 10-hour query benchmark.
- [ ] Evaluation matrix (see Q2 in the Murder Board below).
- [ ] Risk register with at least 4 risks and mitigations.
- [ ] 10-week Gantt chart with WP1→WP10 milestones.

### GitHub Setup

- [ ] Public repository created: `AI-Video-Investigator` under your account.
- [ ] Initial commit pushed with the scaffolding from Section 4 of this document.
- [ ] Repo URL pasted into the cover page of the preparatory report.
- [ ] At least one tagged release: `v0.1.0-wp1`.

---

## 3. The Murder Board — Defense Q&A Simulation

### Q1 — Token Economics

> **"Why not just send the whole video to Gemini 1.5 Pro? It has a 2-million-token context window. Your dual-agent system is over-engineered."**

**Your answer — say this almost verbatim:**

> Three reasons: cost, latency, and architectural discipline.
>
> **On cost:** Gemini 1.5 Pro charges roughly $1.25 per million input tokens. One hour of video sampled at 1 fps is 3,600 frames at ~258 tokens each — about **930K tokens per query-hour**, before the prompt overhead. That's roughly **$1.16 per query per hour of footage**. A 100-vehicle fleet generating 800 hours of footage per day, queried even ten times, is **$9,000 a day**. The economics collapse at fleet scale.
>
> **On latency:** Gemini's reasoning over a multi-hundred-thousand-token video context takes tens of seconds to minutes. That destroys any interactive workflow — analysts will not wait two minutes per query, they'll go back to scrubbing manually.
>
> **On architecture:** this is the canonical **retrieve-then-re-rank** pattern from information retrieval. CLIP encodes the corpus *once*, then every query is a sub-millisecond cosine search. Gemini is invoked only on the top-K candidates that actually need reasoning. We get the precision of a frontier model on the candidates where it matters, at the cost of a small model on the 99.9% of frames that are obviously irrelevant. That isn't over-engineering — it's the same logic Google uses to rank the web before re-ranking with BERT.

> 💡 **Land the dollar number. The lecturer needs to feel the cost.**

---

### Q2 — Metrics Strategy

> **"How exactly do you combine the rank-based scoring from your flagship paper with traditional F1 and accuracy? Aren't you cherry-picking metrics?"**

**Your answer:**

> No — each metric measures a different stage of the pipeline, and the combination is principled.
>
> **Stage 1, CLIP retrieval, is a ranking problem.** I evaluate it with the same rank-based protocol as Galanopoulos et al.: **Recall@1, Recall@5, Recall@10, Mean Reciprocal Rank (MRR), and nDCG**. The right question here is *"did the correct frame appear in the top-K?"* — not whether a single frame was labeled correctly.
>
> **Stage 2, Gemini reasoning, is a classification problem.** Given a candidate frame and a query, did it correctly judge relevance? Here **Precision, Recall, F1, and Accuracy** are the right tools.
>
> **End-to-end, the system is evaluated on Recall@5 with binary relevance** — was at least one truly relevant frame present in the final top-5 output after re-ranking? This gives me a single headline number that captures both stages.
>
> On top of that I report two **system metrics** that no single stage owns: **end-to-end latency in seconds**, and **cost in tokens-per-query**. These are required to defend the dual-agent design itself.
>
> So: rank metrics for retrieval, classification metrics for reasoning, R@5 for the pipeline, and latency-plus-cost for the architecture. Five metric families, each justified by what it measures. No cherry-picking — every baseline is scored on every metric.

**Metric Matrix Reference:**

| Stage | Component | Metrics |
|---|---|---|
| 1 | CLIP Retriever | R@1, R@5, R@10, MRR, nDCG |
| 2 | Gemini Reasoner | Precision, Recall, F1, Accuracy |
| End-to-end | Pipeline | R@5 (binary relevance) |
| System | Architecture | Latency (p95), Tokens/query |

---

### Q3 — GitHub Baseline

> **"What existing open-source code are you building on, and what — exactly — is your unique contribution? Be specific. 'It's a combination' is not a contribution."**

**Your answer:**

> Three open-source foundations, one novel contribution.
>
> **Foundation 1: `rom1504/clip-retrieval`** — a battle-tested CLIP indexing and ANN-search infrastructure. I use it for the retriever, not reinvent it.
>
> **Foundation 2: OpenAI's CLIP (ViT-L/14)** — the encoder itself, used off-the-shelf. No fine-tuning in WP1; fine-tuning is a WP5 risk-mitigation option.
>
> **Foundation 3: Google's Gemini 1.5 Pro API** — invoked via the official SDK as the reasoning agent.
>
> **My contribution is the orchestration layer between them, and it has three concrete pieces:**
>
> 1. **A confidence-gated router** — not every query needs Gemini. If CLIP's top-1 cosine similarity exceeds a learned threshold `τ_high`, we return directly and save the Gemini call. If it falls below `τ_low`, we expand K. Only the ambiguous middle band triggers escalation. This is the token-economics core.
>
> 2. **A domain-specific prompt pack for dashcam/security events** — structured prompts for Gemini covering vehicle interactions, pedestrian events, traffic violations, and object-of-interest queries, with grounded chain-of-thought templates.
>
> 3. **A reproducible dual-agent evaluation harness** — a 10-hour curated benchmark with 100+ natural-language queries and ground-truth frame ranges, scored on the five-metric matrix from Q2. This benchmark itself is a deliverable.
>
> The novelty is not in CLIP and not in Gemini. It is in **(a) the routing decision, (b) the domain prompt engineering, and (c) the benchmark that makes the comparison rigorous.** Without those three, you have a demo. With them, you have a system.

---

## 4. Next Immediate Steps — Bridge to WP2

Push this scaffolding **before Saturday night**. Empty stubs are fine; the presence of structure proves you've thought through the system.

### Target Repository Tree

```
AI-Video-Investigator/
├── README.md                    # Vision, architecture diagram, status badge, quickstart
├── LICENSE                      # MIT
├── .gitignore                   # Python + data + .env
├── requirements.txt             # clip, faiss-cpu, google-generativeai, pydantic, pytest
├── pyproject.toml               # Optional, but a strong signal of maturity
│
├── docs/
│   ├── PRD.md                   # Product Requirements Doc — user, pain, scope, non-goals
│   ├── architecture.md          # Mermaid diagram of CLIP→Router→Gemini flow
│   ├── research_question.md     # The measurable RQ from Section 2 above
│   ├── flagship_paper_notes.md  # Galanopoulos et al. summary + delta analysis
│   ├── risk_register.md         # ≥4 risks, likelihood × impact, mitigation
│   └── prompts/
│       └── prompt_book.md       # Template Gemini prompts per event class
│
├── src/
│   ├── retriever/__init__.py    # CLIP encode + FAISS search
│   ├── router/__init__.py       # Confidence-gated escalation logic
│   ├── reasoner/__init__.py     # Gemini API wrapper
│   ├── pipeline/__init__.py     # End-to-end orchestration
│   └── eval/__init__.py         # Metric implementations (R@K, MRR, nDCG, F1)
│
├── data/
│   └── README.md                # How to acquire BDD100K, expected schema
│
├── evals/
│   ├── README.md                # Benchmark design, query taxonomy
│   ├── queries.example.jsonl    # 5 example queries to prove the schema works
│   └── results/                 # Empty; populated in WP4–WP6
│
└── notebooks/
    └── 00_smoke_test.ipynb      # CLIP loads, Gemini API key works, end-to-end happy path
```

### Skeleton File Templates

#### `README.md` — top of file

```markdown
# AI Video Investigator
> Dual-agent semantic retrieval for long-form security and dashcam video.

**Status:** WP1 — Planning · **Course:** [code] · **Lecturer:** [name]

## The Problem
[2 sentences from your elevator pitch]

## The Approach
CLIP (fast filter) → Confidence-Gated Router → Gemini 1.5 Pro (deep reasoner)

## Success Metrics
| Metric              | Target  | Baseline               |
|---------------------|---------|------------------------|
| Top-5 F1            | ≥ 0.80  | CLIP-only: ~0.65       |
| Latency (p95)       | < 3s    | Gemini-only: ~60s      |
| Cost / query-hour   | < $0.10 | Gemini-only: ~$1.16    |

## Roadmap
- [x] WP1 — Planning & Preparatory Report
- [ ] WP2 — Business Plan
- [ ] WP3 — Data Acquisition & Benchmark Curation
- [ ] WP4 — Retriever Implementation
- [ ] WP5 — Reasoner & Router Integration
- [ ] WP6 — Evaluation Harness
- [ ] WP7 — Baseline Comparisons
- [ ] WP8 — Results Analysis
- [ ] WP9 — Final Report
- [ ] WP10 — Defense
```

#### `docs/PRD.md` — section headers

```markdown
# Product Requirements Document

## 1. User & Persona
## 2. Problem Statement
## 3. Goals & Non-Goals
## 4. Functional Requirements
## 5. Non-Functional Requirements (latency, cost, accuracy)
## 6. Success Metrics
## 7. Out of Scope (WP1)
## 8. Open Questions
```

#### `docs/prompts/prompt_book.md` — initial template

```markdown
# Gemini Prompt Book — v0.1

## Prompt Template: Vehicle Interaction Event
**System:** You are a forensic video analyst...
**User:** Given the frame below and the query "{query}", determine...
**Output schema:** {"relevance": 0-1, "rationale": "...", "objects": [...]}

## Prompt Template: Pedestrian Event
...

## Prompt Template: Object-of-Interest (color / clothing / license plate)
...
```

---

### Why This Scaffolding Wins the Meeting

When the lecturer asks *"are you ready for WP2?"*, you don't argue — you share your screen, show the repo, and let the file tree answer. A populated `docs/` directory is the single strongest signal that you are not bluffing.

---

## Final Tactical Note for Sunday

Lead with the elevator pitch **unprompted** in the first 90 seconds. Get the four success numbers into the room before the lecturer sets the agenda. From that moment on, every question is fought on *your* terrain.

**בהצלחה.**
