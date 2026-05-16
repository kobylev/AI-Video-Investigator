# AI Video Investigator

> Dual-agent semantic retrieval for long-form security and dashcam video.

**Status:** 🔄 WP1 — Planning | **Author:** Koby Lev

## The Problem

Security operators and fleet-safety teams face cognitive overload at industrial scale: scrubbing 12 to 24 hours of dashcam or CCTV footage to find a single event—a near-miss, a red jacket, a midnight license plate. Today's tools force a brutal trade-off. Keyword tagging misses semantic nuance. Feeding raw video into a frontier model like Gemini 1.5 Pro costs over a dollar per query-hour and stalls any interactive workflow.

## The Approach

**AI Video Investigator breaks that trade-off with dual-agent routing.** A CLIP retriever acts as a millisecond semantic filter, surfacing the top-K candidate frames out of tens of thousands. Only those candidates are escalated to Gemini 1.5 Pro for deep reasoning—*"is this actually a hit-and-run, or just a fender-bender?"*

```
CLIP (fast filter) → Confidence-Gated Router → Gemini 1.5 Pro (deep reasoner) → Ranked Output
```

## Success Metrics

| Metric | Target | Baseline |
|--------|--------|----------|
| Top-5 F1 | ≥ 0.80 | CLIP-only: ~0.65 |
| Accuracy Lift | +15 points | over CLIP-alone |
| Latency (p95) | < 3s | Gemini-only: ~60s |
| Cost / query-hour | < $0.10 | Gemini-only: ~$1.16 |
| Token Reduction | > 90% | vs. Gemini-only baseline |

## Research Question

*To what extent does a dual-agent CLIP→Gemini routing architecture improve top-5 retrieval F1 and reduce inference token cost on long-form dashcam footage, compared to (a) a CLIP-only retrieval baseline and (b) a Gemini-only frame-analysis baseline, evaluated on a benchmark of ≥100 natural-language queries over ≥10 hours of curated video?*

## 10-Week Roadmap

- [x] **WP1** — Planning & Preparatory Report
- [ ] **WP2** — Business Plan
- [ ] **WP3** — Data Acquisition & Benchmark Curation
- [ ] **WP4** — Retriever Implementation (CLIP + FAISS)
- [ ] **WP5** — Reasoner & Router Integration (Gemini + confidence gating)
- [ ] **WP6** — Evaluation Harness (R@K, MRR, nDCG, F1, Accuracy)
- [ ] **WP7** — Baseline Comparisons (CLIP-only, Gemini-only)
- [ ] **WP8** — Results Analysis & Ablation Studies
- [ ] **WP9** — Final Report
- [ ] **WP10** — Defense Preparation & Presentation

## Repository Structure

```
.
├── docs/               # Architecture, PRD, research docs, prompts
├── src/                # Source code (retriever, router, reasoner, pipeline, eval)
├── evals/              # Benchmark queries and evaluation results
├── data/               # Dataset acquisition notes (BDD100K)
├── notebooks/          # Experimental notebooks and smoke tests
└── deliverables/       # Final submitted PDFs per work package
```

## Flagship Reference

This work builds upon the retrieve-then-reason paradigm established by:

> Galanopoulos et al., *"An LLM Framework for Long-form Video Retrieval,"* **CVPRW 2025**.

**Key adaptations:**
- **Domain shift:** Security/dashcam footage vs. general long-form video
- **Novel contribution:** Confidence-gated router for token-cost optimization
- **Evaluation:** Dual-baseline comparison (CLIP-only + Gemini-only)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run smoke test (requires GEMINI_API_KEY in environment)
jupyter notebook notebooks/00_smoke_test.ipynb
```

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Academic Project** | 10-week capstone | [Work Package Tracker](WORK_PACKAGES.md)
