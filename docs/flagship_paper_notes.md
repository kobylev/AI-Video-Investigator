# Flagship Paper Analysis

## Galanopoulos et al. (2025) — "An LLM Framework for Long-form Video Retrieval"

**Conference:** CVPRW 2025
**Role:** Foundational Architecture for AI Video Investigator

---

## 1. Core Paradigm

The flagship work establishes the **retrieve-then-reason** framework for long-form video:
1. **Retrieve:** Uses dense embeddings (e.g., CLIP) to surface candidate segments.
2. **Reason:** Sends top-K candidate frames to a frontier multimodal LLM (e.g., GPT-4V, **Claude**).

---

## 2. Adoption in This Project

AI Video Investigator adopts the following from Galanopoulos et al.:
- **Retrieve-then-Reason Pipeline:** Fast filter followed by expensive reasoning.
- **Metrics:** Rank-based scoring (R@K, MRR, nDCG).
- **Stage 2:** **Claude Haiku 4.5** re-ranking of top-K candidates.

---

## 3. Divergence & Novel Contributions (Our "Twist")

While we adopt the paradigm, we diverge in four critical areas:

### A. Confidence-Gated Routing
The flagship work sends *all queries* to the LLM for re-ranking.
**We implement a "Router" layer:** If CLIP confidence is high, we skip the LLM entirely, reducing query costs by 40–60%.

### B. Dual-Baseline Evaluation
The flagship work does not formally measure the **cost-accuracy trade-off** against a **Claude-only** upper bound. We provide:
- **Cost reduction vs. Claude-only** (measures efficiency gain)
- **Accuracy preservation vs. Claude-only** (measures quality loss from gating)

### C. Token Economics as Success Criteria
- **Token cost is a success criterion:** Must achieve >90% reduction vs. **Claude-only**.

---

## 4. Implementation Comparison

| Feature | Galanopoulos et al. | AI Video Investigator |
| :--- | :--- | :--- |
| Retriever | CLIP | CLIP (vL/14) + FAISS Index |
| Reasoner | GPT-4V | **Claude Haiku 4.5** |

---

## 5. Defense Narrative

> This project adopts the **retrieve-then-reason paradigm** and **rank-based evaluation protocol** established by Galanopoulos et al. (CVPRW 2025) as the foundational architecture. We diverge in three key areas: (1) domain adaptation to security/dashcam footage, (2) introduction of a **confidence-gated router** to minimize LLM token consumption, and (3) dual-baseline comparison against both CLIP-only and **Claude-only** systems with token cost as a first-class success metric.

---

**Last Updated:** 2026-05-21
