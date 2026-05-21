# WP3 — Core SDK & Evaluation Strategy

**Status:** ✅ Completed
**Submission Date:** 2026-05-20
**Git Tag:** `v0.3.0-wp3`

---

## Executive Summary
This document establishes the technical foundation for the **AI Video Investigator**. Following the directives for the project defense, we formalize the internal mechanics of the CLIP retrieval model, define a rigorous statistical evaluation framework using 2025-standard benchmarks, and architect the Core SDK's "Prompt Book" for the Anthropic-driven Reasoning Agent.

---

## 1. CLIP Technical ID Card: Architecture & Latent Space

... [CLIP details remain identical as they are provider-neutral] ...

---

## 2. Dataset & Evaluation Methodology

We will not rely on qualitative "eye-balling." Our system will be validated against two rigorous benchmarks.

### A. Target Datasets
1.  **UCF-Crime (Classic Baseline)**
2.  **HIVAU-70k (2025 CVPRW Flagship)**

### B. Statistical Metrics
We will evaluate the pipeline in two stages:

#### Stage 1: CLIP Retriever (Recall@K)
Percentage of cases where the true "Anomaly Frame" is captured within the top 5 or 10 frames retrieved by CLIP.

#### Stage 2: Claude Reasoner (The "Decision")
We will generate a **Confusion Matrix** comparing the CLIP Baseline (Top-1 Score > Threshold) against the Claude-Aided Decision.

| Metric | CLIP Baseline (Expected) | Claude-Aided (Target) |
| :--- | :--- | :--- |
| **Accuracy** | ~72% | **>92%** |
| **Precision** | Low (High False Positives) | High (Filters Noise) |
| **Recall** | High (Captures everything) | High (Maintains safety) |
| **F1-Score** | ~0.65 | **>0.88** |

**The Confusion Matrix Proof:**
By plotting `[TP, FP, FN, TN]`, we will demonstrate that Claude's reasoning reduces **False Positives (FP)** which CLIP often misidentifies due to lack of temporal context.

---

## 3. Core SDK Architecture & Prompt Book

The SDK follows a modular design for maximum reproducibility.

### A. SDK Structure (`src/`)
- `retriever/clip_engine.py`: Handles frame extraction, embedding generation, and FAISS indexing.
- `reasoner/claude_engine.py`: Manages the multi-modal payload delivery to Anthropic Claude Haiku 4.5.
- `router/core.py`: The "Router" logic that executes CLIP search -> Top-K filtering -> Claude verification.

### B. The Claude "System Prompt" (The Security Analyst)
We will use the following system instruction to ensure Claude acts with the required rigor.

```markdown
**Role:** Senior Forensic Security Analyst
**Mission:** Verify if a specific security event is occurring in the provided frames.
**Rules:**
1. **Visual Evidence Only:** Do not hallucinate actions not visible.
2. **Contextual Analysis:** Look for tools, clothing, and body language.
3. **Zero-Trust Policy:** Your default stance is that NO event is occurring.
4. **Structured Output:** Always return a JSON object via the `submit_verdict` tool.
```

---
**Author:** Koby Lev (Senior AI Architect)
**Last Updated:** 2026-05-21
