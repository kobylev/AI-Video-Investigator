# WP3 — Core SDK & Evaluation Strategy

**Status:** 🏗️ In Progress
**Submission Date:** 2026-05-20
**Git Tag:** `v0.3.0-wp3`

---

## Executive Summary
This document establishes the technical foundation for the **AI Video Investigator**. Following the directives for the project defense, we formalize the internal mechanics of the CLIP retrieval model, define a rigorous statistical evaluation framework using 2025-standard benchmarks, and architect the Core SDK's "Prompt Book" for the Gemini-driven Reasoning Agent.

---

## 1. CLIP Technical ID Card: Architecture & Latent Space

To satisfy the technical defense requirements, we provide a deep-dive into the CLIP (Contrastive Language-Image Pre-training) architecture, specifically the `ViT-L/14@336px` variant used in our retriever.

### A. The Dual-Encoder Architecture
CLIP consists of two independent encoders that map high-dimensional data into a shared **joint latent space**.

1.  **Visual Encoder (Vision Transformer - ViT):** 
    - **Mechanism:** Breaks a 336x336 frame into patches (14x14 pixels).
    - **Processing:** Patches are flattened and treated as "tokens" in a Transformer. A special `[CLS]` token aggregates global visual features.
    - **Output:** A fixed-size feature vector (e.g., 768 dimensions).
2.  **Text Encoder (Transformer):**
    - **Mechanism:** Tokenizes the query (e.g., "masked individual entering via window") using Byte-Pair Encoding (BPE).
    - **Output:** A fixed-size feature vector in the *same* 768-dimensional space.

### B. Contrastive Learning & Cosine Similarity
- **The Latent Space:** During pre-training on 400M image-text pairs, CLIP was taught to minimize the distance between matching pairs and maximize the distance between mismatched pairs using the **InfoNCE Loss**.
- **Inference (The "Search"):** 
  - We calculate the **Cosine Similarity** ($S$) between the normalized frame vector ($v$) and text vector ($t$):
    $$S = \frac{v \cdot t}{||v|| \cdot ||t||}$$
  - A score of 1.0 indicates perfect alignment in semantic space; 0.0 indicates orthogonality (no relation).

### C. Zero-Shot Capability
CLIP does not require fine-tuning for new "crimes" or "events." By encoding the text query into the same space as the video frames, we treat the video as a searchable database of semantic concepts.

---

## 2. Dataset & Evaluation Methodology

We will not rely on qualitative "eye-balling." Our system will be validated against two rigorous benchmarks.

### A. Target Datasets
1.  **UCF-Crime (Classic Baseline):**
    - **Reason:** Industry standard for surveillance anomaly detection. Contains 1,900 long videos of real-world crimes (Abuse, Arson, Burglary, etc.).
    - **Usage:** Provides the "Ground Truth" timestamps for event detection.
2.  **HIVAU-70k (2025 CVPRW Flagship):**
    - **Reason:** The *Hierarchical Video Anomaly Understanding* benchmark.
    - **Usage:** Specifically tests the model's ability to move from "something is wrong" (CLIP) to "X is happening because of Y" (Gemini).

### B. Statistical Metrics
We will evaluate the pipeline in two stages:

#### Stage 1: CLIP Retriever (Recall@K)
- **Metric:** **Recall@5** and **Recall@10**.
- **Definition:** Percentage of cases where the true "Anomaly Frame" (per Ground Truth) is captured within the top 5 or 10 frames retrieved by CLIP.
- **Goal:** Ensure the "Searcher" doesn't miss the needle in the haystack.

#### Stage 2: Gemini Reasoner (The "Decision")
We will generate a **Confusion Matrix** comparing the CLIP Baseline (Top-1 Score > Threshold) against the Gemini-Aided Decision.

| Metric | CLIP Baseline (Expected) | Gemini-Aided (Target) |
| :--- | :--- | :--- |
| **Accuracy** | ~72% | **>92%** |
| **Precision** | Low (High False Positives) | High (Filters Noise) |
| **Recall** | High (Captures everything) | High (Maintains safety) |
| **F1-Score** | ~0.65 | **>0.88** |

**The Confusion Matrix Proof:**
By plotting `[TP, FP, FN, TN]`, we will demonstrate that Gemini's reasoning reduces **False Positives (FP)** (e.g., a person carrying a heavy box being flagged as a "theft") which CLIP often misidentifies due to lack of temporal context.

---

## 3. Core SDK Architecture & Prompt Book

The SDK follows a modular design for maximum reproducibility.

### A. SDK Structure (`src/`)
- `retriever/clip_engine.py`: Handles frame extraction, embedding generation, and FAISS indexing.
- `reasoner/gemini_client.py`: Manages the multi-modal payload delivery to Gemini 1.5 Pro.
- `pipeline/orchestrator.py`: The "Router" logic that executes CLIP search -> Top-K filtering -> Gemini verification.

### B. The Gemini "System Prompt" (The Security Analyst)
We will use the following system instruction to ensure Gemini acts with the required rigor.

```markdown
**Role:** Senior Forensic Security Analyst
**Mission:** Verify if a specific security event is occurring in the provided frames.
**Rules:**
1. **Visual Evidence Only:** Do not hallucinate actions not visible. If the frame is blurry, state "Inconclusive."
2. **Contextual Analysis:** Look for tools, clothing (masks/gloves), and body language (hurried, crouching).
3. **Zero-Trust Policy:** Your default stance is that NO event is occurring. You must be "convinced" by the visual evidence to flag an alert.
4. **Structured Output:** Always return a JSON object with:
   - "event_detected": boolean
   - "confidence_score": float (0.0 - 1.0)
   - "reasoning_steps": list of visual observations
   - "timestamp_range": estimated seconds
```

---

## 4. Next Steps (WP3 Execution)
1.  **Scripting:** Implement the `clip_engine.py` using `sentence-transformers` or `open_clip`.
2.  **Data Prep:** Download the UCF-Crime subset and map Ground Truth labels to JSON.
3.  **Smoke Test:** Run CLIP on one video and verify if Recall@10 captures the anomaly.

---
**Author:** Koby Lev (Senior AI Architect)
**Last Updated:** 2026-05-18
