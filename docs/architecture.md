# System Architecture

## AI Video Investigator — Dual-Agent Design

This document describes the high-level architecture of the AI Video Investigator system, a hybrid retrieval-reasoning pipeline optimized for semantic search over long-form security and dashcam footage.

---

## Architecture Overview

The system implements a **retrieve-then-reason** paradigm with confidence-gated routing to balance accuracy, latency, and cost.

```mermaid
graph TD
    A[Natural Language Query] --> B[CLIP Text Encoder]
    B --> C[FAISS ANN Search]
    C --> D{Confidence Router}

    D -->|High Confidence<br/>sim > τ_high| E[Return Top-1]
    D -->|Low Confidence<br/>sim < τ_low| F[Expand K]
    D -->|Ambiguous<br/>τ_low ≤ sim ≤ τ_high| G[Escalate Top-K]

    F --> G
    G --> H[Gemini 1.5 Pro Reasoner]
    H --> I[Re-rank Candidates]
    I --> J[Ranked Output Top-5]
    E --> J

    K[(Video Corpus<br/>10+ hours)] --> L[Frame Extraction<br/>1 fps]
    L --> M[CLIP Image Encoder]
    M --> N[(FAISS Index<br/>Offline)]
    N -.->|Query Time| C

    style D fill:#ff9,stroke:#333,stroke-width:2px
    style H fill:#9cf,stroke:#333,stroke-width:2px
    style N fill:#cfc,stroke:#333,stroke-width:2px
```

---

## Component Descriptions

### 1. Offline Indexing Pipeline (Pre-Processing)

**Input:** Raw video files (MP4, AVI, etc.)
**Process:**
1. Extract frames at 1 fps using ffmpeg
2. Encode each frame with CLIP ViT-L/14 image encoder → 768-dim embedding
3. Build FAISS index (IVF or HNSW) over all frame embeddings
4. Store frame-to-timestamp mapping in metadata database

**Output:** FAISS index + metadata store
**Frequency:** One-time per video; incremental updates for new footage

---

### 2. CLIP Retriever

**Purpose:** Fast semantic filtering to surface top-K candidate frames

**Components:**
- **Text Encoder:** CLIP ViT-L/14 (frozen, off-the-shelf)
- **Index:** FAISS with approximate nearest neighbor (ANN) search
- **Search Strategy:** Cosine similarity in 768-dim embedding space

**Performance:**
- **Latency:** <100ms for 10-hour corpus (36,000 frames)
- **Recall:** Target R@20 ≥ 0.90 (most relevant frame in top-20)

**Limitations:**
- **Semantic ambiguity:** Cannot distinguish "fender-bender" from "hit-and-run"
- **Domain gap:** CLIP trained on web images; dashcam footage may be out-of-distribution

---

### 3. Confidence-Gated Router

**Purpose:** Minimize Gemini API calls while preserving accuracy

**Routing Logic:**

```python
def route(query_embedding, top_k_results):
    top_1_similarity = top_k_results[0].score

    if top_1_similarity > τ_high:
        # High-confidence: CLIP alone is sufficient
        return top_k_results[0]

    elif top_1_similarity < τ_low:
        # Low-confidence: expand search and escalate
        expanded_results = search(query_embedding, k=50)
        return escalate_to_gemini(expanded_results)

    else:
        # Ambiguous: escalate top-K to Gemini
        return escalate_to_gemini(top_k_results)
```

**Thresholds (Initial):**
- `τ_high = 0.85` (90th percentile of positive pairs)
- `τ_low = 0.60` (50th percentile of positive pairs)
- **Tuning:** Learned empirically in WP5 via grid search on validation set

**Impact:**
- Reduces Gemini calls by 40–60% (estimated)
- Preserves accuracy by escalating ambiguous cases

---

### 4. Gemini 1.5 Pro Reasoner

**Purpose:** Deep multimodal reasoning over candidate frames

**API:** Google Generative AI Python SDK
**Model:** `gemini-1.5-pro-latest`
**Input:** Top-K frames (default K=20) + structured prompt
**Output:** Per-frame relevance score (0–1) + rationale + detected objects

**Prompt Structure (Template):**

```
You are a forensic video analyst specializing in security and dashcam footage.

QUERY: "{user_query}"

Analyze the frame below and determine its relevance to the query.

OUTPUT FORMAT (JSON):
{
  "relevance_score": 0.0 to 1.0,
  "rationale": "Brief explanation (1-2 sentences)",
  "detected_objects": ["object1", "object2", ...],
  "timestamp_estimate": "approximate time-of-day if visible"
}

FRAME: [image attachment]
```

**Re-Ranking:** Sort candidates by `relevance_score` descending; return top-5

**Cost Control:**
- Each frame ~258 tokens (vision) + ~150 tokens (prompt) = ~408 tokens/frame
- K=20 → ~8,160 tokens/query
- At $1.25/1M input tokens → ~$0.01 per query

---

## Data Flow

### Query-Time Execution (End-to-End)

1. **User submits query:** `"red sedan running red light at intersection"`
2. **CLIP encoding:** Query text → 768-dim embedding (5ms)
3. **FAISS search:** Retrieve top-20 frames by cosine similarity (50ms)
4. **Router decision:**
   - If top-1 similarity = 0.92 (> τ_high=0.85) → Return immediately
   - Else → Escalate to Gemini
5. **Gemini reasoning:** Analyze 20 frames in parallel (2.5s)
6. **Re-ranking:** Sort by Gemini relevance scores
7. **Return top-5:** Frame IDs, timestamps, scores, rationales

**Total latency:** ~2.6 seconds (well below 3s target)

---

## Scalability Considerations

### Corpus Size vs. Latency

| Video Hours | Frames (1 fps) | FAISS Latency | Gemini Latency | Total (p95) |
|-------------|----------------|---------------|----------------|-------------|
| 10h | 36,000 | 50ms | 2.5s | 2.6s |
| 100h | 360,000 | 120ms | 2.5s | 2.7s |
| 1,000h | 3.6M | 300ms | 2.5s | 2.9s |

**Conclusion:** FAISS scales logarithmically; Gemini latency dominates. System remains <3s for corpora up to 1,000 hours.

---

## Baseline Architectures (For Comparison)

### Baseline 1: CLIP-Only

- **Pipeline:** Query → CLIP text encoder → FAISS search → Top-5
- **Pros:** Sub-100ms latency, near-zero cost
- **Cons:** Cannot handle semantic ambiguity; expected F1 ~0.65

### Baseline 2: Gemini-Only

- **Pipeline:** Query → Extract all frames → Send to Gemini → Rank
- **Pros:** Maximum accuracy potential
- **Cons:** ~$1.16/query-hour, 60+ second latency, prohibitive at scale

### This Work: Dual-Agent Hybrid

- **Pipeline:** CLIP filter → Router → Gemini re-rank
- **Target:** F1 ≥ 0.80, <$0.10/query-hour, <3s latency
- **Rationale:** Captures 90% of Gemini's accuracy at 10% of the cost

---

## Technology Stack

| Component | Technology | Justification |
|-----------|------------|---------------|
| Video Processing | ffmpeg | Industry standard, reliable frame extraction |
| CLIP Encoder | OpenAI CLIP ViT-L/14 | Off-the-shelf, no fine-tuning required |
| Vector Index | FAISS (CPU) | Facebook's battle-tested ANN library |
| Reasoner | Gemini 1.5 Pro API | State-of-art multimodal VLM, long context |
| Backend | Python 3.9+ | Ecosystem compatibility (PyTorch, FAISS, Google SDK) |
| Evaluation | pytest + custom harness | Reproducible metrics (R@K, MRR, nDCG, F1) |

---

## Open Design Questions

1. **Should we batch Gemini requests or process sequentially?**
   - **Trade-off:** Batching reduces latency but may hit rate limits
   - **Resolution:** WP5 experimentation

2. **Should confidence thresholds be event-type-specific?**
   - **Hypothesis:** "Object-of-interest" queries may need lower τ_high than "collision" queries
   - **Resolution:** Ablation study in WP6

3. **Should we cache CLIP embeddings for recurring queries?**
   - **Impact:** Negligible (text encoding is already <5ms)
   - **Decision:** Deprioritized for WP1

---

**Version:** 0.1.0 | **Status:** WP1 — Planning | **Last Updated:** 2026-05-16
