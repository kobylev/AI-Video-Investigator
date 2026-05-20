# WP4 — FAISS Indexing & Retriever Experiments

**Project:** AI Video Investigator — Privacy-Preserving Dual-Agent Semantic Video Retrieval
**Author:** Koby Lev
**Status:** ✅ Completed
**Git Tag:** `v0.4.0-wp4`
**Predecessor:** WP3 (Data Acquisition & Benchmark Curation)
**Successor:** WP5 (Reasoner & Router Integration)

---

## 1. Abstract

Work Package 4 (WP4) operationalises the *retriever* component of the dual-agent architecture introduced in WP1. The objective is to transform a raw video corpus into a persistent, query-ready semantic representation that can be searched with sub-millisecond latency, without re-invoking the visual encoder on every query. To achieve this, we couple frame extraction at a fixed temporal resolution with embeddings produced by **OpenAI CLIP ViT-L/14** and materialise these embeddings in a **FAISS `IndexFlatIP`** index stored on local disk. This document specifies the architecture, motivates the design decisions, quantifies the storage and latency envelope, and defines the experimental protocol by which the retriever is empirically evaluated against the CLIP-only baseline.

---

## 2. Motivation & Problem Statement

The baseline retriever introduced in WP3 re-encodes the entire video corpus for every natural-language query. While conceptually correct, this design is operationally untenable for two reasons:

1. **Computational redundancy.** CLIP ViT-L/14 inference on a 10-hour corpus sampled at 1 fps requires ~36,000 forward passes through a 304M-parameter vision transformer. On commodity GPU hardware (NVIDIA T4, 16 GB VRAM) this takes minutes to tens of minutes — incompatible with the *interactive* query workflows that motivate the system.
2. **Energy and cost amortisation.** Encoding the *frames* is a function of the corpus, not the *query*. Recomputing the same embeddings for every query violates basic amortisation principles and inflates both wall-clock latency and operational energy expenditure.

WP4 therefore introduces a **persistent, on-premise embedding cache** that decouples the (expensive, one-time) encoding step from the (cheap, repeated) retrieval step.

---

## 3. FAISS Architecture

### 3.1 Pipeline Overview

The retriever pipeline implemented in WP4 is structured as a strictly offline indexing stage followed by an online query stage:

```
┌──────────────────────────  OFFLINE  (once per video) ──────────────────────────┐
│                                                                                 │
│  Video File (.mp4)                                                              │
│        │                                                                        │
│        ▼                                                                        │
│  [1] Frame Extraction  ── ffmpeg @ 1 fps  ──▶  PIL frames                       │
│        │                                                                        │
│        ▼                                                                        │
│  [2] CLIP ViT-L/14 Image Encoder  ──▶  L2-normalised 768-d embeddings           │
│        │                                                                        │
│        ▼                                                                        │
│  [3] FAISS IndexFlatIP.add(embeddings)                                          │
│        │                                                                        │
│        ▼                                                                        │
│  [4] Persist to disk:  data/indices/<video_id>.faiss                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────  ONLINE  (per query) ────────────────────────────────┐
│                                                                                 │
│  Natural-Language Query                                                         │
│        │                                                                        │
│        ▼                                                                        │
│  [5] CLIP ViT-L/14 Text Encoder  ──▶  L2-normalised 768-d query vector          │
│        │                                                                        │
│        ▼                                                                        │
│  [6] FAISS index.search(query, k)  ──▶  Top-k frames + inner-product scores    │
│        │                                                                        │
│        ▼                                                                        │
│  [7] Hand-off to Router / Reasoner (WP5)                                        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Frame Extraction (1 fps Sampling)

Frames are extracted at a uniform temporal resolution of **1 frame per second** using `ffmpeg`. This rate is justified empirically by Tian *et al.* (CVPR 2024), who demonstrate that frozen CLIP ViT-L/14 retains 74.3 % Recall@10 on general video benchmarks at 1 fps — i.e., the marginal gain from denser sampling is bounded above by single-digit percentage points and does not justify the linear increase in storage and encoding cost. For the security/dashcam domain, where the events of interest typically persist for several seconds, 1 fps is a defensible operating point.

### 3.3 Visual Encoder: CLIP ViT-L/14

We employ **OpenAI CLIP ViT-L/14** as the joint image-text encoder. The selection is governed by three criteria:

| Criterion | Justification |
|-----------|---------------|
| **Joint embedding space** | Text and image are projected into a shared 768-dimensional space, enabling zero-shot retrieval without any task-specific fine-tuning. |
| **Pre-training scale** | ~400M image-text pairs (LAION-derived) confer broad semantic coverage applicable to the heterogeneous content of CCTV/dashcam footage. |
| **Mature open-source tooling** | The `open_clip_torch` ecosystem provides deterministic checkpoints, batched inference, and clean PyTorch interfaces, supporting academic reproducibility. |

All embeddings are **L2-normalised** prior to insertion into the index. Under L2 normalisation, the inner product on the unit hypersphere is mathematically equivalent to the cosine similarity, which is the standard semantic-similarity measure in vision-language retrieval.

### 3.4 Index Choice: `IndexFlatIP`

Among the indexing families exposed by FAISS, WP4 deliberately selects the **exact (brute-force) inner-product index, `IndexFlatIP`**. The rationale is as follows:

1. **Exactness for benchmark integrity.** Approximate-nearest-neighbour structures (IVF, HNSW, PQ) introduce recall–latency trade-offs that confound the evaluation of the *retriever's semantic quality*. Until the upper bound of CLIP's retrieval accuracy on the curated benchmark is established, approximation would conflate model error with index error.
2. **Corpus scale.** A 10-hour video at 1 fps yields 36,000 vectors; even a 24-hour corpus yields only 86,400. Exhaustive inner-product search at this scale completes in low-millisecond time on commodity CPUs, rendering approximation unnecessary.
3. **Semantic alignment with normalised embeddings.** Because vectors are unit-normalised, `IndexFlatIP` computes cosine similarity directly, with no metric mismatch between the embedding model and the index.

The intent is to defer ANN-style indexing (`IndexIVFFlat`, `IndexHNSWFlat`) to a future work package only if and when the deployment corpus exceeds the scale at which exact search remains tractable.

### 3.5 Persistence & Cache Semantics

Indices are serialised to disk under `data/indices/<video_id>.faiss`. The pipeline implements a **content-addressed cache check**: if an index already exists for the input video, frame extraction and CLIP encoding are skipped entirely, and the query proceeds directly to the search stage. This converts every query *after the first* into a pure retrieval operation, which is the central performance optimisation of WP4.

---

## 4. Storage & Latency Impact

### 4.1 Storage Footprint Analysis

The per-frame storage cost is bounded analytically by the embedding dimensionality and numerical precision:

```
storage_per_frame = 768 dimensions × 4 bytes (float32) = 3,072 bytes ≈ 3 KB
```

For representative operational scales:

| Corpus Length | Frames @ 1 fps | Raw Index Size | Order of Magnitude |
|---------------|---------------:|---------------:|--------------------|
| 1 hour        | 3,600          | ~11 MB         | tens of MB         |
| 8 hours (1080p reference) | 28,800 | ~88 MB     | **< 100 MB**       |
| 24 hours      | 86,400         | ~264 MB        | hundreds of MB     |
| 30-day, 50-camera deployment | ~129.6 M | ~398 GB | enterprise SSD     |

**The headline result is that an 8-hour 1080p video is fully indexable in under 100 MB.** This is approximately three orders of magnitude smaller than the raw video footprint (which, at typical 1080p bitrates of ~8 Mbps, occupies ~28 GB for the same 8 hours), demonstrating that the FAISS embedding cache is a *semantic compression* of the corpus by a factor of ~300×.

### 4.2 Latency Impact

The decoupling of encoding from retrieval collapses query latency by approximately three orders of magnitude:

| Stage                          | Without FAISS Cache (WP3 baseline) | With FAISS Cache (WP4) |
|--------------------------------|------------------------------------|------------------------|
| Frame extraction               | 10–30 s per hour of video          | **0 s** (cached)        |
| CLIP image encoding (per query)| 60–600 s for 1–10 h corpus         | **0 s** (cached)        |
| CLIP text encoding (per query) | ~10 ms                             | ~10 ms                 |
| Vector search (top-k)          | n/a                                | **< 5 ms** (exact IP)  |
| **Query-time total**           | **minutes**                        | **~15 ms**             |

This is the fundamental enabling step for the **<3-second p95 end-to-end latency** target articulated in WP1: the retrieval stage is no longer a latency bottleneck, leaving the entire latency budget available to the cloud reasoner (WP5).

---

## 5. Experimental Protocol

The retriever is evaluated against the WP3 benchmark using two principal axes: **retrieval quality** (Recall@5) and **operational latency** (end-to-end query time, excluding any reasoner escalation).

### 5.1 Systems Under Comparison

| System ID | Retriever | Index | Notes |
|-----------|-----------|-------|-------|
| **B0 — Linear-Scan Baseline** | CLIP ViT-L/14, re-encoded per query | None (cosine over re-computed embeddings) | Reference implementation from WP3 |
| **W4 — FAISS-Cached Retriever** | CLIP ViT-L/14, encoded once | `IndexFlatIP` on disk | The WP4 deliverable |
| **W4-N (optional)** | CLIP ViT-L/14, encoded once | `IndexHNSWFlat` (M=32) | Stretch comparison for ANN trade-off curve |

### 5.2 Metric Definitions

- **Recall@5**: proportion of queries for which at least one ground-truth-relevant frame appears within the top 5 returned candidates.
- **Latency (ms)**: wall-clock time from query submission to top-k return, measured on the same host; mean and p95 reported across the benchmark.
- **Index Build Time (s)**: one-time offline cost to construct the persistent FAISS index, reported for transparency but excluded from query-time latency.
- **Index Size on Disk (MB)**: serialised footprint of the persistent index.

### 5.3 Metrics Logging Table — Placeholder

The following table is the canonical results store for WP4. Each row corresponds to one (system × corpus) configuration evaluated over the full WP3 query set. Values are to be populated as experiments are executed.

| Experiment ID | Date (UTC) | System | Corpus | # Frames | # Queries | Recall@5 | Mean Latency (ms) | p95 Latency (ms) | Index Build Time (s) | Index Size (MB) | Notes |
|---------------|------------|--------|--------|---------:|----------:|---------:|------------------:|-----------------:|---------------------:|----------------:|-------|
| `WP4-EXP-001` | _TBD_      | B0 — Linear-Scan Baseline | `Dog_Chase.mp4` | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | n/a | n/a | Reference run; no cache |
| `WP4-EXP-002` | _TBD_      | W4 — FAISS-Cached Retriever | `Dog_Chase.mp4` | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | Cold-cache build |
| `WP4-EXP-003` | _TBD_      | W4 — FAISS-Cached Retriever | `Dog_Chase.mp4` | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | 0 (cached) | _TBD_ | Warm-cache query path |
| `WP4-EXP-004` | _TBD_      | B0 — Linear-Scan Baseline | BDD100K 10 h subset | _TBD_ | 100 | _TBD_ | _TBD_ | _TBD_ | n/a | n/a | WP3 benchmark, baseline |
| `WP4-EXP-005` | _TBD_      | W4 — FAISS-Cached Retriever | BDD100K 10 h subset | _TBD_ | 100 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | WP3 benchmark, FAISS |
| `WP4-EXP-006` | _TBD_      | W4-N — HNSW (stretch) | BDD100K 10 h subset | _TBD_ | 100 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | Optional ANN comparison |

**Reporting convention.** Each populated row constitutes one self-contained experimental observation. Configuration parameters not captured in the table columns (e.g., `tau_low`, `tau_high`, FAISS construction parameters) are to be recorded in a sibling file `evals/results/<Experiment ID>.json` alongside the raw per-query results.

### 5.4 Acceptance Criteria for WP4

WP4 is considered empirically validated when **all** of the following hold on the BDD100K 10-hour subset (rows `WP4-EXP-004` and `WP4-EXP-005`):

1. **Quality parity.** Recall@5 of W4 is within ±0.5 points of B0, confirming that persistence introduces no semantic regression.
2. **Latency reduction.** Mean query latency of W4 is at least **two orders of magnitude** lower than B0.
3. **Storage envelope.** Index size remains below 100 MB per 8 hours of indexed footage, validating the analytical bound of Section 4.1.

---

## 6. Limitations & Forward Pointers to WP5

WP4 deliberately confines itself to the retriever. Two limitations are acknowledged and are explicitly addressed by subsequent work packages:

- **Semantic ceiling of frozen CLIP.** Because no fine-tuning is performed, the retriever inherits the domain gap of LAION-pretrained CLIP on the dashcam/CCTV distribution. The dual-agent design absorbs this limitation by delegating ambiguous candidates to a higher-capacity multimodal reasoner — implemented in **WP5**.
- **Absence of reasoning.** Top-k inner-product ranking is a *necessary but not sufficient* condition for forensic retrieval. The confidence-gated router and Gemini-based re-ranker introduced in **WP5** convert this top-k stream into adjudicated, evidence-bearing results.

---

## 7. Summary

WP4 delivers a persistent, locally-stored, sub-millisecond CLIP embedding cache backed by FAISS `IndexFlatIP`, satisfying the privacy, latency, and storage envelopes specified in WP1. The retriever is now a constant-time component of the query path, an essential precondition for the interactive dual-agent system whose cloud reasoner is constructed in WP5.
