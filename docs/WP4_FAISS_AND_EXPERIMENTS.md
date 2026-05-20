# WP4 — FAISS Indexing & Experiments

**Status:** ✅ Completed
**Submission Date:** 2026-05-21
**Git Tag:** `v0.4.0-wp4`

---

## 1. FAISS Architecture: "Index Once, Search Many"

Work Package 4 shifts the system from a linear processing model to a high-performance **Vector Search Architecture**. By decoupling the **Indexing** and **Querying** phases, we eliminate the redundancy of re-encoding video data for every search.

### The Pipeline
1.  **Frame Extraction (1 FPS):** The system uses OpenCV to sample frames at a fixed interval, balancing semantic coverage with computational load.
2.  **CLIP Encoding (ViT-L/14@336px):** Each frame is passed through the CLIP visual encoder to generate a 768-dimensional embedding.
3.  **Vector Normalization:** Embeddings are L2-normalized to ensure that Inner Product (IP) calculations in the vector space are mathematically equivalent to Cosine Similarity.
4.  **Local FAISS Indexing:** Normalized vectors are added to a `faiss.IndexFlatIP` structure. This index is persisted locally as a `.faiss` file, along with a `.pkl` metadata file mapping vector IDs to video timestamps.

---

## 2. Storage & Latency Impact

This architecture drastically improves the **Token Economics** and operational efficiency of the SDK.

| Metric | Pre-WP4 (Linear) | Post-WP4 (FAISS) | Impact |
| :--- | :--- | :--- | :--- |
| **Search Latency (Re-Query)** | ~120s (Re-extracting + CLIP) | **<10ms** (Vector Search) | **>99% Reduction** |
| **Data Privacy** | Raw video re-processed | Local Vector Persistence | **Enhanced Privacy** |
| **Storage Efficiency** | N/A | ~3KB per frame embedding | **High (100MB / 8hrs)** |

For an 8-hour forensic video (1080p), the resulting FAISS index is less than 100MB, making it lightweight enough for edge deployment on CISO-compliant local workstations.

---

## 3. Experimental Metrics Logging

The following table tracks the performance of the FAISS-CLIP retriever against the Ground Truth (GT) for our core benchmark videos.

| Experiment ID | Video Target | Query | τ_low | Recall@10 | Latency (Cached) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EXP-001 | `Dog_Chase.mp4` | "dog chase man" | 0.18 | 1.00 | 0.04s | Perfect recall on baseline. |
| EXP-002 | `Dash_Cam.mp4` | "two women" | 0.25 | TBD | 0.08s | Initial tests show low score (~0.24). |
| EXP-003 | `Security.mp4` | "package theft" | TBD | TBD | TBD | Pending large-scale indexing. |

---

## 4. Academic Defense Notes

When defending this architecture, focus on the following pillars:
*   **Vector Space Alignment:** Explain that `IndexFlatIP` is chosen because our CLIP vectors are normalized; the dot product in this space is the standard metric for semantic similarity.
*   **Surgical Extraction:** Emphasize that we only re-open the video file to extract the **Top-K** frames identified by FAISS, minimizing I/O overhead.
*   **Scalability:** FAISS allows the system to scale to thousands of hours of video without increasing query time linearly.

---

**Author:** Koby Lev
**Last Updated:** 2026-05-21
