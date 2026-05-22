# Version 2.0 — OpenCLIP Migration

**Status:** 🚧 In progress on `feature/v2.0-openclip-migration` | **Author:** Koby Lev | **Initiated:** 2026-05-22

---

### Version 2.0: OpenCLIP Integration

#### Architectural Justification

The Stage 1 local semantic filter in WP4 was originally implemented against the **OpenAI CLIP ViT-L/14@336** checkpoint, accessed via the HuggingFace `transformers.CLIPModel` class. While this configuration met the WP6 latency and privacy targets, qualitative review and operator feedback during WP8 stress testing surfaced a recurring failure mode: **precision degradation on visually complex, cluttered, or compositionally nuanced frames** — precisely the high-value cases on which a forensic system must perform best (e.g., a vehicle partially occluded by debris, a pedestrian against a busy street scene, a sign half-obscured by rain).

The root cause is empirically traceable to the training data of the OpenAI checkpoint:

| Dimension | OpenAI CLIP (V1) | OpenCLIP / LAION-2B (V2) |
| :--- | :--- | :--- |
| **Training corpus** | WIT — 400 M image–text pairs, English-centric, proprietary | LAION-2B — 2.32 B image–text pairs, multilingual, open |
| **Compositional coverage** | Limited; weak on cluttered scenes | Substantially broader; better on rare compositions |
| **Embedding dimensionality** | 768 (ViT-L/14) | 1024 (ViT-H/14) |
| **Reproducibility / auditability** | Closed dataset | Fully open, reproducible from the LAION manifest |
| **Reported ImageNet zero-shot** | 75.5% (ViT-L/14@336) | **78.0% (ViT-H/14, laion2b_s32b_b79k)** |

#### V2 Engine Selection

The V2 default checkpoint is **`ViT-H-14` trained on `laion2b_s32b_b79k`** (32 B samples seen, batch size 79 k), loaded via `open_clip.create_model_and_transforms`. Tag verified against `open_clip` 3.3.0's `list_pretrained_tags_by_model("ViT-H-14")` output during the V2.0 smoke test. This checkpoint was selected on three criteria:

1. **Empirical zero-shot strength** on cluttered-scene benchmarks (LAION CLIP-benchmark suite).
2. **Active community maintenance**, ensuring long-term reproducibility.
3. **Compatibility** with our existing CPU/GPU deployment posture — `ViT-H-14` runs on the same T4-class enterprise GPUs used in WP4.

For latency-constrained deployments, the V2 plan also validates a fallback to **`ViT-L-14 / laion2b_s32b_b79k`** (768-dim, drop-in dimensionality with the existing FAISS index) — see the A/B Protocol section below.

#### Architectural Impact

- **Backward compatibility:** The `OpenCLIPEngine` class exposes the identical public interface as the existing `CLIPEngine` (`get_image_embeddings`, `get_text_embeddings`, `compute_similarity`, `get_frames_at_timestamps`). Caller code in `src/retriever/search_index.py` and `src/eval/modes.py` requires **no changes**.
- **Index migration:** If V2 lands on `ViT-H/14`, the on-prem FAISS index must be rebuilt from raw video at the new 1024-dim. A one-shot re-indexing job is documented in the V2 runbook.
- **Audit trail:** Because LAION-2B is fully open, every embedding produced by V2 is traceable to a known training distribution — a meaningful improvement for the GDPR/compliance posture articulated in §2 of the [Final Summary Report](SUMMARY_REPORT.md).

#### Engine Selection at Runtime

A new factory module `src.retriever.factory.build_retriever()` selects between the V1 and V2 engines at runtime via the `RETRIEVER_BACKEND` environment variable:

```python
from src.retriever.factory import build_retriever

# V2 (default)
retriever = build_retriever()                    # OpenCLIP / LAION-2B
# V1 (baseline, for A/B comparison)
retriever = build_retriever(backend="openai_hf") # OpenAI CLIP via transformers
```

Both engines expose an identical public interface, so the FAISS indexer, FastAPI orchestration backend, and the WP6 evaluation harness all consume them interchangeably.

#### A/B Protocol

V2 is validated against V1 by running the existing WP6 harness once per arm and comparing the resulting aggregate metrics with the [evals/compare_v1_v2.py](../evals/compare_v1_v2.py) script.

```powershell
# Arm A — V1 baseline (OpenAI CLIP)
$env:RETRIEVER_BACKEND = "openai_hf"
python -m src.eval.run_benchmark --mode clip_only `
    --queries evals/queries.example.jsonl `
    --output-dir evals/results/v1_baseline --seed 42

# Arm B — V2 candidate (OpenCLIP / LAION-2B)
$env:RETRIEVER_BACKEND = "openclip"
python -m src.eval.run_benchmark --mode clip_only `
    --queries evals/queries.example.jsonl `
    --output-dir evals/results/v2_openclip --seed 42

# Compare and gate
python evals/compare_v1_v2.py
```

> **Wiring note:** `src/eval/run_benchmark.py` currently instantiates `BenchmarkRunner(config)` with no injected retriever. The V2 branch must add a small patch that calls `src.retriever.factory.build_retriever()` and passes it via the `retriever=` parameter — a ~10-line edit. The env-var-driven factory already makes the A/B feasible without that patch in CI-stub mode.

#### Acceptance Criteria for Merge

V2.0 will be merged into `master` only when the WP6 A/B harness empirically demonstrates **all three** of:

1. **Recall@5(V2) ≥ Recall@5(V1) + 0.05** on the existing benchmark query set,
2. **F1@5(V2) ≥ F1@5(V1) + 0.05**, and
3. **Retriever p95 latency(V2) ≤ 1.5 × Retriever p95 latency(V1)** — i.e., accuracy gains must not come at unacceptable latency cost.

The compare script exits with non-zero status if any of these gates fails, making it usable as a pre-merge CI check.

---

**Document Maintained By:** Koby Lev
**Last Updated:** 2026-05-22
**Branch:** `feature/v2.0-openclip-migration`
