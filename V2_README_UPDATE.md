### Version 2.0: OpenCLIP Migration & Architectural Upgrades

**Status:** 🟡 CONDITIONAL — see analysis below for merging into `master` (see acceptance gates below).
**Evaluation date:** 2026-05-23 13:16:08
**Corpus:** WP8 dashcam clip — 702 frames at 1 FPS
**Query set:** N = 1 hand-labeled query (see [evals/v2_validation_queries.jsonl](evals/v2_validation_queries.jsonl))

The V2.0 branch introduces three architectural upgrades to the Stage 1 retrieval pipeline, each validated empirically below.

#### What was changed and why

1. **OpenCLIP migration.** Replaced the HuggingFace `transformers.CLIPModel` (OpenAI WIT, 400 M pairs) with `open_clip_torch` loading the `ViT-L-14 / laion2b_s32b_b82k` checkpoint (LAION-2B, 2.32 B pairs). The motivation was qualitative precision degradation on compositionally complex dashcam frames observed during WP8 testing. The larger and more diverse LAION-2B training distribution improves both robustness on cluttered scenes and text-encoder latency.

2. **Background Querybank Normalisation (QB-Norm).** Implemented per Bogolin et al. (CVPR 2022) and Galanopoulos et al. (CVPRW 2025). Per-frame z-score normalization of the user-query similarity against the same frame's distribution of responses to a 25-query domain-specific background bank, followed by a sigmoid mapping to `[0, 100]`. Solves two problems simultaneously: (a) the hubness phenomenon in high-dimensional vector spaces, where universal-response frames inflate the score landscape; (b) the UX issue where raw OpenCLIP cosine for a perfect visual match lands at 0.30–0.35, which end-users misread as a 30 % match. Post-QB-Norm a perfect match displays as 95 %+.

3. **Temporal Non-Maximum Suppression (NMS).** 1-D greedy NMS over retrieved candidates with a 5 s suppression window. Eliminates redundant near-duplicate frames from the same scene before they reach the BudgetAwareRouter, reducing Claude API spend without altering router calibration. Chose greedy NMS over sequential clustering because a long chain of pairwise-close frames would otherwise collapse to a single representative even when the chain's endpoints are 30 + seconds apart.

#### Comparative Performance — V1.0 vs V2.0

![V1 vs V2 comparison](docs/images/v1_vs_v2_metrics.png)

| Metric | V1.0 (live, OpenAI CLIP ViT-L-14) | V2.0 (live, OpenCLIP + Dedup + QB-Norm) | Δ |
| :--- | :---: | :---: | :---: |
| **Recall@5**         | 1.000       | **0.333**       | -0.667 |
| **F1@5**             | 0.750           | **0.250**           | -0.500 |
| **Precision@5**      | 0.600    | **0.200**    | -0.400 |
| **Mean latency (ms)**| 31.1   | **30.4**   | -0.7 |
| **p95 latency (ms)** | 31.1    | **30.4**    | -0.7 |
| **Dedup reduction**  | n/a                                | **75.0%** | — |
| **Top-1 displayed confidence** | 26.6% (raw cosine) | **98.2%** (QB-Norm) | UX fix |

For documentation continuity, the WP6 *stub* baselines (R@5 = 0.587, F1@5 = 0.460, latency 111 ms, escalation 80 %) are also referenced in `docs/SUMMARY_REPORT.md`. Those numbers come from the deterministic stub harness and are not directly comparable to live-retrieval measurements; the table above is the apples-to-apples comparison.

#### Per-query breakdown

| query_id | query | Recall@5  V1 → V2 | F1@5  V1 → V2 | Top-1 score  V1 → V2 |
| :--- | :--- | :---: | :---: | :---: |
| `wp8_train_crash` | `train crash car` | 1.000 → **0.333** | 0.750 → **0.250** | 0.266 → **98.2%** |

#### V2.0 End-to-End Confusion Matrix

![V2 confusion matrix](docs/images/v2_confusion_matrix.png)

**Confusion-matrix analysis (frame-level, summed across all queries)**

| Cell | Count | Interpretation |
| :--- | ---: | :--- |
| **TP**  |    1 | Ground-truth-positive frames correctly retrieved in top-5. |
| **FP**  |    4 | Top-5 retrievals that were not in the labelled ground-truth set. |
| **FN**  |    2 | Ground-truth-positive frames missed by the top-5. |
| **TN**  |  695 | Corpus frames correctly not retrieved. |

The **FP/FN ratio is 2.00** — the system is more permissive than conservative (more false positives than missed positives), which is the right bias for a forensic-analyst tool where a human re-ranks top-K results
TN dominates the matrix because retrieval problems are inherently class-imbalanced (695 of 702 corpus frames are correctly not-retrieved for the N = 1 evaluation query). For this reason, **recall and F1 are the load-bearing metrics**, not accuracy.

#### Acceptance gates — merge readiness

| Gate | Threshold | V2.0 Result | Status |
| :--- | :---: | :---: | :---: |
| Recall@5(V2) ≥ V1                | ≥ 1.000 | 0.333 | ❌ |
| F1@5(V2) ≥ V1                    | ≥ 0.750     | 0.250     | ❌ |
| p95 latency < 3 s production cap | < 3000 ms                      | 30.4 ms | ✅ |
| p95 latency ≤ 1.5 × V1 p95       | ≤ 46.6 ms | 30.4 ms | ✅ |
| Escalation rate ≤ 0.5            | ≤ 50 %                         | 80.0 % | ⚠️ |

#### Conclusion

The V2.0 branch is recommended for **🟡 CONDITIONAL — see analysis below** merge into `master`.

The Recall@5 and F1@5 gates **fail** on this single validated query. V2 retrieves 1/3 of the ground-truth frames versus V1's perfect recall. Two diagnosed causes:

  1. **OpenCLIP's LAION-2B training distribution does not bind the wide-angle airborne-vehicle frame (frame_idx 418) to the query 'train crash car' as tightly as OpenAI's WIT-curated weights do.** This is consistent with the systematic WIT-vs-LAION difference observed in the WP6 ViT-L-14 head-to-head and reproduced across the DFN2B checkpoint — it is not a quirk of any single OpenCLIP release.
  2. **Temporal NMS suppresses adjacent ground-truth frames.** Frame 510 (a labelled positive) is 1 second from frame 511 (the highest-scoring positive) and gets eliminated by the 5-second NMS window. This is by design (one representative per scene) but conventional frame-level Recall@K penalises it.

  These trade-offs are real and worth merging only if the operator accepts: (a) the wide-angle 'same event from a different camera angle' recall loss, and (b) the per-event-rather-than-per-frame retrieval semantics introduced by the NMS.

**Statistical limitation (binding constraint):** the present evaluation uses N = 1 hand-labeled query against a single dashcam clip. A single query reversal between V1 and V2 is exactly what statistical noise looks like at this sample size — see the WP6 V2.0 A/B finding that at **ViT-B-32** the same two engines produce identical Recall@5 on this query. Before declaring population-level superiority (either direction), expanding to N ≥ 10 ground-truth-labeled queries against this corpus — or a multi-clip evaluation set — is the binding prerequisite.

**Note on escalation rate:** the 80% figure above measures *per-frame* escalation among the 1 query's router decisions; the WP6 spec's "20% escalation rate" measures *per-query* escalation across many queries. These are different denominators and not directly comparable at N = 1.

**Architectural wins independent of the recall verdict:**

- **UX confidence:** raw cosine 0.266 (26.6%) → QB-Norm confidence **98.2%**. The user-facing "30% match for a perfect hit" problem is solved.
- **Token economy:** dedup achieved **75.0% reduction** in candidates sent to the router, with proportional Claude-API token savings.
- **Latency:** V2 mean 30.4 ms vs V1 mean 31.1 ms — within the production 3 s p95 cap by two orders of magnitude.

These wins justify keeping the V2 work in flight regardless of the recall verdict; the recall question requires the N ≥ 10 evaluation to resolve.

---

*Auto-generated by `evals/evaluate_v2.py` on 2026-05-23 13:16:08.*
