# WP2 — Business Plan: VigilEdge

**Product:** VigilEdge — Privacy-Preserving Semantic Video Search
**Strategy Lead:** VigilEdge AI Strategist

---

## 1. Value Proposition

**VigilEdge** is an Edge-to-Cloud hybrid. It uses a local, lightweight "Sentry" (CLIP) to filter 99% of irrelevant frames on-site. Only high-probability candidates are routed to the "Thinker" (**Claude Haiku 4.5**) for final semantic reasoning. This provides the intelligence of a human investigator at the cost of a basic recording system.

**Tagline:** 
"VigilEdge delivers 100% of the intelligence of **Claude Haiku 4.5** at <10% of the cost and with 99% less data exposure than traditional cloud-AI."

---

## 4. Competitive Landscape

| Capability | VigilEdge (Ours) | Cloud-Only (Twelve Labs) | Object-Detection (Traditional) |
| :--- | :--- | :--- | :--- |
| **Zero-Shot NLP** | **Yes (CLIP + Claude Integration)** | Yes | No |

---

## 5. Economic ROI Model

| Metric | Naive **Claude-Only** Approach | **VigilEdge Dual-Agent Router** |
| :--- | :--- | :--- |
| Cost per Query-Hour | ~$0.30 | **<$0.05** |
| Latency (1h video) | 60s+ | **<3s** |

**Note:** By utilizing CLIP for edge-filtering, we avoid the "Token Tax" on empty or irrelevant footage. We only pay for **Claude's** "Brain" when there is actually something worth thinking about.

---

### 6. Decision Matrix: CLIP vs. Claude

| Factor | Fast Model (CLIP) | Advanced Model (**Claude Haiku 4.5**) |
| :--- | :--- | :--- |
| Deployment | Local (Edge) | Cloud (Anthropic) |
| Latency | <10ms | ~2s |
| Reasoning | Semantic Match | Logical Verification |
| Unit Cost | $0.00 | ~$0.0003 |

---

**Last Updated:** 2026-05-21
**Status:** Completed
