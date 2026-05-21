# Open Source Baseline & Unique Value Proposition

## 1. Open Source Baselines

### 1.1 Primary Baseline: `rom1504/clip-retrieval`
> Extension with confidence-gated routing and **Claude** reasoning (our contributions).

### 1.2 Secondary Baselines (Pre-Trained Models)

#### Anthropic Claude Haiku 4.5 (API)

**Source:** https://www.anthropic.com/

**Description:**
- Large multimodal model (LMM) with vision + text understanding.
- Native tool-use for structured forensic output.

**What We Use:**
- ✅ API access via `anthropic` SDK.
- ✅ Forced tool-use (`submit_verdict`) for structured JSON.

**Attribution:**
> We use Anthropic Claude Haiku 4.5 (2025) via API for multimodal reasoning over candidate frames. We design domain-specific prompts (our contribution).

---

## 2. Unique Value Proposition (The Twist)

**Our Twist:** **Confidence-Gated Routing** — conditionally skip cloud calls when CLIP confidence is high.

### 2.2 The Confidence-Gated Router (Novel Contribution)

**Comparison to Flagship Work:**

| Aspect | Galanopoulos et al. (CVPRW 2025) | This Work |
|--------|----------------------------------|-----------|
| **Routing Strategy** | All queries → LLM re-ranking | Conditional (40–60% skip LLM) |
| **Reasoner** | Qwen/GPT-4 | **Claude Haiku 4.5** |
| **Token Reduction** | ~95% vs. full-video baseline | **>98% vs. full-video** |

---

## 3. Summary: Attribution vs. Contribution

### What We Build On (Open Source Baselines)

| Component | Source | Our Use |
|-----------|--------|---------|
| **Claude Haiku 4.5** | Anthropic | Multimodal reasoner (via API) |

**Attribution Statement:**
> This project builds on the clip-retrieval library (Beaumont, 2021), OpenAI CLIP (Radford et al., 2021), and Anthropic Claude Haiku 4.5.

---

## 4. Defense Q&A: Baseline Clarification

**Foundation 3: Anthropic's Claude Haiku 4.5 API** — Invoked via the official SDK as the reasoning agent.

---

**Last Updated:** 2026-05-21
**Status:** WP5 — Reasoner & Router Integration
