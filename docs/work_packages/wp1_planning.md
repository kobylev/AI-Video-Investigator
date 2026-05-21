# WP1 — Planning & Preparatory Report

**Status:** ✅ Completed
**Submission Date:** 2026-05-18
**Git Tag:** `v0.1.0-wp1`

---

## 1. Objectives

- Formalize the primary research question and hypotheses.
- Establish the system architecture (CLIP → Router → **Claude**).
- Define success metrics for both retrieval and reasoning stages.
- Prepare the project roadmap and risk register.

---

## 2. Decision Log

### Decision 3: Cloud Reasoner Model (Claude Haiku 4.5)
- **Option A:** Gemini 1.5 Pro
- **Option B:** **Claude Haiku 4.5 (Selected)**
- **Rationale:** Standardized on Anthropic for superior vision capabilities, forced tool-use reliability, and better developer experience.

---

## 5. Defense Mock Q&A

### Q: "Why not just send the whole video to Claude?"
**A:** Cost and latency. Brute-forcing a 1fps video through Claude vision tokens costs ~$0.30/query-hour. For a fleet of 100 vehicles, this is not economically viable. The dual-agent router reduces this cost by >95% while maintaining accuracy.

---

**Last Updated:** 2026-05-21
