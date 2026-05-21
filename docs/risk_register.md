# Risk Register

## AI Video Investigator — Project Risks & Mitigations

---

## Risk 2: Anthropic API Cost Overrun

### Description
Claude Haiku 4.5 API pricing is highly competitive but still represents a variable OpEx line item. If experiments exceed budget, the project may become financially infeasible.

### Likelihood × Impact
- **Likelihood:** Medium (2)
- **Impact:** Medium (2)
- **Risk Score:** **4 (High Priority)**

### Mitigation Strategy
1. **Cost Monitoring:** Set up **Anthropic dashboard alerts**; log usage to `evals/results/token_log.csv`.
2. **Cost Control Measures:** Cache responses; use K=5 during development.
3. **Fallback Model:** If costs exceed $100, switch to cheaper Anthropic tier or GPT-4o mini.

---

## Risk 4: Anthropic API Rate Limits & Latency Variability

### Description
Anthropic enforces **rate limits**. Additionally, cloud latency may be **variable**, risking the <3s target.

### Likelihood × Impact
- **Likelihood:** High (3)
- **Impact:** Medium (2)
- **Risk Score:** **6 (High Priority)**

### Mitigation Strategy
1. **Parallelize with Backoff:** Use `asyncio` + exponential backoff.
2. **Caching Layer:** Store responses in `evals/results/claude_cache.jsonl`.
3. **Latency Monitoring:** Log p50, p95, p99 latency.

---

## Summary Table

| # | Risk | Likelihood | Impact | Score | Priority | Mitigation Owner |
|---|------|------------|--------|-------|----------|------------------|
| 2 | Anthropic API Cost Overrun | 2 | 2 | 4 | High | Koby Lev |
| 4 | Anthropic Rate Limits & Latency | 3 | 2 | 6 | High | Koby Lev |

**Top 3 Risks (Score ≥ 6):**
1. Dataset Access (6)
2. CLIP Domain Gap (6)
3. Anthropic Rate Limits (6)

---

**Last Updated:** 2026-05-21
**Status:** WP5 — Reasoner & Router Integration
