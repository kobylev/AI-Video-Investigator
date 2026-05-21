# WP5 — The Prompt Book: Forensic Reasoning with Claude Haiku 4.5

**Status:** ✅ Completed
**Agent Role:** Agent 2 (The Reasoner)
**Goal:** High-order semantic verification of "Ambiguous" retrieval candidates.

---

## 1. The Forensic System Prompt
To ensure Claude acts as a strict investigator, we use a class-constant `SYSTEM_PROMPT` that enforces analytical rigor and mandates the use of the `submit_verdict` tool.

**System Prompt:**
> You are an EXPERT FORENSIC VIDEO ANALYST embedded in an automated investigation pipeline. Your task is to inspect ONE or MORE still frames extracted from surveillance footage and decide whether the user's described event is visually occurring in any of them.
>
> **ANALYTICAL GUIDELINES:**
> 1. Reason ONLY from the visual evidence present in the frames. Do NOT infer events from context outside the image.
> 2. Treat ambiguity conservatively.
> 3. Distinguish the queried event from visually similar but distinct activities.
> 4. The reasoning string must cite specific visual cues.
>
> **OUTPUT CONTRACT:**
> You MUST call the `submit_verdict` tool exactly once with the structured fields.

---

## 2. Structured Output (Anthropic Tool Use)
We utilize Claude's native tool-use capability to ensure the verdict matches our JSON schema.

**Tool Schema (submit_verdict):**
```json
{
  "name": "submit_verdict",
  "input_schema": {
    "type": "object",
    "properties": {
      "event_detected": {"type": "boolean"},
      "confidence_score": {"type": "number"},
      "reasoning": {"type": "string"}
    },
    "required": ["event_detected", "confidence_score", "reasoning"]
  }
}
```

---

## 3. Token Economics & Thresholding (τ)
*   **Ambiguous Band:** Escalated to **Claude Haiku 4.5**. *Cost: ~$0.0003.*

### Typical Efficiency Gains:
Claude Haiku 4.5 offers significantly lower latency and cost compared to Gemini 1.5 Pro, achieving a **>99.9% reduction in query-hour costs** when combined with our CLIP-FAISS retriever.

---

## 4. Implementation Details
The `ClaudeReasoner` and `BudgetAwareRouter` are implemented in:
- `src/router/core.py`
- `src/reasoner/claude_engine.py`
