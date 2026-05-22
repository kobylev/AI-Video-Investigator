# Claude Forensic Prompt Book

## Domain-Specific Prompt Templates for Dashcam Video Analysis

This document contains structured prompt templates for the **Claude Haiku 4.5** reasoner, organized by event type. Each template is designed to maximize forensic accuracy while maintaining a strict JSON output contract via Anthropic tool-use.

**Version:** 2.0.0 (WP8 — Full-Stack Closure) | **Last Updated:** 2026-05-22 | **Status:** All Work Packages (WP1–WP8) Complete

> **Document Role within the Project Lineage**
>
> The Prompt Book is the **operational artefact** of the project's reasoning tier. It was first scoped in **WP1** (research question), economically justified in **WP2** (token economics), architecturally placed in **WP3** (SDK boundaries), exercised against the **WP4** FAISS retriever, formally implemented in **WP5** (this document's initial release), empirically validated in **WP6** (benchmark harness), documented in **WP7** (Summary Report), and finally surfaced to the end user through the **WP8** Angular Material frontend. See the [Work Package Integration & Lineage](#work-package-integration--lineage) section at the foot of this document for the per-WP contribution detail.

---

## General Principles

1. **Role Priming:** Every prompt begins with "You are a senior forensic video analyst..."
2. **Structured Output:** All prompts mandate a JSON response via the `submit_verdict` tool.
3. **Chain-of-Thought (CoT):** Prompts enforce explicit step-by-step reasoning before reaching a verdict.
4. **Binary Relevance:** The system relies on the `is_event_present` boolean and `confidence_score` (0.0–1.0).

---

## The Output Contract (ForensicVerdict)

All templates are designed to return a JSON object matching this schema:

```json
{
  "chain_of_thought": "Step-by-step visual audit of the scene...",
  "is_event_present": true/false,
  "confidence_score": 0.0 to 1.0,
  "forensic_summary": "Concise summary of evidence (<=280 chars)"
}
```

---

## Template 1: Vehicle Interaction Event

**Use Cases:** Collision, near-miss, aggressive driving, tailgating, overtaking

### System Prompt

```
You are a senior forensic video analyst specializing in dashcam footage review for fleet safety and insurance investigations. Your task is to determine whether a video frame shows evidence of a specific vehicle interaction event.

ANALYTICAL GUIDELINES:
1. Reason ONLY from the visual evidence present in the frames.
2. Identify all vehicles (type, color, position) and assess their interactions.
3. Look for contextual clues: brake lights, swerving, close proximity, debris.
4. Distinguish between normal traffic flow and aggressive/hazardous maneuvers.

OUTPUT CONTRACT:
You MUST call the `submit_verdict` tool exactly once with your structured findings.
```

### User Prompt Template

```
QUERY: "{user_query}"

Analyze the dashcam frame provided and submit your verdict. Perform a step-by-step visual audit of vehicle positions and behaviors before concluding.
```

---

## Template 2: Pedestrian Event

**Use Cases:** Jaywalking, pedestrian crossing, fall, crowd behavior, pedestrian-vehicle near-miss

### System Prompt

```
You are a senior forensic video analyst specializing in pedestrian safety analysis from surveillance footage. Your task is to identify pedestrian-related events in traffic environments.

ANALYTICAL GUIDELINES:
1. Identify all pedestrians (clothing, activity, position).
2. Assess whether behavior matches the query (e.g., crossing outside crosswalk).
3. Evaluate risk level: is this a safety incident or ambient activity?
4. Cite specific visual cues like crosswalk markings or vehicle proximity.

OUTPUT CONTRACT:
You MUST call the `submit_verdict` tool exactly once.
```

---

## Template 3: Traffic Violation

**Use Cases:** Red light running, stop sign violation, illegal turn, speeding, wrong-way driving

### System Prompt

```
You are a senior forensic video analyst specializing in traffic law enforcement review. Your task is to identify potential traffic violations from dashcam footage.

ANALYTICAL GUIDELINES:
1. Identify the traffic control device (signal, sign, marking).
2. Identify the vehicle(s) potentially in violation.
3. Assess the state of the control device (e.g., signal color) at the exact moment of the vehicle's maneuver.
4. Determine if the action constitutes a violation based on visible evidence.

OUTPUT CONTRACT:
You MUST call the `submit_verdict` tool exactly once.
```

---

## Prompt Engineering Notes

### Why Claude Haiku 4.5?

- **Low Latency:** Sub-second response times for single-frame analysis.
- **Superior Vision:** High-fidelity detection of small objects (license plates, signals) compared to previous generations.
- **Forced Tool Use:** Guarantees 100% JSON compliance, eliminating parsing failures.

### Confidence Calibration

- **0.9–1.0:** Definitive match (Clear evidence, no occlusion).
- **0.7–0.9:** Strong match (Event present, minor ambiguity).
- **0.5–0.7:** Probable match (Likely event, but evidence is partial).
- **<0.5:** Inconclusive or non-event.

---

## Work Package Integration & Lineage

The Prompt Book is not a stand-alone artefact — it is the **terminal point of a multi-stage cascade** whose design constraints, economic envelope, retrieval inputs, and presentation surface were each defined by a distinct work package. The table and per-WP commentary below trace the contribution of each work package to the prompt templates contained in this document.

### Summary Matrix

| WP | Phase | Title | Contribution to the Prompt Book | Status |
| :---: | :--- | :--- | :--- | :---: |
| **WP1** | Planning | Planning & Preparatory Report | Established the research question and the VALID-framework constraints that govern every prompt's role-priming and binary-relevance contract. | ✅ |
| **WP2** | Planning | Business Plan & Token Economics | Imposed the cost ceiling (<$0.10/query-hour) that mandates terse system prompts and ≤280-char `forensic_summary` outputs. | ✅ |
| **WP3** | Planning | Core SDK & Routing Logic | Defined the SDK boundary at which the `PromptManager` dispatches per-event-type templates and enforces the `submit_verdict` tool contract. | ✅ |
| **WP4** | Edge | FAISS Index & CLIP Filtering | Supplies the top-K candidate frames consumed by the user-prompt template — the prompt book is the *consumer* of WP4's retrieval output. | ✅ |
| **WP5** | Cloud | Reasoner & Router Integration | **Original authorship of this document.** Implemented all three templates, the forced-JSON tool-use protocol, and the confidence-calibration ladder. | ✅ |
| **WP6** | Evaluation | Harness & Benchmarks | Empirically validated prompt accuracy — confirmed Recall@5 = 0.587 and 99.97% privacy retention against `clip_only` and `claude_only_stub` baselines. | ✅ |
| **WP7** | Summary | Final Report & Documentation | Codified the prompt book's empirical results into the Final Summary Report and aligned terminology across the documentation tree. | ✅ |
| **WP8** | Frontend | Angular GUI & Backend Orchestration | Exposes the `forensic_summary` field of each `ForensicVerdict` as a *FORENSIC ANALYSIS SUMMARY* card in the user-facing Investigation Report. | ✅ |

### Per-Work-Package Detail

**WP1 — Planning & Preparatory Report.** The dual-agent retrieve-then-reason research hypothesis directly motivated the structural decision to keep prompts *narrow, event-typed, and binary-relevance-oriented*. The role-priming line ("*You are a senior forensic video analyst...*") is a direct artefact of the VALID-framework persona modelling carried out in WP1.

**WP2 — Business Plan & Token Economics.** The mathematical token-economics model derived in WP2 ($C_{\text{naive}} \approx \$84.60$ per query vs. $C_{\text{dual}} \approx \$0.0094$) is the reason the prompt templates are aggressively compressed: every additional sentence of system-prompt boilerplate would erode the >99% cost saving demonstrated by the WP6 benchmarks. The 280-character ceiling on `forensic_summary` is a direct WP2 budget control.

**WP3 — Core SDK & Routing Logic.** WP3 defined the `PromptManager` class responsible for selecting between Template 1 (Vehicle Interaction), Template 2 (Pedestrian), and Template 3 (Traffic Violation) based on the routing metadata attached to each query. The Prompt Book exists as a structured document precisely so the SDK's `PromptManager` can ingest it programmatically.

**WP4 — FAISS Index & Filtering.** WP4 supplies the **input** to every prompt: only frames pre-filtered by the on-premise CLIP ViT-L/14 + FAISS `IndexFlatIP` retriever ever reach a template. The "{user_query}" placeholder in each user-prompt template is bound to the same natural-language string that drove WP4's text-encoder retrieval, ensuring lexical consistency between edge and cloud tiers.

**WP5 — Reasoner & Router Integration (Original Release).** WP5 is the work package in which this document was first authored. The three templates, the `submit_verdict` forced-tool-use protocol, the `ForensicVerdict` Pydantic schema, and the confidence-calibration ladder (0.9–1.0 / 0.7–0.9 / 0.5–0.7 / <0.5) are all WP5 deliverables.

**WP6 — Evaluation Harness & Benchmarks.** The benchmark harness in WP6 evaluated every template against a held-out query set. Empirical findings — Recall@5 of 0.587 and F1@5 of 0.460 on the `dual_agent` configuration, with 80% query on-prem retention and a mean total latency of 111.0 ms — directly validate the prompt designs documented above and confirmed that no further template revisions were required before final delivery.

**WP7 — Summary Report & Documentation.** WP7 absorbed the empirical findings produced by WP6 into the [Final Summary Report](../SUMMARY_REPORT.md), with cross-references to the templates in this document. Terminology and version markers across the documentation tree were normalised at this stage.

**WP8 — Angular GUI & Backend Orchestration.** WP8 closes the loop: the `forensic_summary` string produced by each `ForensicVerdict` is now rendered to the operator as a *FORENSIC ANALYSIS SUMMARY* paragraph beneath each retrieved keyframe in the Investigation Report (see Figures WP8-A and WP8-B in the [Final Summary Report](../SUMMARY_REPORT.md)). The `confidence_score` is surfaced as the percentage *"NN% Match"* badge, and the `is_event_present` boolean drives the *Cloud Verified* chip. The prompt-book contract is therefore now an end-user-facing artefact, not merely an internal SDK convention.

---

## Document Status

The Prompt Book is **frozen for v1.0 delivery**. All template revisions, validation runs, and integration points required for the WP1–WP8 scope are complete. Future revisions (Template 4 for crowd-behaviour analysis, Template 5 for property-damage assessment) are deferred to the post-defence roadmap discussed in [§6 of the Final Summary Report](../SUMMARY_REPORT.md#6-conclusions--future-work).
