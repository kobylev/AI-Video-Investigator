# Work Packages — Documentation Convention

## Organizational Philosophy

This repository follows a **dual-axis organization strategy** to balance code maintainability with temporal project tracking:

### Code Axis (Functional)

The `src/` directory is organized by **system component**:

```
src/
├── retriever/     # CLIP encoder + FAISS index
├── router/        # Confidence-gated escalation logic
├── reasoner/      # Gemini 1.5 Pro API wrapper
├── pipeline/      # End-to-end orchestration
└── eval/          # Metrics: R@K, MRR, nDCG, F1, Accuracy
```

**Rationale:** Functional organization ensures that the codebase remains clean, modular, and maintainable. A developer looking for "how does routing work?" should not have to dig through 10 time-based folders.

---

### Documentation Axis (Temporal)

The `docs/work_packages/` directory is organized by **project phase** (WP1 through WP10):

```
docs/work_packages/
├── wp1_planning.md
├── wp2_business_plan.md
├── wp3_data_acquisition.md
├── wp4_retriever.md
├── wp5_reasoner_router.md
├── wp6_eval_harness.md
├── wp7_baselines.md
├── wp8_results.md
├── wp9_final_report.md
└── wp10_defense.md
```

**Rationale:** Academic projects are inherently temporal. Work packages capture the evolution of the project over 10 weeks, including:
- Decisions made at each phase
- Deliverables submitted
- Lessons learned
- Outputs feeding into the next WP

---

## Git Tags as the Time Axis

To reconcile the functional codebase with the temporal project timeline, **Git tags serve as the time axis**:

- **Tag Format:** `v0.X.0-wpX` (e.g., `v0.1.0-wp1`, `v0.4.0-wp4`)
- **Purpose:** Marks the completion of a work package, freezing the codebase state at that milestone
- **Usage:** A future developer (or the author during writeup) can checkout any tag to see exactly what the system looked like at the end of WP4, WP6, etc.

**Example Workflow:**

```bash
# Complete WP4 (retriever implementation)
git add src/retriever/ evals/
git commit -m "feat(retriever): implement CLIP+FAISS indexing pipeline"
git tag -a v0.4.0-wp4 -m "Complete WP4: Retriever Implementation"
git push origin v0.4.0-wp4
```

---

## Work Package Document Structure

Each `wpX_*.md` file follows a consistent structure defined in `_template.md`:

### Sections

1. **Header** — WP number, title, status, submission date, Git tag
2. **Objectives** — What this WP aimed to achieve (2–4 bullet points)
3. **Deliverables** — List of outputs with file paths (reports, code, benchmarks)
4. **Key Decisions** — Architectural or methodological choices made during this phase
5. **Outputs Feeding Next WP** — What the next WP depends on from this one
6. **Open Issues / Carried Forward** — Unresolved questions or deferred tasks

### Benefits

- **Traceability:** Understand *why* a decision was made and *when*
- **Defense Preparation:** Quickly review the project narrative before WP10
- **Knowledge Transfer:** If another student continues this work, they have a roadmap
- **Academic Integrity:** Documents the research process, not just the final result

---

## Status Lifecycle

Each work package progresses through these states:

| Status | Meaning | Indicator |
|--------|---------|-----------|
| ⏳ Not Started | Work package not yet begun | No file content beyond title |
| 🔄 In Progress | Actively working on this WP | Objectives defined, deliverables TBD |
| ✅ Completed | All deliverables submitted, tag created | Full document with retrospective |

**Current Status (as of 2026-05-16):**
- WP1: 🔄 In Progress (defense scheduled for Sunday)
- WP2–WP10: ⏳ Not Started

---

## Cross-Linking Convention

Work package documents should **cross-link** to:

1. **Source Code:** Link to specific modules (e.g., `src/retriever/clip_encoder.py:42`)
2. **Evaluation Results:** Link to experiment logs (e.g., `evals/results/wp6_baseline_comparison.json`)
3. **Root Docs:** Link to `docs/PRD.md`, `docs/architecture.md`, etc. for foundational context
4. **Deliverables:** Link to final PDFs in `deliverables/wpX/`

**Example:**

> Decision: Use CLIP ViT-L/14 off-the-shelf (no fine-tuning) for WP4 baseline. See implementation in `src/retriever/__init__.py` and evaluation in `evals/results/wp4_clip_baseline.json`. Rationale documented in `docs/flagship_paper_notes.md` (adoption vs. divergence).

---

## Template Usage

When starting a new work package:

1. Copy `_template.md` to `wpX_title.md`
2. Fill in the header with WP number, title, target submission date
3. Set status to 🔄 In Progress
4. Define objectives at the start of the WP
5. Update deliverables and key decisions as work progresses
6. Mark ✅ Completed when all deliverables are submitted
7. Create the corresponding Git tag (`v0.X.0-wpX`)

---

## Why This Matters

Academic software projects often suffer from **temporal amnesia**:

- Six months later: *"Why did we choose FAISS over Annoy?"*
- During defense: *"What changed between WP5 and WP6?"*
- Future work: *"What were the open issues at the end of this project?"*

This work package structure ensures that the **narrative of the project** is captured alongside the code, making the research process transparent, reproducible, and defensible.

---

**Maintained by:** Koby Lev | **Last Updated:** 2026-05-16
