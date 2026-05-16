# Deliverables

## Final Submitted PDFs Per Work Package

This directory holds the **final submitted PDF deliverables** for each work package. These PDFs are the official artifacts submitted for academic review and grading.

---

## Directory Structure

```
deliverables/
├── wp1/    # WP1 — Planning & Preparatory Report
├── wp2/    # WP2 — Business Plan
├── wp3/    # WP3 — Data Acquisition Report (if applicable)
├── wp4/    # WP4 — Retriever Implementation Report
├── wp5/    # WP5 — Reasoner & Router Report
├── wp6/    # WP6 — Evaluation Harness Report
├── wp7/    # WP7 — Baseline Comparisons Report
├── wp8/    # WP8 — Results Analysis Report
├── wp9/    # WP9 — Final Report (comprehensive)
└── wp10/   # WP10 — Defense Presentation (slides PDF)
```

---

## File Naming Convention

Use descriptive names that include the work package number and a brief title:

- `wp1_preparatory_report_koby_lev.pdf`
- `wp2_business_plan_koby_lev.pdf`
- `wp9_final_report_koby_lev.pdf`
- `wp10_defense_slides_koby_lev.pdf`

---

## Gitignore Policy

**Important:** All PDF files in this directory are **gitignored** (see `.gitignore` rule: `deliverables/**/*.pdf`).

**Rationale:**
- PDFs are binary files that bloat Git history
- Final reports may contain sensitive information (if applicable)
- Submission is typically via institutional portal, not GitHub

**Exception:** If you need to version-control a PDF for archival purposes, explicitly force-add it:

```bash
git add -f deliverables/wp9/wp9_final_report_koby_lev.pdf
```

---

## Submission Workflow

1. **Complete work package** (code, experiments, analysis)
2. **Write report** in Markdown or LaTeX (in `docs/work_packages/wpX_*.md` or separate `.tex` file)
3. **Export to PDF** using Pandoc, LaTeX compiler, or Word → PDF
4. **Copy PDF** to `deliverables/wpX/`
5. **Update work package document** with link to PDF and mark status as ✅ Completed
6. **Create Git tag** `v0.X.0-wpX` to mark WP completion
7. **Submit PDF** via institutional portal or email to lecturer

---

## Cross-References

Each work package document in `docs/work_packages/` should link back to its corresponding deliverable PDF:

**Example:**

> Final Report: [deliverables/wp1/wp1_preparatory_report_koby_lev.pdf](wp1/wp1_preparatory_report_koby_lev.pdf)

---

**Maintained by:** Koby Lev | **Last Updated:** 2026-05-16
