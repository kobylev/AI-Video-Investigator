# Work Packages — AI Video Investigator

This document tracks the progress of all 10 work packages for the AI Video Investigator project. Each work package represents a distinct phase of development, research, and delivery.

## Progress Dashboard

| WP # | Title | Status | Git Tag | Document Link |
|------|-------|--------|---------|---------------|
| WP1 | Planning & Preparatory Report | ✅ Completed | `v0.1.0-wp1` | [docs/work_packages/wp1_planning.md](docs/work_packages/wp1_planning.md) |
| WP2 | Business Plan | ✅ Completed | `v0.2.0-wp2` | [docs/WP2_BUSINESS_PLAN.md](docs/WP2_BUSINESS_PLAN.md) |
| WP3 | Data Acquisition & Benchmark Curation | 🏗️ In Progress | `v0.3.0-wp3` | [docs/WP3_CORE_SDK_AND_EVALUATION.md](docs/WP3_CORE_SDK_AND_EVALUATION.md) |
| WP4 | Retriever Implementation | ✅ Completed | `v0.4.0-wp4` | [docs/work_packages/wp4_retriever.md](docs/work_packages/wp4_retriever.md) |
| WP5 | Reasoner & Router Integration | ⏳ Not Started | `v0.5.0-wp5` | [docs/work_packages/wp5_reasoner_router.md](docs/work_packages/wp5_reasoner_router.md) |
| WP6 | Evaluation Harness | ⏳ Not Started | `v0.6.0-wp6` | [docs/work_packages/wp6_eval_harness.md](docs/work_packages/wp6_eval_harness.md) |
| WP7 | Baseline Comparisons | ⏳ Not Started | `v0.7.0-wp7` | [docs/work_packages/wp7_baselines.md](docs/work_packages/wp7_baselines.md) |
| WP8 | Results Analysis | ⏳ Not Started | `v0.8.0-wp8` | [docs/work_packages/wp8_results.md](docs/work_packages/wp8_results.md) |
| WP9 | Final Report | ⏳ Not Started | `v0.9.0-wp9` | [docs/work_packages/wp9_final_report.md](docs/work_packages/wp9_final_report.md) |
| WP10 | Defense Preparation & Presentation | ⏳ Not Started | `v1.0.0-wp10` | [docs/work_packages/wp10_defense.md](docs/work_packages/wp10_defense.md) |

## Git Tag Conventions

This project uses semantic versioning aligned with work package completion:

- **Format:** `v0.X.0-wpX` where X is the work package number
- **Example:** `v0.1.0-wp1` marks the completion of WP1 (Planning)
- **Final Release:** `v1.0.0-wp10` will mark the completion of the final defense

### Tagging Workflow

1. Complete all deliverables for a work package
2. Update the corresponding work package markdown file in `docs/work_packages/`
3. Update the status in this dashboard to ✅ Completed
4. Create a git tag: `git tag -a v0.X.0-wpX -m "Complete WPX: [title]"`
5. Push the tag: `git push origin v0.X.0-wpX`

## Time Axis vs. Functional Axis

**Important architectural decision:** This repository follows a dual organization strategy:

- **Code is organized by function** — `src/retriever/`, `src/router/`, `src/reasoner/` reflect system components
- **Documentation is organized by time** — `docs/work_packages/wp1_*.md` through `wp10_*.md` reflect project phases
- **Git tags serve as the time axis** — they mark completion of temporal milestones

This separation ensures that the codebase remains clean and maintainable while the documentation captures the evolution of the project over the 10-week timeline.
