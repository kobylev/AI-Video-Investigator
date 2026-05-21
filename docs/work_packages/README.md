# Work Packages — AI Video Investigator

This directory contains detailed documentation for each of the 10 work packages (WPs) that make up the AI Video Investigator project.

## Directory Structure

- `wp1_planning.md`: Project scope, architecture, and baseline definitions.
- `wp3_data_acquisition.md`: Dataset subset selection and ground-truth annotation.
- `wp4_retriever.md`: FAISS indexing and CLIP-only baseline.
- `wp5_reasoner_router.md`: **Claude Haiku 4.5** integration and Budget-Aware Router.
- `wp6_eval_harness.md`: Metrics implementation and dual-agent validation.
- `wp7_baselines.md`: Comparison against **Claude-only** and CLIP-only baselines.
- `wp8_results.md`: Final results, ablation studies, and visualizations.

## Core Architecture

The system follows a Dual-Agent cascade:
1. **Retriever (CLIP + FAISS):** Local filter to reduce search space.
2. **Reasoner (**Claude Haiku 4.5**):** Cloud-based agent for high-order verification.

---
**Last Updated:** 2026-05-21
