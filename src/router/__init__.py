"""Confidence-Gated Router Module

This module implements the decision layer between CLIP retrieval and Gemini reasoning,
routing queries based on CLIP confidence scores to minimize API costs while preserving accuracy.

Routing Logic:
    - High confidence (sim > τ_high): Return CLIP top-1 immediately, skip Gemini
    - Low confidence (sim < τ_low): Expand K and escalate to Gemini
    - Ambiguous (τ_low ≤ sim ≤ τ_high): Escalate top-K to Gemini for re-ranking

Thresholds (Initial):
    - τ_high = 0.85 (90th percentile of positive pairs)
    - τ_low = 0.60 (50th percentile of positive pairs)
    - Tuned empirically in WP5 via grid search on validation set

Impact:
    - Reduces Gemini calls by 40–60% (estimated)
    - Preserves accuracy by escalating ambiguous cases
"""
