"""Gemini 1.5 Pro Reasoner Module

This module wraps the Google Generative AI API for deep multimodal reasoning over
candidate frames surfaced by the CLIP retriever.

Components:
    - Gemini API client (google-generativeai SDK)
    - Structured prompt templates per event type (see docs/prompts/prompt_book.md)
    - JSON response parser for relevance scoring
    - Re-ranking logic based on Gemini scores

Input:
    - Top-K candidate frames (default K=20)
    - Natural-language query
    - Event-type-specific prompt template

Output:
    - Per-frame relevance score (0.0–1.0)
    - Rationale (1-2 sentence explanation)
    - Detected objects
    - Confidence level (high, medium, low)

Performance Target:
    - Latency: <2.5s for K=20 frames (parallelized API calls)
    - Cost: ~$0.01 per query (8,160 tokens @ $1.25/1M tokens)
"""
