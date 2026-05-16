"""End-to-End Pipeline Orchestration

This module provides the main entry point for the AI Video Investigator system,
orchestrating the full retrieve → route → reason → rank workflow.

Pipeline Stages:
    1. CLIP Retrieval: Encode query, search FAISS index, return top-K candidates
    2. Confidence Routing: Evaluate CLIP similarity scores, decide escalation
    3. Gemini Reasoning: Analyze escalated candidates with multimodal LLM
    4. Re-Ranking: Sort by Gemini relevance scores, return top-5

Input:
    - Natural-language query (str)
    - Video corpus ID or FAISS index path
    - Configuration (K, τ_high, τ_low, prompt templates)

Output:
    - Ranked list of top-5 frames with:
        - Frame ID
        - Timestamp
        - Relevance score
        - Rationale (if Gemini was invoked)
        - Detected objects
    - System metrics:
        - End-to-end latency (seconds)
        - Tokens consumed
        - Cost (USD)
        - Router decision (skip, expand, escalate)

Usage Example:
    ```python
    from src.pipeline import VideoPipeline

    pipeline = VideoPipeline(corpus_id="bdd100k_10h")
    results = pipeline.search("red car running red light at intersection")

    for rank, result in enumerate(results, 1):
        print(f"{rank}. {result.timestamp} — {result.score:.2f} — {result.rationale}")
    ```
"""
