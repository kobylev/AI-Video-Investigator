"""End-to-End Orchestration Pipeline
Data Flow:
    1. CLIP Encoding: Local vector generation
    2. FAISS Search: Fast retrieval of top-K candidates
    3. Claude Reasoning: Analyze escalated candidates with multimodal LLM
    4. Re-Ranking: Sort by Claude relevance scores, return top-5
"""
