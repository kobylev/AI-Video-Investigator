"""AI Video Investigator — Dual-Agent Semantic Video Retrieval

This package implements a hybrid retrieval-reasoning pipeline for semantic search
over long-form security and dashcam footage.

Modules:
    retriever: CLIP encoder + FAISS index for fast semantic filtering
    router: Confidence-gated escalation logic to minimize LLM calls
    reasoner: Gemini 1.5 Pro API wrapper for deep multimodal reasoning
    pipeline: End-to-end orchestration of retrieval → routing → reasoning
    eval: Evaluation metrics (R@K, MRR, nDCG, F1, Accuracy)
"""

__version__ = "0.1.0"
__author__ = "Koby Lev"
