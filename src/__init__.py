"""AI Video Investigator Source
Modules:
    retriever: CLIP-based frame extraction and FAISS indexing
    router: Confidence-gated routing logic (Edge-to-Cloud)
    reasoner: Claude Haiku 4.5 API wrapper for deep multimodal reasoning
    pipeline: End-to-end orchestration of the Dual-Agent cascade
    eval: Metrics and benchmarking harness
"""
