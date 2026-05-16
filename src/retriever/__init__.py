"""CLIP Retriever Module

This module implements the first stage of the dual-agent pipeline: fast semantic
retrieval using CLIP embeddings and FAISS approximate nearest neighbor search.

Components:
    - CLIP ViT-L/14 image encoder (off-the-shelf)
    - CLIP ViT-L/14 text encoder
    - FAISS index builder (IVF or HNSW)
    - ANN search interface for top-K retrieval

Performance Target:
    - Latency: <100ms for 10-hour corpus (36,000 frames)
    - Recall@20: ≥0.90 (target for downstream reasoning stage)
"""
