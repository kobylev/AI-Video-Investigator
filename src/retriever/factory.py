"""Retrieval-engine factory (V2.0).

Selects the Stage 1 semantic-filter backend at runtime so that the WP6
evaluation harness, the FastAPI server, and the FAISS indexer can all swap
between the V1 (OpenAI CLIP via transformers) and V2 (OpenCLIP / LAION-2B)
engines without code edits.

Resolution order:
    1. Explicit `backend=` argument
    2. RETRIEVER_BACKEND environment variable
    3. Default: "openclip"
"""

from __future__ import annotations

import os
from typing import Literal, Protocol


class RetrievalEngine(Protocol):
    def get_image_embeddings(self, images): ...
    def get_text_embeddings(self, queries): ...
    def compute_similarity(self, image, query) -> float: ...


def build_retriever(
    backend: Literal["openai_hf", "openclip"] | None = None,
) -> RetrievalEngine:
    backend = backend or os.getenv("RETRIEVER_BACKEND", "openclip")
    if backend == "openclip":
        from src.retriever.openclip_engine import OpenCLIPEngine
        return OpenCLIPEngine()
    if backend == "openai_hf":
        from src.retriever.clip_engine import CLIPEngine
        return CLIPEngine()
    raise ValueError(
        f"Unknown retriever backend: {backend!r}. "
        f"Expected one of: 'openai_hf', 'openclip'."
    )
