"""Retrieval-engine factory (V2.0).

Selects the Stage 1 semantic-filter backend at runtime so that the WP6
evaluation harness, the FastAPI server, and the FAISS indexer can all swap
between the V1 (OpenAI CLIP via transformers) and V2 (OpenCLIP / LAION-2B)
engines without code edits.

Resolution order:
    1. Explicit `backend=` argument
    2. RETRIEVER_BACKEND environment variable
    3. Default: "openclip"

Production V2 model: ViT-L-14 / laion2b_s32b_b82k (768-dim, fits 4 GB VRAM).
The engine class itself defaults to ViT-H-14 for users with more VRAM, but
the factory pins L-14 so production deployments are predictable across
hardware tiers.

Relative imports are used so this module works from both `src.retriever.*`
(harness, evaluation scripts) and `retriever.*` (server.py sibling style).
"""

from __future__ import annotations

import os
from typing import Literal, Protocol


class RetrievalEngine(Protocol):
    def get_image_embeddings(self, images): ...
    def get_text_embeddings(self, queries): ...
    def compute_similarity(self, image, query) -> float: ...


# Production V2 checkpoint — pinned for predictability across hardware.
V2_MODEL_NAME = "ViT-L-14"
V2_PRETRAINED = "laion2b_s32b_b82k"


def build_retriever(
    backend: Literal["openai_hf", "openclip"] | None = None,
) -> RetrievalEngine:
    backend = backend or os.getenv("RETRIEVER_BACKEND", "openclip")
    if backend == "openclip":
        from .openclip_engine import OpenCLIPEngine
        return OpenCLIPEngine(model_name=V2_MODEL_NAME, pretrained=V2_PRETRAINED)
    if backend == "openai_hf":
        from .clip_engine import CLIPEngine
        return CLIPEngine()
    raise ValueError(
        f"Unknown retriever backend: {backend!r}. "
        f"Expected one of: 'openai_hf', 'openclip'."
    )
