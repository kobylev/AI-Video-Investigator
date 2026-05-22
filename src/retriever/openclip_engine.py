"""OpenCLIP retrieval engine (V2.0).

Drop-in replacement for src.retriever.clip_engine.CLIPEngine. Public interface
is identical so that FAISS indexing, search, and the WP6 evaluation harness can
swap engines without modification.

Default checkpoint is ViT-H-14 / laion2b_s32b_b82k. The LAION-2B dataset is
~7x larger and substantially more diverse than OpenAI's original 400M-pair
WIT corpus, which is the empirical basis for the precision gains documented
in docs/V2_OPENCLIP_MIGRATION.md.
"""

from __future__ import annotations

from typing import List, Union

import torch
import torch.nn.functional as F
from PIL import Image

import open_clip


class OpenCLIPEngine:
    """OpenCLIP-backed retrieval engine with the same public interface as CLIPEngine."""

    def __init__(
        self,
        model_name: str = "ViT-H-14",
        pretrained: str = "laion2b_s32b_b82k",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.model_name = model_name
        self.pretrained = pretrained

    @torch.inference_mode()
    def get_image_embeddings(
        self, images: Union[Image.Image, List[Image.Image]]
    ) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = [images]
        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        feats = self.model.encode_image(batch)
        return F.normalize(feats, p=2, dim=-1)

    @torch.inference_mode()
    def get_text_embeddings(
        self, queries: Union[str, List[str]]
    ) -> torch.Tensor:
        if isinstance(queries, str):
            queries = [queries]
        tokens = self.tokenizer(queries).to(self.device)
        feats = self.model.encode_text(tokens)
        return F.normalize(feats, p=2, dim=-1)

    def compute_similarity(self, image: Image.Image, query: str) -> float:
        image_emb = self.get_image_embeddings(image)
        text_emb = self.get_text_embeddings(query)
        return (image_emb @ text_emb.T).item()

    def get_frames_at_timestamps(
        self, video_path: str, timestamps: List[float]
    ) -> List[Image.Image]:
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames: List[Image.Image] = []
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames


if __name__ == "__main__":
    engine = OpenCLIPEngine()
    print(f"OpenCLIP engine ready: {engine.model_name} / {engine.pretrained} on {engine.device}")
