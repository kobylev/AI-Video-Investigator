import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
from typing import List, Union, Optional
import os

class CLIPEngine:
    """
    Core retrieval engine utilizing OpenAI's CLIP model.
    Encapsulates embedding generation for both images and text.
    """
    
    def __init__(self, model_id: str = "openai/clip-vit-large-patch14-336"):
        """
        Initialize the CLIP model and processor.
        Default is ViT-L/14@336px for high-fidelity retrieval as per project requirements.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.eval()

    def get_image_embeddings(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """Generates normalized embeddings for one or more images."""
        if isinstance(images, Image.Image):
            images = [images]
            
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            # Handle different return types across transformer versions
            if hasattr(outputs, 'pooler_output'):
                image_features = outputs.pooler_output
            elif isinstance(outputs, torch.Tensor):
                image_features = outputs
            else:
                image_features = outputs[0]
                
            image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features

    def get_text_embeddings(self, queries: Union[str, List[str]]) -> torch.Tensor:
        """Generates normalized embeddings for one or more text queries."""
        if isinstance(queries, str):
            queries = [queries]
            
        inputs = self.processor(text=queries, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            # Handle different return types across transformer versions
            if hasattr(outputs, 'pooler_output'):
                text_features = outputs.pooler_output
            elif isinstance(outputs, torch.Tensor):
                text_features = outputs
            else:
                text_features = outputs[0]
                
            text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features

    def compute_similarity(self, image: Image.Image, query: str) -> float:
        """Computes cosine similarity between a single image and a single query."""
        image_emb = self.get_image_embeddings(image)
        text_emb = self.get_text_embeddings(query)
        
        # Dot product of normalized vectors
        similarity = (image_emb @ text_emb.T).item()
        return similarity

    def get_frames_at_timestamps(self, video_path: str, timestamps: List[float]) -> List[Image.Image]:
        """
        Retrieves actual images for specific timestamps without re-processing the whole video.
        Used for escalating FAISS results to the Reasoner.
        """
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        
        for ts in timestamps:
            frame_idx = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
        return frames

if __name__ == "__main__":
    # Smoke test logic
    engine = CLIPEngine()
    print("CLIP Engine initialized successfully.")
