import faiss
import numpy as np
import pickle
import os
import torch
from typing import List, Tuple, Dict, Any

class VectorSearchIndex:
    """
    Handles persistence and fast similarity search using FAISS.
    Implements 'Index Once, Search Many' architecture.
    """
    
    def __init__(self, index_dir: str = "data/indices"):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        self.index = None
        self.metadata = [] # Stores (timestamp, frame_idx)

    def _get_paths(self, video_path: str) -> Tuple[str, str]:
        """Generates predictable paths for the index and metadata files."""
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        index_path = os.path.join(self.index_dir, f"{base_name}.faiss")
        meta_path = os.path.join(self.index_dir, f"{base_name}.pkl")
        return index_path, meta_path

    def exists(self, video_path: str) -> bool:
        """Checks if a pre-computed index exists for this video."""
        idx_p, meta_p = self._get_paths(video_path)
        return os.path.exists(idx_p) and os.path.exists(meta_p)

    def create(self, embeddings: torch.Tensor, timestamps: List[float]):
        """
        Initializes a new FAISS index from normalized embeddings.
        Uses Inner Product (IP) index which is equivalent to Cosine Similarity for normalized vectors.
        """
        # Convert to float32 numpy for FAISS compatibility
        embeddings_np = embeddings.cpu().numpy().astype('float32')
        dimension = embeddings_np.shape[1]
        
        # IndexFlatIP: Exact search using Inner Product
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings_np)
        self.metadata = timestamps

    def save(self, video_path: str):
        """Persists the FAISS index and metadata to disk."""
        idx_p, meta_p = self._get_paths(video_path)
        faiss.write_index(self.index, idx_p)
        with open(meta_p, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"[*] Index persisted to: {idx_p}")

    def load(self, video_path: str):
        """Loads a pre-computed index and metadata from disk."""
        idx_p, meta_p = self._get_paths(video_path)
        self.index = faiss.read_index(idx_p)
        with open(meta_p, 'rb') as f:
            self.metadata = pickle.load(f)
        print(f"[*] Loaded cached index: {idx_p} ({len(self.metadata)} frames)")

    def search(self, query_embedding: torch.Tensor, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Performs high-speed vector search.
        Returns top candidates with timestamps and confidence scores.
        """
        query_np = query_embedding.cpu().numpy().astype('float32')
        # FAISS search returns: (distances/scores, indices of matching vectors)
        scores, indices = self.index.search(query_np, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1: # FAISS returns -1 if fewer than top_k results exist
                results.append({
                    "timestamp": self.metadata[idx],
                    "score": float(score)
                })
        return results
