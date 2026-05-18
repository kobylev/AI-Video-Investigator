import os
import sys
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from retriever.clip_engine import CLIPEngine
from retriever.video_processor import VideoProcessor

def main():
    parser = argparse.ArgumentParser(description="Video Search POC - Find events in video using CLIP.")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file.")
    parser.add_argument("--query", type=str, required=True, help="Text query to search for.")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames to extract per second (default: 1.0).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for CLIP processing.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to show.")
    
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found at {args.video}")
        return

    print(f"Initializing CLIP Engine...")
    engine = CLIPEngine()
    
    print(f"Initializing Video Processor (Sampling at {args.fps} FPS)...")
    vp = VideoProcessor(sample_rate_fps=args.fps)
    
    print(f"Extracting and processing frames from: {args.video}")
    frames = []
    timestamps = []
    
    # Extract frames
    for img, ts in tqdm(vp.extract_frames(args.video), desc="Extracting Frames"):
        frames.append(img)
        timestamps.append(ts)
    
    if not frames:
        print("Error: No frames were extracted from the video.")
        return

    print(f"Extracted {len(frames)} frames. Generating embeddings...")
    
    # Process in batches
    all_frame_embeddings = []
    for i in tqdm(range(0, len(frames), args.batch_size), desc="Encoding Frames"):
        batch = frames[i:i + args.batch_size]
        batch_embeddings = engine.get_image_embeddings(batch)
        all_frame_embeddings.append(batch_embeddings)
    
    all_frame_embeddings = torch.cat(all_frame_embeddings, dim=0)
    
    # Get text embedding
    print(f"Encoding query: '{args.query}'")
    query_embedding = engine.get_text_embeddings(args.query)
    
    # Compute similarities
    similarities = (all_frame_embeddings @ query_embedding.T).squeeze(1)
    
    # Calculate confidence using logit scale
    logit_scale = engine.model.logit_scale.exp().item()
    logits = similarities * logit_scale
    # Softmax over all frames to see which one stands out most
    probs = F.softmax(logits, dim=0)
    
    # Get top K
    top_values, top_indices = torch.topk(similarities, min(args.top_k, len(similarities)))
    
    print(f"\n--- Search Results for: '{args.query}' ---")
    print(f"{'Rank':<5} | {'Timestamp':<10} | {'Cosine Sim':<12} | {'Confidence':<12} | {'Match %':<8}")
    print("-" * 65)
    
    for rank, idx in enumerate(top_indices, 1):
        ts = timestamps[idx]
        sim = similarities[idx].item()
        prob = probs[idx].item()
        
        # Determine status based on thresholds
        if sim > 0.22:
            status = "High Match"
        elif sim > 0.18:
            status = "Moderate Match"
        else:
            status = "Low Match"
            
        # Format timestamp to MM:SS
        minutes = int(ts // 60)
        seconds = int(ts % 60)
        ts_str = f"{minutes:02d}:{seconds:02d}"
        
        print(f"{rank:<5} | {ts_str:<10} | {sim:.4f}      | {prob:.4f} ({status}) | {prob*100:>6.2f}%")

if __name__ == "__main__":
    main()
