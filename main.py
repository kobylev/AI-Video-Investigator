import argparse
import sys
import os
import torch
from tqdm import tqdm

# Add src to path to ensure clean imports
sys.path.append(os.path.join(os.getcwd(), 'src'))

from retriever.video_processor import VideoProcessor
from retriever.clip_engine import CLIPEngine
from router.core import BudgetAwareRouter
from reasoner.gemini_engine import GeminiReasoner
from utils.metrics import PerformanceTracker

def main():
    parser = argparse.ArgumentParser(description="AI Video Investigator - WP3 SDK Entry Point")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file to analyze.")
    parser.add_argument("--query", type=str, required=True, help="Semantic event query (e.g., 'man jumps over a fence').")
    parser.add_argument("--tau_high", type=float, default=0.25, help="High confidence threshold (Match %).")
    parser.add_argument("--tau_low", type=float, default=0.15, help="Low confidence threshold (Match %).")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames to extract per second.")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found at {args.video}")
        return

    tracker = PerformanceTracker()
    tracker.metrics.start()

    # 1. RETRIEVAL PHASE (CLIP)
    print(f"[*] Initializing AI Video Investigator SDK...")
    print(f"[*] Target Video: {os.path.basename(args.video)}")
    print(f"[*] Query: '{args.query}'")
    
    engine = CLIPEngine()
    processor = VideoProcessor(sample_rate_fps=args.fps)
    
    print(f"[*] Extracting frames at {args.fps} FPS...")
    frames_data = []
    for img, ts in tqdm(processor.extract_frames(args.video), desc="Frame Extraction"):
        frames_data.append((img, ts))
    
    if not frames_data:
        print("[!] No frames extracted. Exiting.")
        return

    print(f"[*] Encoding {len(frames_data)} frames using CLIP ViT-L/14@336px...")
    images = [f[0] for f in frames_data]
    
    # Process in batches to prevent VRAM overflow
    batch_size = 32
    all_img_embs = []
    for i in tqdm(range(0, len(images), batch_size), desc="CLIP Encoding"):
        batch = images[i:i + batch_size]
        all_img_embs.append(engine.get_image_embeddings(batch))
    img_embs = torch.cat(all_img_embs, dim=0)
    
    text_emb = engine.get_text_embeddings(args.query)
    
    # Compute Softmax Probabilities (Match %)
    similarities = (img_embs @ text_emb.T).squeeze(1)
    logit_scale = engine.model.logit_scale.exp().item()
    probs = torch.nn.functional.softmax(similarities * logit_scale, dim=0)
    
    # Package frame results
    ranked_frames = []
    for i, (img, ts) in enumerate(frames_data):
        ranked_frames.append((img, ts, probs[i].item()))

    # 2. ROUTING PHASE (Budget-Aware Router)
    print(f"[*] Executing Confidence-Gated Routing (tau_high={args.tau_high}, tau_low={args.tau_low})...")
    router = BudgetAwareRouter(tau_high=args.tau_high, tau_low=args.tau_low)
    accepted, ambiguous = router.route_request(ranked_frames)
    
    print(f"[*] Router Outcome: {len(accepted)} High Match, {len(ambiguous)} Ambiguous, {len(frames_data)-len(accepted)-len(ambiguous)} Discarded")

    # 3. REASONING PHASE (Gemini 1.5 Pro)
    final_results = accepted
    if ambiguous:
        print(f"[*] Escalating {len(ambiguous)} ambiguous frames to Reasoner (Gemini 1.5 Pro)...")
        reasoner = GeminiReasoner()
        for item in tqdm(ambiguous, desc="Gemini Verification"):
            is_valid, response = reasoner.verify_event(item['frame'], args.query)
            tracker.log_api_usage(response)
            if is_valid:
                item['routing'] = "VERIFIED_BY_REASONER"
                final_results.append(item)
            else:
                # Optional: log rejected frames for audit
                pass

    # 4. OUTPUT & AUDIT
    tracker.metrics.stop()
    print("\n" + "="*50)
    print("            INVESTIGATION REPORT")
    print("="*50)
    
    if not final_results:
        print("[!] SEARCH COMPLETE: No definitive matches found.")
    else:
        # Sort by timestamp for chronological report
        for res in sorted(final_results, key=lambda x: x['timestamp']):
            minutes = int(res['timestamp'] // 60)
            seconds = int(res['timestamp'] % 60)
            print(f"[{minutes:02d}:{seconds:02d}] - Match Probability: {res['score']*100:.1f}% | Method: {res['routing']}")

    print("\n" + "="*50)
    print("            PERFORMANCE AUDIT")
    print("="*50)
    print(f"End-to-End Latency: {tracker.metrics.latency:.2f}s")
    print(f"Reasoner Invoked:   {tracker.metrics.reasoner_invoked}")
    print(f"Tokens Consumed:    {tracker.metrics.input_tokens} Input / {tracker.metrics.output_tokens} Output")
    print(f"Estimated Cost:     ${tracker.metrics.estimated_cost_usd:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
