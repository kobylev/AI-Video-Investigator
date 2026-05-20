import argparse
import sys
import os
import torch
from tqdm import tqdm
from dotenv import load_dotenv

# 1. Environment Initialization
load_dotenv() # Load .env before any SDK components are initialized

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
    parser.add_argument("--tau_high", type=float, default=0.80, help="High confidence threshold (Cosine Similarity).")
    parser.add_argument("--tau_low", type=float, default=0.65, help="Low confidence threshold (Cosine Similarity).")
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
    
    # Compute Raw Cosine Similarities (as requested for thresholds)
    similarities = (img_embs @ text_emb.T).squeeze(1)
    
    # Package frame results
    ranked_frames = []
    best_candidate = {"score": -1.0, "timestamp": 0}
    
    for i, (img, ts) in enumerate(frames_data):
        score = similarities[i].item()
        ranked_frames.append((img, ts, score))
        if score > best_candidate["score"]:
            best_candidate = {"score": score, "timestamp": ts}

    # 2. ROUTING PHASE (Budget-Aware Router)
    print(f"[*] Executing Confidence-Gated Routing (tau_high={args.tau_high}, tau_low={args.tau_low})...")
    router = BudgetAwareRouter(tau_high=args.tau_high, tau_low=args.tau_low)
    accepted, ambiguous = router.route_request(ranked_frames)
    
    print(f"[*] Router Outcome: {len(accepted)} High Match, {len(ambiguous)} Ambiguous, {len(frames_data)-len(accepted)-len(ambiguous)} Discarded")

    # 3. REASONING PHASE (Gemini 1.5 Pro) with Short-Circuit Logic
    final_results = []

    # SHORT-CIRCUIT: If we have High Matches, accept them and SKIP Gemini entirely.
    if accepted:
        print(f"[*] SUCCESS: Found {len(accepted)} High-Confidence matches. Short-circuiting to output.")
        final_results = accepted
    
    # ONLY invoke Reasoner if ZERO High Matches were found but we have Ambiguous candidates.
    elif ambiguous:
        # Pre-flight API key check
        if not os.getenv("GEMINI_API_KEY"):
            print("[!] CRITICAL ERROR: GEMINI_API_KEY not found in .env file.")
            print("[*] Academic Requirement: Please create a .env file with GEMINI_API_KEY=your_key")
            tracker.metrics.stop()
            return

        # SURGICAL ESCALATION: Always take only the top 2 candidates to stay within Free Tier limits
        if len(ambiguous) > 2:
            print(f"[*] INFO: {len(ambiguous)} frames are ambiguous. Capping at top 2 candidates for Reasoning.")
        
        # Sort by score descending and take top 2
        ambiguous = sorted(ambiguous, key=lambda x: x['score'], reverse=True)[:2]

        import time # Added for rate limiting
        print(f"[*] INFO: Escalating {len(ambiguous)} candidates to Reasoner (Gemini 1.5 Pro)...")
        reasoner = GeminiReasoner()
        for item in tqdm(ambiguous, desc="Gemini Verification"):
            is_valid, response = reasoner.verify_event(item['frame'], args.query)
            tracker.log_api_usage(response)
            if is_valid:
                item['routing'] = "VERIFIED_BY_REASONER"
                final_results.append(item)
            
            # Rate Limit Buffer for Free Tier (Safety check)
            time.sleep(3)

    else:
        print("[*] INFO: No candidates found in either High or Ambiguous bands.")

    # 4. OUTPUT & AUDIT
    tracker.metrics.stop()
    print("\n" + "="*50)
    print("            INVESTIGATION REPORT")
    print("="*50)
    
    if not final_results:
        print("[!] SEARCH COMPLETE: No definitive matches found.")
        bc_min = int(best_candidate['timestamp'] // 60)
        bc_sec = int(best_candidate['timestamp'] % 60)
        print(f"[*] Best candidate found at [{bc_min:02d}:{bc_sec:02d}] with score {best_candidate['score']:.4f}")
    else:
        # Sort by timestamp for chronological report
        for res in sorted(final_results, key=lambda x: x['timestamp']):
            minutes = int(res['timestamp'] // 60)
            seconds = int(res['timestamp'] % 60)
            print(f"[{minutes:02d}:{seconds:02d}] - Confidence: {res['score']:.4f} | Method: {res['routing']}")

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
