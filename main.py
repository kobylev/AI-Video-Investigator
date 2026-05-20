import argparse
import sys
import os
import torch
from tqdm import tqdm
from dotenv import load_dotenv

# 1. Environment Initialization
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from retriever.video_processor import VideoProcessor
from retriever.clip_engine import CLIPEngine
from retriever.search_index import VectorSearchIndex
from router.core import BudgetAwareRouter
from reasoner.claude_engine import ClaudeReasoner
from utils.metrics import PerformanceTracker

def main():
    parser = argparse.ArgumentParser(description="AI Video Investigator - WP4 Vector Search Entry Point")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file.")
    parser.add_argument("--query", type=str, required=True, help="Semantic query.")
    parser.add_argument("--tau_high", type=float, default=0.35, help="High match threshold.")
    parser.add_argument("--tau_low", type=float, default=0.25, help="Low match threshold.")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second for indexing.")
    parser.add_argument("--top_k", type=int, default=10, help="Top K results to fetch from FAISS.")
    args = parser.parse_args()

    tracker = PerformanceTracker()
    tracker.metrics.start()

    engine = CLIPEngine()
    index = VectorSearchIndex()

    # --- Phase 1: FAISS Index Management ---
    if index.exists(args.video):
        print(f"[*] Found cached FAISS index. Skipping extraction...")
        index.load(args.video)
    else:
        print(f"[*] No index found. Extracting and encoding video: {os.path.basename(args.video)}")
        processor = VideoProcessor(sample_rate_fps=args.fps)
        
        frames_data = []
        for img, ts in tqdm(processor.extract_frames(args.video), desc="Extracting Frames"):
            frames_data.append((img, ts))
        
        # Batch Encode
        images = [f[0] for f in frames_data]
        batch_size = 32
        all_embs = []
        for i in tqdm(range(0, len(images), batch_size), desc="CLIP Encoding"):
            batch = images[i:i + batch_size]
            all_embs.append(engine.get_image_embeddings(batch))
        
        img_embs = torch.cat(all_embs, dim=0)
        timestamps = [f[1] for f in frames_data]
        
        # Create and Save Index
        index.create(img_embs, timestamps)
        index.save(args.video)

    # --- Phase 2: Vector Search ---
    print(f"[*] Querying FAISS for: '{args.query}'")
    query_emb = engine.get_text_embeddings(args.query)
    search_results = index.search(query_emb, top_k=args.top_k)

    # Convert FAISS results to the format expected by the Router
    # (frame, timestamp, score)
    # We only fetch actual frames for the top candidates found by FAISS
    found_timestamps = [r['timestamp'] for r in search_results]
    found_frames = engine.get_frames_at_timestamps(args.video, found_timestamps)
    
    ranked_candidates = []
    best_candidate = {"score": -1.0, "timestamp": 0}
    
    # Safely zip results with frames to avoid IndexError if extraction fails
    for res, frame in zip(search_results, found_frames):
        score = res['score']
        ranked_candidates.append((frame, res['timestamp'], score))
        if score > best_candidate["score"]:
            best_candidate = {"score": score, "timestamp": res['timestamp']}
            
    # Also track the absolute best from FAISS even if frame extraction failed
    if not ranked_candidates and search_results:
        best_candidate = {"score": search_results[0]['score'], "timestamp": search_results[0]['timestamp']}

    # --- Phase 3: Routing & Reasoning ---
    router = BudgetAwareRouter(tau_high=args.tau_high, tau_low=args.tau_low)
    accepted, ambiguous = router.route_request(ranked_candidates)
    
    print(f"[*] Router Outcome: {len(accepted)} High Match, {len(ambiguous)} Ambiguous")

    final_results = []
    if accepted:
        print(f"[*] SUCCESS: Found {len(accepted)} High-Confidence matches.")
        final_results = accepted
    elif ambiguous:
        # Pre-flight API key check
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("[!] CRITICAL ERROR: ANTHROPIC_API_KEY not found in .env file.")
            print("[*] Academic Requirement: Please create a .env file with ANTHROPIC_API_KEY=your_key")
            tracker.metrics.stop()
            return

        # SURGICAL ESCALATION: Always take only the top 2 candidates to stay within Free Tier limits
        if len(ambiguous) > 2:
            ambiguous = sorted(ambiguous, key=lambda x: x['score'], reverse=True)[:2]
            
        print(f"[*] INFO: Escalating {len(ambiguous)} candidates to Reasoner (Claude Haiku 4.5)...")
        reasoner = ClaudeReasoner()
        for item in tqdm(ambiguous, desc="Verifying"):
            is_valid, response = reasoner.verify_event(item['frame'], args.query)
            tracker.log_api_usage(response)
            if is_valid:
                item['routing'] = "VERIFIED_BY_REASONER"
                final_results.append(item)

    # --- Phase 4: Report ---
    print(f"[*] Search process finished. Generating report...")
    tracker.metrics.stop()
    print("\n" + "="*50)
    print("            INVESTIGATION REPORT (WP4)")
    print("="*50)
    
    if not final_results:
        print("[!] SEARCH COMPLETE: No frames matched your thresholds.")
        if best_candidate["score"] > -1:
            bc_min = int(best_candidate['timestamp'] // 60)
            bc_sec = int(best_candidate['timestamp'] % 60)
            print(f"[*] Best candidate found at [{bc_min:02d}:{bc_sec:02d}] with score {best_candidate['score']:.4f}")
            print(f"[*] Try lowering --tau_low to {max(0.1, best_candidate['score'] - 0.02):.2f} to see this result.")
    else:
        print(f"[*] Found {len(final_results)} matching events:")
        for res in sorted(final_results, key=lambda x: x['timestamp']):
            m = int(res['timestamp'] // 60)
            s = int(res['timestamp'] % 60)
            print(f"[{m:02d}:{s:02d}] - Confidence: {res['score']:.4f} | Method: {res['routing']}")

    print("\n" + "="*50)
    print("            PERFORMANCE AUDIT")
    print("="*50)
    print(f"End-to-End Latency: {tracker.metrics.latency:.2f}s")
    print(f"Reasoner Invoked:   {getattr(tracker.metrics, 'reasoner_invoked', False)}")
    print(f"Estimated Cost:     ${tracker.metrics.estimated_cost_usd:.4f}")
    print("="*50)
    print("[*] SDK execution complete.")

if __name__ == "__main__":
    main()
