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
from reasoner.claude_engine import ClaudeReasoner, ReasonerVerdict
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

    # --- Phase 3: Routing & Reasoning (WP5) ---
    # Phase 3a - Confidence-Gated Router (WP3): the "cheap" gate. Never
    # touches a paid API; inspects CLIP cosine scores and partitions
    # candidates by threshold bands.
    router = BudgetAwareRouter(tau_high=args.tau_high, tau_low=args.tau_low)
    accepted, ambiguous = router.route_request(ranked_candidates)

    print(f"[*] Router Outcome: {len(accepted)} High Match, {len(ambiguous)} Ambiguous")

    # High-confidence CLIP matches always pass through the cascade unchanged.
    final_results = list(accepted)
    if accepted:
        print(f"[*] SUCCESS: Found {len(accepted)} High-Confidence matches.")

    # Phase 3b - Reasoner Escalation (WP5): Claude Haiku 4.5 is invoked ONLY
    # for the AMBIGUOUS subset. Every frame discarded below tau_low and
    # every frame accepted above tau_high is a Reasoner call we did NOT make.
    if ambiguous:
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("[!] CRITICAL ERROR: ANTHROPIC_API_KEY not found in .env file.")
            print("[*] Academic Requirement: add ANTHROPIC_API_KEY=<key> to .env.")
            tracker.metrics.stop()
            return

        # Surgical escalation cap: bound API spend even if FAISS returns an
        # unusually large ambiguous band. Rank by CLIP score so the most
        # promising candidates are reasoned over first.
        MAX_ESCALATIONS = 2
        if len(ambiguous) > MAX_ESCALATIONS:
            ambiguous = sorted(ambiguous, key=lambda x: x['score'], reverse=True)[:MAX_ESCALATIONS]

        print(f"[*] INFO: Escalating {len(ambiguous)} candidates to Reasoner (Claude Haiku 4.5)...")

        try:
            reasoner = ClaudeReasoner()
        except ValueError as e:
            print(f"[!] Reasoner initialization failed: {e}")
            tracker.metrics.stop()
            return

        for item in tqdm(ambiguous, desc="Verifying"):
            try:
                verdict: ReasonerVerdict = reasoner.verify_event(item['frame'], args.query)
            except RuntimeError as e:
                # Retries exhausted or auth error - log and skip this
                # candidate rather than abort the whole investigation.
                print(f"\n[!] Reasoner failed on t={item['timestamp']:.2f}s: {e}")
                continue
            except Exception as e:
                print(f"\n[!] Unexpected Reasoner error on t={item['timestamp']:.2f}s: {e}")
                continue

            tracker.log_api_usage(verdict.raw_response)

            if verdict.is_verified:
                item['routing'] = "VERIFIED_BY_REASONER"
                item['reasoner_confidence'] = verdict.confidence_score
                item['reasoner_reasoning'] = verdict.reasoning
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
            if res['routing'] == "VERIFIED_BY_REASONER":
                # Surface Claude's forensic justification - the academic-defense
                # artifact proving the verdict came from grounded visual reasoning.
                print(f"          | Claude conf: {res['reasoner_confidence']:.2f} | "
                      f"{res['reasoner_reasoning']}")

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
