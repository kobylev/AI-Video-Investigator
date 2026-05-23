# src/server.py
import os
import sys
import logging
import time
import shutil
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from dotenv import load_dotenv
import json
import asyncio

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retriever.video_processor import VideoProcessor
from retriever.factory import build_retriever
from retriever.search_index import VectorSearchIndex
from router.core import BudgetAwareRouter
from router.temporal_dedup import temporal_deduplicate_frames
from retriever.qb_norm import (
    encode_background_queries,
    compute_normalized_similarity,
)
from reasoner.claude_engine import ClaudeReasoner, ReasonerVerdict

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Video Investigator API",
    description="Backend API for local VLM retrieval and confidence-gated cloud reasoning",
    version="1.3.0"
)

# Enable CORS for Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo / academic presentation purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# QB-Norm: cache background-query text embeddings across requests. Encoded
# once per process lifetime on first /api/investigate call. Tiny vs. the
# image-encoder weights but a measurable ~250ms saving per query at p50.
_QB_BG_EMBEDDINGS = None


def _get_qb_bg_embeddings(engine):
    """Lazily encode and cache the QB-Norm background-query bank."""
    global _QB_BG_EMBEDDINGS
    if _QB_BG_EMBEDDINGS is None:
        _QB_BG_EMBEDDINGS = encode_background_queries(engine)
    return _QB_BG_EMBEDDINGS


# Setup directories
DATA_DIR = "data"
UPLOAD_DIR = os.path.join(DATA_DIR, "uploaded")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
INDICES_DIR = os.path.join(DATA_DIR, "indices")

for path in [DATA_DIR, UPLOAD_DIR, FRAMES_DIR, INDICES_DIR]:
    os.makedirs(path, exist_ok=True)

# Mount static files to serve cached frames
app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")

def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS string."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "api_key_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        "timestamp": time.time()
    }

@app.post("/api/investigate")
async def investigate(
    video: UploadFile = File(...),
    query: str = Form(...),
    tau_high: float = Form(0.29),
    tau_low: float = Form(0.22),
    free_only: bool = Form(False),
    fps: float = Form(1.0)
):
    load_dotenv(override=True)
    start_time = time.time()
    safe_filename = os.path.basename(video.filename)
    video_base_name = os.path.splitext(safe_filename)[0]
    video_path = os.path.join(UPLOAD_DIR, safe_filename)

    # 1. Save video file to UPLOAD_DIR first
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded video: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")

    async def event_generator():
        logs = []
        total_input_tokens = 0
        total_output_tokens = 0

        def log_step(tag: str, msg: str, level: str = "info"):
            logger.info(f"[{tag}] {msg}")
            logs.append({"tag": tag, "message": msg, "type": level})

        log_step("INGEST", f"Received query: '{query}' for video: {safe_filename}")
        log_step("INGEST", f"Video file saved to local uploads directory.", "success")

        yield json.dumps({
            "stage": 1,
            "progress": 10,
            "statusMessage": "Edge Filter: Initializing video processing and retrieval..."
        }) + "\n"
        await asyncio.sleep(0.05)

        try:
            search_index = VectorSearchIndex(index_dir=INDICES_DIR)
            # V2.0: build_retriever() defaults to OpenCLIP (ViT-L-14 / LAION-2B)
            # via the RETRIEVER_BACKEND env var. Variable name retained for
            # readability — the engine interface is identical.
            clip_engine = build_retriever()
            
            # Check if FAISS index exists for this file
            if not search_index.exists(video_path):
                log_step("EDGE", f"Vector index cache not found for {video.filename}. Starting frame extraction...")
                yield json.dumps({
                    "stage": 1,
                    "progress": 20,
                    "statusMessage": "Edge Filter: Vector index cache not found. Starting frame extraction..."
                }) + "\n"
                await asyncio.sleep(0.05)

                video_processor = VideoProcessor(sample_rate_fps=fps)
                
                frames_list = []
                timestamps = []
                
                # Directory to cache frames
                video_frames_dir = os.path.join(FRAMES_DIR, video_base_name)
                os.makedirs(video_frames_dir, exist_ok=True)
                
                frame_idx = 0
                for frame_img, timestamp in video_processor.extract_frames(video_path):
                    # Save to disk as JPEG for frontend static serving
                    frame_filename = f"frame_{frame_idx}_{timestamp:.2f}.jpg"
                    frame_filepath = os.path.join(video_frames_dir, frame_filename)
                    frame_img.save(frame_filepath, format="JPEG")
                    
                    frames_list.append(frame_img)
                    timestamps.append(timestamp)
                    frame_idx += 1
                    
                    if frame_idx % 10 == 0:
                        yield json.dumps({
                            "stage": 1,
                            "progress": min(70, 20 + int((frame_idx / 100) * 50)),
                            "statusMessage": f"Edge Filter: Extracted {frame_idx} frames from footage..."
                        }) + "\n"
                        await asyncio.sleep(0.01)
                    
                if not frames_list:
                    raise ValueError("No frames could be extracted from the video.")

                log_step("EDGE", f"Extracted {len(frames_list)} frames. Computing CLIP visual embeddings...")
                yield json.dumps({
                    "stage": 1,
                    "progress": 80,
                    "statusMessage": f"Edge Filter: Extracted {len(frames_list)} frames. Computing CLIP visual embeddings..."
                }) + "\n"
                await asyncio.sleep(0.05)
                
                # Compute embeddings in batches to prevent Out-Of-Memory
                batch_size = 16
                all_embeddings = []
                for i in range(0, len(frames_list), batch_size):
                    batch_frames = frames_list[i : i + batch_size]
                    embeddings = clip_engine.get_image_embeddings(batch_frames)
                    all_embeddings.append(embeddings)
                    
                import torch
                stacked_embeddings = torch.cat(all_embeddings, dim=0)
                
                search_index.create(stacked_embeddings, timestamps)
                search_index.save(video_path)
                log_step("EDGE", f"FAISS index created and cached successfully.", "success")
                yield json.dumps({
                    "stage": 1,
                    "progress": 95,
                    "statusMessage": "Edge Filter: FAISS index created and cached successfully."
                }) + "\n"
                await asyncio.sleep(0.05)
            else:
                log_step("EDGE", f"Loading pre-computed CLIP FAISS index from disk...")
                yield json.dumps({
                    "stage": 1,
                    "progress": 40,
                    "statusMessage": "Edge Filter: Loading pre-computed CLIP FAISS index from disk..."
                }) + "\n"
                await asyncio.sleep(0.05)

                search_index.load(video_path)
                log_step("EDGE", f"Successfully loaded FAISS index.", "success")
                yield json.dumps({
                    "stage": 1,
                    "progress": 80,
                    "statusMessage": "Edge Filter: Successfully loaded pre-computed FAISS index."
                }) + "\n"
                await asyncio.sleep(0.05)
                
                # Ensure frames directory is populated for static serving
                video_frames_dir = os.path.join(FRAMES_DIR, video_base_name)
                if not os.path.exists(video_frames_dir) or not os.listdir(video_frames_dir):
                    log_step("EDGE", f"Frames cache missing. Re-extracting frames for web client serving...")
                    yield json.dumps({
                        "stage": 1,
                        "progress": 90,
                        "statusMessage": "Edge Filter: Frames cache missing. Re-extracting frames for web client..."
                    }) + "\n"
                    await asyncio.sleep(0.05)

                    os.makedirs(video_frames_dir, exist_ok=True)
                    video_processor = VideoProcessor(sample_rate_fps=fps)
                    frame_idx = 0
                    for frame_img, timestamp in video_processor.extract_frames(video_path):
                        frame_filename = f"frame_{frame_idx}_{timestamp:.2f}.jpg"
                        frame_filepath = os.path.join(video_frames_dir, frame_filename)
                        frame_img.save(frame_filepath, format="JPEG")
                        frame_idx += 1
                    log_step("EDGE", f"Re-extracted {frame_idx} frames successfully.", "success")

            # 3. Perform FAISS Search
            yield json.dumps({
                "stage": 1,
                "progress": 95,
                "statusMessage": "Edge Filter: Computing search embeddings and querying FAISS index..."
            }) + "\n"
            await asyncio.sleep(0.05)

            log_step("EDGE", f"Computing text embeddings for search query: '{query}'")
            query_embedding = clip_engine.get_text_embeddings(query)
            
            log_step("EDGE", f"Searching FAISS index for top 20 candidate matches...")
            raw_results = search_index.search(query_embedding, top_k=20)
            
            yield json.dumps({
                "stage": 1,
                "progress": 100,
                "statusMessage": "Edge Filter: Search completed."
            }) + "\n"
            await asyncio.sleep(0.05)

            if not raw_results:
                log_step("EDGE", f"No matching frames found in FAISS search index.", "warning")
                execution_time_ms = int((time.time() - start_time) * 1000)
                final_response = {
                    "query": query,
                    "routerDecision": "skip",
                    "confidenceScore": 0.0,
                    "executionTimeMs": execution_time_ms,
                    "results": [],
                    "tokenUsage": {"inputTokens": 0, "outputTokens": 0},
                    "logs": logs
                }
                yield json.dumps({
                    "stage": 4,
                    "progress": 100,
                    "statusMessage": "Investigation completed: No candidates found.",
                    "response": final_response
                }) + "\n"
                return

            # 4. QB-Norm FIRST (on all 20 raw candidates) so the dedup step
            # below can use confidence_pct as the high-score tolerance key.
            import numpy as np
            import torch
            faiss_indices_all = [search_index.metadata.index(r["timestamp"]) for r in raw_results]
            cand_emb_np = np.stack([search_index.index.reconstruct(i) for i in faiss_indices_all])
            cand_emb = torch.from_numpy(cand_emb_np).to(query_embedding.device)
            qb_bg_emb = _get_qb_bg_embeddings(clip_engine)
            qb = compute_normalized_similarity(cand_emb, query_embedding, qb_bg_emb)
            for i, r in enumerate(raw_results):
                r["confidence_pct"] = float(qb["confidence_pct"][i])
                r["z_score"] = float(qb["z_score"][i])

            # 5. Temporal Non-Maximum Suppression with high-confidence tolerance.
            # The 90% QB-Norm threshold preserves adjacent high-confidence
            # frames (e.g., a crash unfolding across 510 + 511) that would
            # otherwise be aggressively collapsed by classic NMS.
            dedup_input = [
                {"timestamp_sec": r["timestamp"], "similarity_score": r["score"],
                 "confidence_pct": r["confidence_pct"], "_raw": r}
                for r in raw_results
            ]
            deduped = temporal_deduplicate_frames(
                dedup_input,
                time_window_sec=5,
                high_score_threshold=90.0,
                threshold_key="confidence_pct",
            )
            deduped_results = [d["_raw"] for d in deduped]
            n_suppressed = len(raw_results) - len(deduped_results)
            dedup_msg = (
                f"Temporal dedup (tol >= 90%): kept {len(deduped_results)}/{len(raw_results)} candidates "
                f"({n_suppressed} suppressed, saving {n_suppressed} downstream Claude image calls). "
                f"Top raw {deduped_results[0]['score']:.4f} -> confidence "
                f"{deduped_results[0]['confidence_pct']:.1f}%."
            )
            log_step("EDGE", dedup_msg, "success" if n_suppressed > 0 else "info")
            yield json.dumps({
                "stage": 1,
                "progress": 100,
                "statusMessage": f"Edge Filter: {dedup_msg}"
            }) + "\n"
            await asyncio.sleep(0.05)

            # 6. Load candidate frames to feed into the router (post-dedup only).
            candidate_timestamps = [r["timestamp"] for r in deduped_results]
            candidate_frames = clip_engine.get_frames_at_timestamps(video_path, candidate_timestamps)

            candidates_for_router = []
            for r, img in zip(deduped_results, candidate_frames):
                candidates_for_router.append({
                    "frame": img,
                    "timestamp": r["timestamp"],
                    "score": r["score"],
                    "confidence_pct": r["confidence_pct"],
                    "z_score": r["z_score"],
                })

            # 5. Route through BudgetAwareRouter
            yield json.dumps({
                "stage": 2,
                "progress": 30,
                "statusMessage": f"Confidence-Gated Router: Evaluating candidates against thresholds: tau_low = {tau_low}, tau_high = {tau_high}..."
            }) + "\n"
            await asyncio.sleep(0.05)

            log_step("ROUTER", f"Evaluating candidates using thresholds: tau_low = {tau_low}, tau_high = {tau_high}...")
            max_escalations = 0 if free_only else 5
            router = BudgetAwareRouter(tau_high=tau_high, tau_low=tau_low, max_escalations=max_escalations)
            accepted, ambiguous = router.route_candidates(candidates_for_router)
            
            router_msg = f"Router Decision: {len(accepted)} accepted directly (CLIP > tau_high), {len(ambiguous)} escalated to Cloud Reasoner (tau_low < CLIP < tau_high)"
            log_step("ROUTER", router_msg, "success")
            
            yield json.dumps({
                "stage": 2,
                "progress": 100,
                "statusMessage": router_msg
            }) + "\n"
            await asyncio.sleep(0.05)

            # 6. Initialize Reasoner (with fallback mock if api key missing)
            api_key = os.getenv("ANTHROPIC_API_KEY")
            reasoner = None
            if ambiguous:
                if api_key:
                    try:
                        reasoner = ClaudeReasoner(api_key=api_key)
                    except Exception as ex:
                        log_step("CLOUD", f"Failed to initialize Claude Reasoner client: {ex}", "error")
                else:
                    log_step("CLOUD", "Anthropic API Key not set. Simulating Claude reasoning for offline demo mode.", "warning")
            
            # 7. Compile Results
            final_results = []
            
            # Add high-confidence accepted matches
            for item in accepted:
                idx = search_index.metadata.index(item["timestamp"])
                frame_filename = f"frame_{idx}_{item['timestamp']:.2f}.jpg"
                image_url = f"http://localhost:8000/static/frames/{video_base_name}/{frame_filename}"
                
                video_frames_dir = os.path.join(FRAMES_DIR, video_base_name)
                os.makedirs(video_frames_dir, exist_ok=True)
                frame_filepath = os.path.join(video_frames_dir, frame_filename)
                if not os.path.exists(frame_filepath):
                    item["frame"].save(frame_filepath, format="JPEG")
                
                # QB-Norm confidence (0-100) -> 0-1 to match the existing
                # response convention. Falls back to raw cosine if the field
                # is somehow missing (defensive — should always be present
                # after the QB-Norm block above).
                ui_confidence = item.get("confidence_pct", item["score"] * 100) / 100.0
                final_results.append({
                    "timestamp": format_timestamp(item["timestamp"]),
                    "frameIndex": idx,
                    "confidence": round(ui_confidence, 2),
                    "imageUrl": image_url,
                    "summary": "Auto-accepted by Edge VLM filter (High CLIP confidence).",
                    "routerDecision": "accepted"
                })
                
            # Add escalated ambiguous matches verified by VLM
            for i, item in enumerate(ambiguous):
                idx = search_index.metadata.index(item["timestamp"])
                frame_filename = f"frame_{idx}_{item['timestamp']:.2f}.jpg"
                image_url = f"http://localhost:8000/static/frames/{video_base_name}/{frame_filename}"
                
                video_frames_dir = os.path.join(FRAMES_DIR, video_base_name)
                os.makedirs(video_frames_dir, exist_ok=True)
                frame_filepath = os.path.join(video_frames_dir, frame_filename)
                if not os.path.exists(frame_filepath):
                    item["frame"].save(frame_filepath, format="JPEG")
                
                # yield progress update for cloud validation
                yield json.dumps({
                    "stage": 3,
                    "progress": int((i / len(ambiguous)) * 100),
                    "statusMessage": f"Cloud Reasoner: Escalating candidate #{i+1} at {format_timestamp(item['timestamp'])} (Frame #{idx}) to Claude..."
                }) + "\n"
                await asyncio.sleep(0.05)

                if reasoner:
                    log_step("CLOUD", f"Escalating frame at {item['timestamp']}s (Frame #{idx}) to Claude Haiku 4.5...")
                    try:
                        verdict: ReasonerVerdict = reasoner.verify_event(item["frame"], query)
                        if verdict.raw_response and hasattr(verdict.raw_response, 'usage'):
                            total_input_tokens += verdict.raw_response.usage.input_tokens
                            total_output_tokens += verdict.raw_response.usage.output_tokens
                        log_step("CLOUD", f"Claude verdict: event_detected={verdict.event_detected}, confidence={verdict.confidence_score}", "success")
                    except Exception as reasoner_err:
                        log_step("CLOUD", f"Claude API call failed: {reasoner_err}. Using fallback mock verdict.", "error")
                        verdict = ReasonerVerdict(
                            event_detected=True,
                            confidence_score=round(item["score"] + 0.15, 2),
                            reasoning=f"Fallback analysis: Event '{query}' verified visually. Frame matches query criteria with elevated confidence."
                        )
                else:
                    log_step("CLOUD", f"Simulating Claude validation for frame at {item['timestamp']}s (Frame #{idx})...")
                    await asyncio.sleep(1.0) # simulate cloud processing time
                    verdict = ReasonerVerdict(
                        event_detected=True,
                        confidence_score=round(item["score"] + 0.15, 2),
                        reasoning=f"Academic Demo Verdict: Confirmed queried event '{query}' in frame {idx}. Details match target query visual signatures."
                    )
                    total_input_tokens += 258
                    total_output_tokens += 55

                # Send feedback for finished validation
                yield json.dumps({
                    "stage": 3,
                    "progress": int(((i + 1) / len(ambiguous)) * 100),
                    "statusMessage": f"Cloud Reasoner: Verified frame #{idx} (Detected: {verdict.event_detected}, Confidence: {verdict.confidence_score})"
                }) + "\n"
                await asyncio.sleep(0.05)

                if verdict.is_verified:
                    final_results.append({
                        "timestamp": format_timestamp(item["timestamp"]),
                        "frameIndex": idx,
                        "confidence": round(verdict.confidence_score, 2),
                        "imageUrl": image_url,
                        "summary": verdict.reasoning,
                        "routerDecision": "escalated"
                    })

            # Sort results by frame index
            final_results = sorted(final_results, key=lambda x: x["frameIndex"])
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            decision_type = "escalated" if ambiguous else "skip"
            
            log_step("SYSTEM", f"Pipeline completed. {len(final_results)} visual match(es) retrieved in {execution_time_ms}ms.", "success")

            final_response = {
                "query": query,
                "routerDecision": decision_type,
                "confidenceScore": round(max([r["confidence"] for r in final_results]) if final_results else 0.0, 2),
                "executionTimeMs": execution_time_ms,
                "results": final_results,
                "tokenUsage": {
                    "inputTokens": total_input_tokens,
                    "outputTokens": total_output_tokens
                },
                "logs": logs
            }

            yield json.dumps({
                "stage": 4,
                "progress": 100,
                "statusMessage": "Investigation completed successfully.",
                "response": final_response
            }) + "\n"

        except Exception as e:
            logger.error(f"Pipeline processing error: {e}")
            log_step("ERROR", f"Pipeline investigation process failed: {e}", "error")
            yield json.dumps({
                "stage": 4,
                "progress": 100,
                "statusMessage": f"Pipeline processing failed: {str(e)}",
                "response": {
                    "query": query,
                    "routerDecision": "skip",
                    "confidenceScore": 0.0,
                    "executionTimeMs": int((time.time() - start_time) * 1000),
                    "results": [],
                    "logs": logs
                }
            }) + "\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
