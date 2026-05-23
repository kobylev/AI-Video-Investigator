"""Re-label evaluation queries using Claude Haiku 4.5 as an unbiased oracle.

Why this exists
---------------
The companion script ``label_with_v1_oracle.py`` uses V1 OpenAI CLIP as the
labeler, which biases the evaluation AGAINST V2 (V1 has built-in 100%
recall on its own labels). For an academically defensible "V2 vs V1"
comparison, an INDEPENDENT oracle is required.

Claude Haiku 4.5 is the natural choice: it is the project's Stage 2
reasoner, it has no dependence on either CLIP variant, and the existing
``ClaudeReasoner.verify_event()`` enforces a JSON tool-use contract that
yields structured ``{event_detected, confidence_score, reasoning}`` per
frame — exactly the shape needed for labeling.

Methodology
-----------
For each query in the JSONL whose ``label_source`` is one of:

  * ``"v1_oracle_pending"``      (never been labelled)
  * ``"v1_oracle_top3_above_*"`` (labelled by V1 oracle — re-label with Claude)
  * ``"v1_oracle_no_signal"``    (V1 found nothing — re-check with Claude)
  * ``"--force"`` overrides this and re-labels every query.

the script:

  1. Loads BOTH V1 (OpenAI CLIP) and V2 (OpenCLIP) engines + FAISS indices.
  2. Retrieves top-K (default 10) candidate frames from each engine.
  3. Takes the UNION of frame indices — eliminating per-engine candidate
     bias. Typically ~15-20 unique frames per query.
  4. For each unique candidate, loads the JPEG from data/frames/<stem>/
     and sends it to Claude Haiku 4.5 via ``ClaudeReasoner.verify_event``.
  5. Keeps frames where ``event_detected == True`` and
     ``confidence_score >= --min-confidence`` (default 0.7 — strict).
  6. Writes the new labels back into the JSONL with
     ``label_source = "claude_oracle_haiku45_v<MIN_CONF>"``.

Cost
----
  ~9 queries × ~15 unique candidates × ~$0.002 per call ≈ $0.27 total.
  Wall-time on Claude Haiku 4.5: ~6-8 minutes (~3s per single-frame call).

Usage
-----
    python evals/label_with_claude_oracle.py \\
        --queries-jsonl evals/v2_validation_queries.jsonl \\
        --top-k-per-engine 10 \\
        --min-confidence 0.7

Add ``--force`` to re-label queries that already have non-Claude labels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image

from src.reasoner.claude_engine import ClaudeReasoner
from src.retriever.clip_engine import CLIPEngine
from src.retriever.openclip_engine import OpenCLIPEngine
from src.retriever.search_index import VectorSearchIndex


V1_MODEL_ID    = "openai/clip-vit-large-patch14"
V2_MODEL_NAME  = "ViT-L-14"
V2_PRETRAINED  = "laion2b_s32b_b82k"


def load_frame_paths(frames_dir: Path) -> Tuple[List[Path], List[int]]:
    paths = sorted(
        frames_dir.glob("frame_*.jpg"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    indices = [int(p.stem.split("_")[1]) for p in paths]
    return paths, indices


def build_v1_index_in_memory(engine: CLIPEngine, frame_paths: List[Path], batch: int = 4) -> VectorSearchIndex:
    buf, embs = [], []
    for i, p in enumerate(frame_paths):
        buf.append(Image.open(p).convert("RGB"))
        if len(buf) == batch or i == len(frame_paths) - 1:
            embs.append(engine.get_image_embeddings(buf))
            buf = []
    stacked = torch.cat(embs, dim=0)
    timestamps = [float(p.stem.split("_")[2]) for p in frame_paths]
    idx = VectorSearchIndex(index_dir="data/indices")
    idx.create(stacked, timestamps)
    return idx


def top_k_frame_indices(
    engine,
    index: VectorSearchIndex,
    frame_idx_for_pos: List[int],
    query: str,
    top_k: int,
) -> List[int]:
    q_emb = engine.get_text_embeddings(query)
    q_np = q_emb.cpu().numpy().astype("float32")
    _scores, positions = index.index.search(q_np, top_k)
    return [int(frame_idx_for_pos[int(pos)]) for pos in positions[0] if pos != -1]


def frame_path_for_idx(frames_dir: Path, frame_idx: int) -> Path | None:
    matches = list(frames_dir.glob(f"frame_{frame_idx}_*.jpg"))
    return matches[0] if matches else None


def should_relabel(q: Dict[str, Any], force: bool) -> bool:
    if force:
        return True
    src = q.get("label_source", "")
    return (
        src.startswith("v1_oracle")
        or src == "v1_oracle_pending"
        or "claude_oracle" not in src
        and src != "human_wp8_demo"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-jsonl", default="evals/v2_validation_queries.jsonl")
    parser.add_argument("--video", default="data/uploaded/Dash Cam.mp4")
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument("--top-k-per-engine", type=int, default=10,
                        help="How many top candidates to pull from EACH engine "
                             "before unioning.")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                        help="Claude confidence threshold for inclusion in GT.")
    parser.add_argument("--force", action="store_true",
                        help="Re-label every query, including ones already "
                             "labelled by Claude or human.")
    parser.add_argument("--preserve-human", action="store_true", default=True,
                        help="Skip queries whose label_source contains 'human'. "
                             "Defaults to on.")
    args = parser.parse_args(argv)

    queries_path = Path(args.queries_jsonl)
    if not queries_path.exists():
        print(f"ERROR: queries file not found: {queries_path}", file=sys.stderr)
        return 2

    video = Path(args.video)
    frames_dir = Path(args.frames_dir) if args.frames_dir else Path("data/frames") / video.stem
    if not frames_dir.is_dir():
        print(f"ERROR: frames dir not found: {frames_dir}", file=sys.stderr)
        return 2

    # Load queries
    queries: List[Dict[str, Any]] = []
    with open(queries_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    # Filter to ones we want to relabel
    to_relabel = []
    for q in queries:
        if args.preserve_human and "human" in q.get("label_source", ""):
            continue
        if should_relabel(q, args.force):
            to_relabel.append(q)

    if not to_relabel:
        print("No queries to relabel. Use --force to override.")
        return 0

    print(f"Will relabel {len(to_relabel)}/{len(queries)} queries via Claude oracle.")
    print(f"  Min Claude confidence: {args.min_confidence}")
    print(f"  Top-K per engine     : {args.top_k_per_engine}")
    print()

    # --- Load V1 ---
    print(f"Loading V1 (OpenAI CLIP {V1_MODEL_ID}) ...")
    t0 = time.perf_counter()
    v1_engine = CLIPEngine(model_id=V1_MODEL_ID)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")
    paths, frame_idx_for_pos = load_frame_paths(frames_dir)
    print(f"Building V1 in-memory index from {len(paths)} frames ...")
    t0 = time.perf_counter()
    v1_index = build_v1_index_in_memory(v1_engine, paths)
    print(f"  encoded in {time.perf_counter() - t0:.1f}s")

    # --- Load V2 ---
    print(f"Loading V2 (OpenCLIP {V2_MODEL_NAME} / {V2_PRETRAINED}) ...")
    t0 = time.perf_counter()
    v2_engine = OpenCLIPEngine(model_name=V2_MODEL_NAME, pretrained=V2_PRETRAINED)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")
    v2_index = VectorSearchIndex(index_dir="data/indices")
    v2_index.load(str(video))

    # --- Load Claude ---
    print("Loading Claude Reasoner (Haiku 4.5) ...")
    reasoner = ClaudeReasoner()
    print()

    label_tag = f"claude_oracle_haiku45_minconf{args.min_confidence}"

    print(f"Labelling {len(to_relabel)} queries via Claude oracle ...")
    print(f"  {'query_id':<32} {'candidates':>10} {'verified':>9} {'wall (s)':>9}")
    print("-" * 65)

    total_calls = 0
    overall_t0 = time.perf_counter()

    for q in to_relabel:
        query_t0 = time.perf_counter()
        # Pull top-K from BOTH engines, take union — unbiased candidate pool.
        v1_top = top_k_frame_indices(v1_engine, v1_index, frame_idx_for_pos,
                                     q["query"], args.top_k_per_engine)
        v2_top = top_k_frame_indices(v2_engine, v2_index, frame_idx_for_pos,
                                     q["query"], args.top_k_per_engine)
        candidates = sorted(set(v1_top) | set(v2_top))

        verified: List[int] = []
        claude_scores: List[Dict[str, Any]] = []
        for fid in candidates:
            fpath = frame_path_for_idx(frames_dir, fid)
            if fpath is None:
                continue
            img = Image.open(fpath).convert("RGB")
            try:
                verdict = reasoner.verify_event(img, q["query"])
            except Exception as exc:
                print(f"    WARNING: Claude call failed for frame {fid}: {exc}")
                continue
            total_calls += 1
            claude_scores.append({
                "frame": fid,
                "event_detected": bool(verdict.event_detected),
                "confidence": round(float(verdict.confidence_score), 3),
            })
            if verdict.event_detected and verdict.confidence_score >= args.min_confidence:
                verified.append(fid)

        q["ground_truth_frame_indices"] = sorted(verified)
        q["label_source"] = label_tag if verified else "claude_oracle_no_signal"
        q["claude_oracle_audit"] = claude_scores

        wall = time.perf_counter() - query_t0
        print(f"  {q['query_id']:<32} {len(candidates):>10} {len(verified):>9} {wall:>9.1f}")

    print()
    print(f"Total Claude calls: {total_calls}")
    print(f"Total wall time   : {time.perf_counter() - overall_t0:.1f}s")

    # Write back to JSONL
    backup = queries_path.with_suffix(".jsonl.claude_bak")
    queries_path.replace(backup)
    print(f"  original JSONL backed up to {backup}")
    with open(queries_path, "w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"  wrote {len(queries)} queries to {queries_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
