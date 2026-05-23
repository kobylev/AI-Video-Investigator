"""Seed ground-truth labels for queries using V1 OpenAI CLIP as a noisy oracle.

For each query in the JSONL whose `label_source` is `"v1_oracle_pending"`,
this script:

  1. Encodes the query with V1 (openai/clip-vit-large-patch14).
  2. Encodes all corpus frames in data/frames/<video_stem>/ with V1.
  3. Picks the top-K V1 candidates whose raw cosine is at or above
     `--min-score` (default 0.22).
  4. Writes the candidates' frame indices back into the JSONL under
     `ground_truth_frame_indices` and tags `label_source` as
     `"v1_oracle_top{K}_above_{min_score}"` so the bias is transparent.

Why V1 as oracle:

  - V1 is the BASELINE we want to beat. Using V1 to generate ground truth
    biases the eval AGAINST V2 (V1 has a built-in advantage of being the
    labeler). Any V2 competitive result under this disadvantage is
    therefore conservative evidence.

  - Alternative oracles (Claude per frame, human labels) are either
    expensive or unavailable. V1 is the cheapest defensible choice.

  - The methodology is explicitly recorded in the JSONL via `label_source`
    so reviewers see the bias.

Queries with no V1 candidate above `--min-score` get an empty GT list and
their `label_source` becomes `"v1_oracle_no_signal"` — used to test the
specificity of V2 (it should NOT confidently retrieve frames for nonsense
queries either).

Usage:

    python evals/label_with_v1_oracle.py \\
        --queries-jsonl evals/v2_validation_queries.jsonl \\
        --video "data/uploaded/Dash Cam.mp4" \\
        --top-k 3 --min-score 0.22
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image

from src.retriever.clip_engine import CLIPEngine
from src.retriever.search_index import VectorSearchIndex


def load_frame_paths(frames_dir: Path):
    paths = sorted(
        frames_dir.glob("frame_*.jpg"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    indices = [int(p.stem.split("_")[1]) for p in paths]
    return paths, indices


def build_v1_index(engine: CLIPEngine, frame_paths: List[Path], batch: int = 4):
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


def label_one(
    engine: CLIPEngine,
    index: VectorSearchIndex,
    frame_idx_for_pos: List[int],
    query: str,
    top_k: int,
    min_score: float,
) -> Dict[str, Any]:
    q_emb = engine.get_text_embeddings(query)
    q_np = q_emb.cpu().numpy().astype("float32")
    scores, positions = index.index.search(q_np, top_k)
    gt: List[int] = []
    suggested_scores: List[float] = []
    for s, pos in zip(scores[0], positions[0]):
        s = float(s)
        if pos == -1 or s < min_score:
            continue
        gt.append(int(frame_idx_for_pos[int(pos)]))
        suggested_scores.append(s)
    return {"frame_indices": gt, "scores": suggested_scores}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-jsonl", default="evals/v2_validation_queries.jsonl")
    parser.add_argument("--video", default="data/uploaded/Dash Cam.mp4")
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.22)
    parser.add_argument("--only-pending", action="store_true",
                        help="Skip queries whose label_source != 'v1_oracle_pending'. "
                             "Default is on; use --no-only-pending to re-label everything.")
    parser.add_argument("--no-only-pending", dest="only_pending", action="store_false")
    parser.set_defaults(only_pending=True)
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

    queries: List[Dict[str, Any]] = []
    with open(queries_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    pending = [q for q in queries if (not args.only_pending) or q.get("label_source") == "v1_oracle_pending"]
    print(f"Found {len(pending)}/{len(queries)} queries needing V1-oracle labels.")
    if not pending:
        print("Nothing to do.")
        return 0

    print(f"Loading V1 OpenAI CLIP (openai/clip-vit-large-patch14) ...")
    t0 = time.perf_counter()
    engine = CLIPEngine(model_id="openai/clip-vit-large-patch14")
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    print(f"Building V1 index from {frames_dir} ...")
    paths, frame_idx_for_pos = load_frame_paths(frames_dir)
    t0 = time.perf_counter()
    index = build_v1_index(engine, paths)
    print(f"  encoded {len(paths)} frames in {time.perf_counter() - t0:.1f}s")
    print()

    tag = f"v1_oracle_top{args.top_k}_above_{args.min_score:.2f}"
    no_signal_tag = "v1_oracle_no_signal"

    print(f"Labelling {len(pending)} pending queries ...")
    print(f"  {'query_id':<32} {'V1 top score':>14} {'#labels':>8}")
    print("-" * 60)
    for q in pending:
        out = label_one(engine, index, frame_idx_for_pos,
                        q["query"], args.top_k, args.min_score)
        if out["frame_indices"]:
            q["ground_truth_frame_indices"] = out["frame_indices"]
            q["label_source"] = tag
            q["v1_oracle_scores"] = [round(s, 4) for s in out["scores"]]
            top = out["scores"][0]
            print(f"  {q['query_id']:<32} {top:>14.4f} {len(out['frame_indices']):>8}")
        else:
            q["ground_truth_frame_indices"] = []
            q["label_source"] = no_signal_tag
            q["v1_oracle_scores"] = []
            print(f"  {q['query_id']:<32} {'< min':>14} {0:>8}  (no signal)")

    # Write back to JSONL in original order.
    backup = queries_path.with_suffix(".jsonl.bak")
    queries_path.replace(backup)
    print()
    print(f"  original JSONL backed up to {backup}")
    with open(queries_path, "w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"  wrote {len(queries)} labelled queries to {queries_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
