"""Rebuild a FAISS index for a video using the V2 OpenCLIP engine.

V1 (OpenAI CLIP via transformers) and V2 (OpenCLIP / LAION-2B) produce
embeddings in different vector spaces, even when their dimensionality
matches. Querying a V2 text embedding against a V1 image index returns
nonsensical results, so existing V1-built indices MUST be rebuilt before
the server switches backends.

Usage (from repo root):

    # Rebuild the demo Dash Cam index in-place
    python scripts/rebuild_index_v2.py --video "data/uploaded/Dash Cam.mp4"

    # Build for whichever extracted-frames directory you choose
    python scripts/rebuild_index_v2.py --frames "data/frames/Dash Cam" \\
        --video-name "Dash Cam.mp4"

The script writes to data/indices/<basename>.faiss + .pkl, matching the
filename scheme used by VectorSearchIndex._get_paths().
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

# Make `src.*` importable when run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from src.retriever.factory import build_retriever
from src.retriever.search_index import VectorSearchIndex


def _load_frame_paths(frames_dir: Path) -> List[Path]:
    return sorted(
        frames_dir.glob("frame_*.jpg"),
        key=lambda p: int(p.stem.split("_")[1]),
    )


def _timestamp_from_filename(p: Path) -> float:
    # Filenames look like 'frame_<idx>_<seconds>.jpg' — extract <seconds>.
    return float(p.stem.split("_")[2])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True,
                        help="Path to the source video (used only for "
                             "deriving the index basename).")
    parser.add_argument("--frames", default=None,
                        help="Override the frames directory. "
                             "Default: data/frames/<video_stem>/")
    parser.add_argument("--indices-dir", default="data/indices",
                        help="Where to write the .faiss + .pkl files.")
    parser.add_argument("--batch", type=int, default=4,
                        help="Image encoding batch size (default: 4, fits 4 GB).")
    args = parser.parse_args(argv)

    video_path = Path(args.video)
    frames_dir = Path(args.frames) if args.frames else Path("data/frames") / video_path.stem
    if not frames_dir.is_dir():
        print(f"ERROR: frames directory not found: {frames_dir}", file=sys.stderr)
        return 2

    frame_paths = _load_frame_paths(frames_dir)
    if not frame_paths:
        print(f"ERROR: no frame_*.jpg files in {frames_dir}", file=sys.stderr)
        return 2
    timestamps = [_timestamp_from_filename(p) for p in frame_paths]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device       : {device}")
    print(f"Video        : {video_path}")
    print(f"Frames dir   : {frames_dir} ({len(frame_paths)} frames)")
    print(f"Indices dir  : {args.indices_dir}")
    print()

    print("Loading V2 retriever (build_retriever -> OpenCLIP) ...")
    t0 = time.perf_counter()
    engine = build_retriever(backend="openclip")
    print(f"  model loaded in {time.perf_counter() - t0:.1f}s "
          f"({getattr(engine, 'model_name', 'unknown')} / "
          f"{getattr(engine, 'pretrained', 'unknown')})")

    print()
    print(f"Encoding {len(frame_paths)} frames ...")
    t0 = time.perf_counter()
    buf: List[Image.Image] = []
    all_embs = []
    for i, p in enumerate(frame_paths):
        buf.append(Image.open(p).convert("RGB"))
        if len(buf) == args.batch or i == len(frame_paths) - 1:
            emb = engine.get_image_embeddings(buf)
            all_embs.append(emb)
            buf = []
        if (i + 1) % 200 == 0 or i == len(frame_paths) - 1:
            print(f"    encoded {i + 1:>4}/{len(frame_paths)}  "
                  f"({time.perf_counter() - t0:.1f}s elapsed)")
    stacked = torch.cat(all_embs, dim=0)
    encode_s = time.perf_counter() - t0
    print(f"  encoded in {encode_s:.1f}s "
          f"(emb shape = {tuple(stacked.shape)})")

    print()
    print("Persisting FAISS index ...")
    index = VectorSearchIndex(index_dir=args.indices_dir)
    index.create(stacked, timestamps)
    index.save(str(video_path))
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
