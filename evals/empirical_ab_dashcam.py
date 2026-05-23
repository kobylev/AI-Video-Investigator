"""Empirical V1 vs V2 A/B on the WP8 Dash Cam.mp4 corpus.

Bypasses the WP6 harness because src/eval/modes.py does not yet route the
injected retriever into FAISS (that wiring is deferred — see the open
follow-up in docs/V2_OPENCLIP_MIGRATION.md). This script instead:

  1. Loads all on-disk frames from data/frames/Dash Cam/
  2. Encodes them once with each engine (V1 = HF transformers + OpenAI CLIP;
     V2 = OpenCLIP / LAION-2B), at the architecture chosen via --arch.
  3. Builds an in-memory IndexFlatIP per engine.
  4. Runs the WP8-validated ground-truth query ("train crash car" ->
     frame_idx {418, 510, 511}) against each.
  5. Reports Recall@5, F1@5, retrieval latency.

Both arms use the same architecture so the only variable is the training
data (OpenAI 400M WIT vs. LAION-2B 2.32B). --arch b32 picks ViT-B-32
(512-dim, fast, fits CPU); --arch l14 picks ViT-L-14 at 224x224 (768-dim,
GPU-recommended).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Set, Tuple

# Make `src.*` importable when run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np
import torch
from PIL import Image

FRAMES_DIR = Path("data/frames/Dash Cam")
QUERY = "train crash car"
RELEVANT_FRAME_INDICES: Set[int] = {418, 510, 511}
TOP_K = 5

# Per-architecture model selection. Both arms use the same architecture so
# training data is the only variable in the A/B.
ARCH_CONFIG = {
    "b32": {
        "v1_model": "openai/clip-vit-base-patch32",
        "v2_model_name": "ViT-B-32",
        "v2_pretrained": "laion2b_s34b_b79k",
        "default_batch": 8,
        "embedding_dim": 512,
    },
    "l14": {
        "v1_model": "openai/clip-vit-large-patch14",
        "v2_model_name": "ViT-L-14",
        "v2_pretrained": "laion2b_s32b_b82k",
        "default_batch": 4,
        "embedding_dim": 768,
    },
}


def load_frame_paths() -> Tuple[List[Path], List[int]]:
    paths = sorted(
        FRAMES_DIR.glob("frame_*.jpg"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    indices = [int(p.stem.split("_")[1]) for p in paths]
    return paths, indices


def build_index(engine, frame_paths: List[Path], batch: int) -> Tuple[faiss.Index, float]:
    embs: List[np.ndarray] = []
    buf: List[Image.Image] = []
    t0 = time.perf_counter()
    n = len(frame_paths)
    for i, p in enumerate(frame_paths):
        buf.append(Image.open(p).convert("RGB"))
        if len(buf) == batch or i == n - 1:
            emb = engine.get_image_embeddings(buf).cpu().numpy().astype("float32")
            embs.append(emb)
            buf = []
        if (i + 1) % 200 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t0
            print(f"    encoded {i + 1:>4}/{n}  ({elapsed:.1f}s elapsed)")
    all_embs = np.concatenate(embs, axis=0)
    duration = time.perf_counter() - t0
    idx = faiss.IndexFlatIP(all_embs.shape[1])
    idx.add(all_embs)
    return idx, duration


def query_index(
    engine,
    idx: faiss.Index,
    frame_indices: List[int],
    query: str,
    relevant: Set[int],
    k: int = TOP_K,
) -> dict:
    t0 = time.perf_counter()
    q = engine.get_text_embeddings(query).cpu().numpy().astype("float32")
    scores, positions = idx.search(q, k)
    latency_ms = (time.perf_counter() - t0) * 1000
    retrieved = [frame_indices[p] for p in positions[0]]
    hits = [f for f in retrieved if f in relevant]
    recall = len(hits) / len(relevant) if relevant else 0.0
    precision = len(hits) / k if k else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "latency_ms": latency_ms,
        "retrieved": retrieved,
        "scores": [float(s) for s in scores[0]],
        "hits": hits,
        "recall_at_5": recall,
        "precision_at_5": precision,
        "f1_at_5": f1,
    }


def _free_gpu(model_obj) -> None:
    """Aggressively release VRAM before loading the next engine."""
    del model_obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=list(ARCH_CONFIG.keys()), default="b32",
                        help="Architecture for both A/B arms (default: b32).")
    parser.add_argument("--batch", type=int, default=None,
                        help="Encoding batch size; defaults per architecture.")
    parser.add_argument("--v2-pretrained", default=None,
                        help="Override the V2 (OpenCLIP) pretrained tag. "
                             "E.g., 'dfn2b_s39b' or 'datacomp_xl_s13b_b90k' "
                             "for ViT-L-14.")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write the result JSON. "
                             "Default: evals/results/v1_v2_dashcam_<arch>[_<v2tag>]/.")
    args = parser.parse_args(argv)

    cfg = dict(ARCH_CONFIG[args.arch])
    if args.v2_pretrained:
        cfg["v2_pretrained"] = args.v2_pretrained
    batch = args.batch or cfg["default_batch"]
    default_output = f"evals/results/v1_v2_dashcam_{args.arch}"
    if args.v2_pretrained:
        default_output += f"_{args.v2_pretrained}"
    output_dir = Path(args.output_dir or default_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Device : CUDA  ({gpu}, {vram_gb:.1f} GB VRAM)")
    else:
        print("Device : CPU")
    print(f"Arch   : {args.arch}  (batch={batch})")
    print(f"V1     : {cfg['v1_model']}")
    print(f"V2     : {cfg['v2_model_name']} / {cfg['v2_pretrained']}")
    print()

    paths, frame_indices = load_frame_paths()
    print(f"Corpus : {len(paths)} frames from {FRAMES_DIR}")
    print(f"Query  : {QUERY!r}")
    print(f"Ground-truth relevant frame_idx: {sorted(RELEVANT_FRAME_INDICES)}")
    print()

    results: dict = {
        "arch": args.arch,
        "device": device,
        "batch": batch,
        "query": QUERY,
        "relevant": sorted(RELEVANT_FRAME_INDICES),
        "corpus_size": len(paths),
    }

    # ---- V1: HF transformers + OpenAI CLIP ----
    print(f"[V1] Building OpenAI CLIP index ({cfg['v1_model']}) ...")
    from src.retriever.clip_engine import CLIPEngine
    t0 = time.perf_counter()
    v1 = CLIPEngine(model_id=cfg["v1_model"])
    print(f"    model loaded in {time.perf_counter() - t0:.1f}s")
    v1_idx, v1_build_s = build_index(v1, paths, batch=batch)
    print(f"    index built in {v1_build_s:.1f}s")
    v1_q = query_index(v1, v1_idx, frame_indices, QUERY, RELEVANT_FRAME_INDICES)
    print(f"    Recall@5={v1_q['recall_at_5']:.3f}  F1@5={v1_q['f1_at_5']:.3f}  "
          f"latency={v1_q['latency_ms']:.1f}ms")
    print(f"    top-5 retrieved frame_idx: {v1_q['retrieved']}")
    results["v1"] = {"backend": "openai_hf", "model": cfg["v1_model"],
                     "build_seconds": v1_build_s, **v1_q}
    _free_gpu(v1); _free_gpu(v1_idx)
    print()

    # ---- V2: OpenCLIP / LAION-2B ----
    print(f"[V2] Building OpenCLIP index "
          f"({cfg['v2_model_name']} / {cfg['v2_pretrained']}) ...")
    from src.retriever.openclip_engine import OpenCLIPEngine
    t0 = time.perf_counter()
    v2 = OpenCLIPEngine(model_name=cfg["v2_model_name"],
                        pretrained=cfg["v2_pretrained"])
    print(f"    model loaded in {time.perf_counter() - t0:.1f}s")
    v2_idx, v2_build_s = build_index(v2, paths, batch=batch)
    print(f"    index built in {v2_build_s:.1f}s")
    v2_q = query_index(v2, v2_idx, frame_indices, QUERY, RELEVANT_FRAME_INDICES)
    print(f"    Recall@5={v2_q['recall_at_5']:.3f}  F1@5={v2_q['f1_at_5']:.3f}  "
          f"latency={v2_q['latency_ms']:.1f}ms")
    print(f"    top-5 retrieved frame_idx: {v2_q['retrieved']}")
    results["v2"] = {"backend": "openclip",
                     "model": f"{cfg['v2_model_name']} / {cfg['v2_pretrained']}",
                     "build_seconds": v2_build_s, **v2_q}

    # ---- Comparison ----
    print()
    print("=" * 72)
    print(f"  {'Metric':<22}{'V1 (OpenAI)':>18}{'V2 (OpenCLIP)':>18}{'Delta':>12}")
    print("-" * 72)
    for k in ("recall_at_5", "f1_at_5", "precision_at_5", "latency_ms"):
        a, b = results["v1"][k], results["v2"][k]
        print(f"  {k:<22}{a:>18.4f}{b:>18.4f}{b - a:>+12.4f}")
    print("=" * 72)

    out = output_dir / "empirical_ab_result.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
