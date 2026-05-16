# Data Directory

## BDD100K Dataset Acquisition & Management

This directory is the designated location for video data, frame extractions, and FAISS indices. **No data files are committed to Git** (see `.gitignore` rules).

---

## Dataset: BDD100K (Berkeley DeepDrive)

### Overview

**BDD100K** is a large-scale, diverse driving video dataset designed for autonomous driving research.

- **Source:** [BDD100K Official Website](https://www.bdd100k.com/)
- **Size:** 100,000 driving videos (40 seconds each, 720p resolution)
- **Total Duration:** ~1,100 hours
- **License:** Academic and non-commercial use permitted (requires registration)
- **Content:** Dashcam footage from diverse locations, weather conditions, times of day

### Relevance to This Project

BDD100K provides:
- **Domain-appropriate footage:** Real dashcam data (not web images or stock video)
- **Diverse conditions:** Urban, highway, residential; day/night; clear/rain/fog
- **Pre-existing annotations:** Object detection, lane markings (can be leveraged for validation)
- **Academic legitimacy:** Widely cited in CV research (500+ papers)

---

## Acquisition Instructions (WP3)

### Step 1: Register for Access

1. Visit [https://www.bdd100k.com/](https://www.bdd100k.com/)
2. Create an account and request dataset access
3. Accept the terms of use (academic use only)
4. Download the **BDD100K Videos** package (not just annotations)

**Expected Download Size:** ~1.8 TB for full dataset (or select subset)

### Step 2: Select Subset for Project

For this project, we need **≥10 hours of footage** (not the full 1,100 hours).

**Selection Criteria:**
- **Diversity:** Mix of urban, highway, residential scenes
- **Time-of-Day:** Daytime, nighttime, twilight
- **Weather:** Clear, rain, fog (if available)
- **Traffic Density:** Light, moderate, heavy
- **Total:** ~100 videos × 40 seconds = ~67 minutes per 100 videos → select ~900 videos for 10 hours

**Recommended Sampling:**
```python
# Example: stratified sampling by scene type
import random
urban_videos = random.sample(urban_video_list, 300)
highway_videos = random.sample(highway_video_list, 300)
residential_videos = random.sample(residential_video_list, 300)
total_subset = urban_videos + highway_videos + residential_videos  # 900 videos ≈ 10h
```

### Step 3: Download and Extract

Place downloaded videos in:
```
data/raw/bdd100k/videos/
```

**Note:** This directory is gitignored (`data/raw/` in `.gitignore`).

---

## Expected Directory Structure

After WP3 (Data Acquisition) and WP4 (Retriever Implementation), the structure will be:

```
data/
├── README.md                  # This file
├── raw/                       # Original videos (gitignored)
│   └── bdd100k/
│       ├── videos/            # 900 MP4 files (10+ hours)
│       └── metadata.json      # Video IDs, durations, scene types
│
├── processed/                 # Extracted frames (gitignored)
│   └── bdd100k/
│       ├── frames/            # One subdirectory per video
│       │   ├── video_0001/    # 40 frames (1 fps × 40 seconds)
│       │   ├── video_0002/
│       │   └── ...
│       └── frame_index.csv    # Frame ID → Video ID, Timestamp mapping
│
├── embeddings/                # CLIP embeddings (gitignored)
│   └── bdd100k/
│       ├── clip_embeddings.npy   # Numpy array (N × 768)
│       └── frame_ids.npy         # Frame ID mapping
│
└── indices/                   # FAISS indices (gitignored)
    └── bdd100k/
        ├── faiss_index.bin    # FAISS index file
        └── index_metadata.json   # Index config (K, metric, etc.)
```

**Total Disk Space Required:** ~150 GB (videos + frames + embeddings + index)

---

## Frame Extraction (WP4)

Videos will be sampled at **1 fps** (one frame per second) to balance granularity and storage.

**Extraction Command (ffmpeg):**
```bash
ffmpeg -i data/raw/bdd100k/videos/video_0001.mp4 \
       -vf "fps=1" \
       data/processed/bdd100k/frames/video_0001/frame_%04d.jpg
```

**Result:** 40-second video → 40 JPEG frames

**Naming Convention:** `frame_0001.jpg`, `frame_0002.jpg`, ..., `frame_0040.jpg`

---

## CLIP Encoding (WP4)

Each extracted frame is encoded with **CLIP ViT-L/14** to produce a 768-dimensional embedding.

**Pseudocode:**
```python
import open_clip
import torch
from PIL import Image

model, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
model.eval()

embeddings = []
for frame_path in frame_paths:
    image = preprocess(Image.open(frame_path)).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(image)
    embeddings.append(embedding.cpu().numpy())

# Save to disk
np.save("data/embeddings/bdd100k/clip_embeddings.npy", np.vstack(embeddings))
```

**Output:** Numpy array of shape `(N, 768)` where N = total frame count (~36,000 for 10h @ 1fps)

---

## FAISS Indexing (WP4)

Embeddings are indexed with **FAISS** for fast approximate nearest-neighbor search.

**Index Type:** IVF (Inverted File Index) or HNSW (Hierarchical Navigable Small World)

**Pseudocode:**
```python
import faiss

embeddings = np.load("data/embeddings/bdd100k/clip_embeddings.npy")
embeddings = embeddings.astype('float32')

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)

# Build IVF index (100 clusters, 768 dimensions)
quantizer = faiss.IndexFlatIP(768)  # Inner product (cosine after normalization)
index = faiss.IndexIVFFlat(quantizer, 768, 100)
index.train(embeddings)
index.add(embeddings)

# Save to disk
faiss.write_index(index, "data/indices/bdd100k/faiss_index.bin")
```

**Query Example:**
```python
index = faiss.read_index("data/indices/bdd100k/faiss_index.bin")
query_embedding = model.encode_text(clip.tokenize(["red car running red light"]))
distances, indices = index.search(query_embedding.cpu().numpy(), k=20)
```

---

## Data Governance

### Gitignore Rules

The following directories are **excluded from Git** (see `.gitignore`):
- `data/raw/` — Original videos (too large, licensing restrictions)
- `data/processed/` — Extracted frames (derived data, can be regenerated)
- `data/embeddings/` — CLIP embeddings (derived data)
- `data/indices/` — FAISS indices (derived data)

**Rationale:**
- Raw data is too large for Git (100+ GB)
- Derived data can be regenerated from code + raw data
- Academic datasets like BDD100K have licensing terms that may prohibit redistribution

### Reproducibility

To ensure reproducibility **without committing data**:
1. Document acquisition steps in this README (above)
2. Provide video IDs or sampling scripts in `data/bdd100k_video_ids.txt`
3. Commit code for frame extraction, encoding, and indexing (in `src/`)
4. Include checksums or metadata for validation (e.g., `data/metadata.json`)

**Example Metadata:**
```json
{
  "dataset": "BDD100K",
  "subset_version": "v1.0",
  "video_count": 900,
  "total_duration_hours": 10.0,
  "frame_count": 36000,
  "sampling_rate_fps": 1,
  "clip_model": "ViT-L-14",
  "faiss_index_type": "IVFFlat",
  "created_at": "2026-05-20"
}
```

---

## Fallback Datasets (Risk Mitigation)

If BDD100K access is delayed or unavailable (see [Risk Register](../docs/risk_register.md)):

1. **Waymo Open Dataset** — Alternative dashcam source, similar quality
2. **YouTube Dashcam Compilations** — Scrape with `youtube-dl`, check Creative Commons licenses
3. **CARLA Simulator** — Generate synthetic dashcam footage (last resort, less realistic)

---

**Maintained by:** Koby Lev | **Last Updated:** 2026-05-16
