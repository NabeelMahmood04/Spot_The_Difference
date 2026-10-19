# Humanoid-Spatial-AI-and-Navigation-Internship-Challenge-2027
Build a system that compares two phone recordings of the same small indoor area, such as a room, and identifies meaningful changes in 3D: moved furniture, new obstacles, or removed objects. 

## Approach

```
Video 1 ─┐                                       ┌─ Depth Anything V2 (PyTorch) ─┐
         ├─ Extract ─ joint SfM (COLMAP) ────────┤                               ├─ fused metric clouds A, B
Video 2 ─┘   sharp     one shared world frame    └─ SfM-anchored depth warping ──┘  (shared, Z-up, floor at 0)
             keyframes                                                                  │
                         free-space reasoning (PyTorch, both directions) ◄──────────────┘
                                   │  removed = A geometry that B saw through; added = vice versa
                         DBSCAN clusters ─► DINOv2 appearance verification ─► CLIP labels
                                   │                                              │
                         removed/added pairing (CLIP + colour + size + ICP) ─► moved
                                   │
                         OWLv2 landmarks ─► natural-language descriptions ─► 3D HTML, map, evidence, report
```

| Stage | File | What it does |
|---|---|---|
| Extract | [src/Extract.py](src/Extract.py) | Picks the sharpest frame (Laplacian variance) from each time window, so motion blur from phone pans is avoided. |
| Mapping | [src/Mapping.py](src/Mapping.py) | **Joint** Structure-from-Motion (pycolmap) over the keyframes of *both* videos, so every camera from both recordings sits in one coordinate frame. This shared frame is what makes the comparison independent of viewpoint. Each keyframe gets dense depth from **Depth Anything V2 Metric-Indoor-Large**. That depth is then warped onto the SfM points of its frame with a robust (Huber IRLS) fit of a smooth spatial scale field, because monocular depth bends walls differently in each frame. Next the world is gravity-aligned (RANSAC floor plane) and given metric scale. Finally each session is fused into a point cloud, keeping only points confirmed by several of that session's own views. |
| Change detection | [src/Detect_Changes.py](src/Detect_Changes.py), [src/geometry.py](src/geometry.py) | **Visibility reasoning, not cloud differencing.** A point of recording 1 counts as *removed* only if recording 2's cameras **looked through** that spot in several views *and* recording 2 has no geometry there. If recording 2 never saw the spot, or saw it hidden behind something else, it is *unobserved*, which is never a change. The reverse test gives *added*. Candidates are clustered with DBSCAN, then **verified by appearance**: DINOv2 patch features are sampled at exactly the pixels each 3D point projects to, before and after. Depth errors on blank walls, windows and mirrors produce geometric false alarms, but the image content there hasn't changed, so those clusters are rejected. Finally, removed and added clusters that look like the same object are paired into *moved*. |
| Semantics | [src/Semantics.py](src/Semantics.py) | Picks the best unoccluded views of each change and labels them zero-shot with **CLIP ViT-L/14** (prompt ensemble, ~75 household object classes). **OWLv2** detects room landmarks (bed, desk, door, …), which are lifted into 3D with the metric depth. The description then combines the label, the type of change, height and support ("on the floor"), the nearest landmarks, and the displacement. |
| Visualise | [src/Visualise.py](src/Visualise.py) | Produces the interactive 3D HTML, a top-down map, before/after evidence panels, a colour-coded PLY, and the Markdown/JSON report. |

Machine learning (all PyTorch): Depth Anything V2 (dense metric depth), DINOv2 (appearance verification of changes), CLIP (zero-shot labels + appearance embeddings for matching moved objects), OWLv2 (open-vocabulary landmark detection). The geometry itself is also done in batched PyTorch: projection, free-space tests and IRLS depth alignment.

## Run

CPU is enough (tested on 20 cores, no GPU).

```bash
python3 -m pip install --user virtualenv && python3 -m virtualenv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt

.venv/bin/python src/main.py "videos/Video 1.mp4" "videos/Video 2.mp4" --out output
```

The first run takes about 40 minutes. Almost all of that is Depth Anything V2 Large on 133 keyframes; SfM takes about 1 minute. SfM and depth are cached in `output/map/`, so later runs take about 2 minutes. Use `--depth-model depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` for a run roughly 15× faster at lower quality. Use `--rebuild` to recompute everything.

Each stage can also be run on its own: `src/Extract.py`, `src/Mapping.py`, `src/Detect_Changes.py`.

### Outputs (`output/`)

- `changes_3d.html`: interactive 3D view of the shared map with change clusters, 3D boxes, the move arrow, labels, camera paths for both recordings, and landmarks. The full cloud of each recording can be toggled from the legend.
- `topdown.png`: floor-plan view of the changes.
- `evidence/*.jpg`: for each change, keyframes from both recordings looking at the changed spot (present / empty / new position).
- `report.md`, `changes.json`: descriptions, labels, 3D boxes, displacement, and evidence scores.
- `changes.ply`, `map/cloud_A.ply`, `map/cloud_B.ply`, `map/sparse_sfm.ply`: point clouds for Open3D, MeshLab or CloudCompare.

## Results on the included videos

All 133 keyframes (70 + 63) registered into one SfM model, with 1.0 px mean reprojection error. The metric reconstruction gives a 4.0 × 3.6 m room with a ~2.8 m ceiling.

| # | Detected | Assessment against the footage |
|---|---|---|
| 0 | **Moved** backpack, 2.5 m: from the floor by the desk/radiator to hanging on the door hook next to the wardrobe | Correct |
| 1 | **Removed** dark object (bag?) under the desk | Plausible: the object under the desk appears pushed further back |
| 2 | **Removed** small white item hanging above the bedside table ("headphones?") | Correct change; the label is uncertain |
| 3 | **Removed** small item on the floor ("router?") | A pale bag/cloth on the floor shifted; the label is wrong |

The geometric stage produced 15 candidate clusters. Appearance verification rejected 10 of them, which had before/after similarity of 0.82–0.94: the wardrobe mirror, both window panes, blank wall patches and carpet. It kept 5, with similarities of 0.28–0.64. The backpack's removed and added clusters then paired into one move, giving the 4 changes above.

## Limitations

- **Metric scale** comes from a hand-held camera-height prior (`--camera-height`, default 1.45 m). The depth network's own absolute scale was about 2× off on these low-resolution portrait frames. The ceiling height it produces (~2.8 m) is an independent sanity check.
- **Small, flat objects** (e.g. a ~6 cm drink can lying on the carpet) are below the depth tolerance (~6 cm + 3% of range) and are missed.
- Monocular depth on **glass, mirrors and blank walls** is unreliable. Appearance verification removes these false alarms, but changes *on* such surfaces could be rejected too.
- CLIP labels for small or partly observed clusters are uncertain. Descriptions say "possibly …" when confidence is low.
- Moved-object pairing assumes the object is visible in both recordings. If something moved somewhere unobserved, it is reported as removed.
