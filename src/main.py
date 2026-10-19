"""
3D scene change detection from two phone recordings of the same room.

    python src/main.py "videos/Video 1.mp4" "videos/Video 2.mp4" --out output

Pipeline: Extract keyframes -> Mapping (joint SfM + metric monocular depth ->
shared 3D map) -> Detect_Changes (free-space reasoning + clustering + DINOv2
appearance verification) ->
Semantics (CLIP labels, OWLv2 landmarks, moved-object matching, language) ->
Visualise (3D HTML, top-down map, evidence images, report).
Expensive stages (SfM, depth) are cached in the output folder.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parent))

import Detect_Changes
import Extract
import Mapping
import Semantics
import Visualise
from geometry import FrameSet


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs=2, type=Path, help="recording 1 (before), recording 2 (after)")
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--num-frames", type=int, default=60, help="keyframes per video")
    ap.add_argument("--depth-model", default=Mapping.DEPTH_MODEL)
    ap.add_argument("--no-landmarks", action="store_true", help="skip OWLv2 landmark detection")
    ap.add_argument("--rebuild", action="store_true", help="recompute cached frames/SfM/depth")
    args = ap.parse_args()

    out, images, map_dir = args.out, args.out / "images", args.out / "map"
    print("[Extract] keyframes")
    for prefix, video in zip(Mapping.SESSIONS, args.videos):
        folder = images / prefix
        if args.rebuild or not any(folder.glob("*.jpg")):
            if folder.exists():
                for f in folder.glob("*.jpg"):
                    f.unlink()
            paths = Extract.extract_keyframes(video, folder, prefix, args.num_frames)
            print(f"  {video}: {len(paths)} keyframes")
        else:
            print(f"  {folder}: using cached keyframes")

    if args.rebuild or not (map_dir / "cloud_B.ply").exists():
        frames, clouds = Mapping.build_map(images, map_dir, depth_model=args.depth_model,
                                           reuse=not args.rebuild)
    else:
        print("[Mapping] using cached shared map")
        frames = {s: FrameSet.load(map_dir / f"frames_{s}.pt") for s in Mapping.SESSIONS}
        clouds = {s: o3d.io.read_point_cloud(str(map_dir / f"cloud_{s}.ply"))
                  for s in Mapping.SESSIONS}

    print("[Detect_Changes] free-space change reasoning")
    removed, added, status = Detect_Changes.detect(frames, clouds)
    print("[Detect_Changes] DINOv2 appearance verification")
    verifier = Detect_Changes.AppearanceVerifier(images)
    removed = Detect_Changes.verify(removed, frames, verifier, status)
    added = Detect_Changes.verify(added, frames, verifier, status)

    print("[Semantics] CLIP labels / appearance embeddings")
    labeler = Semantics.ClipLabeler()
    info = Semantics.embed_clusters(removed + added, frames, images, labeler)
    embeddings = {cid: v["embedding"] for cid, v in info.items()}
    changes = Detect_Changes.match_moved(removed, added, embeddings)
    landmarks = []
    if not args.no_landmarks:
        print("[Semantics] OWLv2 room landmarks")
        landmarks = Semantics.detect_landmarks(frames["A"], images)
    Semantics.annotate(changes, info, landmarks)

    print("[Visualise] writing outputs")
    Visualise.scene_html(clouds, status, changes, frames, landmarks, out / "changes_3d.html")
    Visualise.topdown(clouds, status, changes, landmarks, out / "topdown.png")
    evidence = Visualise.evidence_images(changes, frames, images, out / "evidence")
    Visualise.export_ply(changes, out / "changes.ply")
    stats = {s: {"keyframes": len(frames[s]), "points": len(clouds[s].points),
                 "unobserved_by_other_session": round(float(np.mean(status[s] == 2)), 3)}
             for s in Mapping.SESSIONS}
    Visualise.write_report(changes, landmarks, evidence, out, stats)

    print(f"\n{len(changes)} change(s):")
    for ch in changes:
        print(f"  #{ch.id} [{ch.kind}] {ch.description}")
    print(f"\nSee {out / 'report.md'}, {out / 'changes_3d.html'}, {out / 'topdown.png'}")


if __name__ == "__main__":
    main()
