"""
The idea here is to take 2 videos of the same room but slightly different
(see video_1 and video_2) and extract the frames from both videos to build
a 3D point cloud of the room based on these videos.

Frames are sampled uniformly in time; within each sampling window the sharpest
frame (highest variance of the Laplacian) is kept so that motion-blurred frames
from fast phone pans do not poison feature matching or depth estimation.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_keyframes(video_path: Path, out_dir: Path, prefix: str,
                      num_frames: int = 60) -> list[Path]:
    """Write ~num_frames sharp keyframes of `video_path` to `out_dir`.

    Returns the written paths in temporal order. File names are
    `<prefix>_<index>.jpg` so frames of both sessions can live in one
    folder for joint Structure-from-Motion.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window = max(1, total // num_frames)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    best, best_score, idx = None, -1.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        score = sharpness(frame)
        if score > best_score:
            best, best_score = frame, score
        idx += 1
        if idx % window == 0:
            path = out_dir / f"{prefix}_{len(written):04d}.jpg"
            cv2.imwrite(str(path), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written.append(path)
            best, best_score = None, -1.0
    if best is not None and best_score > 0:
        path = out_dir / f"{prefix}_{len(written):04d}.jpg"
        cv2.imwrite(str(path), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written.append(path)
    cap.release()
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs=2, type=Path)
    ap.add_argument("--out", type=Path, default=Path("output/images"))
    ap.add_argument("--num-frames", type=int, default=60)
    args = ap.parse_args()
    for prefix, video in zip(("A", "B"), args.videos):
        paths = extract_keyframes(video, args.out / prefix, prefix, args.num_frames)
        print(f"{video}: {len(paths)} keyframes -> {args.out / prefix}")


if __name__ == "__main__":
    main()
