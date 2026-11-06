"""
Visualise detected changes.

* topdown.png     - floor-plan style map with the changes marked.
* evidence/*.jpg  - per change, before/after keyframes from both recordings
                    looking at the changed location.
* changes.ply     - colour-coded change points (open in Open3D/MeshLab/CloudCompare).
* report.md / changes.json - natural-language and structured summaries.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import plotly.graph_objects as go

from Semantics import best_views

COLOURS = {"removed": (220, 40, 40), "added": (30, 170, 60),
          "moved_before": (240, 140, 20), "moved_after": (40, 110, 230)}

def _rgb(c):
    return f"rgb({int(c[0])},{int(c[1])},{int(c[2])})"


def _cluster_colour(change, idx):
    if change.kind == "moved":
        return COLOURS["moved_before" if idx == 0 else "moved_after"]
    return COLOURS[change.kind]

def _box_lines(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    corners = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]

    xs, ys, zs = [], [], []
    for i, j in edges:
        for k in (i, j):
            xs.append(corners[k, 0])
            ys.append(corners[k, 1])
            zs.append(corners[k, 2])
        xs.append(None)
        ys.append(None)
        zs.append(None)
    return xs, ys, zs

def _subsample(n, k, seed=0):
    if n <= k:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, k, replace=False)

#function to produce topdown.png
def topdown(clouds, status, changes, landmarks, path : Path, res : float = 0.007):
    P = np.concatenate([np.asarray(c.points)[status[s] == 0] for s, c in clouds.items()])
    C = np.concatenate([np.asarray(c.colors)[status[s] == 0] for s, c in clouds.items()])
    m = P[:, 2] < np.percentile(P[:, 2], 90)
    P = P[m]
    C = C[m]
    lo = np.percentile(P[:, :2], 0.5, axis=0) - 0.3
    hi = np.percentile(P[:, :2], 99.5, axis=0) + 0.3
    W, H =((hi-lo)/res).astype(int) + 1
    img = np.full((H, W, 3), 255, np.uint8)
    ij = ((P[:, :2] - lo) / res).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
    img[H - 1 - ij[ok, 1], ij[ok, 0]] = (C[order][ok] * 255).astype(np.uint8)[:, ::-1]
    img = (0.55 * img + 0.45 * 255).astype(np.uint8)  # fade background

    def px(xy):
        q = ((np.asarray(xy) - lo) / res).astype(int)
        return int(q[0]), int(H - 1 - q[1])

    for l in landmarks:
        cv2.putText(img, l["label"], px(l["position"][:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90,90,90), 1, cv2.LINE_AA)

    for ch in changes:
        for i, c in enumerate(ch.clusters):
            col = _cluster_colour(ch, i)
            blo, bhi = c.bbox
            cv2.rectangle(img, px([blo[0], bhi[1]]), px([bhi[0], blo[1]]), col, 2)
            for p in c.points:
                cv2.circle(img, px(p[:2]), 2, col, -1)
        if ch.kind == "moved":
            cv2.arrowedLine(img, px(ch.before.centroid[:2]), px(ch.after.centroid[:2]),
                            (0, 0, 0), 3, tipLength=0.05)
        x, y = px(ch.clusters[-1].bbox[1][:2])
        cv2.putText(img, f"#{ch.id} {ch.label}", (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    _cluster_colour(ch, 1)[::-1], 2, cv2.LINE_AA)
    legend = [("removed", COLOURS["removed"]), ("added", COLOURS["added"]),
              ("moved: before", COLOURS["moved_before"]), ("moved: after", COLOURS["moved_after"])]
    strip = np.full((36, img.shape[1], 3), 255, np.uint8)
    x = 10
    for name, col in legend:
        cv2.rectangle(strip, (x, 10), (x + 16, 26), col[::-1], -1)
        cv2.putText(strip, name, (x + 22, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)
        x += 34 + cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    cv2.imwrite(str(path), np.vstack([strip, img]))

def _panel(image_root, view, title, colour, height=420):
    if view is None:
        img = np.full((height, int(height * 0.56), 3), 235, np.uint8)
        cv2.putText(img, "no view", (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)
    else:
        name, (x0, y0, x1, y1) = view
        img = cv2.imread(str(image_root / name))
        cv2.rectangle(img, (x0, y0), (x1, y1), colour[::-1], 3)
        img = cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height))
        title = f"{title} ({name})"
    bar = np.full((28, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])

