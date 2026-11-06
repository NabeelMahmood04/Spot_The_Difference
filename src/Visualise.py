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