"""
Visualise detected changes.

* changes_3d.html - interactive Plotly 3D scene (shared map, change clusters,
                    3D boxes, move arrows, labels, camera paths, landmarks).
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

COLORS = {"removed": (220, 40, 40), "added": (30, 170, 60),
          "moved_before": (240, 140, 20), "moved_after": (40, 110, 230)}


def _rgb(c):
    return f"rgb({int(c[0])},{int(c[1])},{int(c[2])})"


def _cluster_color(change, idx):
    if change.kind == "moved":
        return COLORS["moved_before" if idx == 0 else "moved_after"]
    return COLORS[change.kind]


def _box_lines(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    corners = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        for k in (a, b):
            xs.append(corners[k, 0]); ys.append(corners[k, 1]); zs.append(corners[k, 2])
        xs.append(None); ys.append(None); zs.append(None)
    return xs, ys, zs


def _subsample(n, k, seed=0):
    if n <= k:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, k, replace=False)


def scene_html(clouds, status, changes, frames, landmarks, path: Path,
               max_points: int = 80_000):
    fig = go.Figure()
    P = {s: np.asarray(c.points) for s, c in clouds.items()}
    C = {s: np.asarray(c.colors) for s, c in clouds.items()}
    # shared map: static geometry of both sessions, true colour
    static = np.concatenate([P[s][status[s] == 0] for s in "AB"])
    static_c = np.concatenate([C[s][status[s] == 0] for s in "AB"])
    keep = _subsample(len(static), max_points)
    ceiling = np.percentile(static[:, 2], 97)
    keep = keep[static[keep, 2] < ceiling]      # drop ceiling so the room is visible
    fig.add_trace(go.Scatter3d(
        x=static[keep, 0], y=static[keep, 1], z=static[keep, 2], mode="markers",
        marker=dict(size=1.5, color=[_rgb((c * 255).astype(int)) for c in static_c[keep]]),
        name="shared map (static)", hoverinfo="skip"))
    for s, name in (("A", "recording 1 (before)"), ("B", "recording 2 (after)")):
        k = _subsample(len(P[s]), max_points // 2, seed=1)
        fig.add_trace(go.Scatter3d(
            x=P[s][k, 0], y=P[s][k, 1], z=P[s][k, 2], mode="markers", visible="legendonly",
            marker=dict(size=1.5, color=[_rgb((c * 255).astype(int)) for c in C[s][k]]),
            name=name, hoverinfo="skip"))
        cen = frames[s].centers.numpy()
        fig.add_trace(go.Scatter3d(x=cen[:, 0], y=cen[:, 1], z=cen[:, 2], mode="lines+markers",
                                   marker=dict(size=2), line=dict(width=3),
                                   name=f"camera path {s}"))

    for ch in changes:
        group = f"#{ch.id} {ch.kind}: {ch.label}"
        for i, c in enumerate(ch.clusters):
            col = _rgb(_cluster_color(ch, i))
            k = _subsample(len(c.points), 8000)
            fig.add_trace(go.Scatter3d(
                x=c.points[k, 0], y=c.points[k, 1], z=c.points[k, 2], mode="markers",
                marker=dict(size=2.5, color=col), name=group, legendgroup=group,
                showlegend=i == 0, hovertext=ch.description, hoverinfo="text"))
            lo, hi = c.bbox
            xs, ys, zs = _box_lines(lo, hi)
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                       line=dict(color=col, width=5), legendgroup=group,
                                       showlegend=False, hoverinfo="skip"))
        top = ch.clusters[-1]
        lo, hi = top.bbox
        tag = f"#{ch.id} {ch.label} ({ch.kind})"
        fig.add_trace(go.Scatter3d(x=[(lo[0] + hi[0]) / 2], y=[(lo[1] + hi[1]) / 2],
                                   z=[hi[2] + 0.15], mode="text", text=[tag],
                                   textfont=dict(size=13, color=_rgb(_cluster_color(ch, 1))),
                                   legendgroup=group, showlegend=False,
                                   hovertext=ch.description, hoverinfo="text"))
        if ch.kind == "moved":
            a, b = ch.before.centroid, ch.after.centroid
            fig.add_trace(go.Scatter3d(x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]],
                                       mode="lines", line=dict(color="black", width=6, dash="dash"),
                                       legendgroup=group, showlegend=False, hoverinfo="skip"))
            d = b - a
            fig.add_trace(go.Cone(x=[b[0]], y=[b[1]], z=[b[2]], u=[d[0]], v=[d[1]], w=[d[2]],
                                  sizemode="absolute", sizeref=0.15, anchor="tip",
                                  colorscale=[[0, "black"], [1, "black"]], showscale=False,
                                  legendgroup=group, showlegend=False, hoverinfo="skip"))
    if landmarks:
        L = np.stack([l["position"] for l in landmarks])
        fig.add_trace(go.Scatter3d(x=L[:, 0], y=L[:, 1], z=L[:, 2], mode="markers+text",
                                   text=[l["label"] for l in landmarks],
                                   marker=dict(size=4, color="gray", symbol="diamond"),
                                   textfont=dict(color="gray"), name="landmarks (OWLv2)"))
    fig.update_layout(scene=dict(aspectmode="data", xaxis_title="x (m)", yaxis_title="y (m)",
                                 zaxis_title="z / height (m)"),
                      title="3D scene changes between recording 1 and recording 2",
                      legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(str(path), include_plotlyjs="cdn")


def topdown(clouds, status, changes, landmarks, path: Path, res: float = 0.007):
    P = np.concatenate([np.asarray(c.points)[status[s] == 0] for s, c in clouds.items()])
    C = np.concatenate([np.asarray(c.colors)[status[s] == 0] for s, c in clouds.items()])
    m = P[:, 2] < np.percentile(P[:, 2], 90)         # look under the ceiling
    P, C = P[m], C[m]
    lo = np.percentile(P[:, :2], 0.5, axis=0) - 0.3
    hi = np.percentile(P[:, :2], 99.5, axis=0) + 0.3
    W, H = ((hi - lo) / res).astype(int) + 1
    img = np.full((H, W, 3), 255, np.uint8)
    order = np.argsort(P[:, 2])                       # higher points drawn last
    ij = ((P[order, :2] - lo) / res).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
    img[H - 1 - ij[ok, 1], ij[ok, 0]] = (C[order][ok] * 255).astype(np.uint8)[:, ::-1]
    img = (0.55 * img + 0.45 * 255).astype(np.uint8)  # fade background

    def px(xy):
        q = ((np.asarray(xy) - lo) / res).astype(int)
        return int(q[0]), int(H - 1 - q[1])

    for l in landmarks:
        cv2.putText(img, l["label"], px(l["position"][:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (90, 90, 90), 1, cv2.LINE_AA)
    for ch in changes:
        for i, c in enumerate(ch.clusters):
            col = _cluster_color(ch, i)[::-1]
            blo, bhi = c.bbox
            cv2.rectangle(img, px([blo[0], bhi[1]]), px([bhi[0], blo[1]]), col, 2)
            for p in c.points[_subsample(len(c.points), 3000)]:
                cv2.circle(img, px(p[:2]), 2, col, -1)
        if ch.kind == "moved":
            cv2.arrowedLine(img, px(ch.before.centroid[:2]), px(ch.after.centroid[:2]),
                            (0, 0, 0), 3, tipLength=0.05)
        x, y = px(ch.clusters[-1].bbox[1][:2])
        cv2.putText(img, f"#{ch.id} {ch.label}", (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    _cluster_color(ch, 1)[::-1], 2, cv2.LINE_AA)
    legend = [("removed", COLORS["removed"]), ("added", COLORS["added"]),
              ("moved: before", COLORS["moved_before"]), ("moved: after", COLORS["moved_after"])]
    strip = np.full((36, img.shape[1], 3), 255, np.uint8)
    x = 10
    for name, col in legend:
        cv2.rectangle(strip, (x, 10), (x + 16, 26), col[::-1], -1)
        cv2.putText(strip, name, (x + 22, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)
        x += 34 + cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    cv2.imwrite(str(path), np.vstack([strip, img]))


def _panel(image_root, view, title, color, height=420):
    if view is None:
        img = np.full((height, int(height * 0.56), 3), 235, np.uint8)
        cv2.putText(img, "no view", (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (120, 120, 120), 2)
    else:
        name, (x0, y0, x1, y1) = view
        img = cv2.imread(str(image_root / name))
        cv2.rectangle(img, (x0, y0), (x1, y1), color[::-1], 3)
        img = cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height))
        title = f"{title}  [{name}]"
    bar = np.full((28, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def evidence_images(changes, frames, image_root: Path, out_dir: Path):
    """Before/after keyframes of each changed location from both recordings."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ch in changes:
        panels = []
        if ch.kind == "removed":
            c = ch.before
            seen = (ch.views.get("A") or best_views(c.points, frames["A"], k=1) or [None])[0]
            empty = (best_views(c.points, frames["B"], k=1, mode="free") or [None])[0]
            panels = [_panel(image_root, seen, "recording 1: present", COLORS["removed"]),
                      _panel(image_root, empty, "recording 2: gone", COLORS["removed"])]
        elif ch.kind == "added":
            c = ch.after
            empty = (best_views(c.points, frames["A"], k=1, mode="free") or [None])[0]
            seen = (ch.views.get("B") or best_views(c.points, frames["B"], k=1) or [None])[0]
            panels = [_panel(image_root, empty, "recording 1: empty", COLORS["added"]),
                      _panel(image_root, seen, "recording 2: present", COLORS["added"])]
        else:
            b, a = ch.before, ch.after
            v1 = (ch.views.get("A") or best_views(b.points, frames["A"], k=1) or [None])[0]
            v2 = (best_views(b.points, frames["B"], k=1, mode="free") or [None])[0]
            v3 = (ch.views.get("B") or best_views(a.points, frames["B"], k=1) or [None])[0]
            panels = [_panel(image_root, v1, "rec 1: old position", COLORS["moved_before"]),
                      _panel(image_root, v2, "rec 2: old position empty", COLORS["moved_before"]),
                      _panel(image_root, v3, "rec 2: new position", COLORS["moved_after"])]
        img = np.hstack(panels)
        caption = np.full((60, img.shape[1], 3), 255, np.uint8)
        words, line, y = ch.description.split(), "", 20
        for w in words:            # simple word wrap
            if cv2.getTextSize(line + " " + w, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0] > img.shape[1] - 10:
                cv2.putText(caption, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
                line, y = w, y + 18
            else:
                line = (line + " " + w).strip()
        cv2.putText(caption, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        path = out_dir / f"change_{ch.id:02d}_{ch.kind}_{ch.label.replace(' ', '_')}.jpg"
        cv2.imwrite(str(path), np.vstack([img, caption]))
        paths[ch.id] = path
    return paths


def export_ply(changes, path: Path):
    pts, cols = [], []
    for ch in changes:
        for i, c in enumerate(ch.clusters):
            pts.append(c.points)
            cols.append(np.tile(np.array(_cluster_color(ch, i)) / 255, (len(c.points), 1)))
    pcd = o3d.geometry.PointCloud()
    if pts:
        pcd.points = o3d.utility.Vector3dVector(np.concatenate(pts))
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate(cols))
    o3d.io.write_point_cloud(str(path), pcd)


def write_report(changes, landmarks, evidence, out_dir: Path, stats: dict):
    data = {"stats": stats,
            "landmarks": [{"label": l["label"], "position": np.round(l["position"], 3).tolist(),
                           "support": l["support"]} for l in landmarks],
            "changes": [ch.summary() for ch in changes]}
    (out_dir / "changes.json").write_text(json.dumps(data, indent=2))
    lines = ["# Scene change report", "",
             f"Detected **{len(changes)}** change(s) between recording 1 and recording 2.", ""]
    for ch in changes:
        lines.append(f"## #{ch.id} - {ch.kind.upper()}: {ch.label}")
        lines.append("")
        lines.append(ch.description)
        lines.append("")
        alt = ", ".join(f"{k} {v:.0%}" for k, v in ch.label_scores.items())
        if alt:
            lines.append(f"- CLIP label scores: {alt}")
        for name, c in (("before", ch.before), ("after", ch.after)):
            if c is not None:
                lines.append(f"- {name}: centre {np.round(c.centroid, 2).tolist()} m, size "
                             f"{np.round(c.extent, 2).tolist()} m, free-space evidence {c.evidence:.0%}, "
                             f"DINOv2 before/after similarity {c.appearance_sim:.2f}")
        if ch.kind == "moved":
            lines.append(f"- displacement {np.round(ch.displacement, 2).tolist()} m "
                         f"({np.linalg.norm(ch.displacement):.2f} m), match score {ch.match_score}")
        if ch.id in evidence:
            lines.append(f"\n![change {ch.id}](evidence/{evidence[ch.id].name})")
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines))
