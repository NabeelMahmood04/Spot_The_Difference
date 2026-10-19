"""
Detect geometric changes between the two sessions in the shared 3D frame.

Distinguishing real change from viewpoint/occlusion effects is done with
free-space (visibility) reasoning rather than plain point-cloud differencing:

* A point of session A is a "removed" candidate only if session B's cameras
  *looked through* that location (B's observed depth is clearly behind the
  point in several B views) **and** there is no B geometry nearby.
* If B never saw the location, or only saw it occluded by something in front,
  the point is "unobserved" - not a change. Missing data is never a change.
* Symmetrically, points of B that A's cameras saw through are "added".
* Geometric candidates are verified by appearance: DINOv2 (self-supervised ViT,
  PyTorch) patch features are sampled at the exact pixels each 3D point
  projects to - in views of its own session where it is visible, and in views of
  the other session that look through it. Monocular depth fails on blank walls,
  windows and mirrors, but there the image content is unchanged (wall vs wall,
  high feature similarity); a real change looks different (bag vs carpet).

Candidates are clustered (DBSCAN) into object-level changes, filtered by size
and evidence, and removed/added pairs that look like the same object (CLIP
appearance embedding, colour, size, and ICP shape agreement) are merged into
"moved" changes with an estimated displacement and rotation.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from geometry import FrameSet, project, visibility_stats

DINO_MODEL = "facebook/dinov2-small"


@dataclass
class Cluster:
    id: int
    kind: str                       # "removed" | "added" (geometry present in A | B)
    session: str                    # session whose cloud holds the geometry
    points: np.ndarray = field(repr=False)
    colors: np.ndarray = field(repr=False)
    indices: np.ndarray = field(repr=False)   # into the session's fused cloud
    evidence: float = 0.0           # mean fraction of other-session views seeing free space
    n_free_views: float = 0.0
    appearance_sim: float | None = None   # DINOv2 before/after similarity (low = changed)

    @property
    def centroid(self):
        return self.points.mean(0)

    @property
    def bbox(self):
        lo, hi = np.percentile(self.points, [2, 98], axis=0)
        return lo, hi

    @property
    def extent(self):
        lo, hi = self.bbox
        return hi - lo


@dataclass
class Change:
    id: int
    kind: str                       # "removed" | "added" | "moved"
    clusters: list[Cluster] = field(repr=False)   # [geometry] or [before(A), after(B)]
    displacement: np.ndarray | None = None
    rotation_deg: float | None = None
    match_score: float | None = None
    label: str = "object"
    label_scores: dict = field(default_factory=dict)
    description: str = ""
    views: dict = field(default_factory=dict)    # session -> list of (image name, crop box)

    @property
    def before(self):
        return self.clusters[0] if self.kind != "added" else None

    @property
    def after(self):
        return self.clusters[-1] if self.kind != "removed" else None

    def summary(self) -> dict:
        def box(c):
            if c is None:
                return None
            lo, hi = c.bbox
            return {"min": lo.round(3).tolist(), "max": hi.round(3).tolist(),
                    "centroid": c.centroid.round(3).tolist(), "n_points": len(c.points),
                    "free_space_evidence": round(c.evidence, 3),
                    "appearance_similarity": None if c.appearance_sim is None
                    else round(c.appearance_sim, 3)}
        return {"id": self.id, "kind": self.kind, "label": self.label,
                "label_scores": {k: round(v, 3) for k, v in self.label_scores.items()},
                "description": self.description,
                "before": box(self.before), "after": box(self.after),
                "displacement_m": None if self.displacement is None
                else self.displacement.round(3).tolist(),
                "distance_m": None if self.displacement is None
                else round(float(np.linalg.norm(self.displacement)), 3),
                "rotation_deg": self.rotation_deg, "match_score": self.match_score}


# ------------------------------------------------------------ point evidence
def change_evidence(points: np.ndarray, other_frames: FrameSet, other_points: np.ndarray,
                    tau_abs: float, tau_rel: float):
    """Per-point free-space evidence of `points` against the other session."""
    stats = visibility_stats(torch.from_numpy(points).float(), other_frames,
                             tau_abs=tau_abs, tau_rel=tau_rel)
    free = stats["free"].numpy()
    cons = stats["consistent"].numpy()
    nn_dist, _ = cKDTree(other_points).query(points, k=1, workers=-1)
    return free, cons, nn_dist


def candidate_mask(free, cons, nn_dist, min_free=3, min_ratio=0.7, nn_radius=0.08):
    observed = free + cons
    ratio = free / np.maximum(observed, 1)
    changed = (free >= min_free) & (ratio >= min_ratio) & (nn_dist > nn_radius)
    unobserved = observed < min_free
    return changed, unobserved, ratio


def cluster_candidates(points, colors, mask, ratio, free, kind, session,
                       eps=0.05, min_samples=8, min_points=40, min_extent=0.08,
                       max_extent=2.5, min_height=0.03):
    sel = np.where(mask & (points[:, 2] > min_height))[0]
    if len(sel) < min_points:
        return []
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(points[sel])
    clusters = []
    for lab in set(labels) - {-1}:
        idx = sel[labels == lab]
        if len(idx) < min_points:
            continue
        c = Cluster(0, kind, session, points[idx], colors[idx], idx,
                    float(ratio[idx].mean()), float(free[idx].mean()))
        ext = c.extent
        if ext.max() < min_extent or ext.max() > max_extent:
            continue
        clusters.append(c)
    return clusters


def detect(frames: dict[str, FrameSet], clouds: dict[str, o3d.geometry.PointCloud],
           tau_abs: float = 0.06, tau_rel: float = 0.03, **cluster_kw):
    """Return (removed clusters, added clusters, per-point status for both clouds)."""
    P = {s: np.asarray(c.points) for s, c in clouds.items()}
    C = {s: np.asarray(c.colors) for s, c in clouds.items()}
    out, status = {}, {}
    for src, other, kind in (("A", "B", "removed"), ("B", "A", "added")):
        free, cons, nn = change_evidence(P[src], frames[other], P[other], tau_abs, tau_rel)
        changed, unobserved, ratio = candidate_mask(free, cons, nn)
        out[kind] = cluster_candidates(P[src], C[src], changed, ratio, free, kind, src,
                                       **cluster_kw)
        st = np.zeros(len(P[src]), dtype=np.int8)        # 0 static, 1 changed, 2 unobserved
        st[unobserved] = 2
        for c in out[kind]:
            st[c.indices] = 1
        status[src] = st
        print(f"  {kind:8s}: {changed.sum():7d} candidate points, "
              f"{unobserved.mean() * 100:4.1f}% of session {src} unobserved by {other}, "
              f"{len(out[kind])} clusters")
    for i, c in enumerate(out["removed"] + out["added"]):
        c.id = i
    return out["removed"], out["added"], status


# ------------------------------------------------------ appearance check
class AppearanceVerifier:
    """Per-point before/after appearance comparison with DINOv2 patch features."""

    MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __init__(self, image_root: Path, model_name: str = DINO_MODEL, patch: int = 14):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name).eval()
        self.root, self.patch, self.cache = image_root, patch, {}

    @torch.no_grad()
    def features(self, name: str) -> torch.Tensor:
        """Unit-norm dense feature map [C, h, w] of a keyframe (cached)."""
        if name not in self.cache:
            rgb = cv2.cvtColor(cv2.imread(str(self.root / name)), cv2.COLOR_BGR2RGB)
            H, W = rgb.shape[:2]
            h, w = (H // self.patch) * self.patch, (W // self.patch) * self.patch
            x = torch.from_numpy(cv2.resize(rgb, (w, h))).permute(2, 0, 1).float() / 255
            x = (x - self.MEAN) / self.STD
            tokens = self.model(pixel_values=x[None]).last_hidden_state[0, 1:]
            fmap = tokens.reshape(h // self.patch, w // self.patch, -1).permute(2, 0, 1)
            self.cache[name] = F.normalize(fmap, dim=0)
        return self.cache[name]

    def sample(self, points: np.ndarray, frames: FrameSet, mode: str, max_frames: int = 6):
        """Mean feature per point over the frames where it is visible / seen through."""
        P = torch.from_numpy(points).float()
        uv, z = project(P, frames)
        Fn, H, W = frames.depth.shape
        u, v = uv[..., 0].long(), uv[..., 1].long()
        inside = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        d = frames.depth[torch.arange(Fn)[:, None], v.clamp(0, H - 1), u.clamp(0, W - 1)]
        tau = 0.08 + 0.05 * z
        ok = inside & (((d - z).abs() < tau) if mode == "visible" else (d > z + tau))
        feats, count = None, torch.zeros(len(P))
        for f in ok.sum(1).argsort(descending=True)[:max_frames]:
            m = ok[f]
            if m.sum() < 10:
                continue
            fmap = self.features(frames.names[f])
            grid = torch.stack([uv[f, m, 0] / W * 2 - 1, uv[f, m, 1] / H * 2 - 1], -1)
            s = F.grid_sample(fmap[None], grid[None, None], align_corners=False)[0, :, 0].T
            if feats is None:
                feats = torch.zeros(len(P), s.shape[1])
            feats[m] += s
            count[m] += 1
        if feats is None:
            return None, count > 0
        return F.normalize(feats, dim=1), count > 0

    def similarity(self, cluster: Cluster, frames: dict[str, FrameSet]) -> float | None:
        other = "B" if cluster.session == "A" else "A"
        fa, ma = self.sample(cluster.points, frames[cluster.session], "visible")
        fb, mb = self.sample(cluster.points, frames[other], "free")
        both = ma & mb
        if fa is None or fb is None or both.sum() < 20:
            return None
        return float((fa[both] * fb[both]).sum(1).median())


def verify(clusters: list[Cluster], frames: dict[str, FrameSet], verifier: AppearanceVerifier,
           status: dict[str, np.ndarray] | None = None,
           max_similarity: float = 0.75) -> list[Cluster]:
    """Keep clusters whose location *looks* different between the recordings.

    If `status` is given, points of rejected clusters are reset to static.
    """
    kept = []
    for c in clusters:
        c.appearance_sim = verifier.similarity(c, frames)
        ok = c.appearance_sim is not None and c.appearance_sim < max_similarity
        sim = "n/a" if c.appearance_sim is None else f"{c.appearance_sim:.2f}"
        print(f"  cluster {c.id:2d} {c.kind:7s} at {np.round(c.centroid, 2)}: "
              f"appearance similarity {sim} -> {'change' if ok else 'rejected'}")
        if ok:
            kept.append(c)
        elif status is not None:
            status[c.session][c.indices] = 0
    return kept


# --------------------------------------------------------------- moved pairs
def _color_hist(colors: np.ndarray, bins: int = 8) -> np.ndarray:
    h, _ = np.histogramdd(colors, bins=bins, range=[(0, 1)] * 3)
    h = h.ravel() + 1e-6
    return h / h.sum()


def _icp_fitness(src: Cluster, dst: Cluster, voxel: float = 0.02):
    """Best ICP alignment of src onto dst over a few yaw seeds (objects may rotate)."""
    a = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src.points - src.centroid))
    b = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst.points - dst.centroid))
    best = (0.0, 0.0, np.eye(4))
    for yaw in np.deg2rad(np.arange(0, 360, 45)):
        T0 = np.eye(4)
        T0[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        r = o3d.pipelines.registration.registration_icp(
            a, b, 2.5 * voxel, T0,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        if r.fitness > best[0]:
            T = r.transformation
            best = (r.fitness, float(np.degrees(np.arctan2(T[1, 0], T[0, 0]))), T)
    return best


def match_moved(removed: list[Cluster], added: list[Cluster],
                embeddings: dict[int, np.ndarray] | None = None,
                min_score: float = 0.55, min_distance: float = 0.15) -> list[Change]:
    """Pair removed/added clusters that are the same object -> "moved" changes."""
    changes: list[Change] = []
    used_r, used_a = set(), set()
    if removed and added:
        S = np.zeros((len(removed), len(added)))
        extra = {}
        for i, r in enumerate(removed):
            for j, a in enumerate(added):
                er, ea = np.sort(r.extent), np.sort(a.extent)
                size = float(np.mean(np.minimum(er, ea) / np.maximum(er, ea)))
                color = float(np.sum(np.sqrt(_color_hist(r.colors) * _color_hist(a.colors))))
                fit, yaw, _ = _icp_fitness(r, a)
                app = 0.5
                if embeddings is not None and r.id in embeddings and a.id in embeddings:
                    app = float(np.clip(embeddings[r.id] @ embeddings[a.id], 0, 1))
                S[i, j] = 0.4 * app + 0.2 * color + 0.2 * size + 0.2 * fit
                extra[i, j] = (yaw, app, size, color, fit)
        rows, cols = linear_sum_assignment(-S)
        for i, j in zip(rows, cols):
            r, a = removed[i], added[j]
            disp = a.centroid - r.centroid
            yaw, app, size, color, fit = extra[i, j]
            if S[i, j] < min_score or np.linalg.norm(disp) < min_distance:
                continue
            if embeddings is not None and app < 0.75:      # clearly different objects
                continue
            # yaw is only meaningful when the shapes actually align (same pose class)
            changes.append(Change(0, "moved", [r, a], displacement=disp,
                                  rotation_deg=round(yaw, 1) if fit > 0.85 and size > 0.7 else None,
                                  match_score=round(S[i, j], 3)))
            used_r.add(i)
            used_a.add(j)
    changes += [Change(0, "removed", [c]) for i, c in enumerate(removed) if i not in used_r]
    changes += [Change(0, "added", [c]) for j, c in enumerate(added) if j not in used_a]
    for k, ch in enumerate(changes):
        ch.id = k
    return changes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", type=Path, default=Path("output/map"))
    args = ap.parse_args()
    frames = {s: FrameSet.load(args.map / f"frames_{s}.pt") for s in "AB"}
    clouds = {s: o3d.io.read_point_cloud(str(args.map / f"cloud_{s}.ply")) for s in "AB"}
    removed, added, status = detect(frames, clouds)
    verifier = AppearanceVerifier(args.map.parent / "images")
    removed = verify(removed, frames, verifier, status)
    added = verify(added, frames, verifier, status)
    for ch in match_moved(removed, added):
        print(ch.summary())


if __name__ == "__main__":
    main()
