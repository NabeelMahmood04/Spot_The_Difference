"""
Shared camera / projection utilities (PyTorch, batched over frames).

A `FrameSet` holds every keyframe of one recording session in the shared world
frame produced by joint Structure-from-Motion: intrinsics, world->camera
poses and metric depth maps. `visibility_stats` is the core primitive used both
for multi-view filtering of the fused clouds (Mapping) and for free-space change
reasoning between sessions (Detect_Changes).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class FrameSet:
    names: list[str]          # image names (relative to the image root)
    K: torch.Tensor           # [F, 3, 3] intrinsics
    R: torch.Tensor           # [F, 3, 3] world -> camera rotation
    t: torch.Tensor           # [F, 3]    world -> camera translation
    depth: torch.Tensor       # [F, H, W] metric depth, 0 = invalid

    @property
    def centers(self) -> torch.Tensor:
        """Camera centres in world coordinates, [F, 3]."""
        return -(self.R.transpose(1, 2) @ self.t[..., None])[..., 0]

    def __len__(self):
        return len(self.names)

    def save(self, path: Path):
        torch.save({k: getattr(self, k) for k in ("names", "K", "R", "t", "depth")}, path)

    @classmethod
    def load(cls, path: Path) -> "FrameSet":
        return cls(**torch.load(path, weights_only=False))

    def transformed(self, T: np.ndarray) -> "FrameSet":
        """Apply a rigid world transform x' = T[:3,:3] x + T[:3,3]."""
        Rw = torch.as_tensor(T[:3, :3], dtype=self.R.dtype)
        tw = torch.as_tensor(T[:3, 3], dtype=self.t.dtype)
        # x_cam = R x + t = R Rw^T (x' - tw) + t
        R_new = self.R @ Rw.T
        t_new = self.t - (R_new @ tw)
        return FrameSet(self.names, self.K, R_new, t_new, self.depth)


def backproject(depth: torch.Tensor, K: torch.Tensor, R: torch.Tensor,
                t: torch.Tensor, stride: int = 1):
    """Lift a depth map [H, W] to world points. Returns (points [M,3], (v, u))."""
    H, W = depth.shape
    v, u = torch.meshgrid(torch.arange(0, H, stride), torch.arange(0, W, stride),
                          indexing="ij")
    v, u = v.reshape(-1), u.reshape(-1)
    z = depth[v, u]
    keep = z > 0
    v, u, z = v[keep], u[keep], z[keep]
    x = (u.float() + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v.float() + 0.5 - K[1, 2]) / K[1, 1] * z
    cam = torch.stack([x, y, z], dim=1)
    world = (cam - t) @ R          # R^T (cam - t)
    return world, (v, u)


def project(points: torch.Tensor, frames: FrameSet):
    """Project world points [N,3] into every frame. Returns uv [F,N,2], z [F,N]."""
    cam = torch.einsum("fij,nj->fni", frames.R, points) + frames.t[:, None, :]
    z = cam[..., 2]
    pix = torch.einsum("fij,fnj->fni", frames.K, cam)
    uv = pix[..., :2] / z.clamp(min=1e-6)[..., None]
    return uv, z


def depth_tolerance(z: torch.Tensor, tau_abs: float, tau_rel: float) -> torch.Tensor:
    """Depth agreement threshold; monocular depth error grows with range."""
    return tau_abs + tau_rel * z


@torch.no_grad()
def visibility_stats(points: torch.Tensor, frames: FrameSet, tau_abs: float = 0.05,
                     tau_rel: float = 0.04, min_z: float = 0.15,
                     exclude: torch.Tensor | None = None, chunk: int = 20000,
                     border: int = 4):
    """For each world point, count across `frames` how often it is

    * consistent - observed depth agrees with the point's depth (surface seen there)
    * free       - observed depth is clearly *behind* the point: the camera saw
                   through the point's location, so the space is empty
    * occluded   - observed depth is clearly *in front* of the point (no evidence)

    Depth is compared against 3x3 min/max-pooled depth maps so that pixels on
    depth discontinuities only vote "free" when the whole neighbourhood agrees.
    `exclude` ([N] long) optionally names a frame index per point to skip (the
    frame that generated the point).
    Returns a dict of [N] int tensors: consistent, free, occluded, in_view.
    """
    depth = frames.depth
    Fn, H, W = depth.shape
    big = torch.where(depth > 0, depth, torch.full_like(depth, 1e6))
    d_min = -F.max_pool2d(-big[:, None], 3, 1, 1)[:, 0]
    d_max = F.max_pool2d(depth[:, None], 3, 1, 1)[:, 0]
    invalid = F.max_pool2d((depth <= 0).float()[:, None], 3, 1, 1)[:, 0] > 0

    out = {k: torch.zeros(len(points), dtype=torch.int32)
           for k in ("consistent", "free", "occluded", "in_view")}
    fidx = torch.arange(Fn)[:, None]
    for s in range(0, len(points), chunk):
        p = points[s:s + chunk]
        uv, z = project(p, frames)
        u = uv[..., 0].floor().long()
        v = uv[..., 1].floor().long()
        inside = (z > min_z) & (u >= border) & (u < W - border) & \
                 (v >= border) & (v < H - border)
        if exclude is not None:
            inside &= fidx != exclude[s:s + chunk][None, :]
        uc, vc = u.clamp(0, W - 1), v.clamp(0, H - 1)
        lo = d_min[fidx, vc, uc]
        hi = d_max[fidx, vc, uc]
        inside &= ~invalid[fidx, vc, uc]
        tau = depth_tolerance(z, tau_abs, tau_rel)
        free = inside & (lo > z + tau)
        occl = inside & (hi < z - tau)
        cons = inside & ~free & ~occl
        out["consistent"][s:s + chunk] = cons.sum(0).int()
        out["free"][s:s + chunk] = free.sum(0).int()
        out["occluded"][s:s + chunk] = occl.sum(0).int()
        out["in_view"][s:s + chunk] = inside.sum(0).int()
    return out
