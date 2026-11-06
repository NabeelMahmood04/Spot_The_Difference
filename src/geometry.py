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

#Project world points [N,3] into every frame. Returns uv [F,N,2], z [F,N].
def project(points : torch.tensor, frames : FrameSet):
    cam = torch.einsum("fij,nj->fni", frames.R, points) + frames.t[:, None, :]
    z = cam[..., 2]
    pix = torch.einsum("fij,fnj->fni", frames.K, cam)
    uv = pix[..., :2] / z.clamp(min=1e-6)[..., None]
    return uv, z

