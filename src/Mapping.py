"""
Build a shared, metric, gravity-aligned 3D representation of both recordings.

1. Joint Structure-from-Motion (COLMAP via pycolmap) over the keyframes of
   *both* videos at once. Because the two sessions are matched against each
   other, every camera ends up in one common world frame - this is what makes
   the later comparison meaningful despite different viewpoints.
2. Dense metric depth for every registered keyframe with Depth Anything V2
   (Metric-Indoor, PyTorch / HuggingFace transformers).
3. Monocular depth is locally warped and inconsistent between frames, so each
   depth map is corrected to agree with the SfM sparse points: a robust (Huber
   IRLS, PyTorch) fit of a smooth spatially-varying scale field plus a shift.
4. The world is rotated so that +Z is up and the floor sits at Z = 0 (RANSAC
   floor plane), and the metric scale is fixed from a hand-held camera height
   prior (the depth network's absolute scale is unreliable on phone frames).
5. Each session's depth maps are fused into a coloured point cloud, keeping only
   points confirmed by several views of the same session (removes floaters and
   monocular hallucinations).
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap
import torch
from tqdm import tqdm

from geometry import FrameSet, backproject, visibility_stats

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
SESSIONS = ("A", "B")


# --------------------------------------------------------------------------- SfM
def run_sfm(image_root: Path, work_dir: Path, reuse: bool = True) -> pycolmap.Reconstruction:
    """Joint SfM over `image_root/A/*.jpg` and `image_root/B/*.jpg`.

    One camera per folder (= per video); exhaustive matching so that frames of
    session A are matched directly against frames of session B.
    """
    sparse = work_dir / "sparse"
    if reuse and (sparse / "images.bin").exists():
        return pycolmap.Reconstruction(sparse)
    work_dir.mkdir(parents=True, exist_ok=True)
    db = work_dir / "database.db"
    if db.exists():
        db.unlink()

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_PINHOLE"
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = 8192
    pycolmap.extract_features(db, image_root, camera_mode=pycolmap.CameraMode.PER_FOLDER,
                              reader_options=reader, extraction_options=extraction,
                              device=pycolmap.Device.cpu)

    matching = pycolmap.FeatureMatchingOptions()
    matching.guided_matching = True
    pycolmap.match_exhaustive(db, matching_options=matching, device=pycolmap.Device.cpu)

    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir()
    options = pycolmap.IncrementalPipelineOptions()
    options.min_model_size = 10
    recs = pycolmap.incremental_mapping(db, image_root, sparse, options=options)
    if not recs:
        raise RuntimeError("SfM failed to reconstruct the scene")
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    rec.write(sparse)
    return rec


def _pose(image) -> tuple[np.ndarray, np.ndarray]:
    T = image.cam_from_world()
    return np.asarray(T.rotation.matrix()), np.asarray(T.translation)


def sparse_observations(rec: pycolmap.Reconstruction):
    """Per image: (pixels [M,2], sparse depth [M]) of its triangulated points."""
    obs = {}
    for image in rec.images.values():
        if not image.has_pose:
            continue
        R, t = _pose(image)
        uv, z = [], []
        for p2d in image.points2D:
            if p2d.has_point3D():
                X = rec.points3D[p2d.point3D_id].xyz
                uv.append(p2d.xy)
                z.append((R @ X + t)[2])
        obs[image.name] = (np.array(uv, dtype=np.float32).reshape(-1, 2),
                           np.array(z, dtype=np.float32))
    return obs


# ------------------------------------------------------------------------- depth
class DepthEstimator:
    """Depth Anything V2 (metric indoor) wrapped for single-image inference."""

    def __init__(self, model_name: str = DEPTH_MODEL, device: str = "cpu"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> torch.Tensor:
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        pred = self.model(**inputs).predicted_depth            # [1, h, w] metres
        pred = torch.nn.functional.interpolate(pred[:, None], size=rgb.shape[:2],
                                               mode="bicubic", align_corners=False)
        return pred[0, 0].clamp(min=0).cpu()


def _irls(A: torch.Tensor, y: torch.Tensor, prior_rows: torch.Tensor | None = None,
          iters: int = 10) -> torch.Tensor:
    """Huber-weighted least squares of A x ~= y (optional ridge rows, target 0)."""
    w = torch.ones_like(y)
    for _ in range(iters):
        Aw, yw = A * w[:, None], y * w
        if prior_rows is not None:
            Aw = torch.cat([Aw, prior_rows])
            yw = torch.cat([yw, torch.zeros(len(prior_rows))])
        x = torch.linalg.lstsq(Aw, yw).solution
        r = (A @ x - y).abs()
        delta = 1.345 * torch.median(r).clamp(min=1e-3)
        w = torch.where(r <= delta, torch.ones_like(r), delta / r).sqrt()
    return x


def _poly_basis(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.ones_like(x), x, y, x * y, x * x, y * y], -1)


def correct_depth(pred: torch.Tensor, uv: np.ndarray, z: np.ndarray,
                  ridge: float = 0.3) -> torch.Tensor:
    """Warp a monocular depth map onto the metric SfM points of its frame.

    depth' = s(x, y) * depth + b, with s a quadratic polynomial over normalised
    image coordinates (ridge-regularised towards a constant). Falls back to an
    affine fit with few points and to the raw metric prediction with none.
    """
    H, W = pred.shape
    if len(z) < 15:
        return pred
    u = np.clip(uv[:, 0].astype(int), 0, W - 1)
    v = np.clip(uv[:, 1].astype(int), 0, H - 1)
    d = pred[v, u]
    zt = torch.from_numpy(z)
    ok = (zt > 0.1) & (d > 0.1)
    d, zt = d[ok], zt[ok]
    xs = torch.from_numpy(u[ok.numpy()] / W * 2 - 1).float()
    ys = torch.from_numpy(v[ok.numpy()] / H * 2 - 1).float()

    if len(zt) < 80:
        a, b = _irls(torch.stack([d, torch.ones_like(d)], 1), zt).tolist()
        return (a * pred + b).clamp(min=0)

    A = torch.cat([_poly_basis(xs, ys) * d[:, None], torch.ones_like(d)[:, None]], 1)
    lam = ridge * np.sqrt(len(zt)) * float(d.median())
    prior = torch.zeros(5, 7)
    prior[torch.arange(5), torch.arange(1, 6)] = lam
    coef = _irls(A, zt, prior)
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                            indexing="ij")
    scale = _poly_basis(gx, gy) @ coef[:6]
    scale = scale.clamp(0.5 * coef[0], 2.0 * coef[0])
    return (scale * pred + coef[6]).clamp(min=0)


def edge_mask(depth: torch.Tensor, rel_thresh: float = 0.06) -> torch.Tensor:
    """True on depth discontinuities (where back-projection gives flying pixels)."""
    d = depth.numpy()
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3) / 8
    g = np.sqrt(gx ** 2 + gy ** 2) / np.maximum(d, 1e-3)
    edges = cv2.dilate((g > rel_thresh).astype(np.uint8), np.ones((5, 5), np.uint8))
    return torch.from_numpy(edges > 0)


# ----------------------------------------------------------------- world frame
def gravity_alignment(R_all: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Rigid 4x4 transform taking the world to a Z-up frame with floor at Z=0.

    Initial "up" = average image-up direction of the hand-held phone cameras;
    refined by RANSAC: the lowest large near-horizontal plane is the floor.
    """
    up = np.mean([R.T @ np.array([0.0, -1.0, 0.0]) for R in R_all], axis=0)
    up /= np.linalg.norm(up)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz)).voxel_down_sample(0.05)
    n_total = len(pcd.points)
    rest, floor = pcd, None
    for _ in range(10):
        if len(rest.points) < 0.02 * n_total:
            break
        plane, inl = rest.segment_plane(0.04, 3, 3000)
        n = np.asarray(plane[:3]) / np.linalg.norm(plane[:3])
        if abs(n @ up) > np.cos(np.deg2rad(20)) and len(inl) > 0.02 * n_total:
            n = n if n @ up > 0 else -n
            h = float(np.median(np.asarray(rest.points)[inl] @ n))
            if floor is None or h < floor[1]:
                floor = (n, h)
        rest = rest.select_by_index(inl, invert=True)
    if floor is not None:
        up, height = floor
    else:  # fall back: put the lowest 2% of points on the floor
        height = float(np.percentile(xyz @ up, 2))

    z = up
    x = np.cross([0, 1, 0] if abs(z[1]) < 0.9 else [1, 0, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z])
    T[:3, 3] = -T[:3, :3] @ (z * height)
    return T


def transform_cloud(pcd: o3d.geometry.PointCloud, T: np.ndarray) -> o3d.geometry.PointCloud:
    out = o3d.geometry.PointCloud(pcd)
    out.transform(T)
    return out


# --------------------------------------------------------------------- fusion
def fuse_session(frames: FrameSet, image_root: Path, stride: int = 2,
                 voxel: float = 0.02, min_support: int = 3) -> o3d.geometry.PointCloud:
    """Back-project all depth maps of a session, keep multi-view-consistent points."""
    pts, cols, src = [], [], []
    for i, name in enumerate(frames.names):
        world, (v, u) = backproject(frames.depth[i], frames.K[i], frames.R[i],
                                    frames.t[i], stride)
        rgb = cv2.cvtColor(cv2.imread(str(image_root / name)), cv2.COLOR_BGR2RGB)
        pts.append(world)
        cols.append(torch.from_numpy(rgb[v.numpy(), u.numpy()]).float() / 255)
        src.append(torch.full((len(world),), i, dtype=torch.long))
    pts, cols, src = torch.cat(pts), torch.cat(cols), torch.cat(src)

    stats = visibility_stats(pts, frames, exclude=src)
    keep = (stats["consistent"] >= min_support) & \
           (stats["consistent"] >= 2 * stats["free"])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[keep].double().numpy()))
    pcd.colors = o3d.utility.Vector3dVector(cols[keep].double().numpy())
    pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(20, 2.0)
    return pcd


# ----------------------------------------------------------------------- main
def predict_depths(names: list[str], image_root: Path, cache: Path,
                   model_name: str = DEPTH_MODEL) -> dict[str, torch.Tensor]:
    """Raw metric depth for each keyframe, cached as .npy next to the map."""
    cache.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not (cache / (n.replace("/", "_") + ".npy")).exists()]
    if todo:
        print(f"  running {model_name} on {len(todo)} keyframes (CPU, cached afterwards)")
        estimator = DepthEstimator(model_name)
        for name in tqdm(todo):
            rgb = cv2.cvtColor(cv2.imread(str(image_root / name)), cv2.COLOR_BGR2RGB)
            np.save(cache / (name.replace("/", "_") + ".npy"), estimator(rgb).numpy())
    return {n: torch.from_numpy(np.load(cache / (n.replace("/", "_") + ".npy")))
            for n in names}


def build_map(image_root: Path, out_dir: Path, max_depth: float = 6.0,
              depth_model: str = DEPTH_MODEL, reuse: bool = True,
              camera_height: float = 1.45):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[Mapping] joint Structure-from-Motion over both sessions")
    rec = run_sfm(image_root, out_dir / "colmap", reuse)
    names = sorted(img.name for img in rec.images.values() if img.has_pose)
    by_name = {img.name: img for img in rec.images.values()}
    for s in SESSIONS:
        total = len(list((image_root / s).glob("*.jpg")))
        n = sum(name.startswith(s + "/") for name in names)
        print(f"  session {s}: {n}/{total} keyframes registered in the shared model")
    obs = sparse_observations(rec)

    print("[Mapping] dense metric depth (Depth Anything V2)")
    preds = predict_depths(names, image_root, out_dir / "depth_raw", depth_model)

    # global metric scale of the SfM model: monocular metric depth / SfM depth
    ratios = []
    for name in names:
        uv, z = obs[name]
        H, W = preds[name].shape
        d = preds[name][np.clip(uv[:, 1].astype(int), 0, H - 1),
                        np.clip(uv[:, 0].astype(int), 0, W - 1)].numpy()
        ok = (z > 0) & (d > 0)
        ratios.append(d[ok] / z[ok])
    scale = float(np.median(np.concatenate(ratios)))
    print(f"  SfM -> depth-model units scale factor: {scale:.3f}")

    depths = {}
    for name in names:
        uv, z = obs[name]
        depths[name] = correct_depth(preds[name], uv, z * scale)

    # frame sets in the (provisionally scaled) SfM world
    R_all = np.stack([_pose(by_name[n])[0] for n in names])
    t_all = np.stack([_pose(by_name[n])[1] for n in names]) * scale
    frame_sets = {}
    for s in SESSIONS:
        idx = [i for i, n in enumerate(names) if n.startswith(s + "/")]
        Ks = []
        for i in idx:
            f, cx, cy = by_name[names[i]].camera.params[:3]
            Ks.append([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        frame_sets[s] = FrameSet(names=[names[i] for i in idx],
                                 K=torch.tensor(Ks, dtype=torch.float32),
                                 R=torch.from_numpy(R_all[idx]).float(),
                                 t=torch.from_numpy(t_all[idx]).float(),
                                 depth=torch.stack([depths[names[i]] for i in idx]))

    # gravity + floor from a coarse dense fusion of both sessions
    print("[Mapping] gravity alignment and metric scale")
    coarse = np.concatenate([np.asarray(fuse_session(frame_sets[s], image_root, stride=6,
                                                     voxel=0.05).points) for s in SESSIONS])
    T = gravity_alignment(R_all, coarse)

    # Metric scale. Monocular "metric" depth is unreliable on low-res phone
    # frames with an unusual FOV (it overestimated ~2x here), so the scale is
    # fixed by a hand-held camera height prior; ceiling height is reported as an
    # independent sanity check.
    heights = np.concatenate([(fs.transformed(T).centers[:, 2]).numpy()
                              for fs in frame_sets.values()])
    s_metric = camera_height / float(np.median(heights))
    up = (T[:3, :3] @ coarse.T).T[:, 2] + T[2, 3]
    ceiling = float(np.percentile(up, 99)) * s_metric
    print(f"  depth-model scale correction x{s_metric:.3f} "
          f"(camera height prior {camera_height:.2f} m) -> ceiling at ~{ceiling:.2f} m")
    S = np.diag([s_metric] * 3 + [1.0])
    T = S @ T                                   # world_from_sfm, metric
    np.save(out_dir / "world_from_sfm.npy", T)

    print("[Mapping] fusing dense, multi-view-consistent point clouds")
    clouds = {}
    for s in SESSIONS:
        fs = frame_sets[s]
        depth = fs.depth * s_metric
        for i in range(len(depth)):
            d = depth[i]
            d[(d > max_depth) | (d < 0.1) | edge_mask(d)] = 0
        rigid = T.copy()                        # T = S @ T0 acts rigidly on the scaled world
        rigid[:3, :3] /= s_metric
        fs = FrameSet(fs.names, fs.K, fs.R, fs.t * s_metric, depth).transformed(rigid)
        frame_sets[s] = fs
        fs.save(out_dir / f"frames_{s}.pt")
        clouds[s] = fuse_session(fs, image_root)
        o3d.io.write_point_cloud(str(out_dir / f"cloud_{s}.ply"), clouds[s])
        h = fs.centers[:, 2]
        print(f"  session {s}: {len(clouds[s].points)} points, camera height "
              f"{h.min():.2f}-{h.max():.2f} m")

    xyz = np.stack([p.xyz for p in rec.points3D.values()]) * scale
    sparse_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    sparse_pcd.colors = o3d.utility.Vector3dVector(
        np.stack([p.color for p in rec.points3D.values()]) / 255.0)
    o3d.io.write_point_cloud(str(out_dir / "sparse_sfm.ply"), transform_cloud(sparse_pcd, T))
    return frame_sets, clouds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, default=Path("output/images"))
    ap.add_argument("--out", type=Path, default=Path("output/map"))
    ap.add_argument("--depth-model", default=DEPTH_MODEL)
    ap.add_argument("--camera-height", type=float, default=1.45,
                    help="typical height the phone was held at (m), fixes metric scale")
    args = ap.parse_args()
    build_map(args.images, args.out, depth_model=args.depth_model,
              camera_height=args.camera_height)


if __name__ == "__main__":
    main()
