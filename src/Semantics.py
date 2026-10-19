"""
Semantic understanding of the detected changes.

* Best views: each changed 3D cluster is projected into the keyframes of the
  session in which it exists; frames where its points are actually visible
  (depth-consistent, i.e. not occluded) and large on screen are selected and
  cropped.
* Labelling: CLIP (PyTorch, zero-shot) classifies the crops against an indoor
  object vocabulary using prompt ensembling; image embeddings averaged over
  views double as an appearance descriptor for matching moved objects.
* Landmarks: OWLv2 open-vocabulary detection on a subset of keyframes finds
  room landmarks (bed, desk, door, ...), lifted to 3D with the metric depth, so
  that changes can be described relative to the room.
* Descriptions: natural-language sentences combining label, change type,
  support (floor / surface height), nearest landmarks, and displacement.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.cluster import DBSCAN

from geometry import FrameSet, backproject, project

CLIP_MODEL = "openai/clip-vit-large-patch14"
OWL_MODEL = "google/owlv2-base-patch16-ensemble"

VOCABULARY = [
    "backpack", "bag", "handbag", "suitcase", "cardboard box", "storage box", "basket",
    "laundry basket", "office chair", "chair", "stool", "table", "desk", "laptop",
    "computer monitor", "keyboard", "computer mouse", "speaker", "lamp", "desk lamp",
    "fan", "heater", "bottle", "water bottle", "mug", "cup", "bowl", "plate",
    "book", "stack of books", "notebook", "paper", "pillow", "cushion", "blanket",
    "duvet", "towel", "jacket", "coat", "hoodie", "t-shirt", "pile of clothes",
    "shoes", "sneakers", "boots", "slippers", "hat", "umbrella", "plant",
    "trash can", "plastic bag", "guitar", "ball", "toy", "teddy bear",
    "headphones", "game controller", "phone", "charger", "cable", "tripod",
    "camera", "clock", "picture frame", "bin", "printer", "router", "skateboard",
    "bicycle helmet", "hairdryer", "tote bag", "folded clothes",
]
TEMPLATES = ["a photo of a {}.", "a photo of a {} in a bedroom.", "a cropped photo of a {}.",
             "a close-up photo of a {}.", "a photo of a {} on the floor."]
LANDMARKS = ["bed", "desk", "office chair", "wardrobe", "door", "window", "radiator",
             "computer monitor", "shelf", "mirror", "chest of drawers", "sofa", "table"]


def _as_tensor(out):
    """transformers v5 may return model outputs instead of raw feature tensors."""
    if isinstance(out, torch.Tensor):
        return out
    for key in ("image_embeds", "text_embeds", "pooler_output"):
        if getattr(out, key, None) is not None:
            return getattr(out, key)
    return out[0]


# ---------------------------------------------------------------- best views
def best_views(points: np.ndarray, frames: FrameSet, k: int = 3, margin: float = 0.25,
               min_side: int = 64, mode: str = "visible"):
    """Pick the k keyframes where the cluster is most visible; return crop boxes.

    mode="visible": the cluster's surface is seen (depth agrees, not occluded).
    mode="free":    the frame looks *through* the cluster's location (used to show
                    the empty space in the other session as evidence).
    """
    P = torch.from_numpy(points).float()
    uv, z = project(P, frames)
    Fn, H, W = frames.depth.shape
    u, v = uv[..., 0].long(), uv[..., 1].long()
    inside = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    d = frames.depth[torch.arange(Fn)[:, None], v.clamp(0, H - 1), u.clamp(0, W - 1)]
    tau = 0.08 + 0.05 * z
    visible = inside & (((d - z).abs() < tau) if mode == "visible" else (d > z + tau))
    scores = []
    for f in range(Fn):
        m = visible[f]
        if m.sum() < 20:
            scores.append(-1)
            continue
        cu, cv_ = uv[f, m, 0].float().mean(), uv[f, m, 1].float().mean()
        centred = 1 - 0.5 * (abs(cu / W - 0.5) + abs(cv_ / H - 0.5))
        frac = m.float().mean()                       # unoccluded fraction
        scores.append(float(m.sum()) * float(frac) * float(centred))
    order = np.argsort(scores)[::-1]
    views = []
    for f in order[:k]:
        if scores[f] <= 0:
            break
        m = visible[f]
        x0, y0 = uv[f, m].min(0).values.tolist()
        x1, y1 = uv[f, m].max(0).values.tolist()
        w, h = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side_w = max(w * (1 + 2 * margin), min_side)
        side_h = max(h * (1 + 2 * margin), min_side)
        box = [int(max(0, cx - side_w / 2)), int(max(0, cy - side_h / 2)),
               int(min(W, cx + side_w / 2)), int(min(H, cy + side_h / 2))]
        views.append((frames.names[f], box))
    return views


# ---------------------------------------------------------------------- CLIP
class ClipLabeler:
    def __init__(self, model_name: str = CLIP_MODEL, vocabulary=VOCABULARY):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model_name).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.vocabulary = list(vocabulary)
        with torch.no_grad():
            embs = []
            for name in self.vocabulary:
                t = self.processor(text=[tp.format(name) for tp in TEMPLATES],
                                   return_tensors="pt", padding=True)
                e = _as_tensor(self.model.get_text_features(**t))
                e = torch.nn.functional.normalize(e, dim=-1).mean(0)
                embs.append(torch.nn.functional.normalize(e, dim=0))
            self.text = torch.stack(embs)

    @torch.no_grad()
    def embed(self, crops: list[np.ndarray]) -> torch.Tensor:
        inp = self.processor(images=crops, return_tensors="pt")
        e = _as_tensor(self.model.get_image_features(**inp))
        return torch.nn.functional.normalize(e, dim=-1)

    def classify(self, crops: list[np.ndarray], top: int = 3):
        """Average CLIP embedding over crops -> (label scores, unit embedding)."""
        e = torch.nn.functional.normalize(self.embed(crops).mean(0), dim=0)
        probs = (100 * e @ self.text.T).softmax(-1)
        best = probs.topk(top)
        scores = {self.vocabulary[i]: float(p) for p, i in zip(best.values, best.indices)}
        return scores, e.numpy()


def load_crop(image_root: Path, name: str, box) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(str(image_root / name)), cv2.COLOR_BGR2RGB)
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]


def embed_clusters(clusters, frames: dict[str, FrameSet], image_root: Path,
                   labeler: ClipLabeler):
    """CLIP label + embedding for each cluster from its best views."""
    info = {}
    for c in clusters:
        views = best_views(c.points, frames[c.session])
        if not views:
            continue
        crops = [load_crop(image_root, n, b) for n, b in views]
        scores, emb = labeler.classify(crops)
        info[c.id] = {"views": views, "scores": scores, "embedding": emb}
    return info


# ----------------------------------------------------------------- landmarks
@torch.no_grad()
def detect_landmarks(frames: FrameSet, image_root: Path, every: int = 4,
                     threshold: float = 0.3, queries=LANDMARKS):
    """OWLv2 detections lifted to 3D and fused across views -> [(label, xyz)]."""
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    processor = Owlv2Processor.from_pretrained(OWL_MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL).eval()
    hits = {q: [] for q in queries}
    for f in range(0, len(frames), every):
        name = frames.names[f]
        img = cv2.cvtColor(cv2.imread(str(image_root / name)), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        inputs = processor(text=[queries], images=img, return_tensors="pt")
        outputs = model(**inputs)
        side = max(H, W)          # OWLv2 pads to a square at the bottom/right
        res = processor.post_process_grounded_object_detection(
            outputs, threshold=threshold, target_sizes=torch.tensor([[side, side]]))[0]
        world, (vv, uu) = backproject(frames.depth[f], frames.K[f], frames.R[f], frames.t[f])
        dmap = torch.full((H, W, 3), float("nan"))
        dmap[vv, uu] = world
        for box, lab, score in zip(res["boxes"], res["labels"], res["scores"]):
            x0, y0, x1, y1 = [int(round(float(t))) for t in box]
            x0, x1 = max(0, x0), min(W, x1)
            y0, y1 = max(0, y0), min(H, y1)
            # central half of the box to avoid background bleeding in
            cx0, cx1 = x0 + (x1 - x0) // 4, x1 - (x1 - x0) // 4
            cy0, cy1 = y0 + (y1 - y0) // 4, y1 - (y1 - y0) // 4
            pts = dmap[cy0:cy1, cx0:cx1].reshape(-1, 3)
            pts = pts[~torch.isnan(pts[:, 0])]
            if len(pts) < 20:
                continue
            hits[queries[int(lab)]].append((pts.median(0).values.numpy(), float(score)))
    landmarks = []
    for q, hs in hits.items():
        if len(hs) < 2:
            continue
        X = np.stack([h[0] for h in hs])
        S = np.array([h[1] for h in hs])
        labels = DBSCAN(eps=0.5, min_samples=2).fit_predict(X[:, :2])
        for lab in set(labels) - {-1}:
            m = labels == lab
            landmarks.append({"label": q, "position": np.average(X[m], 0, S[m]),
                              "support": int(m.sum()), "score": float(S[m].max())})
    return landmarks


# --------------------------------------------------------------- language
PLURALS = {"headphones", "shoes", "sneakers", "boots", "slippers", "folded clothes"}


def _noun_phrase(label: str) -> str:
    if label in PLURALS:
        return f"a pair of {label}" if label != "folded clothes" else "some folded clothes"
    return ("an " if label[0].lower() in "aeiou" else "a ") + label


def _place(cluster, landmarks, label: str) -> str:
    """e.g. 'on the floor next to the desk' / 'about 0.9-1.4 m above the floor near the door'."""
    lo, hi = cluster.bbox
    c = cluster.centroid
    support = "on the floor" if lo[2] < 0.15 else f"about {lo[2]:.1f}-{hi[2]:.1f} m above the floor"
    cands = [l for l in landmarks if l["label"] != label]
    if not cands:
        return support
    d = np.array([np.linalg.norm((l["position"] - c)[:2]) for l in cands])
    order = np.argsort(d)
    near = cands[order[0]]
    if d[order[0]] < 1.0:
        rel = f"next to the {near['label']}"
    else:
        rel = f"about {d[order[0]]:.1f} m from the {near['label']}"
    if len(order) > 1 and d[order[1]] < 1.5 and cands[order[1]]["label"] != near["label"]:
        rel += f" and the {cands[order[1]]['label']}"
    return f"{support} {rel}"


def describe(change, landmarks) -> str:
    label = change.label
    conf = max(change.label_scores.values()) if change.label_scores else 0
    obj = _noun_phrase(label) if conf >= 0.25 else f"an object (possibly {_noun_phrase(label)})"
    obj = obj[0].upper() + obj[1:]
    size = " x ".join(f"{e:.2f}" for e in np.sort(change.clusters[0].extent)[::-1])
    if change.kind == "removed":
        return (f"{obj} ({size} m) was removed. In the first recording it was "
                f"{_place(change.before, landmarks, label)}; in the second recording that "
                f"space is visibly empty.")
    if change.kind == "added":
        return (f"{obj} ({size} m) was added {_place(change.after, landmarks, label)}; "
                f"the first recording saw empty space there.")
    dist = float(np.linalg.norm(change.displacement))
    return (f"{obj} was moved {dist:.2f} m. Before, it was "
            f"{_place(change.before, landmarks, label)}; now it is "
            f"{_place(change.after, landmarks, label)}.")


def annotate(changes, cluster_info, landmarks):
    """Attach labels, best views and descriptions to the changes in place."""
    for ch in changes:
        scores, views = {}, {}
        for c in ch.clusters:
            info = cluster_info.get(c.id)
            if info is None:
                continue
            views[c.session] = info["views"]
            for k, v in info["scores"].items():
                scores[k] = scores.get(k, 0) + v / len(ch.clusters)
        ch.views = views
        if scores:
            ch.label_scores = dict(sorted(scores.items(), key=lambda kv: -kv[1])[:3])
            ch.label = next(iter(ch.label_scores))
        ch.description = describe(ch, landmarks)
    return changes
